import torch
import torch.nn as nn
import torch.nn.functional as F


class LogicalLayer(nn.Module):
    """
    ALRL logical layer:
    - conjunction (AND) units
    - disjunction (OR) units
    - Gumbel-Softmax feature selection
    """

    def __init__(self, input_dim, num_and, num_or, m=2):
        super().__init__()

        self.input_dim = input_dim
        self.num_and = num_and
        self.num_or = num_or
        self.m = m

        self.and_pi = nn.Parameter(
            torch.randn(num_and, m, input_dim) * 0.01
        )

        self.or_pi = nn.Parameter(
            torch.randn(num_or, m, input_dim) * 0.01
        )

    def get_selectors(self, pi, tau, training=True):
        if training:
            return F.gumbel_softmax(
                pi,
                tau=tau,
                hard=True,
                dim=-1
            )

        indexes = torch.argmax(pi, dim=-1)

        return F.one_hot(
            indexes,
            num_classes=self.input_dim
        ).float()

    def forward(self, x, tau=1.0, training=True):
        and_selector = self.get_selectors(
            self.and_pi, tau, training
        )

        or_selector = self.get_selectors(
            self.or_pi, tau, training
        )

        # [batch, units, m]
        and_selected = torch.einsum(
            "bf,umf->bum",
            x,
            and_selector
        )

        or_selected = torch.einsum(
            "bf,umf->bum",
            x,
            or_selector
        )

        # AND
        and_output = torch.prod(
            and_selected,
            dim=-1
        )

        # OR
        or_output = (
            1.0
            -
            torch.prod(
                1.0 - or_selected,
                dim=-1
            )
        )

        return torch.cat(
            [and_output, or_output],
            dim=1
        )


class SingleStageLogicalModule(nn.Module):
    """
    First logical layer:
    each physical SWaT stage learns its own internal logic.
    """

    def __init__(
        self,
        stage_dims,
        num_and=24,
        num_or=24,
        m=2
    ):
        super().__init__()

        self.stage_layers = nn.ModuleList(
            [
                LogicalLayer(
                    input_dim=dim,
                    num_and=num_and,
                    num_or=num_or,
                    m=m
                )
                for dim in stage_dims
            ]
        )

    def forward(
        self,
        stage_inputs,
        tau=1.0,
        training=True
    ):
        stage_outputs = []

        for layer, x in zip(
            self.stage_layers,
            stage_inputs
        ):
            stage_outputs.append(
                layer(
                    x,
                    tau=tau,
                    training=training
                )
            )

        # h1 = concatenate all stage outputs
        h1 = torch.cat(
            stage_outputs,
            dim=1
        )

        return h1, stage_outputs


class ALRL(nn.Module):
    """
    Current ALRL student reconstruction.

    Structure:
        binary stage inputs
            ↓
        single-stage logical learning
            ↓
        h1
            ↓
        multistage logical learning
            ↓
        h2

    Residual interpretation used here:
        final_logic = concat(h1, h2)

    This preserves simple single-stage logic while also using
    cross-stage logical features. The paper explicitly states
    that residual connections are used, but does not disclose
    a standalone residual equation, so this should be treated
    as a reconstruction rather than claimed as exact author code.
    """

    def __init__(
        self,
        stage_dims,
        num_classes,
        single_and=24,
        single_or=24,
        multi_and=64,
        multi_or=64,
        m=2
    ):
        super().__init__()

        self.stage_dims = stage_dims
        self.num_classes = num_classes

        # Layer 1: single-stage
        self.single_stage = SingleStageLogicalModule(
            stage_dims=stage_dims,
            num_and=single_and,
            num_or=single_or,
            m=m
        )

        # SWaT: 6 * (24 + 24) = 288
        self.h1_dim = (
            len(stage_dims)
            *
            (single_and + single_or)
        )

        # Layer 2: multistage
        self.multistage = LogicalLayer(
            input_dim=self.h1_dim,
            num_and=multi_and,
            num_or=multi_or,
            m=m
        )

        # 64 + 64 = 128
        self.h2_dim = (
            multi_and + multi_or
        )

        # Residual reconstruction:
        # preserve h1 and append h2.
        self.final_logic_dim = (
            self.h1_dim + self.h2_dim
        )

        # Prediction weights.
        # softplus is applied in forward so the effective
        # classifier weights are nonnegative.
        self.raw_classifier_weights = nn.Parameter(
            torch.zeros(
                num_classes,
                self.final_logic_dim
            )
        )

    def forward(
        self,
        stage_inputs,
        tau=1.0,
        training=True
    ):
        # Single-stage features
        h1, stage_outputs = self.single_stage(
            stage_inputs,
            tau=tau,
            training=training
        )

        # Cross-stage features
        h2 = self.multistage(
            h1,
            tau=tau,
            training=training
        )

        # Residual / skip path reconstruction
        final_logic = torch.cat(
            [h1, h2],
            dim=1
        )

        classifier_weights = F.softplus(
            self.raw_classifier_weights
        )

        logits = (
            final_logic
            @ classifier_weights.T
        )

        return {
            "logits": logits,
            "h1": h1,
            "h2": h2,
            "final_logic": final_logic,
            "stage_outputs": stage_outputs
        }
