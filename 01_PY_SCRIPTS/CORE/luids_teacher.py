import numpy as np
import torch
import torch.nn as nn


# ============================================================
# LU-IDS Teacher Reconstruction
#
# Paper-disclosed parts:
#   1) raw ICS features are encoded with IEEE754
#   2) encoded input goes into a residual network
#   3) the residual network has 4 residual blocks
#   4) each residual block has 2 convolution layers
#   5) each convolution uses a 3 x 1 kernel
#
# Paper-undisclosed parts reconstructed here:
#   - channel width = 64
#   - BatchNorm + ReLU
#   - one 3x1 stem convolution
#   - flatten + linear classifier
#
# We deliberately KEEP spatial/device/bit positions before the
# classifier instead of global-average-pooling them away.
#
# This is a transparent LU-IDS-style reconstruction,
# not a claim of the exact private author implementation.
# ============================================================


def ieee754_encode_numpy(values):
    """
    Convert float values into IEEE754 single-precision bit vectors.

    Input:
        [N, F] or [F]

    Output:
        uint8 bits [N, F, 32] or [F, 32]
    """

    x = np.asarray(values, dtype=np.float32)

    # Reinterpret the same 32 raw bits as uint32.
    u = x.view(np.uint32)

    shifts = np.arange(
        31, -1, -1,
        dtype=np.uint32
    )

    bits = (
        (u[..., None] >> shifts) & 1
    ).astype(np.uint8)

    return bits


class ResidualBlock(nn.Module):
    """Two 3x1 convolutions plus identity shortcut."""

    def __init__(self, channels=64):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class LUIDSTeacher(nn.Module):
    """
    SWaT input:
        [batch, 51 devices, 32 IEEE754 bits]

    Internal:
        [B, 1, 51, 32]
        -> stem 3x1 convolution
        -> 4 residual blocks
        -> flatten while preserving positions
        -> 36-class classifier
    """

    def __init__(
        self,
        num_classes=36,
        channels=64,
        num_devices=51,
        ieee_bits=32
    ):
        super().__init__()

        self.num_classes = num_classes
        self.channels = channels
        self.num_devices = num_devices
        self.ieee_bits = ieee_bits

        self.stem = nn.Sequential(
            nn.Conv2d(
                1,
                channels,
                kernel_size=(3, 1),
                padding=(1, 0),
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.Sequential(
            ResidualBlock(channels),
            ResidualBlock(channels),
            ResidualBlock(channels),
            ResidualBlock(channels)
        )

        flattened_dim = (
            channels
            * num_devices
            * ieee_bits
        )

        self.classifier = nn.Linear(
            flattened_dim,
            num_classes
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"Expected [B, F, 32], got {tuple(x.shape)}"
            )

        if x.shape[1] != self.num_devices:
            raise ValueError(
                f"Expected {self.num_devices} devices, "
                f"got {x.shape[1]}"
            )

        if x.shape[2] != self.ieee_bits:
            raise ValueError(
                f"Expected {self.ieee_bits} IEEE754 bits, "
                f"got {x.shape[2]}"
            )

        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.blocks(x)

        # Preserve absolute device and IEEE754-bit positions.
        x = torch.flatten(x, 1)

        return self.classifier(x)
