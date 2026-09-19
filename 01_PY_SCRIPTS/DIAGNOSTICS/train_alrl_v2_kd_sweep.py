import gc
import json
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from alrl_model import ALRL
from swat_data import (
    get_stage_feature_columns,
    SWaTBinaryDataset,
)


# ============================================================
# Focused KD sweep for ALRL V2
#
# Motivation:
#   Reconstructed Teacher logits are much larger than the
#   undisclosed original LU-IDS logits.
#
#   Diagnostic:
#       T=10 -> mean top1 probability ~0.847  (still sharp)
#       T=20 -> mean top1 probability ~0.400  (much softer)
#
# Therefore, instead of blindly using only paper's alpha=0.2,
# T=10, this script performs a SMALL controlled sweep.
#
# Phase A: screen several (alpha, T) settings for first 20 epochs
#          of the SAME 50-epoch lr/tau schedule.
#
# Phase B: retrain the two best KD settings for full 50 epochs.
#
# Existing reference:
#   V2 no-KD best Macro-F1 = 0.9864
#   Teacher Macro-F1       = 0.9949
#
# This is a reconstruction diagnostic, NOT a claim that these
# hyperparameters are the authors' settings.
# ============================================================


SEED = 42

TRAIN_FILE = "ALRL_v2_fold4_train_binary.csv"
VAL_FILE = "ALRL_v2_fold4_val_binary.csv"

TEACHER_TRAIN_LOGITS_FILE = "LU_IDS_stratified_fold4_train_logits.npy"
TEACHER_VAL_LOGITS_FILE = "LU_IDS_stratified_fold4_val_logits.npy"

TEACHER_TRAIN_KEYS_FILE = "LU_IDS_stratified_fold4_train_keys.csv"
TEACHER_VAL_KEYS_FILE = "LU_IDS_stratified_fold4_val_keys.csv"

SCREEN_RESULTS_FILE = "ALRL_v2_kd_screen_results.csv"
FULL_RESULTS_FILE = "ALRL_v2_kd_full_results.csv"
SUMMARY_FILE = "ALRL_v2_kd_sweep_summary.txt"

SCREEN_EPOCHS = 20
FULL_EPOCHS = 50

BATCH_SIZE = 32

INITIAL_LR = 0.01
FINAL_LR = 0.001

INITIAL_TAU = 1.0
FINAL_TAU = 0.0001

V2_NODISTILL_F1 = 0.9864
V2_NODISTILL_ACC = 0.9932
TEACHER_F1 = 0.9949


# ------------------------------------------------------------
# Focused search space
#
# alpha=0 is included as a short-run reference.
#
# Based on Teacher softness:
#   T=10 is too sharp
#   T=20 is useful middle ground
#   T=30 tests a slightly softer target
#
# Because V2 no-KD is already very strong, alpha values are kept
# smaller than paper's 0.2 for most candidates.
# ------------------------------------------------------------

SCREEN_CONFIGS = [
    {"alpha": 0.00, "T": 20.0},  # no-KD short-run reference

    {"alpha": 0.02, "T": 10.0},
    {"alpha": 0.05, "T": 10.0},

    {"alpha": 0.02, "T": 20.0},
    {"alpha": 0.05, "T": 20.0},
    {"alpha": 0.10, "T": 20.0},

    {"alpha": 0.02, "T": 30.0},
    {"alpha": 0.05, "T": 30.0},
    {"alpha": 0.10, "T": 30.0},
]


PRACTICAL_ATTACK_IDS = [
    1, 2, 3, 4, 6, 7, 8, 10, 11, 13,
    16, 17, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41,
]

CLASSES = ["Normal"] + [
    f"Attack_{attack_id}"
    for attack_id in PRACTICAL_ATTACK_IDS
]

CLASS_TO_ID = {
    class_name: class_id
    for class_id, class_name in enumerate(CLASSES)
}

NUM_CLASSES = len(CLASSES)
assert NUM_CLASSES == 36


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Load data once
# ============================================================

train_df = pd.read_csv(
    TRAIN_FILE,
    low_memory=False,
)

val_df = pd.read_csv(
    VAL_FILE,
    low_memory=False,
)

teacher_train_keys = pd.read_csv(
    TEACHER_TRAIN_KEYS_FILE,
)

teacher_val_keys = pd.read_csv(
    TEACHER_VAL_KEYS_FILE,
)

for frame in [
    train_df,
    val_df,
    teacher_train_keys,
    teacher_val_keys,
]:
    frame["Timestamp"] = pd.to_datetime(
        frame["Timestamp"],
        errors="raise",
    )


student_train_keys = (
    train_df[
        ["Timestamp", "Attack_Type", "Fold"]
    ]
    .reset_index(drop=True)
)

student_val_keys = (
    val_df[
        ["Timestamp", "Attack_Type", "Fold"]
    ]
    .reset_index(drop=True)
)

if not student_train_keys.equals(
    teacher_train_keys.reset_index(drop=True)
):
    raise RuntimeError(
        "Teacher TRAIN logits are not aligned with V2 Student rows."
    )

if not student_val_keys.equals(
    teacher_val_keys.reset_index(drop=True)
):
    raise RuntimeError(
        "Teacher VAL logits are not aligned with V2 Student rows."
    )


teacher_train_logits = np.load(
    TEACHER_TRAIN_LOGITS_FILE
).astype(np.float32)

teacher_val_logits = np.load(
    TEACHER_VAL_LOGITS_FILE
).astype(np.float32)

assert teacher_train_logits.shape == (
    len(train_df),
    NUM_CLASSES,
)

assert teacher_val_logits.shape == (
    len(val_df),
    NUM_CLASSES,
)


stage_columns = get_stage_feature_columns(
    train_df
)

val_stage_columns = get_stage_feature_columns(
    val_df
)

for stage in range(1, 7):
    if stage_columns[stage] != val_stage_columns[stage]:
        raise RuntimeError(
            f"Stage {stage} feature mismatch."
        )

stage_dims = [
    len(stage_columns[stage])
    for stage in range(1, 7)
]

assert sum(stage_dims) == 298

print("Train rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Stage dimensions:", stage_dims)
print("Total V2 binary features:", sum(stage_dims))


# ============================================================
# Dataset
# ============================================================

class DistillationDataset(Dataset):

    def __init__(
        self,
        dataframe,
        stage_columns,
        class_to_id,
        teacher_logits,
    ):
        self.student = SWaTBinaryDataset(
            dataframe,
            stage_columns,
            class_to_id,
        )

        self.teacher_logits = torch.from_numpy(
            teacher_logits
        )

        if len(self.student) != len(self.teacher_logits):
            raise RuntimeError(
                "Student/teacher dataset length mismatch."
            )

    def __len__(self):
        return len(self.student)

    def __getitem__(self, index):
        stage_inputs, label = self.student[index]

        return (
            stage_inputs,
            label,
            self.teacher_logits[index],
        )


train_dataset = DistillationDataset(
    train_df,
    stage_columns,
    CLASS_TO_ID,
    teacher_train_logits,
)

val_dataset = DistillationDataset(
    val_df,
    stage_columns,
    CLASS_TO_ID,
    teacher_val_logits,
)


def build_loaders(seed=SEED):
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


# ============================================================
# Model / loss helpers
# ============================================================

def build_model():
    return ALRL(
        stage_dims=stage_dims,
        num_classes=NUM_CLASSES,
        single_and=24,
        single_or=24,
        multi_and=64,
        multi_or=64,
        m=2,
    ).to(device)


hard_criterion = nn.CrossEntropyLoss()


def move_stages(stage_inputs):
    return [
        x.to(device)
        for x in stage_inputs
    ]


def compute_losses(
    student_logits,
    teacher_logits,
    labels,
    alpha,
    T,
):
    hard_loss = hard_criterion(
        student_logits,
        labels,
    )

    # For alpha=0, avoid unnecessary KD computation.
    if alpha == 0.0:
        zero = torch.zeros(
            (),
            device=student_logits.device,
            dtype=student_logits.dtype,
        )

        return (
            hard_loss,
            hard_loss,
            zero,
        )

    teacher_prob = F.softmax(
        teacher_logits / T,
        dim=1,
    )

    student_log_prob = F.log_softmax(
        student_logits / T,
        dim=1,
    )

    # Same paper-style soft cross-entropy form used previously.
    distill_loss = -(
        teacher_prob
        *
        student_log_prob
    ).sum(
        dim=1
    ).mean()

    total_loss = (
        (T ** 2)
        *
        alpha
        *
        distill_loss
        +
        (1.0 - alpha)
        *
        hard_loss
    )

    return (
        total_loss,
        hard_loss,
        distill_loss,
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    loader,
    alpha,
    T,
):
    model.eval()

    y_true = []
    y_pred = []

    total_loss_sum = 0.0
    hard_loss_sum = 0.0
    distill_loss_sum = 0.0

    with torch.no_grad():

        for (
            stage_inputs,
            labels,
            teacher_logits,
        ) in loader:

            stage_inputs = move_stages(
                stage_inputs
            )

            labels = labels.to(
                device
            )

            teacher_logits = teacher_logits.to(
                device=device,
                dtype=torch.float32,
            )

            output = model(
                stage_inputs,
                tau=FINAL_TAU,
                training=False,
            )

            student_logits = output["logits"]

            (
                total_loss,
                hard_loss,
                distill_loss,
            ) = compute_losses(
                student_logits,
                teacher_logits,
                labels,
                alpha,
                T,
            )

            n = labels.size(0)

            total_loss_sum += total_loss.item() * n
            hard_loss_sum += hard_loss.item() * n
            distill_loss_sum += distill_loss.item() * n

            pred = torch.argmax(
                student_logits,
                dim=1,
            )

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                pred.cpu().numpy()
            )

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    n_total = len(
        loader.dataset
    )

    return {
        "loss":
            total_loss_sum / n_total,

        "hard_loss":
            hard_loss_sum / n_total,

        "distill_loss":
            distill_loss_sum / n_total,

        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "targets":
            y_true,

        "predictions":
            y_pred,
    }


# ============================================================
# One training run
#
# schedule_total_epochs remains 50 even for the 20-epoch
# screen. Therefore screening epochs exactly match the first
# 20 epochs of the full 50-epoch lr/tau schedule.
# ============================================================

def run_training(
    alpha,
    T,
    epochs_to_run,
    save_prefix=None,
):

    set_seed(SEED)

    train_loader, val_loader = build_loaders(
        SEED
    )

    model = build_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=INITIAL_LR,
    )

    best_f1 = -1.0
    best_epoch = None
    best_state = None
    best_result = None

    history = []

    print("\n-----------------------------------")
    print(
        f"RUN alpha={alpha:.3f}, T={T:.1f}, "
        f"epochs={epochs_to_run}"
    )
    print("-----------------------------------")

    for epoch in range(
        epochs_to_run
    ):

        model.train()

        # IMPORTANT:
        # schedule is always based on the original 50 epochs.
        progress = (
            epoch
            /
            (FULL_EPOCHS - 1)
        )

        tau = (
            INITIAL_TAU
            *
            (
                FINAL_TAU
                /
                INITIAL_TAU
            )
            **
            progress
        )

        lr = (
            INITIAL_LR
            +
            (
                FINAL_LR
                -
                INITIAL_LR
            )
            *
            progress
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        total_sum = 0.0
        hard_sum = 0.0
        distill_sum = 0.0

        for (
            stage_inputs,
            labels,
            teacher_logits,
        ) in train_loader:

            stage_inputs = move_stages(
                stage_inputs
            )

            labels = labels.to(
                device
            )

            teacher_logits = teacher_logits.to(
                device=device,
                dtype=torch.float32,
            )

            optimizer.zero_grad()

            output = model(
                stage_inputs,
                tau=tau,
                training=True,
            )

            student_logits = output["logits"]

            (
                loss,
                hard_loss,
                distill_loss,
            ) = compute_losses(
                student_logits,
                teacher_logits,
                labels,
                alpha,
                T,
            )

            loss.backward()
            optimizer.step()

            n = labels.size(0)

            total_sum += loss.item() * n
            hard_sum += hard_loss.item() * n
            distill_sum += distill_loss.item() * n

        validation = evaluate(
            model,
            val_loader,
            alpha,
            T,
        )

        train_n = len(
            train_dataset
        )

        row = {
            "epoch":
                epoch + 1,

            "train_total":
                total_sum / train_n,

            "train_hard":
                hard_sum / train_n,

            "train_distill":
                distill_sum / train_n,

            "val_total":
                validation["loss"],

            "val_hard":
                validation["hard_loss"],

            "val_distill":
                validation["distill_loss"],

            "accuracy":
                validation["accuracy"],

            "macro_f1":
                validation["f1"],

            "tau":
                tau,

            "lr":
                lr,
        }

        history.append(
            row
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Acc={validation['accuracy']:.4f} | "
            f"F1={validation['f1']:.4f} | "
            f"Hard={validation['hard_loss']:.4f} | "
            f"tau={tau:.6f}"
        )

        if validation["f1"] > best_f1:

            best_f1 = validation["f1"]
            best_epoch = epoch + 1
            best_result = deepcopy(
                validation
            )

            if save_prefix is not None:
                best_state = {
                    key:
                        value.detach().cpu().clone()
                    for key, value
                    in model.state_dict().items()
                }

    if save_prefix is not None:

        checkpoint_file = (
            f"{save_prefix}_best.pt"
        )

        history_file = (
            f"{save_prefix}_history.csv"
        )

        report_file = (
            f"{save_prefix}_classification_report.csv"
        )

        cm_file = (
            f"{save_prefix}_confusion_matrix.csv"
        )

        torch.save(
            {
                "model_state_dict":
                    best_state,

                "alpha":
                    alpha,

                "T":
                    T,

                "best_epoch":
                    best_epoch,

                "best_accuracy":
                    best_result["accuracy"],

                "best_macro_f1":
                    best_result["f1"],

                "stage_dims":
                    stage_dims,

                "stage_columns":
                    stage_columns,

                "classes":
                    CLASSES,

                "class_to_id":
                    CLASS_TO_ID,

                "seed":
                    SEED,
            },
            checkpoint_file,
        )

        pd.DataFrame(
            history
        ).to_csv(
            history_file,
            index=False,
        )

        report = classification_report(
            best_result["targets"],
            best_result["predictions"],
            labels=list(range(NUM_CLASSES)),
            target_names=CLASSES,
            zero_division=0,
            output_dict=True,
        )

        pd.DataFrame(
            report
        ).transpose().to_csv(
            report_file
        )

        cm = confusion_matrix(
            best_result["targets"],
            best_result["predictions"],
            labels=list(range(NUM_CLASSES)),
        )

        pd.DataFrame(
            cm,
            index=CLASSES,
            columns=CLASSES,
        ).to_csv(
            cm_file
        )

    result = {
        "alpha":
            alpha,

        "T":
            T,

        "epochs_run":
            epochs_to_run,

        "best_epoch":
            best_epoch,

        "best_accuracy":
            best_result["accuracy"],

        "best_precision_macro":
            best_result["precision"],

        "best_recall_macro":
            best_result["recall"],

        "best_macro_f1":
            best_result["f1"],
    }

    del model
    del optimizer
    del train_loader
    del val_loader

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# Phase A: screen
# ============================================================

print("\n===================================")
print("PHASE A: 20-EPOCH KD SCREEN")
print("===================================")

screen_results = []

for config in SCREEN_CONFIGS:

    result = run_training(
        alpha=config["alpha"],
        T=config["T"],
        epochs_to_run=SCREEN_EPOCHS,
        save_prefix=None,
    )

    screen_results.append(
        result
    )


screen_df = pd.DataFrame(
    screen_results
)

screen_df = screen_df.sort_values(
    "best_macro_f1",
    ascending=False,
).reset_index(
    drop=True
)

screen_df.to_csv(
    SCREEN_RESULTS_FILE,
    index=False,
)


print("\n===================================")
print("SCREEN RANKING")
print("===================================")

print(
    screen_df.to_string(
        index=False
    )
)


# ============================================================
# Pick top 2 actual KD configs (exclude alpha=0)
# ============================================================

kd_only = screen_df[
    screen_df["alpha"] > 0
].copy()

top_configs = (
    kd_only
    .head(2)
    [
        ["alpha", "T"]
    ]
    .to_dict(
        orient="records"
    )
)


# ============================================================
# Phase B: full 50 epochs for top two
# ============================================================

print("\n===================================")
print("PHASE B: FULL 50-EPOCH RETRAIN")
print("===================================")

full_results = []

for rank, config in enumerate(
    top_configs,
    start=1,
):

    alpha = float(
        config["alpha"]
    )

    T = float(
        config["T"]
    )

    prefix = (
        f"ALRL_v2_kd_rank{rank}"
        f"_a{alpha:.3f}"
        f"_T{int(T)}"
    )

    result = run_training(
        alpha=alpha,
        T=T,
        epochs_to_run=FULL_EPOCHS,
        save_prefix=prefix,
    )

    result["rank_from_screen"] = rank

    full_results.append(
        result
    )


full_df = pd.DataFrame(
    full_results
)

full_df = full_df.sort_values(
    "best_macro_f1",
    ascending=False,
).reset_index(
    drop=True
)

full_df.to_csv(
    FULL_RESULTS_FILE,
    index=False,
)


print("\n===================================")
print("FULL-RUN RANKING")
print("===================================")

print(
    full_df.to_string(
        index=False
    )
)


# ============================================================
# Final interpretation
# ============================================================

best_kd_f1 = float(
    full_df.iloc[0][
        "best_macro_f1"
    ]
)

best_kd_acc = float(
    full_df.iloc[0][
        "best_accuracy"
    ]
)

best_alpha = float(
    full_df.iloc[0][
        "alpha"
    ]
)

best_T = float(
    full_df.iloc[0][
        "T"
    ]
)


print("\n===================================")
print("FINAL KD SWEEP SUMMARY")
print("===================================")

print(
    f"Best KD alpha: {best_alpha}"
)

print(
    f"Best KD T:     {best_T}"
)

print(
    f"Best KD Accuracy: {best_kd_acc:.4f}"
)

print(
    f"Best KD Macro F1: {best_kd_f1:.4f}"
)

print(
    f"\nV2 no-KD Macro F1: "
    f"{V2_NODISTILL_F1:.4f}"
)

print(
    f"KD gain over V2 no-KD: "
    f"{best_kd_f1 - V2_NODISTILL_F1:+.4f}"
)

print(
    f"Teacher Macro F1: "
    f"{TEACHER_F1:.4f}"
)


if best_kd_f1 > V2_NODISTILL_F1:
    conclusion = (
        "KD improves the reconstructed V2 Student."
    )
else:
    conclusion = (
        "No tested KD setting beats the V2 no-KD Student; "
        "retain V2 no-KD as the best reconstructed Student."
    )


print(
    "\nConclusion:",
    conclusion
)


summary_lines = [
    "ALRL V2 KD SWEEP SUMMARY",
    "=========================",
    "",
    f"V2 no-KD Accuracy: {V2_NODISTILL_ACC:.4f}",
    f"V2 no-KD Macro F1: {V2_NODISTILL_F1:.4f}",
    f"Teacher Macro F1: {TEACHER_F1:.4f}",
    "",
    f"Best KD alpha: {best_alpha}",
    f"Best KD T: {best_T}",
    f"Best KD Accuracy: {best_kd_acc:.6f}",
    f"Best KD Macro F1: {best_kd_f1:.6f}",
    (
        "KD Macro F1 gain over V2 no-KD: "
        f"{best_kd_f1 - V2_NODISTILL_F1:+.6f}"
    ),
    "",
    conclusion,
]


Path(
    SUMMARY_FILE
).write_text(
    "\n".join(
        summary_lines
    ),
    encoding="utf-8",
)


print("\nSaved:")
print(SCREEN_RESULTS_FILE)
print(FULL_RESULTS_FILE)
print(SUMMARY_FILE)
