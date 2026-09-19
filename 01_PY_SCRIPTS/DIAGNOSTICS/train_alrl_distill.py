import random
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
    confusion_matrix
)

from alrl_model import ALRL
from swat_data import (
    get_stage_feature_columns,
    SWaTBinaryDataset
)


# ============================================================
# ALRL Knowledge Distillation
#
# Paper objective:
#
# Lloss =
#     T^2 * alpha * Ldist
#     +
#     (1-alpha) * Lhard
#
# Paper-selected:
#     alpha = 0.2
#     T = 10
#
# Student:
#     ALRL logical model
#
# Teacher:
#     precomputed logits from BEST LU-IDS-style teacher
# ============================================================


SEED = 42

TRAIN_FILE = "ALRL_stratified_fold4_train_binary.csv"
VAL_FILE = "ALRL_stratified_fold4_val_binary.csv"

TEACHER_TRAIN_LOGITS_FILE = (
    "LU_IDS_stratified_fold4_train_logits.npy"
)

TEACHER_VAL_LOGITS_FILE = (
    "LU_IDS_stratified_fold4_val_logits.npy"
)

TEACHER_TRAIN_KEYS_FILE = (
    "LU_IDS_stratified_fold4_train_keys.csv"
)

TEACHER_VAL_KEYS_FILE = (
    "LU_IDS_stratified_fold4_val_keys.csv"
)

BEST_MODEL_FILE = (
    "ALRL_stratified_fold4_best_distill.pt"
)

FINAL_MODEL_FILE = (
    "ALRL_stratified_fold4_final_distill.pt"
)

HISTORY_FILE = (
    "ALRL_stratified_fold4_distill_history.csv"
)

BEST_CM_FILE = (
    "ALRL_stratified_fold4_distill_confusion_best.csv"
)

BEST_REPORT_FILE = (
    "ALRL_stratified_fold4_distill_report_best.csv"
)


EPOCHS = 50
BATCH_SIZE = 32

INITIAL_LR = 0.01
FINAL_LR = 0.001

INITIAL_TAU = 1.0
FINAL_TAU = 0.0001

ALPHA = 0.2
DISTILL_T = 10.0


PRACTICAL_ATTACK_IDS = [
    1, 2, 3, 4, 6, 7, 8, 10, 11, 13,
    16, 17, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41
]

CLASSES = (
    ["Normal"]
    +
    [f"Attack_{attack_id}" for attack_id in PRACTICAL_ATTACK_IDS]
)

class_to_id = {
    class_name: class_id
    for class_id, class_name in enumerate(CLASSES)
}

NUM_CLASSES = len(CLASSES)
assert NUM_CLASSES == 36


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Load Student binary data
# ============================================================

train_df = pd.read_csv(
    TRAIN_FILE,
    low_memory=False
)

val_df = pd.read_csv(
    VAL_FILE,
    low_memory=False
)

print("Train samples:", len(train_df))
print("Validation samples:", len(val_df))


# ============================================================
# Verify teacher row alignment
# ============================================================

teacher_train_keys = pd.read_csv(
    TEACHER_TRAIN_KEYS_FILE
)

teacher_val_keys = pd.read_csv(
    TEACHER_VAL_KEYS_FILE
)

for frame in [
    train_df,
    val_df,
    teacher_train_keys,
    teacher_val_keys
]:
    frame["Timestamp"] = pd.to_datetime(
        frame["Timestamp"],
        errors="raise"
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
        "Teacher TRAIN logits are not aligned "
        "with Student TRAIN rows."
    )

if not student_val_keys.equals(
    teacher_val_keys.reset_index(drop=True)
):
    raise RuntimeError(
        "Teacher VAL logits are not aligned "
        "with Student VAL rows."
    )

print("Teacher/Student train alignment: OK")
print("Teacher/Student val alignment:   OK")


# ============================================================
# Load Teacher logits
# ============================================================

teacher_train_logits = np.load(
    TEACHER_TRAIN_LOGITS_FILE
).astype(np.float32)

teacher_val_logits = np.load(
    TEACHER_VAL_LOGITS_FILE
).astype(np.float32)

if teacher_train_logits.shape != (
    len(train_df),
    NUM_CLASSES
):
    raise RuntimeError(
        f"Bad teacher train logits shape: "
        f"{teacher_train_logits.shape}"
    )

if teacher_val_logits.shape != (
    len(val_df),
    NUM_CLASSES
):
    raise RuntimeError(
        f"Bad teacher val logits shape: "
        f"{teacher_val_logits.shape}"
    )

print(
    "Teacher train logits:",
    teacher_train_logits.shape
)

print(
    "Teacher val logits:",
    teacher_val_logits.shape
)


# ============================================================
# Stage mapping
# ============================================================

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

print("Stage dimensions:", stage_dims)
print("Total binary features:", sum(stage_dims))


# ============================================================
# Dataset wrapper
# ============================================================

class DistillationDataset(Dataset):

    def __init__(
        self,
        dataframe,
        stage_columns,
        class_to_id,
        teacher_logits
    ):
        self.student = SWaTBinaryDataset(
            dataframe,
            stage_columns,
            class_to_id
        )

        self.teacher_logits = torch.from_numpy(
            teacher_logits
        )

        if len(self.student) != len(
            self.teacher_logits
        ):
            raise RuntimeError(
                "Student/teacher length mismatch."
            )

    def __len__(self):
        return len(self.student)

    def __getitem__(self, index):
        stage_inputs, label = self.student[index]

        return (
            stage_inputs,
            label,
            self.teacher_logits[index]
        )


train_dataset = DistillationDataset(
    train_df,
    stage_columns,
    class_to_id,
    teacher_train_logits
)

val_dataset = DistillationDataset(
    val_df,
    stage_columns,
    class_to_id,
    teacher_val_logits
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Student
# ============================================================

model = ALRL(
    stage_dims=stage_dims,
    num_classes=NUM_CLASSES,
    single_and=24,
    single_or=24,
    multi_and=64,
    multi_or=64,
    m=2
).to(device)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Student trainable parameters:", trainable_params)
print("Distillation alpha:", ALPHA)
print("Distillation T:", DISTILL_T)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=INITIAL_LR
)

hard_criterion = nn.CrossEntropyLoss()


def move_stages_to_device(stage_inputs):
    return [
        x.to(device)
        for x in stage_inputs
    ]


# ============================================================
# Distillation objective
# ============================================================

def compute_losses(
    student_logits,
    teacher_logits,
    labels
):
    # Lhard
    hard_loss = hard_criterion(
        student_logits,
        labels
    )

    # Teacher soft targets.
    teacher_prob = F.softmax(
        teacher_logits / DISTILL_T,
        dim=1
    )

    # Student soft distribution.
    student_log_prob = F.log_softmax(
        student_logits / DISTILL_T,
        dim=1
    )

    # Paper-style cross entropy between teacher soft target
    # and student soft prediction.
    distill_loss = -(
        teacher_prob
        *
        student_log_prob
    ).sum(
        dim=1
    ).mean()

    # Eq. (11)
    total_loss = (
        (DISTILL_T ** 2)
        *
        ALPHA
        *
        distill_loss
        +
        (1.0 - ALPHA)
        *
        hard_loss
    )

    return (
        total_loss,
        hard_loss,
        distill_loss
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader):
    model.eval()

    targets = []
    predictions = []

    total_loss_sum = 0.0
    hard_loss_sum = 0.0
    distill_loss_sum = 0.0

    with torch.no_grad():

        for (
            stage_inputs,
            labels,
            teacher_logits
        ) in loader:

            stage_inputs = move_stages_to_device(
                stage_inputs
            )

            labels = labels.to(device)

            teacher_logits = teacher_logits.to(
                device=device,
                dtype=torch.float32
            )

            output = model(
                stage_inputs,
                tau=FINAL_TAU,
                training=False
            )

            student_logits = output["logits"]

            (
                total_loss,
                hard_loss,
                distill_loss
            ) = compute_losses(
                student_logits,
                teacher_logits,
                labels
            )

            batch_size = labels.size(0)

            total_loss_sum += (
                total_loss.item() * batch_size
            )

            hard_loss_sum += (
                hard_loss.item() * batch_size
            )

            distill_loss_sum += (
                distill_loss.item() * batch_size
            )

            pred = torch.argmax(
                student_logits,
                dim=1
            )

            predictions.extend(
                pred.cpu().numpy()
            )

            targets.extend(
                labels.cpu().numpy()
            )

    y_true = np.asarray(targets)
    y_pred = np.asarray(predictions)

    n = len(loader.dataset)

    return {
        "loss":
            total_loss_sum / n,

        "hard_loss":
            hard_loss_sum / n,

        "distill_loss":
            distill_loss_sum / n,

        "accuracy":
            accuracy_score(y_true, y_pred),

        "precision":
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "targets": y_true,
        "predictions": y_pred
    }


# ============================================================
# Train Student with KD
# ============================================================

best_f1 = -1.0
history = []

print("\n===================================")
print("START ALRL KNOWLEDGE DISTILLATION")
print("===================================")

for epoch in range(EPOCHS):
    model.train()

    progress = epoch / (EPOCHS - 1)

    tau = (
        INITIAL_TAU
        *
        (
            FINAL_TAU / INITIAL_TAU
        ) ** progress
    )

    lr = (
        INITIAL_LR
        +
        (FINAL_LR - INITIAL_LR) * progress
    )

    for group in optimizer.param_groups:
        group["lr"] = lr

    total_sum = 0.0
    hard_sum = 0.0
    distill_sum = 0.0

    for (
        stage_inputs,
        labels,
        teacher_logits
    ) in train_loader:

        stage_inputs = move_stages_to_device(
            stage_inputs
        )

        labels = labels.to(device)

        teacher_logits = teacher_logits.to(
            device=device,
            dtype=torch.float32
        )

        optimizer.zero_grad()

        output = model(
            stage_inputs,
            tau=tau,
            training=True
        )

        student_logits = output["logits"]

        (
            loss,
            hard_loss,
            distill_loss
        ) = compute_losses(
            student_logits,
            teacher_logits,
            labels
        )

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_sum += (
            loss.item() * batch_size
        )

        hard_sum += (
            hard_loss.item() * batch_size
        )

        distill_sum += (
            distill_loss.item() * batch_size
        )

    n_train = len(train_dataset)

    train_total = total_sum / n_train
    train_hard = hard_sum / n_train
    train_distill = distill_sum / n_train

    val_result = evaluate(
        model,
        val_loader
    )

    print(
        f"Epoch {epoch + 1:02d}/50 | "
        f"TrainTotal={train_total:.4f} | "
        f"Hard={train_hard:.4f} | "
        f"Distill={train_distill:.4f} | "
        f"ValTotal={val_result['loss']:.4f} | "
        f"Acc={val_result['accuracy']:.4f} | "
        f"MacroF1={val_result['f1']:.4f} | "
        f"tau={tau:.6f} | "
        f"lr={lr:.5f}"
    )

    history.append(
        {
            "epoch": epoch + 1,
            "train_total_loss": train_total,
            "train_hard_loss": train_hard,
            "train_distill_loss": train_distill,
            "val_total_loss": val_result["loss"],
            "val_hard_loss": val_result["hard_loss"],
            "val_distill_loss": val_result["distill_loss"],
            "accuracy": val_result["accuracy"],
            "precision_macro": val_result["precision"],
            "recall_macro": val_result["recall"],
            "f1_macro": val_result["f1"],
            "tau": tau,
            "lr": lr
        }
    )

    if val_result["f1"] > best_f1:
        best_f1 = val_result["f1"]

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch + 1,

                "validation_accuracy":
                    val_result["accuracy"],

                "validation_f1_macro":
                    val_result["f1"],

                "alpha":
                    ALPHA,

                "distill_temperature":
                    DISTILL_T,

                "stage_dims":
                    stage_dims,

                "stage_columns":
                    stage_columns,

                "classes":
                    CLASSES,

                "class_to_id":
                    class_to_id,

                "trainable_parameters":
                    trainable_params,

                "seed":
                    SEED
            },
            BEST_MODEL_FILE
        )


torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "epoch":
            EPOCHS,

        "alpha":
            ALPHA,

        "distill_temperature":
            DISTILL_T,

        "stage_dims":
            stage_dims,

        "stage_columns":
            stage_columns,

        "classes":
            CLASSES,

        "class_to_id":
            class_to_id,

        "trainable_parameters":
            trainable_params,

        "seed":
            SEED
    },
    FINAL_MODEL_FILE
)

pd.DataFrame(
    history
).to_csv(
    HISTORY_FILE,
    index=False
)


# ============================================================
# Best distilled Student
# ============================================================

checkpoint = torch.load(
    BEST_MODEL_FILE,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

best_result = evaluate(
    model,
    val_loader
)

print("\n===================================")
print("BEST DISTILLED STUDENT RESULT")
print("===================================")

print("Best epoch:", checkpoint["epoch"])
print(f"Accuracy:        {best_result['accuracy']:.4f}")
print(f"Macro Precision: {best_result['precision']:.4f}")
print(f"Macro Recall:    {best_result['recall']:.4f}")
print(f"Macro F1:        {best_result['f1']:.4f}")
print(f"alpha:           {ALPHA}")
print(f"T:               {DISTILL_T}")


cm = confusion_matrix(
    best_result["targets"],
    best_result["predictions"],
    labels=list(range(NUM_CLASSES))
)

pd.DataFrame(
    cm,
    index=CLASSES,
    columns=CLASSES
).to_csv(
    BEST_CM_FILE
)

report = classification_report(
    best_result["targets"],
    best_result["predictions"],
    labels=list(range(NUM_CLASSES)),
    target_names=CLASSES,
    zero_division=0,
    output_dict=True
)

pd.DataFrame(
    report
).transpose().to_csv(
    BEST_REPORT_FILE
)


print("\n===================================")
print("BEST DISTILLED PER-CLASS METRICS")
print("===================================")

for class_name in CLASSES:
    metrics = report[class_name]

    print(
        f"{class_name:12s} | "
        f"P={metrics['precision']:.3f} | "
        f"R={metrics['recall']:.3f} | "
        f"F1={metrics['f1-score']:.3f} | "
        f"N={int(metrics['support'])}"
    )


print("\n===================================")
print("BASELINE COMPARISON")
print("===================================")

print("No-distillation baseline:")
print("Accuracy = 0.9068")
print("Macro F1 = 0.8517")

print(
    "Distilled Macro F1 gain = "
    f"{best_result['f1'] - 0.8517:+.4f}"
)
