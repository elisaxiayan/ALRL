import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from alrl_model import ALRL
from swat_data import (
    get_stage_feature_columns,
    SWaTBinaryDataset
)


# ============================================================
# Paper-like ALRL Student Training (No Distillation)
#
# Inputs:
#   ALRL_paper_like_fold4_train_binary.csv
#   ALRL_paper_like_fold4_val_binary.csv
#
# Model:
#   alrl_model.py
#
# Current model reconstruction:
#   Single-stage -> h1
#   Multistage  -> h2
#   Residual reconstruction -> concat(h1, h2)
#   Prediction -> 36 classes
#
# This script deliberately trains WITHOUT knowledge distillation.
# It is the alpha=0 Student baseline.
#
# Paper settings used:
#   epochs = 50
#   batch size = 32
#   optimizer = Adam
#   learning rate = 0.01 -> 0.001 linearly
#   Gumbel temperature = 1 -> 0.0001
#
# Outputs:
#   ALRL_paper_like_fold4_best_nodistill.pt
#   ALRL_paper_like_fold4_final_nodistill.pt
#   ALRL_paper_like_fold4_training_history.csv
#   ALRL_paper_like_fold4_confusion_matrix_best.csv
#   ALRL_paper_like_fold4_confusion_matrix_final.csv
#   ALRL_paper_like_fold4_classification_report_best.csv
#   ALRL_paper_like_fold4_classification_report_final.csv
#   ALRL_paper_like_class_mapping.json
# ============================================================


# ============================================================
# 0. Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 1. Files / hyperparameters
# ============================================================

TRAIN_FILE = "ALRL_paper_like_fold4_train_binary.csv"
VAL_FILE = "ALRL_paper_like_fold4_val_binary.csv"

BEST_MODEL_FILE = "ALRL_paper_like_fold4_best_nodistill.pt"
FINAL_MODEL_FILE = "ALRL_paper_like_fold4_final_nodistill.pt"

HISTORY_FILE = "ALRL_paper_like_fold4_training_history.csv"

BEST_CM_FILE = "ALRL_paper_like_fold4_confusion_matrix_best.csv"
FINAL_CM_FILE = "ALRL_paper_like_fold4_confusion_matrix_final.csv"

BEST_REPORT_FILE = "ALRL_paper_like_fold4_classification_report_best.csv"
FINAL_REPORT_FILE = "ALRL_paper_like_fold4_classification_report_final.csv"

CLASS_MAPPING_FILE = "ALRL_paper_like_class_mapping.json"


EPOCHS = 50
BATCH_SIZE = 32

INITIAL_LR = 0.01
FINAL_LR = 0.001

INITIAL_TAU = 1.0
FINAL_TAU = 0.0001


# Explicit, stable class order.
PRACTICAL_ATTACK_IDS = [
    1, 2, 3, 4, 6, 7, 8, 10, 11, 13,
    16, 17, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41
]

CLASSES = (
    ["Normal"]
    +
    [
        f"Attack_{attack_id}"
        for attack_id in PRACTICAL_ATTACK_IDS
    ]
)


# ============================================================
# 2. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# 3. Load binary data
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

print(
    "Train Normal:",
    (train_df["Attack_Type"] == "Normal").sum()
)

print(
    "Validation Normal:",
    (val_df["Attack_Type"] == "Normal").sum()
)


# ============================================================
# 4. Verify class set and mapping
# ============================================================

observed_train = set(
    train_df["Attack_Type"].unique()
)

observed_val = set(
    val_df["Attack_Type"].unique()
)

expected = set(CLASSES)

if observed_train != expected:
    raise ValueError(
        "Training class set mismatch.\n"
        f"Missing: {sorted(expected - observed_train)}\n"
        f"Unexpected: {sorted(observed_train - expected)}"
    )

if observed_val != expected:
    raise ValueError(
        "Validation class set mismatch.\n"
        f"Missing: {sorted(expected - observed_val)}\n"
        f"Unexpected: {sorted(observed_val - expected)}"
    )


class_to_id = {
    class_name: class_id
    for class_id, class_name in enumerate(CLASSES)
}

id_to_class = {
    class_id: class_name
    for class_name, class_id in class_to_id.items()
}

num_classes = len(CLASSES)

assert num_classes == 36


with open(
    CLASS_MAPPING_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        class_to_id,
        f,
        indent=2
    )


print("Class count:", num_classes)
print("Class order:", CLASSES)


# ============================================================
# 5. Split 125 binary features into 6 physical stages
# ============================================================

stage_columns = get_stage_feature_columns(
    train_df
)

stage_dims = [
    len(stage_columns[stage])
    for stage in range(1, 7)
]


print("Stage dimensions:", stage_dims)
print("Total binary features:", sum(stage_dims))

assert sum(stage_dims) == 125


# Make sure val has exactly the same stage features.
val_stage_columns = get_stage_feature_columns(
    val_df
)

for stage in range(1, 7):
    if stage_columns[stage] != val_stage_columns[stage]:
        raise ValueError(
            f"Train/validation feature mismatch in Stage {stage}"
        )


# ============================================================
# 6. Dataset / DataLoader
# ============================================================

train_dataset = SWaTBinaryDataset(
    train_df,
    stage_columns,
    class_to_id
)

val_dataset = SWaTBinaryDataset(
    val_df,
    stage_columns,
    class_to_id
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
# 7. ALRL Student
# ============================================================

model = ALRL(
    stage_dims=stage_dims,
    num_classes=num_classes,

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

print("Trainable parameters:", trainable_params)


# ============================================================
# 8. Optimizer / hard-label loss
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=INITIAL_LR
)

criterion = nn.CrossEntropyLoss()


# ============================================================
# Helper
# ============================================================

def move_stages_to_device(stage_inputs):
    return [
        x.to(device)
        for x in stage_inputs
    ]


# ============================================================
# 9. Evaluation
# ============================================================

def evaluate(model, loader):

    model.eval()

    predictions = []
    targets = []

    total_loss = 0.0


    with torch.no_grad():

        for stage_inputs, labels in loader:

            stage_inputs = move_stages_to_device(
                stage_inputs
            )

            labels = labels.to(device)


            output = model(
                stage_inputs,
                tau=FINAL_TAU,
                training=False
            )

            logits = output["logits"]


            loss = criterion(
                logits,
                labels
            )

            total_loss += (
                loss.item()
                *
                labels.size(0)
            )


            pred = torch.argmax(
                logits,
                dim=1
            )


            predictions.extend(
                pred.cpu().numpy()
            )

            targets.extend(
                labels.cpu().numpy()
            )


    predictions = np.asarray(
        predictions
    )

    targets = np.asarray(
        targets
    )


    result = {
        "loss":
            total_loss / len(loader.dataset),

        "accuracy":
            accuracy_score(
                targets,
                predictions
            ),

        "precision":
            precision_score(
                targets,
                predictions,
                average="macro",
                zero_division=0
            ),

        "recall":
            recall_score(
                targets,
                predictions,
                average="macro",
                zero_division=0
            ),

        "f1":
            f1_score(
                targets,
                predictions,
                average="macro",
                zero_division=0
            ),

        "targets":
            targets,

        "predictions":
            predictions
    }

    return result


# ============================================================
# 10. Save detailed metrics
# ============================================================

def save_detailed_metrics(
    result,
    cm_file,
    report_file
):

    targets = result["targets"]
    predictions = result["predictions"]


    cm = confusion_matrix(
        targets,
        predictions,
        labels=list(
            range(num_classes)
        )
    )


    cm_df = pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES
    )

    cm_df.to_csv(
        cm_file
    )


    report = classification_report(
        targets,
        predictions,
        labels=list(
            range(num_classes)
        ),
        target_names=CLASSES,
        zero_division=0,
        output_dict=True
    )


    report_df = (
        pd.DataFrame(report)
        .transpose()
    )

    report_df.to_csv(
        report_file
    )


# ============================================================
# 11. Training
# ============================================================

best_f1 = -1.0
best_epoch = None

history = []


print("\n===================================")
print("START PAPER-LIKE ALRL TRAINING")
print("===================================")


for epoch in range(EPOCHS):

    model.train()


    progress = (
        epoch
        /
        (EPOCHS - 1)
    )


    # Gumbel temperature:
    # 1 -> 0.0001
    tau = (
        INITIAL_TAU
        *
        (
            FINAL_TAU
            /
            INITIAL_TAU
        ) ** progress
    )


    # Learning rate:
    # 0.01 -> 0.001 linearly
    lr = (
        INITIAL_LR
        +
        (
            FINAL_LR
            -
            INITIAL_LR
        )
        * progress
    )


    for group in optimizer.param_groups:
        group["lr"] = lr


    running_loss = 0.0


    for stage_inputs, labels in train_loader:

        stage_inputs = move_stages_to_device(
            stage_inputs
        )

        labels = labels.to(device)


        optimizer.zero_grad()


        output = model(
            stage_inputs,
            tau=tau,
            training=True
        )


        logits = output["logits"]


        # alpha = 0 baseline:
        # only hard classification loss
        loss = criterion(
            logits,
            labels
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            *
            labels.size(0)
        )


    train_loss = (
        running_loss
        /
        len(train_dataset)
    )


    val_result = evaluate(
        model,
        val_loader
    )


    print(
        f"Epoch {epoch + 1:02d}/50 | "
        f"TrainLoss={train_loss:.4f} | "
        f"ValLoss={val_result['loss']:.4f} | "
        f"Acc={val_result['accuracy']:.4f} | "
        f"MacroF1={val_result['f1']:.4f} | "
        f"tau={tau:.6f} | "
        f"lr={lr:.5f}"
    )


    history.append(
        {
            "epoch":
                epoch + 1,

            "train_loss":
                train_loss,

            "val_loss":
                val_result["loss"],

            "accuracy":
                val_result["accuracy"],

            "precision_macro":
                val_result["precision"],

            "recall_macro":
                val_result["recall"],

            "f1_macro":
                val_result["f1"],

            "tau":
                tau,

            "lr":
                lr
        }
    )


    # Save best validation Macro-F1 checkpoint.
    if val_result["f1"] > best_f1:

        best_f1 = val_result["f1"]
        best_epoch = epoch + 1

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "stage_dims":
                    stage_dims,

                "stage_columns":
                    stage_columns,

                "classes":
                    CLASSES,

                "class_to_id":
                    class_to_id,

                "epoch":
                    best_epoch,

                "validation_accuracy":
                    val_result["accuracy"],

                "validation_f1_macro":
                    best_f1,

                "trainable_parameters":
                    trainable_params,

                "seed":
                    SEED
            },

            BEST_MODEL_FILE
        )


# ============================================================
# 12. Save final epoch checkpoint
# ============================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "stage_dims":
            stage_dims,

        "stage_columns":
            stage_columns,

        "classes":
            CLASSES,

        "class_to_id":
            class_to_id,

        "epoch":
            EPOCHS,

        "trainable_parameters":
            trainable_params,

        "seed":
            SEED
    },

    FINAL_MODEL_FILE
)


# ============================================================
# 13. Save training history
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    HISTORY_FILE,
    index=False
)


# ============================================================
# 14. Evaluate FINAL epoch
# ============================================================

final_result = evaluate(
    model,
    val_loader
)

save_detailed_metrics(
    final_result,
    FINAL_CM_FILE,
    FINAL_REPORT_FILE
)


# ============================================================
# 15. Reload and evaluate BEST epoch
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

save_detailed_metrics(
    best_result,
    BEST_CM_FILE,
    BEST_REPORT_FILE
)


# ============================================================
# 16. Print summary
# ============================================================

print("\n===================================")
print("BEST VALIDATION RESULT")
print("===================================")

print(
    "Best epoch:",
    checkpoint["epoch"]
)

print(
    f"Accuracy:        "
    f"{best_result['accuracy']:.4f}"
)

print(
    f"Macro Precision: "
    f"{best_result['precision']:.4f}"
)

print(
    f"Macro Recall:    "
    f"{best_result['recall']:.4f}"
)

print(
    f"Macro F1:        "
    f"{best_result['f1']:.4f}"
)


print("\n===================================")
print("FINAL EPOCH RESULT")
print("===================================")

print(
    f"Accuracy:        "
    f"{final_result['accuracy']:.4f}"
)

print(
    f"Macro Precision: "
    f"{final_result['precision']:.4f}"
)

print(
    f"Macro Recall:    "
    f"{final_result['recall']:.4f}"
)

print(
    f"Macro F1:        "
    f"{final_result['f1']:.4f}"
)


# ============================================================
# 17. Per-class BEST performance
# ============================================================

print("\n===================================")
print("BEST MODEL PER-CLASS METRICS")
print("===================================")


report = classification_report(
    best_result["targets"],
    best_result["predictions"],
    labels=list(range(num_classes)),
    target_names=CLASSES,
    zero_division=0,
    output_dict=True
)


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
print("FILES SAVED")
print("===================================")

for filename in [
    BEST_MODEL_FILE,
    FINAL_MODEL_FILE,
    HISTORY_FILE,
    BEST_CM_FILE,
    FINAL_CM_FILE,
    BEST_REPORT_FILE,
    FINAL_REPORT_FILE,
    CLASS_MAPPING_FILE
]:
    print(filename)
