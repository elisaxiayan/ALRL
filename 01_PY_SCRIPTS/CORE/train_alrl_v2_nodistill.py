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
# ALRL V2 Student Baseline (NO Knowledge Distillation)
#
# Inputs:
#   ALRL_v2_fold4_train_binary.csv
#   ALRL_v2_fold4_val_binary.csv
#
# Purpose:
#   Test whether the richer V2 binarization itself improves ALRL.
#
# Everything else remains the same as the previous stratified
# no-distillation baseline:
#   - same Fold 4
#   - same 36 classes
#   - same ALRL architecture
#   - 24 AND + 24 OR units per physical stage
#   - 64 AND + 64 OR multistage units
#   - m = 2
#   - Adam
#   - batch size 32
#   - 50 epochs
#   - lr 0.01 -> 0.001
#   - Gumbel tau 1 -> 0.0001
#
# IMPORTANT:
#   alpha = 0 here.
#   We intentionally do NOT use Teacher/KD yet, so V2 can be
#   compared directly with the old V1 baseline Macro-F1 0.8517.
# ============================================================


SEED = 42

TRAIN_FILE = "ALRL_v2_fold4_train_binary.csv"
VAL_FILE = "ALRL_v2_fold4_val_binary.csv"

BEST_MODEL_FILE = "ALRL_v2_fold4_best_nodistill.pt"
FINAL_MODEL_FILE = "ALRL_v2_fold4_final_nodistill.pt"

HISTORY_FILE = "ALRL_v2_fold4_training_history.csv"

BEST_CM_FILE = "ALRL_v2_fold4_confusion_matrix_best.csv"
FINAL_CM_FILE = "ALRL_v2_fold4_confusion_matrix_final.csv"

BEST_REPORT_FILE = "ALRL_v2_fold4_classification_report_best.csv"
FINAL_REPORT_FILE = "ALRL_v2_fold4_classification_report_final.csv"

CLASS_MAPPING_FILE = "ALRL_v2_class_mapping.json"


EPOCHS = 50
BATCH_SIZE = 32

INITIAL_LR = 0.01
FINAL_LR = 0.001

INITIAL_TAU = 1.0
FINAL_TAU = 0.0001


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

V1_BASELINE_ACCURACY = 0.9068
V1_BASELINE_MACRO_F1 = 0.8517


# ============================================================
# 0. Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# 1. Load V2 binary data
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
# 2. Stable class mapping
# ============================================================

expected_classes = set(
    CLASSES
)

train_classes = set(
    train_df["Attack_Type"].unique()
)

val_classes = set(
    val_df["Attack_Type"].unique()
)


if train_classes != expected_classes:
    raise ValueError(
        "Training class set mismatch.\n"
        f"Missing: {sorted(expected_classes - train_classes)}\n"
        f"Unexpected: {sorted(train_classes - expected_classes)}"
    )


if val_classes != expected_classes:
    raise ValueError(
        "Validation class set mismatch.\n"
        f"Missing: {sorted(expected_classes - val_classes)}\n"
        f"Unexpected: {sorted(val_classes - expected_classes)}"
    )


class_to_id = {
    class_name: class_id
    for class_id, class_name in enumerate(CLASSES)
}

NUM_CLASSES = len(
    CLASSES
)

assert NUM_CLASSES == 36


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


print(
    "Class count:",
    NUM_CLASSES
)


# ============================================================
# 3. Split V2 binary features into 6 physical stages
#
# swat_data.py groups by device prefixes, so newly added
# SENSOR_ATTACK_RANGE features stay in their original stage.
# ============================================================

stage_columns = get_stage_feature_columns(
    train_df
)

val_stage_columns = get_stage_feature_columns(
    val_df
)


for stage in range(1, 7):

    if (
        stage_columns[stage]
        !=
        val_stage_columns[stage]
    ):

        raise ValueError(
            f"Train/validation feature mismatch in Stage {stage}"
        )


stage_dims = [
    len(
        stage_columns[stage]
    )
    for stage in range(1, 7)
]


total_features = sum(
    stage_dims
)


print(
    "Stage dimensions:",
    stage_dims
)

print(
    "Total binary features:",
    total_features
)


assert total_features == 298, (
    f"Expected 298 V2 binary features, got {total_features}"
)


# ============================================================
# 4. Dataset / DataLoader
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
# 5. ALRL Student
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
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print(
    "Trainable parameters:",
    trainable_params
)


# ============================================================
# 6. Optimizer / hard classification loss
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=INITIAL_LR
)

criterion = nn.CrossEntropyLoss()


def move_stages_to_device(
    stage_inputs
):

    return [
        tensor.to(device)
        for tensor in stage_inputs
    ]


# ============================================================
# 7. Evaluation
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()

    predictions = []
    targets = []

    total_loss = 0.0


    with torch.no_grad():

        for (
            stage_inputs,
            labels
        ) in loader:

            stage_inputs = (
                move_stages_to_device(
                    stage_inputs
                )
            )

            labels = labels.to(
                device
            )


            output = model(
                stage_inputs,
                tau=FINAL_TAU,
                training=False
            )


            logits = output[
                "logits"
            ]


            loss = criterion(
                logits,
                labels
            )


            total_loss += (
                loss.item()
                *
                labels.size(0)
            )


            prediction = torch.argmax(
                logits,
                dim=1
            )


            predictions.extend(
                prediction
                .cpu()
                .numpy()
            )

            targets.extend(
                labels
                .cpu()
                .numpy()
            )


    predictions = np.asarray(
        predictions
    )

    targets = np.asarray(
        targets
    )


    return {
        "loss":
            total_loss
            /
            len(
                loader.dataset
            ),

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


# ============================================================
# 8. Save detailed metrics
# ============================================================

def save_detailed_metrics(
    result,
    confusion_file,
    report_file
):

    cm = confusion_matrix(
        result[
            "targets"
        ],
        result[
            "predictions"
        ],
        labels=list(
            range(
                NUM_CLASSES
            )
        )
    )


    pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES
    ).to_csv(
        confusion_file
    )


    report = classification_report(
        result[
            "targets"
        ],
        result[
            "predictions"
        ],
        labels=list(
            range(
                NUM_CLASSES
            )
        ),
        target_names=CLASSES,
        zero_division=0,
        output_dict=True
    )


    pd.DataFrame(
        report
    ).transpose().to_csv(
        report_file
    )


# ============================================================
# 9. Train
# ============================================================

best_f1 = -1.0

history = []


print("\n===================================")
print("START ALRL V2 NO-DISTILL TRAINING")
print("===================================")


for epoch in range(
    EPOCHS
):

    model.train()


    progress = (
        epoch
        /
        (
            EPOCHS
            -
            1
        )
    )


    # Gumbel temperature:
    # 1.0 -> 0.0001
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


    # Linear learning-rate decay:
    # 0.01 -> 0.001
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


    for group in (
        optimizer.param_groups
    ):

        group[
            "lr"
        ] = lr


    running_loss = 0.0


    for (
        stage_inputs,
        labels
    ) in train_loader:

        stage_inputs = (
            move_stages_to_device(
                stage_inputs
            )
        )

        labels = labels.to(
            device
        )


        optimizer.zero_grad()


        output = model(
            stage_inputs,
            tau=tau,
            training=True
        )


        logits = output[
            "logits"
        ]


        # alpha = 0 baseline:
        # hard-label classification only
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
        len(
            train_dataset
        )
    )


    validation = evaluate(
        model,
        val_loader
    )


    print(
        f"Epoch {epoch + 1:02d}/50 | "
        f"TrainLoss={train_loss:.4f} | "
        f"ValLoss={validation['loss']:.4f} | "
        f"Acc={validation['accuracy']:.4f} | "
        f"MacroF1={validation['f1']:.4f} | "
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
                validation[
                    "loss"
                ],

            "accuracy":
                validation[
                    "accuracy"
                ],

            "precision_macro":
                validation[
                    "precision"
                ],

            "recall_macro":
                validation[
                    "recall"
                ],

            "f1_macro":
                validation[
                    "f1"
                ],

            "tau":
                tau,

            "lr":
                lr
        }
    )


    if (
        validation[
            "f1"
        ]
        >
        best_f1
    ):

        best_f1 = (
            validation[
                "f1"
            ]
        )


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch + 1,

                "validation_accuracy":
                    validation[
                        "accuracy"
                    ],

                "validation_f1_macro":
                    validation[
                        "f1"
                    ],

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

                "total_binary_features":
                    total_features,

                "seed":
                    SEED
            },

            BEST_MODEL_FILE
        )


# ============================================================
# 10. Save final checkpoint / history
# ============================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "epoch":
            EPOCHS,

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

        "total_binary_features":
            total_features,

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
# 11. Final-epoch evaluation
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
# 12. Best checkpoint evaluation
# ============================================================

checkpoint = torch.load(
    BEST_MODEL_FILE,
    map_location=device,
    weights_only=False
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
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
# 13. Summary
# ============================================================

print("\n===================================")
print("BEST V2 VALIDATION RESULT")
print("===================================")

print(
    "Best epoch:",
    checkpoint[
        "epoch"
    ]
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
print("FINAL V2 EPOCH RESULT")
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
# 14. Per-class best result
# ============================================================

print("\n===================================")
print("BEST V2 PER-CLASS METRICS")
print("===================================")


report = classification_report(
    best_result[
        "targets"
    ],
    best_result[
        "predictions"
    ],
    labels=list(
        range(
            NUM_CLASSES
        )
    ),
    target_names=CLASSES,
    zero_division=0,
    output_dict=True
)


for class_name in CLASSES:

    metrics = report[
        class_name
    ]


    print(
        f"{class_name:12s} | "
        f"P={metrics['precision']:.3f} | "
        f"R={metrics['recall']:.3f} | "
        f"F1={metrics['f1-score']:.3f} | "
        f"N={int(metrics['support'])}"
    )


# ============================================================
# 15. Direct V1 vs V2 comparison
# ============================================================

print("\n===================================")
print("V1 vs V2 BASELINE COMPARISON")
print("===================================")

print(
    "V1 no-distill Accuracy:",
    f"{V1_BASELINE_ACCURACY:.4f}"
)

print(
    "V2 no-distill Accuracy:",
    f"{best_result['accuracy']:.4f}"
)

print(
    "Accuracy gain:",
    f"{best_result['accuracy'] - V1_BASELINE_ACCURACY:+.4f}"
)


print(
    "\nV1 no-distill Macro F1:",
    f"{V1_BASELINE_MACRO_F1:.4f}"
)

print(
    "V2 no-distill Macro F1:",
    f"{best_result['f1']:.4f}"
)

print(
    "Macro F1 gain:",
    f"{best_result['f1'] - V1_BASELINE_MACRO_F1:+.4f}"
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

    print(
        filename
    )
