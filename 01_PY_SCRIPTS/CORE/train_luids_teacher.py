import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from luids_teacher import (
    LUIDSTeacher,
    ieee754_encode_numpy
)


# ============================================================
# Train LU-IDS-style Teacher on stratified Fold 4
# ============================================================

SEED = 42
VALIDATION_FOLD = 4

RAW_FILE = "ALRL_paper_like_stratified_5fold.csv"

STUDENT_TRAIN_FILE = "ALRL_stratified_fold4_train_binary.csv"
STUDENT_VAL_FILE = "ALRL_stratified_fold4_val_binary.csv"

BEST_MODEL_FILE = "LU_IDS_stratified_fold4_best.pt"
FINAL_MODEL_FILE = "LU_IDS_stratified_fold4_final.pt"
HISTORY_FILE = "LU_IDS_stratified_fold4_history.csv"

TRAIN_LOGITS_FILE = "LU_IDS_stratified_fold4_train_logits.npy"
VAL_LOGITS_FILE = "LU_IDS_stratified_fold4_val_logits.npy"

TRAIN_KEYS_FILE = "LU_IDS_stratified_fold4_train_keys.csv"
VAL_KEYS_FILE = "LU_IDS_stratified_fold4_val_keys.csv"

CLASS_MAPPING_FILE = "LU_IDS_stratified_class_mapping.json"


# Paper does not disclose a complete LU-IDS optimizer schedule.
# Reconstruction used here:
# Adam, batch 32, 50 epochs, lr 0.001 -> 0.0001.
EPOCHS = 50
BATCH_SIZE = 32
INITIAL_LR = 0.001
FINAL_LR = 0.0001
CHANNELS = 64


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


SENSORS = [
    "FIT101", "LIT101",
    "AIT201", "AIT202", "AIT203", "FIT201",
    "DPIT301", "FIT301", "LIT301",
    "AIT401", "AIT402", "FIT401", "LIT401",
    "AIT501", "AIT502", "AIT503", "AIT504",
    "FIT501", "FIT502", "FIT503", "FIT504",
    "PIT501", "PIT502", "PIT503",
    "FIT601"
]

ACTUATORS = [
    "MV101", "P101", "P102",
    "MV201", "P201", "P202", "P203", "P204", "P205", "P206",
    "MV301", "MV302", "MV303", "MV304", "P301", "P302",
    "P401", "P402", "P403", "P404", "UV401",
    "P501", "P502",
    "P601", "P602", "P603"
]

DEVICE_COLUMNS = SENSORS + ACTUATORS
assert len(DEVICE_COLUMNS) == 51


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


class IEEE754Dataset(Dataset):
    def __init__(
        self,
        dataframe,
        device_columns,
        class_to_id
    ):
        values = dataframe[
            device_columns
        ].to_numpy(dtype=np.float32)

        bits = ieee754_encode_numpy(values)

        if bits.shape[1:] != (51, 32):
            raise RuntimeError(
                f"Unexpected IEEE754 shape: {bits.shape}"
            )

        self.bits = torch.from_numpy(bits)

        labels = (
            dataframe["Attack_Type"]
            .map(class_to_id)
            .to_numpy(dtype=np.int64)
        )

        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.bits[index],
            self.labels[index]
        )


# ============================================================
# Load raw stratified data
# ============================================================

raw_df = pd.read_csv(
    RAW_FILE,
    low_memory=False
)

raw_df["Timestamp"] = pd.to_datetime(
    raw_df["Timestamp"],
    errors="raise"
)

train_df = raw_df[
    raw_df["Fold"] != VALIDATION_FOLD
].copy()

val_df = raw_df[
    raw_df["Fold"] == VALIDATION_FOLD
].copy()


print("Raw train samples:", len(train_df))
print("Raw validation samples:", len(val_df))
print("Device count:", len(DEVICE_COLUMNS))
print("Class count:", NUM_CLASSES)


# ============================================================
# Verify exact row alignment with Student binary data
# ============================================================

student_train_keys = pd.read_csv(
    STUDENT_TRAIN_FILE,
    usecols=["Timestamp", "Attack_Type", "Fold"]
)

student_val_keys = pd.read_csv(
    STUDENT_VAL_FILE,
    usecols=["Timestamp", "Attack_Type", "Fold"]
)

student_train_keys["Timestamp"] = pd.to_datetime(
    student_train_keys["Timestamp"],
    errors="raise"
)

student_val_keys["Timestamp"] = pd.to_datetime(
    student_val_keys["Timestamp"],
    errors="raise"
)

raw_train_keys = (
    train_df[
        ["Timestamp", "Attack_Type", "Fold"]
    ]
    .reset_index(drop=True)
)

raw_val_keys = (
    val_df[
        ["Timestamp", "Attack_Type", "Fold"]
    ]
    .reset_index(drop=True)
)

if not raw_train_keys.equals(
    student_train_keys.reset_index(drop=True)
):
    raise RuntimeError(
        "Teacher raw TRAIN rows are not aligned "
        "with Student binary TRAIN rows."
    )

if not raw_val_keys.equals(
    student_val_keys.reset_index(drop=True)
):
    raise RuntimeError(
        "Teacher raw VAL rows are not aligned "
        "with Student binary VAL rows."
    )

print("Teacher/Student train row alignment: OK")
print("Teacher/Student val row alignment:   OK")


raw_train_keys.to_csv(
    TRAIN_KEYS_FILE,
    index=False
)

raw_val_keys.to_csv(
    VAL_KEYS_FILE,
    index=False
)

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


# ============================================================
# Data
# ============================================================

train_dataset = IEEE754Dataset(
    train_df,
    DEVICE_COLUMNS,
    class_to_id
)

val_dataset = IEEE754Dataset(
    val_df,
    DEVICE_COLUMNS,
    class_to_id
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# Model
# ============================================================

model = LUIDSTeacher(
    num_classes=NUM_CLASSES,
    channels=CHANNELS
).to(device)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Teacher trainable parameters:", trainable_params)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=INITIAL_LR
)

criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()

    all_targets = []
    all_predictions = []
    total_loss = 0.0

    with torch.no_grad():
        for bits, labels in loader:
            bits = bits.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            logits = model(bits)
            loss = criterion(logits, labels)

            total_loss += (
                loss.item() * labels.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_targets.extend(
                labels.cpu().numpy()
            )
            all_predictions.extend(
                predictions.cpu().numpy()
            )

    y_true = np.asarray(all_targets)
    y_pred = np.asarray(all_predictions)

    return {
        "loss":
            total_loss / len(loader.dataset),

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


best_f1 = -1.0
history = []

print("\n===================================")
print("START LU-IDS TEACHER TRAINING")
print("===================================")

for epoch in range(EPOCHS):
    model.train()

    progress = epoch / (EPOCHS - 1)

    lr = (
        INITIAL_LR
        +
        (FINAL_LR - INITIAL_LR) * progress
    )

    for group in optimizer.param_groups:
        group["lr"] = lr

    running_loss = 0.0

    for bits, labels in train_loader:
        bits = bits.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        logits = model(bits)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += (
            loss.item() * labels.size(0)
        )

    train_loss = (
        running_loss / len(train_dataset)
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
        f"lr={lr:.6f}"
    )

    history.append(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_result["loss"],
            "accuracy": val_result["accuracy"],
            "precision_macro": val_result["precision"],
            "recall_macro": val_result["recall"],
            "f1_macro": val_result["f1"],
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

                "classes":
                    CLASSES,

                "class_to_id":
                    class_to_id,

                "device_columns":
                    DEVICE_COLUMNS,

                "channels":
                    CHANNELS,

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

        "classes":
            CLASSES,

        "class_to_id":
            class_to_id,

        "device_columns":
            DEVICE_COLUMNS,

        "channels":
            CHANNELS,

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
print("BEST TEACHER RESULT")
print("===================================")

print("Best epoch:", checkpoint["epoch"])
print(f"Accuracy:        {best_result['accuracy']:.4f}")
print(f"Macro Precision: {best_result['precision']:.4f}")
print(f"Macro Recall:    {best_result['recall']:.4f}")
print(f"Macro F1:        {best_result['f1']:.4f}")


# ============================================================
# Collect BEST teacher logits in exact Student row order
# ============================================================

def collect_logits(model, dataset):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    model.eval()
    chunks = []

    with torch.no_grad():
        for bits, _ in loader:
            bits = bits.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True
            )

            logits = model(bits)

            chunks.append(
                logits.cpu()
                .numpy()
                .astype(np.float32)
            )

    return np.concatenate(
        chunks,
        axis=0
    )


train_logits = collect_logits(
    model,
    train_dataset
)

val_logits = collect_logits(
    model,
    val_dataset
)

assert train_logits.shape == (
    len(train_dataset),
    NUM_CLASSES
)

assert val_logits.shape == (
    len(val_dataset),
    NUM_CLASSES
)

np.save(
    TRAIN_LOGITS_FILE,
    train_logits
)

np.save(
    VAL_LOGITS_FILE,
    val_logits
)

print("\n===================================")
print("TEACHER LOGITS SAVED")
print("===================================")

print(
    TRAIN_LOGITS_FILE,
    train_logits.shape
)

print(
    VAL_LOGITS_FILE,
    val_logits.shape
)


print("\n===================================")
print("BEST TEACHER PER-CLASS METRICS")
print("===================================")

report = classification_report(
    best_result["targets"],
    best_result["predictions"],
    labels=list(range(NUM_CLASSES)),
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
