import json
import re
import numpy as np
import pandas as pd

SEED = 42
INPUT_FILE = "ALRL_paper_like_47044_5fold.csv"

FOLD_OUTPUT = "ALRL_paper_like_stratified_5fold.csv"
TRAIN_OUTPUT = "ALRL_stratified_fold4_train_binary.csv"
VAL_OUTPUT = "ALRL_stratified_fold4_val_binary.csv"
CONFIG_OUTPUT = "ALRL_stratified_fold4_binarization_config.json"

VALIDATION_FOLD = 4
LOW_QUANTILE = 0.05
HIGH_QUANTILE = 0.95

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

assert len(SENSORS) == 25
assert len(ACTUATORS) == 26


def state_to_text(value):
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if float(value).is_integer():
            return str(int(value))
        text = f"{float(value):.10g}"
        return re.sub(r"[^0-9A-Za-z]+", "_", text)
    return re.sub(r"[^0-9A-Za-z]+", "_", str(value))


print("===================================")
print("1. LOAD PAPER-LIKE DATASET")
print("===================================")

df = pd.read_csv(INPUT_FILE, low_memory=False)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="raise")

if "Fold" in df.columns:
    df = df.drop(columns=["Fold"])

print("Input shape without old Fold:", df.shape)
print("Total:", len(df))
print("Normal:", (df["Attack_Type"] == "Normal").sum())
print("Attack:", (df["Attack_Type"] != "Normal").sum())
print("Classes:", df["Attack_Type"].nunique())

assert len(df) == 47044
assert (df["Attack_Type"] == "Normal").sum() == 26877
assert (df["Attack_Type"] != "Normal").sum() == 20167
assert df["Attack_Type"].nunique() == 36

print("\n===================================")
print("2. BUILD SHUFFLED STRATIFIED FOLDS")
print("===================================")

rng = np.random.default_rng(SEED)
fold_parts = []

for class_name, group in df.groupby("Attack_Type", sort=False):
    group = group.copy()

    unique_times = group["Timestamp"].drop_duplicates().to_numpy()
    rng.shuffle(unique_times)

    time_to_fold = {}
    for rank, timestamp in enumerate(unique_times):
        time_to_fold[pd.Timestamp(timestamp)] = int(rank % 5)

    group["Fold"] = group["Timestamp"].map(time_to_fold).astype(int)
    fold_parts.append(group)

df = pd.concat(fold_parts, ignore_index=True)

timestamp_fold_counts = (
    df.groupby(["Attack_Type", "Timestamp"])["Fold"].nunique()
)
leak_count = int((timestamp_fold_counts > 1).sum())

print("Exact timestamp leakage across folds:", leak_count)
assert leak_count == 0

print("\nFold totals:")
print(df["Fold"].value_counts().sort_index().to_string())

print("\nNormal per fold:")
print(
    df[df["Attack_Type"] == "Normal"]["Fold"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nAttack per fold:")
print(
    df[df["Attack_Type"] != "Normal"]["Fold"]
    .value_counts()
    .sort_index()
    .to_string()
)

class_fold_table = pd.crosstab(df["Attack_Type"], df["Fold"])
missing_class_fold_cells = int((class_fold_table == 0).sum().sum())

print("\nMissing class/fold cells:", missing_class_fold_cells)
assert missing_class_fold_cells == 0

df.to_csv(FOLD_OUTPUT, index=False)

train = df[df["Fold"] != VALIDATION_FOLD].copy()
val = df[df["Fold"] == VALIDATION_FOLD].copy()

print("\n===================================")
print("3. FOLD 4 TRAIN / VALIDATION")
print("===================================")

print("Train rows:", len(train))
print("Validation rows:", len(val))
print("Train Normal:", (train["Attack_Type"] == "Normal").sum())
print("Validation Normal:", (val["Attack_Type"] == "Normal").sum())
print("Train Attack:", (train["Attack_Type"] != "Normal").sum())
print("Validation Attack:", (val["Attack_Type"] != "Normal").sum())

print("\n===================================")
print("4. FIT SENSOR BOUNDARIES")
print("===================================")

normal_train = train[train["Attack_Type"] == "Normal"].copy()
sensor_config = {}

for sensor in SENSORS:
    values = pd.to_numeric(normal_train[sensor], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Invalid sensor values in {sensor}")

    low = float(values.quantile(LOW_QUANTILE))
    high = float(values.quantile(HIGH_QUANTILE))

    sensor_config[sensor] = {
        "low": low,
        "high": high
    }

print("\n===================================")
print("5. FIT ACTUATOR STATES")
print("===================================")

actuator_config = {}

for actuator in ACTUATORS:
    values = pd.to_numeric(train[actuator], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Invalid actuator values in {actuator}")

    states = sorted(values.unique().tolist())

    actuator_config[actuator] = {
        "states": [float(x) for x in states]
    }

config = {
    "source_file": INPUT_FILE,
    "fold_file": FOLD_OUTPUT,
    "fold_method": "class-stratified shuffled unique-timestamp groups",
    "seed": SEED,
    "validation_fold": VALIDATION_FOLD,
    "sensor_method": "train_normal_q05_q95_reconstruction",
    "low_quantile": LOW_QUANTILE,
    "high_quantile": HIGH_QUANTILE,
    "sensors": sensor_config,
    "actuators": actuator_config
}

with open(CONFIG_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)


def transform_to_binary(data):
    metadata = (
        data[["Timestamp", "Attack_Type", "Fold"]]
        .reset_index(drop=True)
    )

    sensor_columns = {}

    for sensor in SENSORS:
        values = pd.to_numeric(
            data[sensor],
            errors="raise"
        ).reset_index(drop=True)

        low = sensor_config[sensor]["low"]
        high = sensor_config[sensor]["high"]

        sensor_columns[f"{sensor}_LOW"] = (values < low).astype("int8")
        sensor_columns[f"{sensor}_NORMAL"] = (
            (values >= low) & (values <= high)
        ).astype("int8")
        sensor_columns[f"{sensor}_HIGH"] = (values > high).astype("int8")

    actuator_columns = {}

    for actuator in ACTUATORS:
        values = pd.to_numeric(
            data[actuator],
            errors="raise"
        ).reset_index(drop=True)

        train_states = actuator_config[actuator]["states"]

        observed_states = set(
            float(x) for x in values.unique().tolist()
        )
        allowed_states = set(
            float(x) for x in train_states
        )

        unseen = observed_states - allowed_states
        if unseen:
            raise ValueError(
                f"{actuator} has unseen states in validation: {sorted(unseen)}"
            )

        for state in train_states:
            state_text = state_to_text(state)
            actuator_columns[f"{actuator}_EQ_{state_text}"] = (
                values == state
            ).astype("int8")

    return pd.concat(
        [
            metadata,
            pd.DataFrame(sensor_columns),
            pd.DataFrame(actuator_columns)
        ],
        axis=1
    )


print("\n===================================")
print("6. BINARIZE FOLD 4")
print("===================================")

train_binary = transform_to_binary(train)
val_binary = transform_to_binary(val)

feature_columns = [
    col for col in train_binary.columns
    if col not in {"Timestamp", "Attack_Type", "Fold"}
]

print("Train binary shape:", train_binary.shape)
print("Validation binary shape:", val_binary.shape)
print("Binary feature count:", len(feature_columns))

for sensor in SENSORS:
    cols = [
        f"{sensor}_LOW",
        f"{sensor}_NORMAL",
        f"{sensor}_HIGH"
    ]

    assert (train_binary[cols].sum(axis=1) == 1).all()
    assert (val_binary[cols].sum(axis=1) == 1).all()

for actuator in ACTUATORS:
    cols = [
        col for col in feature_columns
        if col.startswith(f"{actuator}_EQ_")
    ]

    assert (train_binary[cols].sum(axis=1) == 1).all()
    assert (val_binary[cols].sum(axis=1) == 1).all()

train_binary.to_csv(TRAIN_OUTPUT, index=False)
val_binary.to_csv(VAL_OUTPUT, index=False)

print("\n===================================")
print("7. STAGE 1 AUDIT")
print("===================================")

print(
    f"FIT101: low={sensor_config['FIT101']['low']:.6f}, "
    f"high={sensor_config['FIT101']['high']:.6f}"
)

print(
    f"LIT101: low={sensor_config['LIT101']['low']:.6f}, "
    f"high={sensor_config['LIT101']['high']:.6f}"
)

print("MV101 train states:", actuator_config["MV101"]["states"])

print("\n===================================")
print("STRATIFIED PREPARATION COMPLETE")
print("===================================")

print("Saved:", FOLD_OUTPUT)
print("Saved:", TRAIN_OUTPUT)
print("Saved:", VAL_OUTPUT)
print("Saved:", CONFIG_OUTPUT)
print(
    "\nOnly the CV protocol changed. "
    "The 47,044 reconstructed rows and q05/q95 method remain unchanged."
)
