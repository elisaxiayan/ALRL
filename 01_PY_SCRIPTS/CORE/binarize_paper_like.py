import json
import re
import numpy as np
import pandas as pd


# ============================================================
# Paper-like SWaT Binarization
#
# Input:
#   ALRL_paper_like_47044_5fold.csv
#
# Outputs:
#   ALRL_paper_like_fold4_train_binary.csv
#   ALRL_paper_like_fold4_val_binary.csv
#   ALRL_paper_like_fold4_binarization_config.json
#
# IMPORTANT:
# The ALRL paper specifies the STRUCTURE of binarization:
#
#   sensor:
#     [s < LL, s > HH, LL <= s <= HH, ...]
#
#   actuator:
#     [a = ON, a = OFF]
#
# but it does NOT publish a complete LL/HH table for all SWaT
# sensors or the exact raw-state mapping for every actuator.
#
# Therefore this script is a transparent reconstruction:
#
#   sensors:
#     train-Normal 5th / 95th percentiles
#
#   actuators:
#     one-hot encode every discrete state observed in TRAIN only
#
# This avoids validation leakage and keeps one fixed config that
# can later be replaced by exact domain thresholds if obtained.
# ============================================================


INPUT_FILE = "ALRL_paper_like_47044_5fold.csv"

TRAIN_OUTPUT = "ALRL_paper_like_fold4_train_binary.csv"
VAL_OUTPUT = "ALRL_paper_like_fold4_val_binary.csv"
CONFIG_OUTPUT = "ALRL_paper_like_fold4_binarization_config.json"

VALIDATION_FOLD = 4

LOW_QUANTILE = 0.05
HIGH_QUANTILE = 0.95


# ============================================================
# 1. SWaT device lists
# ============================================================

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


# ============================================================
# Helper: make safe state text for column names
# ============================================================

def state_to_text(value):
    """
    Convert actuator state values into stable column-name text.

    Examples:
        1.0 -> "1"
        2.0 -> "2"
        0.0 -> "0"
    """

    if isinstance(value, (np.integer, int)):
        return str(int(value))

    if isinstance(value, (np.floating, float)):
        if float(value).is_integer():
            return str(int(value))

        text = f"{float(value):.10g}"
        return re.sub(r"[^0-9A-Za-z]+", "_", text)

    return re.sub(
        r"[^0-9A-Za-z]+",
        "_",
        str(value)
    )


# ============================================================
# 2. Load paper-like dataset
# ============================================================

print("===================================")
print("1. LOAD PAPER-LIKE DATASET")
print("===================================")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="raise"
)

print("Input shape:", df.shape)

print(
    "Total:",
    len(df)
)

print(
    "Normal:",
    (df["Attack_Type"] == "Normal").sum()
)

print(
    "Attack:",
    (df["Attack_Type"] != "Normal").sum()
)

print(
    "Classes:",
    df["Attack_Type"].nunique()
)

assert len(df) == 47044
assert df["Attack_Type"].nunique() == 36
assert "Fold" in df.columns


# ============================================================
# 3. Split Fold 4 validation
#
# Paper:
# 5-fold CV -> 80% training / 20% validation
#
# We fit ALL binarization parameters using training only.
# ============================================================

train = df[
    df["Fold"] != VALIDATION_FOLD
].copy()

val = df[
    df["Fold"] == VALIDATION_FOLD
].copy()


print("\n===================================")
print("2. TRAIN / VALIDATION SPLIT")
print("===================================")

print(
    "Train rows:",
    len(train)
)

print(
    "Validation rows:",
    len(val)
)

print(
    "Train Normal:",
    (train["Attack_Type"] == "Normal").sum()
)

print(
    "Validation Normal:",
    (val["Attack_Type"] == "Normal").sum()
)

print(
    "Train Attack:",
    (train["Attack_Type"] != "Normal").sum()
)

print(
    "Validation Attack:",
    (val["Attack_Type"] != "Normal").sum()
)


# ============================================================
# 4. Validate device columns
# ============================================================

missing_sensors = [
    s for s in SENSORS
    if s not in df.columns
]

missing_actuators = [
    a for a in ACTUATORS
    if a not in df.columns
]

if missing_sensors:
    raise ValueError(
        f"Missing sensor columns: {missing_sensors}"
    )

if missing_actuators:
    raise ValueError(
        f"Missing actuator columns: {missing_actuators}"
    )


# ============================================================
# 5. Fit sensor thresholds using TRAIN Normal only
#
# Reconstruction:
#   low  = 5th percentile
#   high = 95th percentile
#
# IMPORTANT:
# This is NOT claimed as the exact paper LL/HH table.
# ============================================================

print("\n===================================")
print("3. FIT SENSOR BOUNDARIES")
print("===================================")

normal_train = train[
    train["Attack_Type"] == "Normal"
].copy()

if len(normal_train) == 0:
    raise ValueError(
        "No Normal samples found in training folds."
    )


sensor_config = {}

for sensor in SENSORS:

    values = pd.to_numeric(
        normal_train[sensor],
        errors="coerce"
    )

    if values.isna().any():
        raise ValueError(
            f"NaN/non-numeric values found in sensor {sensor}"
        )

    low = float(
        values.quantile(
            LOW_QUANTILE
        )
    )

    high = float(
        values.quantile(
            HIGH_QUANTILE
        )
    )

    if low > high:
        raise RuntimeError(
            f"Invalid thresholds for {sensor}: "
            f"low={low}, high={high}"
        )

    sensor_config[sensor] = {
        "low": low,
        "high": high
    }


# ============================================================
# 6. Fit actuator states using TRAIN only
#
# We preserve every observed discrete state.
# This avoids guessing a universal ON/OFF mapping for raw SWaT.
# ============================================================

print("\n===================================")
print("4. FIT ACTUATOR STATES")
print("===================================")

actuator_config = {}

for actuator in ACTUATORS:

    values = pd.to_numeric(
        train[actuator],
        errors="coerce"
    )

    if values.isna().any():
        raise ValueError(
            f"NaN/non-numeric values found in actuator {actuator}"
        )

    states = sorted(
        values.unique().tolist()
    )

    actuator_config[actuator] = {
        "states": [
            float(x)
            for x in states
        ]
    }


# ============================================================
# 7. Save one frozen binarization config
# ============================================================

config = {
    "input_file": INPUT_FILE,
    "validation_fold": VALIDATION_FOLD,

    "sensor_method": "train_normal_q05_q95_reconstruction",

    "low_quantile": LOW_QUANTILE,
    "high_quantile": HIGH_QUANTILE,

    "sensors": sensor_config,
    "actuators": actuator_config
}


with open(
    CONFIG_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=2
    )


# ============================================================
# 8. Transform raw values -> binary features
# ============================================================

def transform_to_binary(data):

    pieces = []

    # --------------------------------------------------------
    # Keep metadata
    # --------------------------------------------------------

    metadata = data[
        [
            "Timestamp",
            "Attack_Type",
            "Fold"
        ]
    ].reset_index(drop=True)

    pieces.append(metadata)


    # --------------------------------------------------------
    # Sensors:
    #
    # LOW / NORMAL / HIGH
    # --------------------------------------------------------

    sensor_columns = {}

    for sensor in SENSORS:

        values = pd.to_numeric(
            data[sensor],
            errors="raise"
        ).reset_index(drop=True)

        low = sensor_config[sensor]["low"]
        high = sensor_config[sensor]["high"]

        sensor_columns[
            f"{sensor}_LOW"
        ] = (
            values < low
        ).astype("int8")

        sensor_columns[
            f"{sensor}_NORMAL"
        ] = (
            (values >= low)
            &
            (values <= high)
        ).astype("int8")

        sensor_columns[
            f"{sensor}_HIGH"
        ] = (
            values > high
        ).astype("int8")


    pieces.append(
        pd.DataFrame(
            sensor_columns
        )
    )


    # --------------------------------------------------------
    # Actuators:
    #
    # one-hot for every state observed in TRAIN
    # --------------------------------------------------------

    actuator_columns = {}

    for actuator in ACTUATORS:

        values = pd.to_numeric(
            data[actuator],
            errors="raise"
        ).reset_index(drop=True)

        train_states = (
            actuator_config[
                actuator
            ]["states"]
        )

        # Check for unseen validation states.
        observed_states = set(
            float(x)
            for x in values.unique().tolist()
        )

        allowed_states = set(
            float(x)
            for x in train_states
        )

        unseen = (
            observed_states
            -
            allowed_states
        )

        if unseen:
            raise ValueError(
                f"{actuator} contains unseen states "
                f"outside training: {sorted(unseen)}"
            )


        for state in train_states:

            state_text = state_to_text(
                state
            )

            column_name = (
                f"{actuator}_EQ_{state_text}"
            )

            actuator_columns[
                column_name
            ] = (
                values == state
            ).astype("int8")


    pieces.append(
        pd.DataFrame(
            actuator_columns
        )
    )


    # --------------------------------------------------------
    # Concatenate once to avoid pandas fragmentation warnings
    # --------------------------------------------------------

    result = pd.concat(
        pieces,
        axis=1
    )

    return result


# ============================================================
# 9. Transform train and validation with SAME config
# ============================================================

print("\n===================================")
print("5. BINARIZE TRAIN / VALIDATION")
print("===================================")

train_binary = transform_to_binary(
    train
)

val_binary = transform_to_binary(
    val
)


# ============================================================
# 10. Audit binary features
# ============================================================

feature_columns = [
    col
    for col in train_binary.columns
    if col not in {
        "Timestamp",
        "Attack_Type",
        "Fold"
    }
]

print(
    "Train binary shape:",
    train_binary.shape
)

print(
    "Validation binary shape:",
    val_binary.shape
)

print(
    "Binary feature count:",
    len(feature_columns)
)


# Every sensor should activate exactly one of LOW/NORMAL/HIGH.
for sensor in SENSORS:

    cols = [
        f"{sensor}_LOW",
        f"{sensor}_NORMAL",
        f"{sensor}_HIGH"
    ]

    train_sum = (
        train_binary[cols]
        .sum(axis=1)
    )

    val_sum = (
        val_binary[cols]
        .sum(axis=1)
    )

    if not (train_sum == 1).all():
        raise RuntimeError(
            f"Train sensor encoding invalid: {sensor}"
        )

    if not (val_sum == 1).all():
        raise RuntimeError(
            f"Validation sensor encoding invalid: {sensor}"
        )


# Every actuator should activate exactly one observed-state bit.
for actuator in ACTUATORS:

    prefix = f"{actuator}_EQ_"

    cols = [
        col
        for col in feature_columns
        if col.startswith(prefix)
    ]

    train_sum = (
        train_binary[cols]
        .sum(axis=1)
    )

    val_sum = (
        val_binary[cols]
        .sum(axis=1)
    )

    if not (train_sum == 1).all():
        raise RuntimeError(
            f"Train actuator encoding invalid: {actuator}"
        )

    if not (val_sum == 1).all():
        raise RuntimeError(
            f"Validation actuator encoding invalid: {actuator}"
        )


# ============================================================
# 11. Save binary datasets
# ============================================================

train_binary.to_csv(
    TRAIN_OUTPUT,
    index=False
)

val_binary.to_csv(
    VAL_OUTPUT,
    index=False
)


# ============================================================
# 12. Print important Stage 1 audit info
# ============================================================

print("\n===================================")
print("6. STAGE 1 AUDIT")
print("===================================")

for sensor in [
    "FIT101",
    "LIT101"
]:
    print(
        f"{sensor}: "
        f"low={sensor_config[sensor]['low']:.6f}, "
        f"high={sensor_config[sensor]['high']:.6f}"
    )


print(
    "MV101 train states:",
    actuator_config[
        "MV101"
    ]["states"]
)


# Attack 1 sample preview in validation if available,
# otherwise preview from train.
attack1_preview = val_binary[
    val_binary["Attack_Type"] == "Attack_1"
]

if len(attack1_preview) == 0:
    attack1_preview = train_binary[
        train_binary["Attack_Type"] == "Attack_1"
    ]


preview_cols = [
    "Timestamp",
    "FIT101_LOW",
    "FIT101_NORMAL",
    "FIT101_HIGH",
    "LIT101_LOW",
    "LIT101_NORMAL",
    "LIT101_HIGH"
]

preview_cols += [
    col
    for col in feature_columns
    if col.startswith(
        "MV101_EQ_"
    )
]


print("\nAttack_1 Stage 1 preview:")

print(
    attack1_preview[
        preview_cols
    ]
    .head(5)
    .to_string(index=False)
)


# ============================================================
# 13. Final
# ============================================================

print("\n===================================")
print("PAPER-LIKE BINARIZATION COMPLETE")
print("===================================")

print("Saved:", TRAIN_OUTPUT)
print("Saved:", VAL_OUTPUT)
print("Saved:", CONFIG_OUTPUT)

print(
    "\nNOTE:"
)

print(
    "The binarization STRUCTURE follows ALRL, "
    "but q05/q95 sensor boundaries are our reconstruction "
    "because the paper does not publish the full SWaT LL/HH table."
)
