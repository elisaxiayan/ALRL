from pathlib import Path
import json
import re
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


# ============================================================
# ALRL Binarization V2
#
# Goal:
#   Preserve the same 47,044 reconstructed samples and the same
#   shuffled stratified Fold 4, but enrich the sensor binary
#   representation with TRAINING-ONLY attack-range conditions.
#
# Paper-supported idea:
#   Besides LL / HH operational boundaries, sensor conditions may
#   also be defined from numerical fluctuation ranges observed
#   during known attacks.
#
# Reconstruction used here:
#   1) Base conditions per sensor:
#        LOW    : value < Normal-train q05
#        NORMAL : q05 <= value <= q95
#        HIGH   : value > Normal-train q95
#
#   2) Additional attack-derived interval conditions:
#        For every attack class and sensor:
#        - derive q05..q95 interval from TRAINING attack samples
#        - measure how often it covers the target attack
#        - measure how often it covers all other training classes
#        - rank by specificity
#        - keep at most TOP_K_RANGES_PER_ATTACK
#
#   3) Actuators:
#        retain the current observed-state one-hot reconstruction.
#
# IMPORTANT:
#   This is NOT claimed as the authors' exact private threshold set.
#   It is a transparent implementation of the paper's stated
#   "known-attack fluctuation range" idea, using training data only.
#
# Outputs:
#   ALRL_v2_fold4_train_binary.csv
#   ALRL_v2_fold4_val_binary.csv
#   ALRL_v2_binarization_config.json
#   ALRL_v2_selected_attack_ranges.csv
#   ALRL_v2_collision_summary.txt
# ============================================================


RAW_FILE = "ALRL_paper_like_stratified_5fold.csv"

TRAIN_OUTPUT = "ALRL_v2_fold4_train_binary.csv"
VAL_OUTPUT = "ALRL_v2_fold4_val_binary.csv"

CONFIG_OUTPUT = "ALRL_v2_binarization_config.json"
RANGE_REPORT_OUTPUT = "ALRL_v2_selected_attack_ranges.csv"
COLLISION_SUMMARY_OUTPUT = "ALRL_v2_collision_summary.txt"

VALIDATION_FOLD = 4

BASE_LOW_QUANTILE = 0.05
BASE_HIGH_QUANTILE = 0.95

ATTACK_RANGE_LOW_QUANTILE = 0.05
ATTACK_RANGE_HIGH_QUANTILE = 0.95

TOP_K_RANGES_PER_ATTACK = 5

# Minimum target-attack coverage for a candidate interval.
MIN_ATTACK_COVERAGE = 0.60

# Minimum target-vs-other specificity advantage.
MIN_SPECIFICITY_SCORE = 0.10

# Avoid unstable intervals from extremely tiny classes.
MIN_ATTACK_TRAIN_SAMPLES = 20


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


PRACTICAL_ATTACK_IDS = [
    1, 2, 3, 4, 6, 7, 8, 10, 11, 13,
    16, 17, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41
]

ATTACK_CLASSES = [
    f"Attack_{attack_id}"
    for attack_id in PRACTICAL_ATTACK_IDS
]


def state_to_text(value):
    if isinstance(value, (np.integer, int)):
        return str(int(value))

    if isinstance(value, (np.floating, float)):
        if float(value).is_integer():
            return str(int(value))

        text = f"{float(value):.10g}"
        return re.sub(
            r"[^0-9A-Za-z]+",
            "_",
            text
        )

    return re.sub(
        r"[^0-9A-Za-z]+",
        "_",
        str(value)
    )


# ============================================================
# 1. Load same stratified raw dataset
# ============================================================

print("===================================")
print("1. LOAD STRATIFIED RAW DATA")
print("===================================")

df = pd.read_csv(
    RAW_FILE,
    low_memory=False
)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="raise"
)

assert len(df) == 47044
assert df["Attack_Type"].nunique() == 36
assert "Fold" in df.columns

train = df[
    df["Fold"] != VALIDATION_FOLD
].copy()

val = df[
    df["Fold"] == VALIDATION_FOLD
].copy()

print("Train rows:", len(train))
print("Validation rows:", len(val))

print(
    "Train Normal:",
    (train["Attack_Type"] == "Normal").sum()
)

print(
    "Validation Normal:",
    (val["Attack_Type"] == "Normal").sum()
)


# ============================================================
# 2. Base Normal LL / HH reconstruction
# ============================================================

print("\n===================================")
print("2. FIT BASE SENSOR BOUNDARIES")
print("===================================")

normal_train = train[
    train["Attack_Type"] == "Normal"
].copy()

sensor_base_config = {}

for sensor in SENSORS:

    values = pd.to_numeric(
        normal_train[sensor],
        errors="raise"
    )

    low = float(
        values.quantile(
            BASE_LOW_QUANTILE
        )
    )

    high = float(
        values.quantile(
            BASE_HIGH_QUANTILE
        )
    )

    sensor_base_config[sensor] = {
        "low": low,
        "high": high
    }


# ============================================================
# 3. Discover attack-derived interval conditions
#
# Candidate score:
#
#   target_coverage - other_coverage
#
# A high score means:
#   "this sensor interval often occurs in this attack, but not
#    very often in other classes."
#
# ALL statistics are computed from TRAINING ONLY.
# ============================================================

print("\n===================================")
print("3. DISCOVER ATTACK-RANGE CONDITIONS")
print("===================================")

candidate_rows = []


for attack_class in ATTACK_CLASSES:

    attack_df = train[
        train["Attack_Type"] == attack_class
    ]

    other_df = train[
        train["Attack_Type"] != attack_class
    ]


    if len(attack_df) < MIN_ATTACK_TRAIN_SAMPLES:
        print(
            f"Skip {attack_class}: only "
            f"{len(attack_df)} train samples"
        )
        continue


    attack_candidates = []


    for sensor in SENSORS:

        target_values = pd.to_numeric(
            attack_df[sensor],
            errors="raise"
        )

        other_values = pd.to_numeric(
            other_df[sensor],
            errors="raise"
        )


        low = float(
            target_values.quantile(
                ATTACK_RANGE_LOW_QUANTILE
            )
        )

        high = float(
            target_values.quantile(
                ATTACK_RANGE_HIGH_QUANTILE
            )
        )


        # Ignore invalid intervals.
        if low > high:
            continue


        target_coverage = float(
            (
                (target_values >= low)
                &
                (target_values <= high)
            ).mean()
        )


        other_coverage = float(
            (
                (other_values >= low)
                &
                (other_values <= high)
            ).mean()
        )


        specificity_score = (
            target_coverage
            -
            other_coverage
        )


        attack_candidates.append(
            {
                "source_attack":
                    attack_class,

                "sensor":
                    sensor,

                "low":
                    low,

                "high":
                    high,

                "target_coverage":
                    target_coverage,

                "other_coverage":
                    other_coverage,

                "specificity_score":
                    specificity_score,

                "attack_train_samples":
                    len(attack_df)
            }
        )


    # Prefer truly discriminative conditions.
    qualifying = [
        row
        for row in attack_candidates
        if (
            row["target_coverage"]
            >=
            MIN_ATTACK_COVERAGE
            and
            row["specificity_score"]
            >=
            MIN_SPECIFICITY_SCORE
        )
    ]


    qualifying = sorted(
        qualifying,
        key=lambda row: (
            row["specificity_score"],
            row["target_coverage"],
            -row["other_coverage"]
        ),
        reverse=True
    )


    selected_for_attack = (
        qualifying[
            :TOP_K_RANGES_PER_ATTACK
        ]
    )


    # If nothing passes the threshold, keep the single best
    # positive candidate so the attack is not automatically
    # discarded from the range-learning process.
    if (
        len(selected_for_attack) == 0
        and
        len(attack_candidates) > 0
    ):

        fallback = max(
            attack_candidates,
            key=lambda row:
                row["specificity_score"]
        )

        if (
            fallback[
                "specificity_score"
            ] > 0
        ):
            fallback = dict(
                fallback
            )

            fallback[
                "fallback_selected"
            ] = True

            selected_for_attack = [
                fallback
            ]


    for row in selected_for_attack:

        row = dict(row)

        row.setdefault(
            "fallback_selected",
            False
        )

        candidate_rows.append(
            row
        )


# ============================================================
# 4. Deduplicate near-identical conditions
#
# Same sensor + rounded low/high = same logical condition.
# Keep the highest-specificity version and retain source attack.
# ============================================================

dedup_map = {}


for row in candidate_rows:

    key = (
        row["sensor"],
        round(
            row["low"],
            6
        ),
        round(
            row["high"],
            6
        )
    )


    if key not in dedup_map:

        dedup_map[key] = dict(
            row
        )

        dedup_map[key][
            "source_attacks"
        ] = [
            row["source_attack"]
        ]


    else:

        dedup_map[key][
            "source_attacks"
        ].append(
            row["source_attack"]
        )


        if (
            row["specificity_score"]
            >
            dedup_map[key][
                "specificity_score"
            ]
        ):

            previous_attacks = (
                dedup_map[key][
                    "source_attacks"
                ]
            )

            dedup_map[key] = dict(
                row
            )

            dedup_map[key][
                "source_attacks"
            ] = previous_attacks


selected_ranges = list(
    dedup_map.values()
)


# Stable ordering by sensor then score.
selected_ranges = sorted(
    selected_ranges,
    key=lambda row: (
        row["sensor"],
        -row["specificity_score"],
        row["low"],
        row["high"]
    )
)


# Assign stable feature names.
sensor_range_counter = defaultdict(
    int
)


for row in selected_ranges:

    sensor = row["sensor"]

    index = sensor_range_counter[
        sensor
    ]

    sensor_range_counter[
        sensor
    ] += 1

    row[
        "feature_name"
    ] = (
        f"{sensor}_ATTACK_RANGE_{index:03d}"
    )

    row[
        "source_attacks"
    ] = ",".join(
        sorted(
            set(
                row[
                    "source_attacks"
                ]
            )
        )
    )


range_report = pd.DataFrame(
    selected_ranges
)


range_report.to_csv(
    RANGE_REPORT_OUTPUT,
    index=False
)


print(
    "Selected attack-derived sensor ranges:",
    len(selected_ranges)
)

print(
    "Sensors receiving extra ranges:",
    len(
        set(
            row["sensor"]
            for row in selected_ranges
        )
    )
)


attack_range_counts = (
    range_report[
        "source_attack"
    ]
    .value_counts()
    if len(range_report) > 0
    else pd.Series(dtype=int)
)


print(
    "Attack classes represented by selected ranges:",
    len(
        attack_range_counts
    )
)


# ============================================================
# 5. Actuator state configuration
# ============================================================

print("\n===================================")
print("4. FIT ACTUATOR STATES")
print("===================================")

actuator_config = {}


for actuator in ACTUATORS:

    values = pd.to_numeric(
        train[actuator],
        errors="raise"
    )

    states = sorted(
        values.unique().tolist()
    )


    actuator_config[
        actuator
    ] = {
        "states": [
            float(state)
            for state in states
        ]
    }


# ============================================================
# 6. Transform
# ============================================================

def transform_to_binary(data):

    metadata = (
        data[
            [
                "Timestamp",
                "Attack_Type",
                "Fold"
            ]
        ]
        .reset_index(
            drop=True
        )
    )


    binary_columns = {}


    # --------------------------------------------------------
    # Base LOW/NORMAL/HIGH sensor conditions
    # --------------------------------------------------------

    for sensor in SENSORS:

        values = pd.to_numeric(
            data[sensor],
            errors="raise"
        ).reset_index(
            drop=True
        )


        low = (
            sensor_base_config[
                sensor
            ][
                "low"
            ]
        )

        high = (
            sensor_base_config[
                sensor
            ][
                "high"
            ]
        )


        binary_columns[
            f"{sensor}_LOW"
        ] = (
            values < low
        ).astype(
            "int8"
        )


        binary_columns[
            f"{sensor}_NORMAL"
        ] = (
            (
                values >= low
            )
            &
            (
                values <= high
            )
        ).astype(
            "int8"
        )


        binary_columns[
            f"{sensor}_HIGH"
        ] = (
            values > high
        ).astype(
            "int8"
        )


    # --------------------------------------------------------
    # Additional attack-range sensor conditions
    #
    # These do NOT replace LOW/NORMAL/HIGH.
    # They are extra interpretable Boolean conditions.
    # --------------------------------------------------------

    for row in selected_ranges:

        sensor = row[
            "sensor"
        ]

        feature_name = row[
            "feature_name"
        ]

        low = row[
            "low"
        ]

        high = row[
            "high"
        ]


        values = pd.to_numeric(
            data[sensor],
            errors="raise"
        ).reset_index(
            drop=True
        )


        binary_columns[
            feature_name
        ] = (
            (
                values >= low
            )
            &
            (
                values <= high
            )
        ).astype(
            "int8"
        )


    # --------------------------------------------------------
    # Actuator one-hot reconstruction
    # --------------------------------------------------------

    for actuator in ACTUATORS:

        values = pd.to_numeric(
            data[actuator],
            errors="raise"
        ).reset_index(
            drop=True
        )


        train_states = (
            actuator_config[
                actuator
            ][
                "states"
            ]
        )


        observed_states = set(
            float(x)
            for x in values.unique().tolist()
        )


        allowed_states = set(
            float(x)
            for x in train_states
        )


        unseen_states = (
            observed_states
            -
            allowed_states
        )


        if unseen_states:

            raise RuntimeError(
                f"{actuator} has unseen validation states: "
                f"{sorted(unseen_states)}"
            )


        for state in train_states:

            state_text = (
                state_to_text(
                    state
                )
            )


            binary_columns[
                f"{actuator}_EQ_{state_text}"
            ] = (
                values == state
            ).astype(
                "int8"
            )


    return pd.concat(
        [
            metadata,
            pd.DataFrame(
                binary_columns
            )
        ],
        axis=1
    )


print("\n===================================")
print("5. BUILD V2 BINARY DATA")
print("===================================")


train_binary = transform_to_binary(
    train
)

val_binary = transform_to_binary(
    val
)


metadata_cols = {
    "Timestamp",
    "Attack_Type",
    "Fold"
}


feature_cols = [
    col
    for col in train_binary.columns
    if col not in metadata_cols
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
    "Total binary features:",
    len(feature_cols)
)

print(
    "Base features:",
    125
)

print(
    "Added attack-range features:",
    len(feature_cols) - 125
)


# ============================================================
# 7. Save V2 data
# ============================================================

train_binary.to_csv(
    TRAIN_OUTPUT,
    index=False
)

val_binary.to_csv(
    VAL_OUTPUT,
    index=False
)


config = {
    "raw_file":
        RAW_FILE,

    "validation_fold":
        VALIDATION_FOLD,

    "base_sensor_method":
        "train-normal q05/q95 reconstruction",

    "base_low_quantile":
        BASE_LOW_QUANTILE,

    "base_high_quantile":
        BASE_HIGH_QUANTILE,

    "attack_range_low_quantile":
        ATTACK_RANGE_LOW_QUANTILE,

    "attack_range_high_quantile":
        ATTACK_RANGE_HIGH_QUANTILE,

    "top_k_ranges_per_attack":
        TOP_K_RANGES_PER_ATTACK,

    "min_attack_coverage":
        MIN_ATTACK_COVERAGE,

    "min_specificity_score":
        MIN_SPECIFICITY_SCORE,

    "min_attack_train_samples":
        MIN_ATTACK_TRAIN_SAMPLES,

    "base_sensor_boundaries":
        sensor_base_config,

    "selected_attack_ranges":
        selected_ranges,

    "actuators":
        actuator_config
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
# 8. Immediate collision diagnostic
#
# This lets us decide whether V2 actually preserved more class
# information BEFORE spending time retraining ALRL.
# ============================================================

print("\n===================================")
print("6. V2 COLLISION DIAGNOSTIC")
print("===================================")


def packed_keys(frame):

    x = frame[
        feature_cols
    ].to_numpy(
        dtype=np.uint8
    )


    if not np.isin(
        x,
        [0, 1]
    ).all():

        raise RuntimeError(
            "Non-binary value found."
        )


    packed = np.packbits(
        x,
        axis=1,
        bitorder="big"
    )


    return [
        row.tobytes()
        for row in packed
    ]


train_keys = packed_keys(
    train_binary
)

val_keys = packed_keys(
    val_binary
)


train_pattern_labels = defaultdict(
    Counter
)


for key, label in zip(
    train_keys,
    train_binary[
        "Attack_Type"
    ].tolist()
):

    train_pattern_labels[
        key
    ][
        label
    ] += 1


status_counts = Counter()


for key, true_label in zip(
    val_keys,
    val_binary[
        "Attack_Type"
    ].tolist()
):

    if key not in train_pattern_labels:

        status = "unseen"

    else:

        train_classes = set(
            train_pattern_labels[
                key
            ].keys()
        )


        if (
            len(train_classes) == 1
            and
            true_label in train_classes
        ):

            status = "clean_same"

        elif true_label in train_classes:

            status = (
                "ambiguous_including_true"
            )

        else:

            status = "wrong_only"


    status_counts[
        status
    ] += 1


for status in [
    "clean_same",
    "ambiguous_including_true",
    "wrong_only",
    "unseen"
]:

    count = status_counts[
        status
    ]

    print(
        f"{status:26s}: "
        f"{count:5d} "
        f"({count / len(val_binary):.4%})"
    )


# ------------------------------------------------------------
# Empirical finite-dataset collision ceiling
# ------------------------------------------------------------

all_binary = pd.concat(
    [
        train_binary,
        val_binary
    ],
    ignore_index=True
)


all_keys = (
    train_keys
    +
    val_keys
)


all_pattern_labels = defaultdict(
    Counter
)


for key, label in zip(
    all_keys,
    all_binary[
        "Attack_Type"
    ].tolist()
):

    all_pattern_labels[
        key
    ][
        label
    ] += 1


collision_rows = 0
majority_correct = 0


for class_counts in (
    all_pattern_labels.values()
):

    total = sum(
        class_counts.values()
    )

    majority = max(
        class_counts.values()
    )


    majority_correct += majority


    if len(
        class_counts
    ) > 1:

        collision_rows += total


collision_rate = (
    collision_rows
    /
    len(all_binary)
)


empirical_ceiling = (
    majority_correct
    /
    len(all_binary)
)


print(
    "\nRows in cross-class identical V2 patterns:",
    collision_rows
)

print(
    "V2 cross-class collision row rate:",
    f"{collision_rate:.4%}"
)

print(
    "V2 empirical deterministic accuracy ceiling:",
    f"{empirical_ceiling:.6f}"
)


# ============================================================
# 9. Compare against V1 known result
# ============================================================

V1_COLLISION_RATE = 0.350310
V1_EMPIRICAL_CEILING = 0.925028


print("\n===================================")
print("7. V1 vs V2 REPRESENTATION")
print("===================================")

print(
    f"V1 collision rate: "
    f"{V1_COLLISION_RATE:.4%}"
)

print(
    f"V2 collision rate: "
    f"{collision_rate:.4%}"
)

print(
    "Collision-rate change: "
    f"{collision_rate - V1_COLLISION_RATE:+.4%}"
)


print(
    f"\nV1 empirical accuracy ceiling: "
    f"{V1_EMPIRICAL_CEILING:.6f}"
)

print(
    f"V2 empirical accuracy ceiling: "
    f"{empirical_ceiling:.6f}"
)

print(
    "Ceiling change: "
    f"{empirical_ceiling - V1_EMPIRICAL_CEILING:+.6f}"
)


# ============================================================
# 10. Save summary
# ============================================================

summary_lines = [
    "ALRL BINARIZATION V2 SUMMARY",
    "============================",
    "",
    f"Train rows: {len(train_binary)}",
    f"Validation rows: {len(val_binary)}",
    f"Total binary features: {len(feature_cols)}",
    f"Added attack-range features: {len(feature_cols) - 125}",
    f"Selected attack-derived ranges: {len(selected_ranges)}",
    "",
    f"V1 collision rate: {V1_COLLISION_RATE:.6%}",
    f"V2 collision rate: {collision_rate:.6%}",
    (
        "Collision-rate change: "
        f"{collision_rate - V1_COLLISION_RATE:+.6%}"
    ),
    "",
    (
        "V1 empirical deterministic accuracy ceiling: "
        f"{V1_EMPIRICAL_CEILING:.6f}"
    ),
    (
        "V2 empirical deterministic accuracy ceiling: "
        f"{empirical_ceiling:.6f}"
    ),
    (
        "Ceiling change: "
        f"{empirical_ceiling - V1_EMPIRICAL_CEILING:+.6f}"
    ),
    "",
    "Validation status:",
    (
        "clean_same = "
        f"{status_counts['clean_same']}"
    ),
    (
        "ambiguous_including_true = "
        f"{status_counts['ambiguous_including_true']}"
    ),
    (
        "wrong_only = "
        f"{status_counts['wrong_only']}"
    ),
    (
        "unseen = "
        f"{status_counts['unseen']}"
    )
]


Path(
    COLLISION_SUMMARY_OUTPUT
).write_text(
    "\n".join(
        summary_lines
    ),
    encoding="utf-8"
)


print("\n===================================")
print("V2 PREPARATION COMPLETE")
print("===================================")

print("Saved:", TRAIN_OUTPUT)
print("Saved:", VAL_OUTPUT)
print("Saved:", CONFIG_OUTPUT)
print("Saved:", RANGE_REPORT_OUTPUT)
print("Saved:", COLLISION_SUMMARY_OUTPUT)
