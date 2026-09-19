import hashlib
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


# ============================================================
# Diagnose information loss caused by current ALRL binarization
#
# Inputs:
#   ALRL_stratified_fold4_train_binary.csv
#   ALRL_stratified_fold4_val_binary.csv
#   ALRL_paper_like_stratified_5fold.csv
#
# Outputs:
#   ALRL_binary_collision_by_class.csv
#   ALRL_binary_collision_patterns.csv
#   ALRL_binary_diagnosis_summary.txt
#
# Important:
#   This script DOES NOT train any model.
#   It only diagnoses whether different classes collapse into
#   identical binary representations.
# ============================================================


TRAIN_BINARY_FILE = "ALRL_stratified_fold4_train_binary.csv"
VAL_BINARY_FILE = "ALRL_stratified_fold4_val_binary.csv"
RAW_FILE = "ALRL_paper_like_stratified_5fold.csv"

CLASS_REPORT_FILE = "ALRL_binary_collision_by_class.csv"
PATTERN_REPORT_FILE = "ALRL_binary_collision_patterns.csv"
SUMMARY_FILE = "ALRL_binary_diagnosis_summary.txt"


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

assert len(SENSORS) == 25
assert len(ACTUATORS) == 26
assert len(DEVICE_COLUMNS) == 51


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


# ============================================================
# 1. Load data
# ============================================================

print("===================================")
print("1. LOAD DATA")
print("===================================")

train = pd.read_csv(
    TRAIN_BINARY_FILE,
    low_memory=False
)

val = pd.read_csv(
    VAL_BINARY_FILE,
    low_memory=False
)

raw = pd.read_csv(
    RAW_FILE,
    low_memory=False
)

print("Binary train:", train.shape)
print("Binary val:  ", val.shape)
print("Raw dataset: ", raw.shape)


META_COLUMNS = {
    "Timestamp",
    "Attack_Type",
    "Fold"
}

feature_cols = [
    col
    for col in train.columns
    if col not in META_COLUMNS
]

val_feature_cols = [
    col
    for col in val.columns
    if col not in META_COLUMNS
]

print("Binary feature count:", len(feature_cols))

assert len(feature_cols) == 125

if feature_cols != val_feature_cols:
    raise RuntimeError(
        "Train/validation binary feature columns do not match."
    )


# ============================================================
# 2. Convert 125 binary features -> compact exact keys
#
# np.packbits turns 125 bits into 16 bytes.
# Identical binary vectors produce identical keys.
# ============================================================

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
            "Non-binary value found in binary dataset."
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
    train
)

val_keys = packed_keys(
    val
)


# ============================================================
# 3. Build map:
#
# exact train binary pattern
#      ->
# class frequencies
# ============================================================

print("\n===================================")
print("2. BUILD TRAIN PATTERN MAP")
print("===================================")

train_pattern_labels = defaultdict(
    Counter
)

for key, label in zip(
    train_keys,
    train["Attack_Type"].tolist()
):

    train_pattern_labels[
        key
    ][
        label
    ] += 1


unique_train_patterns = len(
    train_pattern_labels
)

ambiguous_train_patterns = sum(
    1
    for class_counts
    in train_pattern_labels.values()
    if len(class_counts) > 1
)


print(
    "Unique train binary patterns:",
    unique_train_patterns
)

print(
    "Train patterns shared by >1 class:",
    ambiguous_train_patterns
)


# ============================================================
# 4. Validation collision analysis
#
# clean_same:
#   pattern exists in training and only belongs to true class
#
# ambiguous_including_true:
#   exact pattern exists under multiple training classes,
#   including true class
#
# wrong_only:
#   exact pattern exists in train but only under other classes
#
# unseen:
#   exact pattern never appeared in training
# ============================================================

print("\n===================================")
print("3. VALIDATION COLLISION ANALYSIS")
print("===================================")

status_counts = Counter()

per_class = {
    class_name: Counter()
    for class_name in CLASSES
}


for key, true_label in zip(
    val_keys,
    val["Attack_Type"].tolist()
):

    if key not in train_pattern_labels:

        status = "unseen"

    else:

        class_counts = (
            train_pattern_labels[
                key
            ]
        )

        train_classes = set(
            class_counts.keys()
        )


        if (
            len(train_classes) == 1
            and true_label in train_classes
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

    per_class[
        true_label
    ][
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
        f"({count / len(val):.4%})"
    )


# ============================================================
# 5. Exact-pattern majority classifier
#
# This is NOT ALRL.
#
# It predicts the majority training class for an exact binary
# pattern. It is a representation diagnostic only.
# ============================================================

known_correct = 0
known_total = 0


for key, true_label in zip(
    val_keys,
    val["Attack_Type"].tolist()
):

    if key not in train_pattern_labels:
        continue


    prediction = (
        train_pattern_labels[
            key
        ]
        .most_common(1)[0][0]
    )


    known_total += 1

    if prediction == true_label:
        known_correct += 1


majority_known_accuracy = (
    known_correct
    /
    known_total
    if known_total > 0
    else float("nan")
)


print(
    "\nExact-pattern majority accuracy "
    "(seen validation patterns only): "
    f"{majority_known_accuracy:.4f}"
)

print(
    "Seen validation samples:",
    known_total,
    "/",
    len(val)
)


# ============================================================
# 6. Empirical deterministic ceiling caused by exact binary
# collisions on the reconstructed 47,044-row dataset.
#
# For every identical 125-bit vector, a deterministic model can
# assign only one class.
#
# Choosing the majority class for each vector gives the best
# possible accuracy on these exact collisions.
# ============================================================

print("\n===================================")
print("4. EMPIRICAL COLLISION CEILING")
print("===================================")

all_binary = pd.concat(
    [
        train,
        val
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


binary_empirical_ceiling = (
    majority_correct
    /
    len(all_binary)
)


print(
    "Rows in cross-class identical binary patterns:",
    collision_rows
)

print(
    "Cross-class collision row rate:",
    f"{collision_rows / len(all_binary):.4%}"
)

print(
    "Empirical deterministic accuracy ceiling "
    "from exact binary collisions:",
    f"{binary_empirical_ceiling:.6f}"
)


# ============================================================
# 7. Raw-feature exact collision comparison
#
# FIX:
# np.ascontiguousarray() is used before .view(np.uint8).
#
# Your previous error:
#   ValueError: last axis must be contiguous
#
# happened because pandas -> numpy does not guarantee the
# last axis is contiguous in memory.
# ============================================================

print("\n===================================")
print("5. RAW FEATURE COLLISION COMPARISON")
print("===================================")


missing_devices = [
    device
    for device in DEVICE_COLUMNS
    if device not in raw.columns
]

if missing_devices:
    raise RuntimeError(
        "Missing raw device columns: "
        f"{missing_devices}"
    )


raw_values = raw[
    DEVICE_COLUMNS
].to_numpy(
    dtype=np.float32
)


# ------------------------------------------------------------
# IMPORTANT FIX
# ------------------------------------------------------------

raw_values = np.ascontiguousarray(
    raw_values
)


# reinterpret exact float32 bytes
raw_bytes = (
    raw_values
    .view(np.uint8)
    .reshape(
        len(raw_values),
        -1
    )
)


raw_keys = [
    hashlib.blake2b(
        row.tobytes(),
        digest_size=16
    ).digest()

    for row in raw_bytes
]


raw_pattern_labels = defaultdict(
    Counter
)


for key, label in zip(
    raw_keys,
    raw["Attack_Type"].tolist()
):

    raw_pattern_labels[
        key
    ][
        label
    ] += 1


raw_collision_rows = 0
raw_majority_correct = 0


for class_counts in (
    raw_pattern_labels.values()
):

    total = sum(
        class_counts.values()
    )

    majority = max(
        class_counts.values()
    )


    raw_majority_correct += majority


    if len(
        class_counts
    ) > 1:

        raw_collision_rows += total


raw_empirical_ceiling = (
    raw_majority_correct
    /
    len(raw)
)


print(
    "Raw device features used:",
    len(DEVICE_COLUMNS)
)

print(
    "Raw cross-class collision row rate:",
    f"{raw_collision_rows / len(raw):.4%}"
)

print(
    "Raw empirical deterministic accuracy ceiling:",
    f"{raw_empirical_ceiling:.6f}"
)


# ============================================================
# 8. Per-class collision report
# ============================================================

class_rows = []


for class_name in CLASSES:

    counts = per_class[
        class_name
    ]

    n = int(
        (
            val[
                "Attack_Type"
            ]
            ==
            class_name
        ).sum()
    )


    row = {
        "class":
            class_name,

        "n_val":
            n
    }


    for status in [
        "clean_same",
        "ambiguous_including_true",
        "wrong_only",
        "unseen"
    ]:

        count = counts[
            status
        ]


        row[
            status
        ] = count


        row[
            f"{status}_rate"
        ] = (
            count / n
            if n > 0
            else np.nan
        )


    row[
        "problem_rate"
    ] = (
        row[
            "ambiguous_including_true_rate"
        ]
        +
        row[
            "wrong_only_rate"
        ]
    )


    class_rows.append(
        row
    )


class_report = pd.DataFrame(
    class_rows
)


class_report.to_csv(
    CLASS_REPORT_FILE,
    index=False
)


# ============================================================
# 9. Save most ambiguous patterns
# ============================================================

pattern_rows = []


for class_counts in (
    all_pattern_labels.values()
):

    if len(
        class_counts
    ) <= 1:

        continue


    sorted_counts = (
        class_counts
        .most_common()
    )


    pattern_rows.append(
        {
            "total_rows":
                sum(
                    class_counts.values()
                ),

            "num_classes":
                len(
                    class_counts
                ),

            "classes":
                " | ".join(
                    f"{label}:{count}"
                    for label, count
                    in sorted_counts
                )
        }
    )


pattern_report = pd.DataFrame(
    pattern_rows
)


if len(
    pattern_report
) > 0:

    pattern_report = (
        pattern_report
        .sort_values(
            [
                "total_rows",
                "num_classes"
            ],
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


pattern_report.to_csv(
    PATTERN_REPORT_FILE,
    index=False
)


# ============================================================
# 10. Print classes most affected by ambiguity
# ============================================================

print("\n===================================")
print("6. CLASSES WITH MOST BINARY AMBIGUITY")
print("===================================")


worst = (
    class_report
    .sort_values(
        "problem_rate",
        ascending=False
    )
    .head(12)
)


print(
    worst[
        [
            "class",
            "n_val",
            "clean_same_rate",
            "ambiguous_including_true_rate",
            "wrong_only_rate",
            "unseen_rate",
            "problem_rate"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 11. Save summary
# ============================================================

summary_lines = [
    "ALRL BINARY REPRESENTATION DIAGNOSIS",
    "====================================",
    "",
    f"Binary features: {len(feature_cols)}",
    f"Train rows: {len(train)}",
    f"Validation rows: {len(val)}",
    "",
    f"Unique train patterns: {unique_train_patterns}",
    f"Ambiguous train patterns: {ambiguous_train_patterns}",
    "",
    (
        "Validation clean_same: "
        f"{status_counts['clean_same']} "
        f"({status_counts['clean_same']/len(val):.4%})"
    ),
    (
        "Validation ambiguous_including_true: "
        f"{status_counts['ambiguous_including_true']} "
        f"({status_counts['ambiguous_including_true']/len(val):.4%})"
    ),
    (
        "Validation wrong_only: "
        f"{status_counts['wrong_only']} "
        f"({status_counts['wrong_only']/len(val):.4%})"
    ),
    (
        "Validation unseen: "
        f"{status_counts['unseen']} "
        f"({status_counts['unseen']/len(val):.4%})"
    ),
    "",
    (
        "Exact-pattern majority accuracy on seen val patterns: "
        f"{majority_known_accuracy:.6f}"
    ),
    "",
    (
        "Binary cross-class collision row rate: "
        f"{collision_rows / len(all_binary):.6%}"
    ),
    (
        "Binary empirical deterministic ceiling: "
        f"{binary_empirical_ceiling:.6f}"
    ),
    "",
    (
        "Raw cross-class collision row rate: "
        f"{raw_collision_rows / len(raw):.6%}"
    ),
    (
        "Raw empirical deterministic ceiling: "
        f"{raw_empirical_ceiling:.6f}"
    )
]


Path(
    SUMMARY_FILE
).write_text(
    "\n".join(
        summary_lines
    ),
    encoding="utf-8"
)


print("\n===================================")
print("FILES SAVED")
print("===================================")

print(
    CLASS_REPORT_FILE
)

print(
    PATTERN_REPORT_FILE
)

print(
    SUMMARY_FILE
)
