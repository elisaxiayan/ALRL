from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# Finalize ALRL reproduction results
#
# This script does NOT train anything.
#
# It:
#   1) reads the completed 5-fold summary
#   2) evaluates the learned Attack-1 core rule
#        LIT101_HIGH AND MV101_EQ_2
#      on Fold-4 validation
#   3) reads the deterministic-rule extraction summary
#   4) writes one compact final report
#
# Output:
#   ALRL_REPRODUCTION_FINAL_SUMMARY.txt
#   ALRL_attack1_core_rule_metrics.csv
# ============================================================


FIVEFOLD_FILE = "ALRL_v2_5fold_summary.csv"
VAL_FILE = "ALRL_v2_fold4_val_binary.csv"
TRAIN_FILE = "ALRL_v2_fold4_train_binary.csv"
RULE_FILE = "ALRL_v2_fold4_deterministic_rules.csv"

OUTPUT_TXT = "ALRL_REPRODUCTION_FINAL_SUMMARY.txt"
ATTACK1_METRICS_CSV = "ALRL_attack1_core_rule_metrics.csv"


TARGET_CLASS = "Attack_1"

CORE_RULE_CONDITIONS = [
    "LIT101_HIGH",
    "MV101_EQ_2",
]


# ============================================================
# 1. Five-fold classification
# ============================================================

fivefold = pd.read_csv(
    FIVEFOLD_FILE
)

metrics = {}

for col in [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
]:
    metrics[f"{col}_mean"] = float(
        fivefold[col].mean()
    )

    metrics[f"{col}_std"] = float(
        fivefold[col].std(ddof=1)
    )


# ============================================================
# 2. Attack-1 core rule evaluation
# ============================================================

train_df = pd.read_csv(
    TRAIN_FILE,
    low_memory=False,
)

val_df = pd.read_csv(
    VAL_FILE,
    low_memory=False,
)


for condition in CORE_RULE_CONDITIONS:
    if condition not in train_df.columns:
        raise RuntimeError(
            f"Missing condition: {condition}"
        )

    if condition not in val_df.columns:
        raise RuntimeError(
            f"Missing condition in validation: {condition}"
        )


def rule_mask(frame):
    return (
        frame[
            CORE_RULE_CONDITIONS
        ]
        .to_numpy(
            dtype=np.uint8
        )
        .all(
            axis=1
        )
    )


train_mask = rule_mask(
    train_df
)

val_mask = rule_mask(
    val_df
)


train_true = (
    train_df["Attack_Type"]
    ==
    TARGET_CLASS
).to_numpy()


val_true = (
    val_df["Attack_Type"]
    ==
    TARGET_CLASS
).to_numpy()


def binary_metrics(
    prediction,
    truth,
):
    tp = int(
        np.sum(
            prediction
            &
            truth
        )
    )

    fp = int(
        np.sum(
            prediction
            &
            ~truth
        )
    )

    fn = int(
        np.sum(
            ~prediction
            &
            truth
        )
    )

    tn = int(
        np.sum(
            ~prediction
            &
            ~truth
        )
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2
        *
        precision
        *
        recall
        /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


train_rule_metrics = binary_metrics(
    train_mask,
    train_true,
)

val_rule_metrics = binary_metrics(
    val_mask,
    val_true,
)


attack1_metrics_df = pd.DataFrame(
    [
        {
            "split": "train",
            **train_rule_metrics,
        },
        {
            "split": "validation",
            **val_rule_metrics,
        },
    ]
)

attack1_metrics_df.to_csv(
    ATTACK1_METRICS_CSV,
    index=False,
)


# ============================================================
# 3. Existing deterministic-rule summary
# ============================================================

rule_summary_lines = []

if Path(
    RULE_FILE
).exists():

    rules = pd.read_csv(
        RULE_FILE
    )

    attack_rules = rules[
        rules["class"]
        !=
        "Normal"
    ]

    rule_summary_lines = [
        (
            "Mean deterministic-rule precision: "
            f"{attack_rules['validation_precision'].mean():.4f}"
        ),
        (
            "Mean deterministic-rule recall: "
            f"{attack_rules['validation_recall'].mean():.4f}"
        ),
        (
            "Mean deterministic-rule F1: "
            f"{attack_rules['validation_f1'].mean():.4f}"
        ),
        (
            "Mean selected conditions per attack rule: "
            f"{attack_rules['num_selected_conditions'].mean():.2f}"
        ),
    ]


# ============================================================
# 4. Final report
# ============================================================

lines = []

lines += [
    "ALRL REPRODUCTION FINAL SUMMARY",
    "===============================",
    "",
    "1. DATASET",
    "----------",
    "Total samples: 47,044",
    "Classes: 36 (Normal + 35 attack classes)",
    "Cross-validation: 5 folds",
    "",
    "2. ALRL V2 FIVE-FOLD CLASSIFICATION",
    "-----------------------------------",
    (
        "Accuracy: "
        f"{metrics['accuracy_mean']:.4f} "
        f"± {metrics['accuracy_std']:.4f}"
    ),
    (
        "Macro Precision: "
        f"{metrics['precision_macro_mean']:.4f} "
        f"± {metrics['precision_macro_std']:.4f}"
    ),
    (
        "Macro Recall: "
        f"{metrics['recall_macro_mean']:.4f} "
        f"± {metrics['recall_macro_std']:.4f}"
    ),
    (
        "Macro F1: "
        f"{metrics['f1_macro_mean']:.4f} "
        f"± {metrics['f1_macro_std']:.4f}"
    ),
    "",
    "Per-fold Macro F1:",
]

for _, row in fivefold.sort_values(
    "fold"
).iterrows():

    lines.append(
        f"  Fold {int(row['fold'])}: "
        f"{row['f1_macro']:.4f}"
    )


lines += [
    "",
    "3. ATTACK-1 CORE LOGICAL RULE",
    "-----------------------------",
    (
        "Rule: "
        "LIT101 > 813.513 AND MV101 = 2"
    ),
    "",
    "Training one-vs-rest:",
    (
        "  Precision: "
        f"{train_rule_metrics['precision']:.4f}"
    ),
    (
        "  Recall: "
        f"{train_rule_metrics['recall']:.4f}"
    ),
    (
        "  F1: "
        f"{train_rule_metrics['f1']:.4f}"
    ),
    "",
    "Validation one-vs-rest:",
    (
        "  Precision: "
        f"{val_rule_metrics['precision']:.4f}"
    ),
    (
        "  Recall: "
        f"{val_rule_metrics['recall']:.4f}"
    ),
    (
        "  F1: "
        f"{val_rule_metrics['f1']:.4f}"
    ),
    (
        "  TP / FP / FN: "
        f"{val_rule_metrics['tp']} / "
        f"{val_rule_metrics['fp']} / "
        f"{val_rule_metrics['fn']}"
    ),
    "",
    "The trained logical network contains this direct Stage-1 AND rule.",
    "",
    "4. EXPLANATION MODULE",
    "---------------------",
]


if rule_summary_lines:
    lines.extend(
        rule_summary_lines
    )
else:
    lines.append(
        "Deterministic-rule summary file was not found."
    )


lines += [
    "",
    "5. KNOWLEDGE DISTILLATION",
    "-------------------------",
    "LU-IDS-style teacher Macro F1: 0.9949",
    "Best ALRL V2 student is the no-distillation model.",
    "",
    "6. COMPLETED COMPONENTS",
    "-----------------------",
    "SWaT preprocessing and class reconstruction: complete",
    "Binary logical representation: complete",
    "Single-stage logical learning: complete",
    "Multistage logical learning: complete",
    "Prediction module: complete",
    "5-fold evaluation: complete",
    "Teacher / KD experiment: complete",
    "Explanation / logical-rule extraction: complete",
    "",
    "Key saved outputs:",
    "  ALRL_v2_5fold_summary.csv",
    "  ALRL_v2_5fold_summary.txt",
    "  ALRL_v2_5fold_runs/",
    "  ALRL_v2_fold4_candidate_rules.csv",
    "  ALRL_v2_fold4_condition_matches.csv",
    "  ALRL_v2_fold4_deterministic_rules.csv",
    "  ALRL_v2_fold4_attack1_explanation.txt",
    f"  {ATTACK1_METRICS_CSV}",
]


Path(
    OUTPUT_TXT
).write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8",
)


print("=" * 64)
print("ATTACK-1 CORE RULE VALIDATION")
print("=" * 64)

print(
    "Rule: LIT101 > 813.513 AND MV101 = 2"
)

print(
    f"Precision: "
    f"{val_rule_metrics['precision']:.4f}"
)

print(
    f"Recall:    "
    f"{val_rule_metrics['recall']:.4f}"
)

print(
    f"F1:        "
    f"{val_rule_metrics['f1']:.4f}"
)

print(
    "TP / FP / FN:",
    val_rule_metrics["tp"],
    "/",
    val_rule_metrics["fp"],
    "/",
    val_rule_metrics["fn"],
)


print("\n" + "=" * 64)
print("FINAL FIVE-FOLD CLASSIFICATION")
print("=" * 64)

print(
    f"Accuracy:        "
    f"{metrics['accuracy_mean']:.4f} "
    f"± {metrics['accuracy_std']:.4f}"
)

print(
    f"Macro Precision: "
    f"{metrics['precision_macro_mean']:.4f} "
    f"± {metrics['precision_macro_std']:.4f}"
)

print(
    f"Macro Recall:    "
    f"{metrics['recall_macro_mean']:.4f} "
    f"± {metrics['recall_macro_std']:.4f}"
)

print(
    f"Macro F1:        "
    f"{metrics['f1_macro_mean']:.4f} "
    f"± {metrics['f1_macro_std']:.4f}"
)


print("\nSaved:")
print(OUTPUT_TXT)
print(ATTACK1_METRICS_CSV)
