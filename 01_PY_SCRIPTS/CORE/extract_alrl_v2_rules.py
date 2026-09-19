from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from alrl_model import ALRL


CHECKPOINT_FILE = "ALRL_v2_fold4_best_nodistill.pt"
TRAIN_BINARY_FILE = "ALRL_v2_fold4_train_binary.csv"
VAL_BINARY_FILE = "ALRL_v2_fold4_val_binary.csv"
CONFIG_FILE = "ALRL_v2_binarization_config.json"

CANDIDATE_OUT = "ALRL_v2_fold4_candidate_rules.csv"
CONDITION_OUT = "ALRL_v2_fold4_condition_matches.csv"
RULE_OUT = "ALRL_v2_fold4_deterministic_rules.csv"
ATTACK1_OUT = "ALRL_v2_fold4_attack1_explanation.txt"

TOP_K_RULES = 10
MATCH_THRESHOLD = 0.90

SINGLE_AND = 24
SINGLE_OR = 24
MULTI_AND = 64
MULTI_OR = 64
M = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location="cpu",
    weights_only=False,
)

train_df = pd.read_csv(TRAIN_BINARY_FILE, low_memory=False)
val_df = pd.read_csv(VAL_BINARY_FILE, low_memory=False)

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

classes = checkpoint["classes"]
class_to_id = checkpoint["class_to_id"]
stage_dims = checkpoint["stage_dims"]
stage_columns = {
    int(k): v
    for k, v in checkpoint["stage_columns"].items()
}

print("Checkpoint epoch:", checkpoint["epoch"])
print("Classes:", len(classes))
print("Stage dimensions:", stage_dims)
print("Train rows:", len(train_df))
print("Validation rows:", len(val_df))

model = ALRL(
    stage_dims=stage_dims,
    num_classes=len(classes),
    single_and=SINGLE_AND,
    single_or=SINGLE_OR,
    multi_and=MULTI_AND,
    multi_or=MULTI_OR,
    m=M,
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

range_by_feature = {
    row["feature_name"]: row
    for row in config.get("selected_attack_ranges", [])
}

base_boundaries = config.get(
    "base_sensor_boundaries",
    config.get("base_sensor_config", {})
)

def fmt_number(x):
    x = float(x)
    if abs(x) >= 100:
        return f"{x:.3f}"
    if abs(x) >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"

def semantic_condition(feature_name):
    if feature_name in range_by_feature:
        row = range_by_feature[feature_name]
        return (
            f"{fmt_number(row['low'])} <= "
            f"{row['sensor']} <= "
            f"{fmt_number(row['high'])}"
        )

    for suffix in ["_LOW", "_NORMAL", "_HIGH"]:
        if feature_name.endswith(suffix):
            sensor = feature_name[:-len(suffix)]
            if sensor in base_boundaries:
                low = base_boundaries[sensor]["low"]
                high = base_boundaries[sensor]["high"]

                if suffix == "_LOW":
                    return f"{sensor} < {fmt_number(low)}"

                if suffix == "_HIGH":
                    return f"{sensor} > {fmt_number(high)}"

                return (
                    f"{fmt_number(low)} <= "
                    f"{sensor} <= "
                    f"{fmt_number(high)}"
                )

    if "_EQ_" in feature_name:
        device_name, state = feature_name.split("_EQ_", 1)
        return f"{device_name} = {state}"

    return feature_name

def leaf(name):
    return {"op": "LEAF", "name": name}

def node(op, children):
    return {"op": op, "children": children}

def expression_to_text(expr):
    if expr["op"] == "LEAF":
        return semantic_condition(expr["name"])

    symbol = " AND " if expr["op"] == "AND" else " OR "

    return (
        "("
        + symbol.join(
            expression_to_text(child)
            for child in expr["children"]
        )
        + ")"
    )

def collect_leaves(expr):
    if expr["op"] == "LEAF":
        return [expr["name"]]

    leaves = []
    for child in expr["children"]:
        leaves.extend(collect_leaves(child))
    return leaves


# ============================================================
# Trace h1 rules
# ============================================================

h1_rules = []
h1_metadata = []

for stage_idx in range(6):
    stage_number = stage_idx + 1
    cols = stage_columns[stage_number]
    layer = model.single_stage.stage_layers[stage_idx]

    and_idx = torch.argmax(
        layer.and_pi.detach().cpu(),
        dim=-1
    ).numpy()

    or_idx = torch.argmax(
        layer.or_pi.detach().cpu(),
        dim=-1
    ).numpy()

    for unit in range(SINGLE_AND):
        names = [cols[int(i)] for i in and_idx[unit]]
        h1_rules.append(
            node("AND", [leaf(name) for name in names])
        )
        h1_metadata.append({
            "level": "h1",
            "stage": stage_number,
            "logical_type": "AND",
            "unit": unit,
        })

    for unit in range(SINGLE_OR):
        names = [cols[int(i)] for i in or_idx[unit]]
        h1_rules.append(
            node("OR", [leaf(name) for name in names])
        )
        h1_metadata.append({
            "level": "h1",
            "stage": stage_number,
            "logical_type": "OR",
            "unit": unit,
        })

assert len(h1_rules) == 288


# ============================================================
# Trace h2 rules
# ============================================================

h2_rules = []
h2_metadata = []

multi = model.multistage

multi_and_idx = torch.argmax(
    multi.and_pi.detach().cpu(),
    dim=-1
).numpy()

multi_or_idx = torch.argmax(
    multi.or_pi.detach().cpu(),
    dim=-1
).numpy()

for unit in range(MULTI_AND):
    child_indices = [int(i) for i in multi_and_idx[unit]]

    h2_rules.append(
        node(
            "AND",
            [h1_rules[i] for i in child_indices]
        )
    )

    h2_metadata.append({
        "level": "h2",
        "stage": "multi",
        "logical_type": "AND",
        "unit": unit,
        "h1_sources": ",".join(str(i) for i in child_indices),
    })

for unit in range(MULTI_OR):
    child_indices = [int(i) for i in multi_or_idx[unit]]

    h2_rules.append(
        node(
            "OR",
            [h1_rules[i] for i in child_indices]
        )
    )

    h2_metadata.append({
        "level": "h2",
        "stage": "multi",
        "logical_type": "OR",
        "unit": unit,
        "h1_sources": ",".join(str(i) for i in child_indices),
    })

assert len(h2_rules) == 128

final_rules = h1_rules + h2_rules
final_metadata = h1_metadata + h2_metadata

classifier_weights = F.softplus(
    model.raw_classifier_weights.detach().cpu()
).numpy()

assert classifier_weights.shape == (len(classes), 416)


# ============================================================
# Top-10 candidate rules
# ============================================================

candidate_rows = []
top_feature_indices_by_class = {}

for class_name in classes:
    class_id = class_to_id[class_name]
    weights = classifier_weights[class_id]

    top_indices = np.argsort(weights)[::-1][:TOP_K_RULES]
    top_feature_indices_by_class[class_name] = top_indices.tolist()

    for rank, feature_idx in enumerate(top_indices, start=1):
        feature_idx = int(feature_idx)
        expr = final_rules[feature_idx]
        meta = final_metadata[feature_idx]

        candidate_rows.append({
            "class": class_name,
            "rank": rank,
            "final_logic_index": feature_idx,
            "weight": float(weights[feature_idx]),
            "level": meta["level"],
            "stage": meta["stage"],
            "logical_type": meta["logical_type"],
            "expression": expression_to_text(expr),
            "leaf_conditions": " | ".join(collect_leaves(expr)),
        })

candidate_df = pd.DataFrame(candidate_rows)
candidate_df.to_csv(CANDIDATE_OUT, index=False)


# ============================================================
# Independent condition matching > 0.9 on training attack
# ============================================================

condition_rows = []
selected_conditions_by_class = {}

for class_name in classes:
    leaf_names = []

    for idx in top_feature_indices_by_class[class_name]:
        leaf_names.extend(
            collect_leaves(final_rules[idx])
        )

    leaf_names = list(dict.fromkeys(leaf_names))

    train_class = train_df[
        train_df["Attack_Type"] == class_name
    ]

    val_class = val_df[
        val_df["Attack_Type"] == class_name
    ]

    selected = []

    for condition in leaf_names:
        if condition not in train_df.columns:
            raise RuntimeError(
                f"Missing binary condition: {condition}"
            )

        train_match = float(train_class[condition].mean())
        val_match = float(val_class[condition].mean())

        keep = train_match > MATCH_THRESHOLD

        if keep:
            selected.append(condition)

        condition_rows.append({
            "class": class_name,
            "condition": condition,
            "semantic_condition": semantic_condition(condition),
            "train_attack_match": train_match,
            "validation_attack_match": val_match,
            "selected_gt_0_9": keep,
        })

    selected_conditions_by_class[class_name] = selected

condition_df = pd.DataFrame(condition_rows)
condition_df.to_csv(CONDITION_OUT, index=False)


# ============================================================
# Deterministic AND rule evaluation on validation
# ============================================================

rule_rows = []

for class_name in classes:
    selected = selected_conditions_by_class[class_name]

    true_mask = (
        val_df["Attack_Type"] == class_name
    ).to_numpy()

    if selected:
        rule_mask = (
            val_df[selected]
            .to_numpy(dtype=np.uint8)
            .all(axis=1)
        )
    else:
        rule_mask = np.zeros(len(val_df), dtype=bool)

    tp = int(np.sum(rule_mask & true_mask))
    fp = int(np.sum(rule_mask & ~true_mask))
    fn = int(np.sum(~rule_mask & true_mask))

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
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    train_class = train_df[
        train_df["Attack_Type"] == class_name
    ]

    if selected:
        train_rule_match = float(
            train_class[selected]
            .to_numpy(dtype=np.uint8)
            .all(axis=1)
            .mean()
        )
    else:
        train_rule_match = 0.0

    semantic_conditions = [
        semantic_condition(c)
        for c in selected
    ]

    readable_rule = (
        " AND ".join(semantic_conditions)
        if semantic_conditions
        else "(no condition exceeded 0.9)"
    )

    rule_rows.append({
        "class": class_name,
        "num_selected_conditions": len(selected),
        "selected_binary_conditions": " | ".join(selected),
        "deterministic_rule": readable_rule,
        "train_attack_rule_match": train_rule_match,
        "validation_precision": precision,
        "validation_recall": recall,
        "validation_f1": f1,
        "validation_tp": tp,
        "validation_fp": fp,
        "validation_fn": fn,
    })

rule_df = pd.DataFrame(rule_rows)
rule_df.to_csv(RULE_OUT, index=False)


# ============================================================
# Attack 1 readable output
# ============================================================

attack_name = "Attack_1"

attack_candidates = candidate_df[
    candidate_df["class"] == attack_name
].copy()

attack_conditions = condition_df[
    condition_df["class"] == attack_name
].copy()

attack_selected = attack_conditions[
    attack_conditions["selected_gt_0_9"]
].sort_values(
    "train_attack_match",
    ascending=False
)

attack_rule = rule_df[
    rule_df["class"] == attack_name
].iloc[0]

print("\n" + "=" * 70)
print("ATTACK 1 - TOP 10 CANDIDATE LOGICAL RULES")
print("=" * 70)

for _, row in attack_candidates.iterrows():
    print(
        f"#{int(row['rank']):02d} | "
        f"weight={row['weight']:.6f} | "
        f"{row['level']} {row['logical_type']} | "
        f"{row['expression']}"
    )

print("\n" + "=" * 70)
print("ATTACK 1 - CONDITIONS WITH TRAIN MATCH > 0.9")
print("=" * 70)

if len(attack_selected) == 0:
    print("None.")
else:
    for _, row in attack_selected.iterrows():
        print(
            f"{row['semantic_condition']} | "
            f"train={row['train_attack_match']:.4f} | "
            f"val={row['validation_attack_match']:.4f}"
        )

print("\n" + "=" * 70)
print("ATTACK 1 - DETERMINISTIC RULE")
print("=" * 70)

print(attack_rule["deterministic_rule"])
print(
    "\nTrain Attack-1 rule match:",
    f"{attack_rule['train_attack_rule_match']:.4f}"
)
print(
    "Validation precision:",
    f"{attack_rule['validation_precision']:.4f}"
)
print(
    "Validation recall:",
    f"{attack_rule['validation_recall']:.4f}"
)
print(
    "Validation F1:",
    f"{attack_rule['validation_f1']:.4f}"
)

paper_relevant = attack_conditions[
    attack_conditions["condition"].str.startswith(
        ("LIT101", "MV101", "FIT101")
    )
].sort_values(
    "train_attack_match",
    ascending=False
)

print("\n" + "=" * 70)
print("ATTACK 1 - LIT101 / MV101 / FIT101 CONDITIONS FOUND")
print("=" * 70)

if len(paper_relevant) == 0:
    print(
        "No LIT101/MV101/FIT101 leaves appeared "
        "inside our top-10 candidate rules."
    )
else:
    print(
        paper_relevant[
            [
                "condition",
                "semantic_condition",
                "train_attack_match",
                "validation_attack_match",
                "selected_gt_0_9",
            ]
        ].to_string(index=False)
    )

lines = [
    "ALRL V2 FOLD-4 ATTACK 1 EXPLANATION",
    "===================================",
    "",
    "Top 10 candidate logical rules:",
]

for _, row in attack_candidates.iterrows():
    lines.append(
        f"#{int(row['rank']):02d} "
        f"weight={row['weight']:.6f}: "
        f"{row['expression']}"
    )

lines += [
    "",
    "Independent conditions with training attack match > 0.9:",
]

for _, row in attack_selected.iterrows():
    lines.append(
        f"- {row['semantic_condition']} "
        f"(train={row['train_attack_match']:.4f}, "
        f"val={row['validation_attack_match']:.4f})"
    )

lines += [
    "",
    "Deterministic rule:",
    str(attack_rule["deterministic_rule"]),
    "",
    (
        "Validation precision: "
        f"{attack_rule['validation_precision']:.6f}"
    ),
    (
        "Validation recall: "
        f"{attack_rule['validation_recall']:.6f}"
    ),
    (
        "Validation F1: "
        f"{attack_rule['validation_f1']:.6f}"
    ),
    "",
    (
        "NOTE: actuator states are printed as observed numeric "
        "states; no universal ON/OFF mapping is assumed."
    ),
]

Path(ATTACK1_OUT).write_text(
    "\n".join(lines),
    encoding="utf-8"
)

attack_rules = rule_df[
    rule_df["class"] != "Normal"
]

print("\n" + "=" * 70)
print("DETERMINISTIC RULE SUMMARY - 35 ATTACK CLASSES")
print("=" * 70)

print(
    "Mean validation rule precision:",
    f"{attack_rules['validation_precision'].mean():.4f}"
)
print(
    "Mean validation rule recall:",
    f"{attack_rules['validation_recall'].mean():.4f}"
)
print(
    "Mean validation rule F1:",
    f"{attack_rules['validation_f1'].mean():.4f}"
)
print(
    "Mean selected conditions per rule:",
    f"{attack_rules['num_selected_conditions'].mean():.2f}"
)

print("\nSaved:")
print(CANDIDATE_OUT)
print(CONDITION_OUT)
print(RULE_OUT)
print(ATTACK1_OUT)
