import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from alrl_model import ALRL


CHECKPOINT_FILE = "ALRL_v2_fold4_best_nodistill.pt"
TRAIN_FILE = "ALRL_v2_fold4_train_binary.csv"
VAL_FILE = "ALRL_v2_fold4_val_binary.csv"
CONFIG_FILE = "ALRL_v2_binarization_config.json"

TARGET_CLASS = "Attack_1"
TARGET_DEVICE = "MV101"
RELATED_SENSOR = "LIT101"

TOP_N_DEVICE_RULES = 30


checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location="cpu",
    weights_only=False,
)

train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
val_df = pd.read_csv(VAL_FILE, low_memory=False)

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

classes = checkpoint["classes"]
class_to_id = checkpoint["class_to_id"]
stage_dims = checkpoint["stage_dims"]
stage_columns = {
    int(k): v
    for k, v in checkpoint["stage_columns"].items()
}

model = ALRL(
    stage_dims=stage_dims,
    num_classes=len(classes),
    single_and=24,
    single_or=24,
    multi_and=64,
    multi_or=64,
    m=2,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


range_by_feature = {
    row["feature_name"]: row
    for row in config.get("selected_attack_ranges", [])
}

base_boundaries = config.get(
    "base_sensor_boundaries",
    {}
)


def semantic(feature_name):
    if feature_name in range_by_feature:
        row = range_by_feature[feature_name]
        return (
            f"{row['low']:.3f} <= {row['sensor']} <= "
            f"{row['high']:.3f}"
        )

    if feature_name.endswith("_HIGH"):
        sensor = feature_name[:-5]
        if sensor in base_boundaries:
            return (
                f"{sensor} > "
                f"{base_boundaries[sensor]['high']:.3f}"
            )

    if feature_name.endswith("_LOW"):
        sensor = feature_name[:-4]
        if sensor in base_boundaries:
            return (
                f"{sensor} < "
                f"{base_boundaries[sensor]['low']:.3f}"
            )

    if feature_name.endswith("_NORMAL"):
        sensor = feature_name[:-7]
        if sensor in base_boundaries:
            low = base_boundaries[sensor]["low"]
            high = base_boundaries[sensor]["high"]
            return f"{low:.3f} <= {sensor} <= {high:.3f}"

    if "_EQ_" in feature_name:
        dev, state = feature_name.split("_EQ_", 1)
        return f"{dev} = {state}"

    return feature_name


def leaf(name):
    return {
        "op": "LEAF",
        "name": name,
    }


def node(op, children):
    return {
        "op": op,
        "children": children,
    }


def leaves(expr):
    if expr["op"] == "LEAF":
        return [expr["name"]]

    out = []
    for child in expr["children"]:
        out.extend(leaves(child))
    return out


def text(expr):
    if expr["op"] == "LEAF":
        return semantic(expr["name"])

    symbol = " AND " if expr["op"] == "AND" else " OR "

    return (
        "("
        + symbol.join(text(child) for child in expr["children"])
        + ")"
    )


# ============================================================
# Reconstruct h1
# ============================================================

h1 = []
h1_meta = []

for stage_idx in range(6):
    stage = stage_idx + 1
    cols = stage_columns[stage]
    layer = model.single_stage.stage_layers[stage_idx]

    and_idx = torch.argmax(
        layer.and_pi.detach(),
        dim=-1,
    ).cpu().numpy()

    or_idx = torch.argmax(
        layer.or_pi.detach(),
        dim=-1,
    ).cpu().numpy()

    for unit in range(24):
        names = [
            cols[int(i)]
            for i in and_idx[unit]
        ]

        h1.append(
            node(
                "AND",
                [leaf(name) for name in names],
            )
        )

        h1_meta.append(
            f"h1 stage{stage} AND unit{unit}"
        )

    for unit in range(24):
        names = [
            cols[int(i)]
            for i in or_idx[unit]
        ]

        h1.append(
            node(
                "OR",
                [leaf(name) for name in names],
            )
        )

        h1_meta.append(
            f"h1 stage{stage} OR unit{unit}"
        )


# ============================================================
# Reconstruct h2
# ============================================================

h2 = []
h2_meta = []

multi = model.multistage

and_idx = torch.argmax(
    multi.and_pi.detach(),
    dim=-1,
).cpu().numpy()

or_idx = torch.argmax(
    multi.or_pi.detach(),
    dim=-1,
).cpu().numpy()

for unit in range(64):
    indices = [
        int(i)
        for i in and_idx[unit]
    ]

    h2.append(
        node(
            "AND",
            [h1[i] for i in indices],
        )
    )

    h2_meta.append(
        f"h2 AND unit{unit} <- {indices}"
    )

for unit in range(64):
    indices = [
        int(i)
        for i in or_idx[unit]
    ]

    h2.append(
        node(
            "OR",
            [h1[i] for i in indices],
        )
    )

    h2_meta.append(
        f"h2 OR unit{unit} <- {indices}"
    )


final_rules = h1 + h2
final_meta = h1_meta + h2_meta

weights = F.softplus(
    model.raw_classifier_weights.detach()
).cpu().numpy()

class_id = class_to_id[TARGET_CLASS]
class_weights = weights[class_id]

ranking = np.argsort(
    class_weights
)[::-1]

rank_lookup = {
    int(feature_idx): rank
    for rank, feature_idx in enumerate(
        ranking,
        start=1,
    )
}


# ============================================================
# 1. MV101 binary state statistics
# ============================================================

print("=" * 72)
print("1. MV101 BINARY STATE MATCHING")
print("=" * 72)

mv_cols = [
    col
    for col in train_df.columns
    if col.startswith("MV101_")
]

attack_train = train_df[
    train_df["Attack_Type"] == TARGET_CLASS
]

attack_val = val_df[
    val_df["Attack_Type"] == TARGET_CLASS
]

normal_train = train_df[
    train_df["Attack_Type"] == "Normal"
]


for col in mv_cols:
    print(
        f"{col:20s} | "
        f"Attack1 train={attack_train[col].mean():.4f} | "
        f"Attack1 val={attack_val[col].mean():.4f} | "
        f"Normal train={normal_train[col].mean():.4f}"
    )


# ============================================================
# 2. LIT101 matching
# ============================================================

print("\n" + "=" * 72)
print("2. LIT101 CONDITIONS")
print("=" * 72)

lit_cols = [
    col
    for col in train_df.columns
    if col.startswith("LIT101_")
]

for col in lit_cols:
    print(
        f"{col:28s} | "
        f"Attack1 train={attack_train[col].mean():.4f} | "
        f"Attack1 val={attack_val[col].mean():.4f} | "
        f"Normal train={normal_train[col].mean():.4f} | "
        f"{semantic(col)}"
    )


# ============================================================
# 3. All final rules that contain MV101
# ============================================================

device_rules = []

for idx, expr in enumerate(final_rules):
    rule_leaves = leaves(expr)

    if any(
        name.startswith(TARGET_DEVICE)
        for name in rule_leaves
    ):
        device_rules.append({
            "index": idx,
            "rank": rank_lookup[idx],
            "weight": float(class_weights[idx]),
            "meta": final_meta[idx],
            "contains_lit101": any(
                name.startswith(RELATED_SENSOR)
                for name in rule_leaves
            ),
            "expression": text(expr),
        })


device_rules = sorted(
    device_rules,
    key=lambda row: row["rank"],
)


print("\n" + "=" * 72)
print("3. HIGHEST-RANKED ATTACK-1 RULES CONTAINING MV101")
print("=" * 72)

if not device_rules:
    print("No learned final logical feature contains MV101.")
else:
    for row in device_rules[:TOP_N_DEVICE_RULES]:
        print(
            f"rank={row['rank']:3d} | "
            f"weight={row['weight']:.6f} | "
            f"LIT101={row['contains_lit101']} | "
            f"{row['meta']} | "
            f"{row['expression']}"
        )


# ============================================================
# 4. Rules containing BOTH MV101 and LIT101
# ============================================================

both = [
    row
    for row in device_rules
    if row["contains_lit101"]
]


print("\n" + "=" * 72)
print("4. RULES CONTAINING BOTH MV101 AND LIT101")
print("=" * 72)

if not both:
    print("No final logical feature contains both MV101 and LIT101.")
else:
    for row in both:
        print(
            f"rank={row['rank']:3d} | "
            f"weight={row['weight']:.6f} | "
            f"{row['meta']} | "
            f"{row['expression']}"
        )


# ============================================================
# 5. Summary
# ============================================================

print("\n" + "=" * 72)
print("5. SUMMARY")
print("=" * 72)

print(
    "Number of final logical features containing MV101:",
    len(device_rules),
)

if device_rules:
    print(
        "Best Attack-1 classifier rank of an MV101 rule:",
        device_rules[0]["rank"],
    )

print(
    "Number of final logical features containing both MV101 and LIT101:",
    len(both),
)

if both:
    print(
        "Best Attack-1 classifier rank of an MV101+LIT101 rule:",
        both[0]["rank"],
    )
