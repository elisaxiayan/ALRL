
from pathlib import Path
import gc, json, random, re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from alrl_model import ALRL
from swat_data import get_stage_feature_columns, SWaTBinaryDataset


# ============================================================
# ALRL V2 - Complete 5-fold evaluation, NO KD
#
# Saves/resumes Fold 0-3 and reuses the already completed Fold 4
# when its checkpoint/report are present.
#
# For EACH new validation fold:
#   1) fit q05/q95 from training Normal only
#   2) learn attack-range features from training attack labels only
#   3) transform train/validation using that fold's config
#   4) train ALRL for 50 epochs
#   5) keep best validation Macro-F1 checkpoint
#
# No validation labels are used to choose thresholds/ranges.
# ============================================================

RAW_FILE = "ALRL_paper_like_stratified_5fold.csv"

OUT_DIR = Path("ALRL_v2_5fold_runs")
OUT_DIR.mkdir(exist_ok=True)

SUMMARY_CSV = "ALRL_v2_5fold_summary.csv"
SUMMARY_TXT = "ALRL_v2_5fold_summary.txt"

OLD_FOLD4_CKPT = Path("ALRL_v2_fold4_best_nodistill.pt")
OLD_FOLD4_REPORT = Path("ALRL_v2_fold4_classification_report_best.csv")

SEED = 42
EPOCHS = 50
BATCH_SIZE = 32
LR0, LR1 = 0.01, 0.001
TAU0, TAU1 = 1.0, 0.0001

BASE_Q_LOW, BASE_Q_HIGH = 0.05, 0.95
ATTACK_Q_LOW, ATTACK_Q_HIGH = 0.05, 0.95
TOP_K = 5
MIN_COVERAGE = 0.60
MIN_SPECIFICITY = 0.10
MIN_ATTACK_N = 20

SENSORS = [
    "FIT101","LIT101",
    "AIT201","AIT202","AIT203","FIT201",
    "DPIT301","FIT301","LIT301",
    "AIT401","AIT402","FIT401","LIT401",
    "AIT501","AIT502","AIT503","AIT504",
    "FIT501","FIT502","FIT503","FIT504",
    "PIT501","PIT502","PIT503",
    "FIT601"
]

ACTUATORS = [
    "MV101","P101","P102",
    "MV201","P201","P202","P203","P204","P205","P206",
    "MV301","MV302","MV303","MV304","P301","P302",
    "P401","P402","P403","P404","UV401",
    "P501","P502",
    "P601","P602","P603"
]

ATTACK_IDS = [
    1,2,3,4,6,7,8,10,11,13,16,17,19,20,21,22,23,24,25,
    26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41
]

ATTACK_CLASSES = [f"Attack_{i}" for i in ATTACK_IDS]
CLASSES = ["Normal"] + ATTACK_CLASSES
CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

print("Device:", device)


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def state_text(v):
    x = float(v)
    if x.is_integer():
        return str(int(x))
    return re.sub(r"[^0-9A-Za-z]+", "_", f"{x:.10g}")


df = pd.read_csv(RAW_FILE, low_memory=False)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="raise")

assert len(df) == 47044
assert df["Attack_Type"].nunique() == 36
assert set(df["Fold"].unique()) == {0,1,2,3,4}

print("Frozen dataset:", df.shape)


# ============================================================
# V2 fitting
# ============================================================

def fit_v2(train):
    normal = train[train["Attack_Type"] == "Normal"]

    base = {}
    for s in SENSORS:
        x = pd.to_numeric(normal[s], errors="raise")
        base[s] = {
            "low": float(x.quantile(BASE_Q_LOW)),
            "high": float(x.quantile(BASE_Q_HIGH)),
        }

    candidates = []

    for attack in ATTACK_CLASSES:
        target = train[train["Attack_Type"] == attack]
        other = train[train["Attack_Type"] != attack]

        if len(target) < MIN_ATTACK_N:
            continue

        rows = []

        for s in SENSORS:
            xt = pd.to_numeric(target[s], errors="raise")
            xo = pd.to_numeric(other[s], errors="raise")

            low = float(xt.quantile(ATTACK_Q_LOW))
            high = float(xt.quantile(ATTACK_Q_HIGH))

            tc = float(((xt >= low) & (xt <= high)).mean())
            oc = float(((xo >= low) & (xo <= high)).mean())
            score = tc - oc

            rows.append({
                "source_attack": attack,
                "sensor": s,
                "low": low,
                "high": high,
                "target_coverage": tc,
                "other_coverage": oc,
                "specificity_score": score,
                "fallback_selected": False,
            })

        good = [
            r for r in rows
            if r["target_coverage"] >= MIN_COVERAGE
            and r["specificity_score"] >= MIN_SPECIFICITY
        ]

        good.sort(
            key=lambda r: (
                r["specificity_score"],
                r["target_coverage"],
                -r["other_coverage"]
            ),
            reverse=True
        )

        chosen = good[:TOP_K]

        if not chosen and rows:
            best = max(rows, key=lambda r: r["specificity_score"])
            if best["specificity_score"] > 0:
                best = dict(best)
                best["fallback_selected"] = True
                chosen = [best]

        candidates.extend(chosen)

    dedup = {}

    for row in candidates:
        key = (
            row["sensor"],
            round(row["low"], 6),
            round(row["high"], 6)
        )

        if key not in dedup:
            dedup[key] = dict(row)
            dedup[key]["source_attacks"] = [row["source_attack"]]
        else:
            dedup[key]["source_attacks"].append(row["source_attack"])
            if row["specificity_score"] > dedup[key]["specificity_score"]:
                attacks = dedup[key]["source_attacks"]
                dedup[key] = dict(row)
                dedup[key]["source_attacks"] = attacks

    ranges = list(dedup.values())
    ranges.sort(
        key=lambda r: (
            r["sensor"],
            -r["specificity_score"],
            r["low"],
            r["high"]
        )
    )

    per_sensor = defaultdict(int)

    for row in ranges:
        s = row["sensor"]
        idx = per_sensor[s]
        per_sensor[s] += 1
        row["feature_name"] = f"{s}_ATTACK_RANGE_{idx:03d}"
        row["source_attacks"] = ",".join(
            sorted(set(row["source_attacks"]))
        )

    actuators = {}

    for a in ACTUATORS:
        x = pd.to_numeric(train[a], errors="raise")
        actuators[a] = {
            "states": [float(v) for v in sorted(x.unique().tolist())]
        }

    return base, ranges, actuators


def transform(data, base, ranges, actuator_cfg):
    meta = data[
        ["Timestamp", "Attack_Type", "Fold"]
    ].reset_index(drop=True)

    cols = {}

    for s in SENSORS:
        x = pd.to_numeric(data[s], errors="raise").reset_index(drop=True)
        lo, hi = base[s]["low"], base[s]["high"]

        cols[f"{s}_LOW"] = (x < lo).astype("int8")
        cols[f"{s}_NORMAL"] = ((x >= lo) & (x <= hi)).astype("int8")
        cols[f"{s}_HIGH"] = (x > hi).astype("int8")

    for row in ranges:
        s = row["sensor"]
        x = pd.to_numeric(data[s], errors="raise").reset_index(drop=True)

        cols[row["feature_name"]] = (
            (x >= row["low"]) & (x <= row["high"])
        ).astype("int8")

    for a in ACTUATORS:
        x = pd.to_numeric(data[a], errors="raise").reset_index(drop=True)
        states = actuator_cfg[a]["states"]

        unseen = set(float(v) for v in x.unique()) - set(states)
        if unseen:
            raise RuntimeError(f"{a} unseen states: {sorted(unseen)}")

        for state in states:
            cols[f"{a}_EQ_{state_text(state)}"] = (
                x == state
            ).astype("int8")

    return pd.concat([meta, pd.DataFrame(cols)], axis=1)


# ============================================================
# ALRL evaluation
# ============================================================

def move_stages(xs):
    return [x.to(device) for x in xs]


def evaluate(model, loader):
    model.eval()

    y_true, y_pred = [], []
    loss_sum = 0.0

    with torch.no_grad():
        for stage_inputs, labels in loader:
            stage_inputs = move_stages(stage_inputs)
            labels = labels.to(device)

            logits = model(
                stage_inputs,
                tau=TAU1,
                training=False
            )["logits"]

            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)

            pred = logits.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "loss": loss_sum / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "targets": y_true,
        "predictions": y_pred,
    }


# ============================================================
# One fold, resume-friendly
# ============================================================

def run_fold(fold):
    fold_dir = OUT_DIR / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    result_file = fold_dir / "result.json"

    if result_file.exists():
        print(f"\nFold {fold} already complete -> skip.")
        return json.loads(result_file.read_text(encoding="utf-8"))

    print("\n" + "="*60)
    print(f"FOLD {fold}")
    print("="*60)

    train_raw = df[df["Fold"] != fold].copy()
    val_raw = df[df["Fold"] == fold].copy()

    print("Train:", len(train_raw), "Val:", len(val_raw))

    base, ranges, actuator_cfg = fit_v2(train_raw)

    print("Selected attack ranges:", len(ranges))

    train_bin = transform(train_raw, base, ranges, actuator_cfg)
    val_bin = transform(val_raw, base, ranges, actuator_cfg)

    stage_cols = get_stage_feature_columns(train_bin)
    val_stage_cols = get_stage_feature_columns(val_bin)

    for stage in range(1, 7):
        assert stage_cols[stage] == val_stage_cols[stage]

    stage_dims = [len(stage_cols[i]) for i in range(1, 7)]
    total_features = sum(stage_dims)

    print("Stage dims:", stage_dims)
    print("Binary features:", total_features)

    pd.DataFrame(ranges).to_csv(
        fold_dir / "selected_attack_ranges.csv",
        index=False
    )

    config = {
        "fold": fold,
        "base_sensor_boundaries": base,
        "selected_ranges": ranges,
        "actuators": actuator_cfg,
        "stage_dims": stage_dims,
        "total_binary_features": total_features,
    }

    (fold_dir / "binarization_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8"
    )

    train_ds = SWaTBinaryDataset(
        train_bin, stage_cols, CLASS_TO_ID
    )
    val_ds = SWaTBinaryDataset(
        val_bin, stage_cols, CLASS_TO_ID
    )

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=g
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    seed_all()

    model = ALRL(
        stage_dims=stage_dims,
        num_classes=NUM_CLASSES,
        single_and=24,
        single_or=24,
        multi_and=64,
        multi_or=64,
        m=2
    ).to(device)

    params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR0
    )

    best = None
    best_state = None
    history = []

    for epoch in range(EPOCHS):
        model.train()

        p = epoch / (EPOCHS - 1)
        tau = TAU0 * (TAU1 / TAU0) ** p
        lr = LR0 + (LR1 - LR0) * p

        for group in optimizer.param_groups:
            group["lr"] = lr

        train_loss_sum = 0.0

        for stage_inputs, labels in train_loader:
            stage_inputs = move_stages(stage_inputs)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(
                stage_inputs,
                tau=tau,
                training=True
            )["logits"]

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)

        train_loss = train_loss_sum / len(train_ds)
        val_result = evaluate(model, val_loader)

        print(
            f"Fold {fold} | Epoch {epoch+1:02d}/50 | "
            f"Train={train_loss:.4f} | "
            f"Val={val_result['loss']:.4f} | "
            f"Acc={val_result['accuracy']:.4f} | "
            f"F1={val_result['f1']:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_result["loss"],
            "accuracy": val_result["accuracy"],
            "precision_macro": val_result["precision"],
            "recall_macro": val_result["recall"],
            "f1_macro": val_result["f1"],
            "tau": tau,
            "lr": lr,
        })

        if best is None or val_result["f1"] > best["f1"]:
            best = {
                "epoch": epoch + 1,
                "accuracy": val_result["accuracy"],
                "precision": val_result["precision"],
                "recall": val_result["recall"],
                "f1": val_result["f1"],
                "targets": val_result["targets"].copy(),
                "predictions": val_result["predictions"].copy(),
            }

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    pd.DataFrame(history).to_csv(
        fold_dir / "training_history.csv",
        index=False
    )

    torch.save(
        {
            "model_state_dict": best_state,
            "fold": fold,
            "best_epoch": best["epoch"],
            "validation_accuracy": best["accuracy"],
            "validation_f1_macro": best["f1"],
            "stage_dims": stage_dims,
            "stage_columns": stage_cols,
            "classes": CLASSES,
            "class_to_id": CLASS_TO_ID,
            "total_binary_features": total_features,
            "trainable_parameters": params,
            "seed": SEED,
        },
        fold_dir / "best_model.pt"
    )

    report = classification_report(
        best["targets"],
        best["predictions"],
        labels=list(range(NUM_CLASSES)),
        target_names=CLASSES,
        zero_division=0,
        output_dict=True
    )

    pd.DataFrame(report).transpose().to_csv(
        fold_dir / "classification_report.csv"
    )

    cm = confusion_matrix(
        best["targets"],
        best["predictions"],
        labels=list(range(NUM_CLASSES))
    )

    pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES
    ).to_csv(
        fold_dir / "confusion_matrix.csv"
    )

    result = {
        "fold": fold,
        "reused": False,
        "best_epoch": int(best["epoch"]),
        "accuracy": float(best["accuracy"]),
        "precision_macro": float(best["precision"]),
        "recall_macro": float(best["recall"]),
        "f1_macro": float(best["f1"]),
        "train_rows": int(len(train_raw)),
        "val_rows": int(len(val_raw)),
        "binary_features": int(total_features),
        "trainable_parameters": int(params),
        "selected_attack_ranges": int(len(ranges)),
    }

    result_file.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    del train_bin, val_bin, train_ds, val_ds
    del train_loader, val_loader, model, optimizer

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# Reuse already-completed Fold 4
# ============================================================

def read_existing_fold4():
    if not (
        OLD_FOLD4_CKPT.exists()
        and OLD_FOLD4_REPORT.exists()
    ):
        return None

    ckpt = torch.load(
        OLD_FOLD4_CKPT,
        map_location="cpu",
        weights_only=False
    )

    report = pd.read_csv(
        OLD_FOLD4_REPORT,
        index_col=0
    )

    # sklearn output_dict CSV stores accuracy in the precision cell.
    return {
        "fold": 4,
        "reused": True,
        "best_epoch": int(ckpt["epoch"]),
        "accuracy": float(report.loc["accuracy", "precision"]),
        "precision_macro": float(report.loc["macro avg", "precision"]),
        "recall_macro": float(report.loc["macro avg", "recall"]),
        "f1_macro": float(report.loc["macro avg", "f1-score"]),
        "train_rows": 37686,
        "val_rows": 9358,
        "binary_features": int(
            ckpt.get("total_binary_features", 298)
        ),
        "trainable_parameters": int(
            ckpt.get("trainable_parameters", 117312)
        ),
        "selected_attack_ranges": 173,
    }


# ============================================================
# Run Fold 0-3, reuse Fold 4
# ============================================================

results = []

for fold in range(4):
    results.append(run_fold(fold))

fold4 = read_existing_fold4()

if fold4 is None:
    print("\nExisting Fold 4 output not found; training Fold 4.")
    fold4 = run_fold(4)
else:
    print(
        "\nReused existing Fold 4:",
        f"Acc={fold4['accuracy']:.4f}",
        f"F1={fold4['f1_macro']:.4f}"
    )

results.append(fold4)


# ============================================================
# Final 5-fold aggregate
# ============================================================

res = pd.DataFrame(results).sort_values("fold").reset_index(drop=True)
res.to_csv(SUMMARY_CSV, index=False)

metric_names = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro"
]

agg = {}

for m in metric_names:
    agg[m + "_mean"] = float(res[m].mean())
    agg[m + "_std"] = float(res[m].std(ddof=1))

print("\n" + "="*60)
print("FINAL ALRL V2 FIVE-FOLD RESULT")
print("="*60)

print(
    res[
        [
            "fold",
            "best_epoch",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "binary_features",
            "selected_attack_ranges",
            "reused",
        ]
    ].to_string(index=False)
)

print("\nMean ± std")
print(
    f"Accuracy:        {agg['accuracy_mean']:.4f} "
    f"± {agg['accuracy_std']:.4f}"
)
print(
    f"Macro Precision: {agg['precision_macro_mean']:.4f} "
    f"± {agg['precision_macro_std']:.4f}"
)
print(
    f"Macro Recall:    {agg['recall_macro_mean']:.4f} "
    f"± {agg['recall_macro_std']:.4f}"
)
print(
    f"Macro F1:        {agg['f1_macro_mean']:.4f} "
    f"± {agg['f1_macro_std']:.4f}"
)

summary = [
    "ALRL V2 FIVE-FOLD SUMMARY",
    "=========================",
    "",
    res.to_string(index=False),
    "",
    (
        f"Accuracy mean±std: "
        f"{agg['accuracy_mean']:.6f} ± {agg['accuracy_std']:.6f}"
    ),
    (
        f"Macro Precision mean±std: "
        f"{agg['precision_macro_mean']:.6f} ± "
        f"{agg['precision_macro_std']:.6f}"
    ),
    (
        f"Macro Recall mean±std: "
        f"{agg['recall_macro_mean']:.6f} ± "
        f"{agg['recall_macro_std']:.6f}"
    ),
    (
        f"Macro F1 mean±std: "
        f"{agg['f1_macro_mean']:.6f} ± "
        f"{agg['f1_macro_std']:.6f}"
    ),
    "",
    "KD disabled.",
    "V2 preprocessing refit independently inside each training fold.",
]

Path(SUMMARY_TXT).write_text(
    "\n".join(summary),
    encoding="utf-8"
)

print("\nSaved:", SUMMARY_CSV)
print("Saved:", SUMMARY_TXT)
print("Per-fold directory:", OUT_DIR)
