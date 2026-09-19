import math
import numpy as np
import pandas as pd

TRAIN_LOGITS_FILE = "LU_IDS_stratified_fold4_train_logits.npy"
VAL_LOGITS_FILE = "LU_IDS_stratified_fold4_val_logits.npy"
TRAIN_DATA_FILE = "ALRL_v2_fold4_train_binary.csv"
VAL_DATA_FILE = "ALRL_v2_fold4_val_binary.csv"
OUTPUT_FILE = "KD_teacher_temperature_diagnostic.csv"

PRACTICAL_ATTACK_IDS = [
    1,2,3,4,6,7,8,10,11,13,16,17,19,20,21,22,23,24,25,
    26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41
]
CLASSES = ["Normal"] + [f"Attack_{i}" for i in PRACTICAL_ATTACK_IDS]
CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
TEMPERATURES = [1,2,5,10,20,50,100]

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def analyze(split, logits, labels):
    row_range = logits.max(axis=1) - logits.min(axis=1)
    print(f"\n=== {split.upper()} ===")
    print("shape:", logits.shape)
    print("global min/max:", float(logits.min()), float(logits.max()))
    print("mean |logit|:", float(np.mean(np.abs(logits))))
    print("mean row range:", float(row_range.mean()))
    print("T  mean_top1  mean_true  norm_entropy  mean_top2_gap")
    rows = []
    for T in TEMPERATURES:
        p = softmax(logits / T)
        sorted_p = np.sort(p, axis=1)
        top1 = p.max(axis=1)
        true_p = p[np.arange(len(labels)), labels]
        ent = -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)
        norm_ent = ent / math.log(NUM_CLASSES)
        gap = sorted_p[:, -1] - sorted_p[:, -2]
        pred = logits.argmax(axis=1)
        acc = float(np.mean(pred == labels))
        row = {
            "split": split,
            "temperature": T,
            "teacher_accuracy": acc,
            "mean_top1_probability": float(top1.mean()),
            "mean_true_class_probability": float(true_p.mean()),
            "mean_normalized_entropy": float(norm_ent.mean()),
            "mean_top2_probability_gap": float(gap.mean()),
            "mean_row_logit_range": float(row_range.mean()),
            "mean_abs_logit": float(np.mean(np.abs(logits))),
        }
        rows.append(row)
        print(
            f"{T:<3} "
            f"{row['mean_top1_probability']:.4f}      "
            f"{row['mean_true_class_probability']:.4f}      "
            f"{row['mean_normalized_entropy']:.4f}         "
            f"{row['mean_top2_probability_gap']:.4f}"
        )
    return rows

train_logits = np.load(TRAIN_LOGITS_FILE).astype(np.float64)
val_logits = np.load(VAL_LOGITS_FILE).astype(np.float64)

train_df = pd.read_csv(TRAIN_DATA_FILE, usecols=["Attack_Type"])
val_df = pd.read_csv(VAL_DATA_FILE, usecols=["Attack_Type"])

train_labels = train_df["Attack_Type"].map(CLASS_TO_ID).to_numpy(dtype=np.int64)
val_labels = val_df["Attack_Type"].map(CLASS_TO_ID).to_numpy(dtype=np.int64)

assert train_logits.shape == (len(train_labels), NUM_CLASSES)
assert val_logits.shape == (len(val_labels), NUM_CLASSES)

rows = analyze("train", train_logits, train_labels)
rows += analyze("validation", val_logits, val_labels)

result = pd.DataFrame(rows)
result.to_csv(OUTPUT_FILE, index=False)

t10 = result[(result["split"]=="validation") & (result["temperature"]==10)].iloc[0]

print("\n=== T=10 INTERPRETATION ===")
print("Validation mean top1 probability:", f"{t10['mean_top1_probability']:.4f}")
print("Validation normalized entropy:", f"{t10['mean_normalized_entropy']:.4f}")

if t10["mean_top1_probability"] > 0.80:
    print("T=10 is still VERY SHARP for this reconstructed Teacher.")
elif t10["mean_top1_probability"] > 0.50:
    print("T=10 is moderately sharp for this reconstructed Teacher.")
else:
    print("T=10 produces fairly soft Teacher targets.")

print("Saved:", OUTPUT_FILE)
