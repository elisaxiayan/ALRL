ALRL REPRODUCTION CORE PACKAGE
==============================

01_PY_SCRIPTS/
  CORE/
    Core model, preprocessing, training, 5-fold evaluation,
    explanation, and final-summary scripts.

  DIAGNOSTICS/
    Collision, KD-temperature, Attack-1 MV101 and other
    diagnostic scripts. These are useful for understanding
    how the reproduction was debugged, but are not required
    for the final inference result.

02_RESULTS/
  CORE/
    Final 5-fold classification results, Fold-4 best model,
    rule extraction outputs, Attack-1 rule validation, and
    per-fold checkpoints/configs/reports.

  TEACHER_KD/
    LU-IDS-style teacher outputs, teacher logits/alignment
    files, KD experiment outputs and KD diagnostics.

03_FROZEN_DATA/
  Frozen datasets / binary representations used by the final
  reproduction. Raw original SWaT Excel files are intentionally
  not copied here because they can be much larger and are not
  necessary for downloading the core reproduction package.

MOST IMPORTANT FILES
--------------------

Model code:
  01_PY_SCRIPTS/CORE/alrl_model.py
  01_PY_SCRIPTS/CORE/swat_data.py

Final experiment:
  01_PY_SCRIPTS/CORE/run_alrl_v2_5fold.py
  01_PY_SCRIPTS/CORE/extract_alrl_v2_rules.py
  01_PY_SCRIPTS/CORE/finalize_alrl_reproduction.py

Final results:
  02_RESULTS/CORE/ALRL_REPRODUCTION_FINAL_SUMMARY.txt
  02_RESULTS/CORE/ALRL_v2_5fold_summary.csv
  02_RESULTS/CORE/ALRL_attack1_core_rule_metrics.csv

Final 5-fold result:
  Accuracy        = 0.9919 ± 0.0011
  Macro Precision = 0.9871 ± 0.0025
  Macro Recall    = 0.9824 ± 0.0043
  Macro F1        = 0.9844 ± 0.0021

Attack-1 learned core rule:
  LIT101 > 813.513 AND MV101 = 2

Fold-4 Attack-1 rule validation:
  Precision = 0.9788
  Recall    = 0.9840
  F1        = 0.9814
