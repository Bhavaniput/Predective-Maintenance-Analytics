"""
Predictive Maintenance - Steps 9-13
=====================================
Continues directly from train_pipeline.py (Steps 1-8). Re-derives the same
train/test split and the same three candidate models (via common/data_prep,
so nothing is duplicated or allowed to drift), then:

  9.  Validates model stability with stratified cross-validation on the
      TRAINING data only.
  10. Analyzes recall/errors, and selects an operating threshold using
      cross-validated out-of-fold probabilities on the TRAINING data only
      (never the test set) - then reports that threshold's performance,
      once, on the held-out test set.
  11. Computes global + local explanations (SHAP if available, else a
      tested permutation-based fallback - see common/explainability.py).
  12. Attaches illustrative maintenance recommendations (common/maintenance.py).
  13. Saves the FINAL pipeline + chosen threshold + full metadata, and
      verifies the saved file reloads and predicts correctly.

Run after train_pipeline.py:  python validate_explain_finalize.py
"""

import sys
import os
import json
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
)

from common.data_prep import (
    RANDOM_STATE, load_and_split, make_preprocessor, CATEGORICAL_COLS,
    ENGINEERED_NUMERICAL_COLS, FEATURE_ORDER, RAW_NUMERICAL_COLS,
)
from common.explainability import (
    global_importance_permutation, global_importance_shap, SHAP_AVAILABLE,
    explain_local, reference_values_from,
)
from common.maintenance import recommendation_for

np.random.seed(RANDOM_STATE)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJECT_DIR, "figures")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
sns.set_style("whitegrid")

X_train, X_test, y_train, y_test = load_and_split()

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting (HistGB)": HistGradientBoostingClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
}

# ---------------------------------------------------------------------------
# STEP 9 - VALIDATE MODEL STABILITY (stratified CV, TRAIN data only)
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 9 - VALIDATE MODEL STABILITY (5-fold stratified CV on TRAIN only)")
print("=" * 70)
print(
    "Each fold re-fits the full Pipeline (scaler + one-hot-encoder + model)\n"
    "from scratch on that fold's training portion, so preprocessing statistics\n"
    "never see the held-out fold -> no leakage across folds. The test set is\n"
    "not touched anywhere in this step.\n"
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {"recall": "recall", "precision": "precision", "f1": "f1", "roc_auc": "roc_auc"}

cv_summary = {}
for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocessor", make_preprocessor()), ("classifier", clf)])
    cv_results = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    summary = {}
    print(f"\n--- {name} (5-fold CV on training data) ---")
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        summary[metric] = {"mean": float(scores.mean()), "std": float(scores.std())}
        print(f"{metric:10s}: mean={scores.mean():.4f}  std={scores.std():.4f}  folds={np.round(scores, 3).tolist()}")
    cv_summary[name] = summary

with open(os.path.join(FIG_DIR, "cv_stability_results.json"), "w") as f:
    json.dump(cv_summary, f, indent=2)

print(
    "\nOverfitting check: compare each model's CV-mean Recall/F1/ROC-AUC above "
    "to its single train/test-split numbers from train_pipeline.py (Step 8). "
    "A CV mean close to the single-split test score, with a modest standard "
    "deviation across folds, indicates the result is stable rather than a "
    "lucky/unlucky split. A large std or a big gap between CV mean and the "
    "single-split score would indicate instability or overfitting - inspect "
    "figures/cv_stability_results.json and the printed fold-by-fold scores "
    "above for the actual numbers produced on this run."
)

best_model_name = "Gradient Boosting (HistGB)"  # carried over from Step 8 selection
best_clf = models[best_model_name]

# ---------------------------------------------------------------------------
# STEP 10 - RECALL / ERROR EVALUATION + THRESHOLD SELECTION
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 10 - RECALL / ERROR EVALUATION + THRESHOLD SELECTION")
print("=" * 70)
print(
    "Threshold selection uses cross_val_predict on the TRAINING set only "
    "(out-of-fold probabilities), never the final test set - the test set "
    "is reserved for a single, final, honest performance read-out below."
)

best_pipe_template = Pipeline(steps=[("preprocessor", make_preprocessor()), ("classifier", best_clf)])
train_oof_proba = cross_val_predict(
    best_pipe_template, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
)[:, 1]

candidate_thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
threshold_rows = []
for t in candidate_thresholds:
    preds = (train_oof_proba >= t).astype(int)
    threshold_rows.append({
        "threshold": t,
        "precision": precision_score(y_train, preds, zero_division=0),
        "recall": recall_score(y_train, preds, zero_division=0),
        "f1": f1_score(y_train, preds, zero_division=0),
    })
threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(os.path.join(FIG_DIR, "threshold_analysis_train_oof.csv"), index=False)
print("\nThreshold sweep on TRAIN out-of-fold probabilities:")
print(threshold_df.to_string(index=False))

# Selection rule, decided on TRAIN data only: the lowest threshold that
# still achieves at least 0.80 recall, breaking ties toward higher F1 -
# i.e. "don't miss failures, but don't lower the bar further than needed."
MIN_RECALL_TARGET = 0.80
eligible = threshold_df[threshold_df["recall"] >= MIN_RECALL_TARGET]
if len(eligible) > 0:
    selected_threshold = float(eligible.sort_values(["recall", "f1"], ascending=[True, False]).iloc[0]["threshold"])
else:
    # No threshold hit the recall target: fall back to the threshold with max recall.
    selected_threshold = float(threshold_df.sort_values("recall", ascending=False).iloc[0]["threshold"])

print(
    f"\nSelected operating threshold = {selected_threshold} "
    f"(lowest threshold reaching >= {MIN_RECALL_TARGET:.0%} recall on train "
    "out-of-fold predictions, chosen for highest F1 among ties). This value "
    "was picked without looking at the test set."
)

# Final, single, honest read on the held-out TEST set using the fitted
# Step-8 pipeline and the threshold chosen above.
import joblib as _joblib_check  # noqa
best_pipeline = joblib.load(os.path.join(MODEL_DIR, "predictive_maintenance_model.pkl"))
test_proba = best_pipeline.predict_proba(X_test)[:, 1]
test_pred_default = (test_proba >= 0.5).astype(int)
test_pred_selected = (test_proba >= selected_threshold).astype(int)

print(f"\n--- TEST SET performance at default threshold 0.5 ---")
print(confusion_matrix(y_test, test_pred_default))
print(classification_report(y_test, test_pred_default, target_names=["No Failure", "Failure"], zero_division=0))

print(f"\n--- TEST SET performance at SELECTED threshold {selected_threshold} ---")
cm_selected = confusion_matrix(y_test, test_pred_selected)
print(cm_selected)
print(classification_report(y_test, test_pred_selected, target_names=["No Failure", "Failure"], zero_division=0))

tn, fp, fn, tp = cm_selected.ravel()
print(
    f"\nFalse negatives (missed failures): {fn} of {fn+tp} actual failures in the test set.\n"
    f"False positives (false alarms): {fp} of {tn+fp} actual healthy machines in the test set.\n"
    "Recall matters most here because a false negative means an unplanned "
    "stoppage on the floor - the exact event this project exists to prevent "
    "- while a false positive only costs one avoidable inspection."
)

# Confusion matrix @ selected threshold
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_selected, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"])
ax.set_title(f"Test Confusion Matrix @ threshold={selected_threshold}")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "11_confusion_matrix_selected_threshold.png"), dpi=120)
plt.close()

# ROC curve (test set)
fpr, tpr, _ = roc_curve(y_test, test_proba)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr, tpr, label=f"HistGB (AUC={roc_auc_score(y_test, test_proba):.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - Test Set"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "12_roc_curve_test.png"), dpi=120)
plt.close()

# Precision-Recall curve (test set), with selected threshold marked
prec, rec, pr_thresholds = precision_recall_curve(y_test, test_proba)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(rec, prec, label="Precision-Recall")
idx = np.argmin(np.abs(pr_thresholds - selected_threshold)) if len(pr_thresholds) else None
if idx is not None:
    ax.scatter([rec[idx]], [prec[idx]], color="red", zorder=5,
               label=f"Selected threshold={selected_threshold}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve - Test Set"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "13_precision_recall_curve_test.png"), dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# STEP 11 - EXPLAINABILITY (global + local)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 11 - EXPLAINABILITY (SHAP if available, else tested permutation fallback)")
print("=" * 70)
print(f"SHAP installed in this environment: {SHAP_AVAILABLE}")

if SHAP_AVAILABLE:
    try:
        global_imp = global_importance_shap(best_pipeline, X_train, X_test)
        global_method = "shap"
    except Exception as e:
        print(f"SHAP global importance failed ({e}); using permutation fallback.")
        global_imp = global_importance_permutation(best_pipeline, X_test, y_test)
        global_method = "permutation"
else:
    print("shap not installed in this sandbox (no network access to PyPI) -> using the tested permutation-importance fallback for global explainability.")
    global_imp = global_importance_permutation(best_pipeline, X_test, y_test)
    global_method = "permutation"

print(f"\nGlobal feature importance ({global_method}), evaluated on the TEST set:")
print(global_imp)

fig, ax = plt.subplots(figsize=(7, 5))
global_imp.plot(kind="barh", ax=ax)
ax.invert_yaxis()
ax.set_title(f"Global Feature Importance ({global_method})")
ax.set_xlabel("Mean |impact| on predicted failure probability" if global_method == "shap" else "Drop in Recall when shuffled")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "14_global_importance.png"), dpi=120)
plt.close()

print(
    "\nInterpretation note: these values describe ASSOCIATION the model has "
    "learned between each input and predicted failure risk, not a proven "
    "physical cause of failure - the dataset is observational, not a "
    "controlled experiment."
)

# Local explanation example - a handful of test-set rows, varied by predicted risk
print("\n--- Local explanation examples (test set) ---")
example_idx = (
    list(pd.Series(test_proba, index=X_test.index).sort_values(ascending=False).head(2).index) +
    list(pd.Series(test_proba, index=X_test.index).sort_values(ascending=True).head(1).index)
)
local_method_used = None
for idx in example_idx:
    raw_row = X_test.loc[idx, RAW_NUMERICAL_COLS + CATEGORICAL_COLS].to_dict()
    proba, contributions, method = explain_local(best_pipeline, raw_row, X_train)
    local_method_used = method
    actual_label = "Failure" if y_test.loc[idx] == 1 else "No Failure"
    print(f"\nRow {idx} | actual={actual_label} | predicted probability={proba:.3f} | method={method}")
    print(contributions)

fig, axes = plt.subplots(1, len(example_idx), figsize=(5 * len(example_idx), 4))
if len(example_idx) == 1:
    axes = [axes]
for ax, idx in zip(axes, example_idx):
    raw_row = X_test.loc[idx, RAW_NUMERICAL_COLS + CATEGORICAL_COLS].to_dict()
    proba, contributions, method = explain_local(best_pipeline, raw_row, X_train)
    contributions.sort_values().plot(kind="barh", ax=ax, color=contributions.sort_values().apply(lambda v: "#C44E52" if v > 0 else "#4C72B0"))
    ax.set_title(f"Row {idx}\nP(failure)={proba:.2f}")
    ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "15_local_explanation_examples.png"), dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# STEP 12 - MAINTENANCE SCHEDULING INSIGHTS
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 12 - MAINTENANCE SCHEDULING INSIGHTS (illustrative, not validated)")
print("=" * 70)

LOW_MAX = 0.30
HIGH_MIN = max(0.60, selected_threshold)  # keep High-risk aligned with the validated decision point
print(f"Risk categories used by the app: LOW < {LOW_MAX:.0%}  |  MEDIUM {LOW_MAX:.0%}-{HIGH_MIN:.0%}  |  HIGH >= {HIGH_MIN:.0%}")

for p in [0.05, 0.45, 0.85]:
    cat, rec = recommendation_for(p, low_max=LOW_MAX, high_min=HIGH_MIN)
    print(f"  P(failure)={p:.0%} -> {cat}: {rec}")

print(
    "\nThese categories/recommendations are illustrative project output for "
    "an academic demo - not a validated industrial maintenance standard and "
    "not based on real maintenance-cost data (see README limitations)."
)

# ---------------------------------------------------------------------------
# STEP 13 - SAVE FINAL PIPELINE + THRESHOLD + METADATA
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 13 - SAVE FINAL PIPELINE")
print("=" * 70)

model_path = os.path.join(MODEL_DIR, "predictive_maintenance_model.pkl")
joblib.dump(best_pipeline, model_path)  # re-save for a clean, final artifact

final_metrics_selected_threshold = {
    "precision": float(precision_score(y_test, test_pred_selected, zero_division=0)),
    "recall": float(recall_score(y_test, test_pred_selected, zero_division=0)),
    "f1": float(f1_score(y_test, test_pred_selected, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, test_proba)),
    "confusion_matrix": cm_selected.tolist(),
}

metadata = {
    "best_model_name": best_model_name,
    "numerical_cols_raw": RAW_NUMERICAL_COLS,
    "engineered_numerical_cols": ENGINEERED_NUMERICAL_COLS,
    "categorical_cols": CATEGORICAL_COLS,
    "feature_order": FEATURE_ORDER,
    "target_col": "Machine failure",
    "random_state": RANDOM_STATE,
    "cv_stability": cv_summary,
    "selected_threshold": selected_threshold,
    "threshold_selection_method": (
        f"lowest threshold with train-CV out-of-fold recall >= {MIN_RECALL_TARGET:.0%}, "
        "tie-broken by F1; selected using training data only"
    ),
    "risk_bands": {"low_max": LOW_MAX, "high_min": HIGH_MIN},
    "test_metrics_default_threshold_0.5": {
        "precision": float(precision_score(y_test, test_pred_default, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred_default, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred_default, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_proba)),
    },
    "test_metrics_selected_threshold": final_metrics_selected_threshold,
    "global_explainability_method": global_method,
    "local_explainability_method_last_run": local_method_used,
    "shap_available_at_build_time": SHAP_AVAILABLE,
    "reference_values": reference_values_from(X_train),
}
with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved final pipeline to: {model_path}")
print(f"Saved final metadata (incl. selected_threshold={selected_threshold}) to model_metadata.json")

# --- Verify the saved artifact reloads and predicts correctly ---
print("\n--- Verifying saved pipeline reload ---")
reloaded = joblib.load(model_path)
sample_row = X_test.iloc[[0]]
orig_proba = best_pipeline.predict_proba(sample_row)[0, 1]
reload_proba = reloaded.predict_proba(sample_row)[0, 1]
assert abs(orig_proba - reload_proba) < 1e-9, "Reloaded model prediction does not match original!"
print(f"OK: reloaded pipeline reproduces identical prediction ({reload_proba:.6f}).")

print("\nDone with Steps 9-13.")
