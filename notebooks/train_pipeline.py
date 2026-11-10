"""
Predictive Maintenance - AI4I 2020 Dataset
============================================
Steps 1-8: objective, dataset understanding, EDA, cleaning, trend analysis,
feature engineering, train/test split, and model training/comparison.

This script shares its data-loading, feature-engineering, and split logic
with notebooks/validate_explain_finalize.py (Steps 9-13) and app.py (Step 14)
via common/data_prep.py, so the three cannot silently drift apart.

Run from the `notebooks/` folder:  python train_pipeline.py

NOTE: this script trains the three candidate models and reports their
comparison, but the FINAL deployed pipeline (with its chosen decision
threshold) is saved by validate_explain_finalize.py, which performs the
additional Step 9-10 validation before anything is written to disk for the
Streamlit app. Run this script first, then that one.
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

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.inspection import permutation_importance

from common.data_prep import (
    RANDOM_STATE, load_raw_data, build_model_frame, load_and_split,
    make_preprocessor, ID_COLS, CATEGORICAL_COLS, RAW_NUMERICAL_COLS,
    ENGINEERED_NUMERICAL_COLS, FEATURE_ORDER, TARGET_COL, FAILURE_MODE_COLS,
)

np.random.seed(RANDOM_STATE)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJECT_DIR, "figures")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# STEP 2 - DATASET UNDERSTANDING
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 2 - DATASET UNDERSTANDING")
print("=" * 70)

df = load_raw_data()

print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumn names:\n{list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nMissing values per column:\n{df.isnull().sum()}")
print(f"\nTotal duplicate rows: {df.duplicated().sum()}")
print(f"\nDescriptive statistics (numeric):\n{df.describe()}")

print(f"\nIdentifier columns (excluded): {ID_COLS}")
print(f"Categorical feature columns: {CATEGORICAL_COLS}")
print(f"Numerical feature columns: {RAW_NUMERICAL_COLS}")
print(f"Target column: '{TARGET_COL}'")
print(f"Failure-mode sub-flag columns (excluded, leakage): {FAILURE_MODE_COLS}")
print(f"\nTarget class balance:\n{df[TARGET_COL].value_counts()}")
print(f"Failure rate: {df[TARGET_COL].mean() * 100:.2f}%")
print(f"\nProduct 'Type' distribution:\n{df['Type'].value_counts()}")

flag_sum = df[FAILURE_MODE_COLS].sum(axis=1)
mismatch_a = ((flag_sum > 0) & (df[TARGET_COL] == 0)).sum()
mismatch_b = ((flag_sum == 0) & (df[TARGET_COL] == 1)).sum()
print(f"\nRows where a failure-mode flag fired but Machine failure = 0: {mismatch_a}")
print(f"Rows where NO failure-mode flag fired but Machine failure = 1: {mismatch_b}")
print("-> Sub-flags are almost a direct encoding of the target -> DROPPED from features.")

# ---------------------------------------------------------------------------
# STEP 3 - HISTORICAL FAILURE ANALYSIS (EDA)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3 - HISTORICAL FAILURE ANALYSIS (EDA)")
print("=" * 70)

fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x=TARGET_COL, data=df, ax=ax, palette=["#4C72B0", "#C44E52"])
ax.set_xticklabels(["No Failure (0)", "Failure (1)"])
ax.set_title("Failure vs No-Failure Distribution")
ax.set_ylabel("Number of Machines")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_failure_distribution.png"), dpi=120)
plt.close()

fail_by_type = df.groupby("Type")[TARGET_COL].mean().sort_values(ascending=False) * 100
fig, ax = plt.subplots(figsize=(5, 4))
fail_by_type.plot(kind="bar", ax=ax, color="#DD8452")
ax.set_title("Failure Rate (%) by Product Type")
ax.set_ylabel("Failure Rate (%)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_failure_rate_by_type.png"), dpi=120)
plt.close()
print(f"\nFailure rate by Type (%):\n{fail_by_type}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(x=TARGET_COL, y="Air temperature [K]", data=df, ax=axes[0])
axes[0].set_title("Air Temperature vs Failure")
sns.boxplot(x=TARGET_COL, y="Process temperature [K]", data=df, ax=axes[1])
axes[1].set_title("Process Temperature vs Failure")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_temperature_vs_failure.png"), dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(5, 4))
sns.boxplot(x=TARGET_COL, y="Torque [Nm]", data=df, ax=ax)
ax.set_title("Torque vs Failure")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_torque_vs_failure.png"), dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(5, 4))
sns.boxplot(x=TARGET_COL, y="Rotational speed [rpm]", data=df, ax=ax)
ax.set_title("Rotational Speed vs Failure")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "05_rpm_vs_failure.png"), dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(5, 4))
sns.boxplot(x=TARGET_COL, y="Tool wear [min]", data=df, ax=ax)
ax.set_title("Tool Wear vs Failure")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "06_toolwear_vs_failure.png"), dpi=120)
plt.close()

corr_cols = RAW_NUMERICAL_COLS + [TARGET_COL]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Heatmap (Sensor Features & Target)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "07_correlation_heatmap.png"), dpi=120)
plt.close()
print(f"\nCorrelation of each numeric feature with target:\n{corr[TARGET_COL].sort_values(ascending=False)}")

# ---------------------------------------------------------------------------
# STEP 4 - DATA CLEANING
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4 - DATA CLEANING")
print("=" * 70)

print(f"Missing values total: {df.isnull().sum().sum()} (none -> no imputation required)")
range_checks = {c: (df[c].min(), df[c].max()) for c in RAW_NUMERICAL_COLS}
for k, v in range_checks.items():
    print(f"  {k}: min={v[0]}, max={v[1]}")
print(
    "\nOutlier policy: sensor outliers are RETAINED. An unusually high torque or "
    "tool-wear reading is frequently the genuine high-risk condition this model "
    "must learn to flag, not a data-entry error - deleting it would remove real "
    "failure signal rather than noise."
)

X_all, y_all = build_model_frame(df)
print(f"\nColumns retained for modeling: {list(X_all.columns)}")

# ---------------------------------------------------------------------------
# STEP 5 - FAILURE TREND ANALYSIS
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5 - FAILURE TREND ANALYSIS")
print("=" * 70)

group_means = df.groupby(TARGET_COL)[RAW_NUMERICAL_COLS].mean()
print(f"\nMean sensor values by failure status:\n{group_means}")

findings = []
if group_means.loc[1, "Torque [Nm]"] > group_means.loc[0, "Torque [Nm]"]:
    findings.append(f"Failed machines show higher average torque ({group_means.loc[1,'Torque [Nm]']:.1f} vs {group_means.loc[0,'Torque [Nm]']:.1f} Nm).")
if group_means.loc[1, "Tool wear [min]"] > group_means.loc[0, "Tool wear [min]"]:
    findings.append(f"Failed machines show higher average tool wear ({group_means.loc[1,'Tool wear [min]']:.1f} vs {group_means.loc[0,'Tool wear [min]']:.1f} min).")
if group_means.loc[1, "Process temperature [K]"] > group_means.loc[0, "Process temperature [K]"]:
    findings.append(f"Failed machines show slightly higher process temperature ({group_means.loc[1,'Process temperature [K]']:.1f} vs {group_means.loc[0,'Process temperature [K]']:.1f} K).")
findings.append(f"Failure rate by Type ranges {fail_by_type.min():.2f}%-{fail_by_type.max():.2f}%.")
for f in findings:
    print(" -", f)

with open(os.path.join(FIG_DIR, "eda_findings.json"), "w") as f:
    json.dump({"findings": findings, "failure_rate_by_type": fail_by_type.to_dict()}, f, indent=2)

# ---------------------------------------------------------------------------
# STEP 6 - FEATURE ENGINEERING (see common/data_prep.py:engineer_features)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6 - FEATURE ENGINEERING")
print("=" * 70)
print("Engineered: Temp_diff_K, Mechanical_power_W, ToolWear_x_Torque (see common/data_prep.py)")
print(X_all[["Temp_diff_K", "Mechanical_power_W", "ToolWear_x_Torque"]].describe())

# ---------------------------------------------------------------------------
# STEP 7 - TRAIN/TEST SPLIT
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7 - TRAIN/TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = load_and_split()
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train failure rate: {y_train.mean()*100:.2f}%  |  Test failure rate: {y_test.mean()*100:.2f}%")
print("Stratified split used because the target is imbalanced.")

# ---------------------------------------------------------------------------
# STEP 8 - MODEL TRAINING & COMPARISON
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 8 - MODEL TRAINING & COMPARISON")
print("=" * 70)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting (HistGB)": HistGradientBoostingClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
}

results = {}
fitted_pipelines = {}

for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocessor", make_preprocessor()), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
    }
    results[name] = metrics
    fitted_pipelines[name] = pipe
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["No Failure", "Failure"], zero_division=0))

results_df = pd.DataFrame(results).T.sort_values("Recall", ascending=False)
print("\n=== Model comparison (sorted by Recall) ===")
print(results_df)
results_df.to_csv(os.path.join(FIG_DIR, "model_comparison.csv"))

fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
for ax, (name, pipe) in zip(axes, fitted_pipelines.items()):
    cm = confusion_matrix(y_test, pipe.predict(X_test))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"])
    ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "08_confusion_matrices.png"), dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
for name, pipe in fitted_pipelines.items():
    y_proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['ROC_AUC']:.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "09_roc_curves.png"), dpi=120)
plt.close()

results_df["score"] = 0.5 * results_df["Recall"] + 0.3 * results_df["F1"] + 0.2 * results_df["ROC_AUC"]
best_model_name = results_df["score"].idxmax()
best_pipeline = fitted_pipelines[best_model_name]
print(f"\n>>> Selected model: {best_model_name}")
print("Selection rule: score = 0.5*Recall + 0.3*F1 + 0.2*ROC-AUC (recall weighted highest).")

clf = best_pipeline.named_steps["classifier"]
try:
    if hasattr(clf, "feature_importances_"):
        ohe = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
        all_names = ENGINEERED_NUMERICAL_COLS + cat_names
        importances = pd.Series(clf.feature_importances_, index=all_names).sort_values(ascending=False)
        title = f"Feature Importance (impurity-based) - {best_model_name}"
    else:
        perm = permutation_importance(best_pipeline, X_test, y_test, scoring="recall",
                                       n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
        importances = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
        title = f"Permutation Importance (impact on Recall) - {best_model_name}"
    print(f"\n{title}:\n{importances}")
    fig, ax = plt.subplots(figsize=(7, 5))
    importances.plot(kind="barh", ax=ax); ax.invert_yaxis(); ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "10_feature_importance.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"(Feature importance skipped: {e})")

# Save a baseline pipeline/metadata (default 0.5 threshold). This gets
# OVERWRITTEN by validate_explain_finalize.py once Steps 9-10 pick a
# validated threshold - run that script next.
import joblib
joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "predictive_maintenance_model.pkl"))
metadata = {
    "best_model_name": best_model_name,
    "numerical_cols_raw": RAW_NUMERICAL_COLS,
    "engineered_numerical_cols": ENGINEERED_NUMERICAL_COLS,
    "categorical_cols": CATEGORICAL_COLS,
    "feature_order": FEATURE_ORDER,
    "target_col": TARGET_COL,
    "test_metrics": results[best_model_name],
    "random_state": RANDOM_STATE,
    "selected_threshold": 0.5,
    "threshold_status": "default (not yet validated) - run validate_explain_finalize.py",
}
with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved baseline pipeline + metadata.")
print("Next: run notebooks/validate_explain_finalize.py for Steps 9-13.")
print("Done.")
