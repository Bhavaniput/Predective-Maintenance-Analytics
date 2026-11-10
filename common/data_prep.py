"""
Shared data-loading, feature-engineering and train/test-split logic.

This module is the SINGLE source of truth for how raw AI4I 2020 rows become
model-ready features. It is imported by:
  - notebooks/train_pipeline.py        (Steps 2-8)
  - notebooks/validate_explain_finalize.py  (Steps 9-13)
  - app.py                              (Step 14, for feature engineering only)
  - tests/test_predictions.py           (Step 15)

Keeping this logic in one place guarantees the Streamlit app engineers new
inputs in EXACTLY the same way the training pipeline did - the single most
common source of train/serve skew.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

RANDOM_STATE = 42

# Column roles, verified against the actual AI4I 2020 file (see notebook Step 2)
ID_COLS = ["UDI", "Product ID"]
CATEGORICAL_COLS = ["Type"]
RAW_NUMERICAL_COLS = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]
TARGET_COL = "Machine failure"
# Failure-mode sub-flags: near-perfectly determine the target -> excluded as
# ordinary predictors to avoid data leakage (see README / notebook Step 2).
FAILURE_MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

ENGINEERED_NUMERICAL_COLS = RAW_NUMERICAL_COLS + [
    "Temp_diff_K", "Mechanical_power_W", "ToolWear_x_Torque"
]
FEATURE_ORDER = ENGINEERED_NUMERICAL_COLS + CATEGORICAL_COLS


def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_raw_data(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(project_root(), "data", "ai4i2020.csv")
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    return df


def engineer_features(df):
    """Add the three physically-motivated health features (Step 6).

    Built only from raw sensor columns - never from the target, row
    identifiers, or the failure-mode flags - so this cannot leak label
    information, whether called during training or at inference time.
    """
    df = df.copy()
    df["Temp_diff_K"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Mechanical_power_W"] = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * 2 * np.pi / 60)
    df["ToolWear_x_Torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    return df


def build_model_frame(df):
    """Drop identifiers + leakage columns, engineer features, return X (raw+engineered) and y."""
    model_df = df.drop(columns=[c for c in ID_COLS + FAILURE_MODE_COLS if c in df.columns])
    model_df = engineer_features(model_df)
    X = model_df[FEATURE_ORDER]
    y = model_df[TARGET_COL]
    return X, y


def load_and_split(test_size=0.2, random_state=RANDOM_STATE):
    """Load the CSV and produce the SAME stratified train/test split used
    throughout the project (Step 7). Deterministic given random_state, so
    re-running this in a different script reproduces the identical split
    without needing to persist X_train/X_test to disk.
    """
    df = load_raw_data()
    X, y = build_model_frame(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def make_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), ENGINEERED_NUMERICAL_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])


def engineer_single_input(raw_dict):
    """Turn a single raw UI input (dict of the 6 raw fields) into a one-row
    DataFrame with engineered features, in the exact column order the
    pipeline expects. Used by app.py and the test script.
    """
    row = pd.DataFrame([raw_dict])
    row = engineer_features(row)
    return row[FEATURE_ORDER]
