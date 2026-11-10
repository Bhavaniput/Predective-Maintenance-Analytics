"""
Step 11 - Explainability (global + local)
==========================================

Preferred approach: SHAP (model-agnostic `shap.Explainer` wrapping the
pipeline's `predict_proba`, so it works through the scaler/one-hot-encoder
without needing to touch internal model structure).

IMPORTANT / HONESTY NOTE FOR THE ACADEMIC WRITE-UP:
This project was developed in a sandboxed execution environment with no
outbound internet access to PyPI, so `pip install shap` / `pip install lime`
could not be run here, and the SHAP code path below is UNTESTED in this
environment (see README "What remains untested"). It is written to the
documented SHAP API and is expected to work in a normal internet-connected
environment (`pip install shap`).

A fallback path is implemented and ACTUALLY TESTED here: a permutation-based
local/global attribution ("occlusion" style). For a given prediction, each
raw input field is temporarily replaced with its typical (median/mode) value
from a background sample, the engineered features are recomputed, and the
resulting shift in predicted probability is reported as that field's
contribution. This is a simpler, more transparent cousin of SHAP/LIME: it
answers "how much did this specific value, versus a typical value, move the
predicted risk?" for one prediction at a time. It is not a Shapley-value
calculation and does not satisfy SHAP's formal efficiency/consistency
properties, but it is a legitimate, widely-used explanation technique
(sometimes called occlusion-based or leave-one-out feature attribution).

Both paths are correlational, not causal: a feature moving the predicted
probability does not prove it causes failure, only that the model's learned
association with it does.
"""

import numpy as np
import pandas as pd

from common.data_prep import (
    FEATURE_ORDER, RAW_NUMERICAL_COLS, CATEGORICAL_COLS,
    ENGINEERED_NUMERICAL_COLS, engineer_features, engineer_single_input,
)

RAW_INPUT_FIELDS = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]

SHAP_AVAILABLE = False
try:
    import shap  # noqa: F401
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# GLOBAL explanation
# ---------------------------------------------------------------------------
def global_importance_permutation(pipeline, X, y, scoring="recall", n_repeats=10, random_state=42):
    """Tested fallback: scikit-learn permutation importance on engineered
    features, measured by the drop in the given scoring metric when each
    column is shuffled. Works for ANY fitted pipeline/model.
    """
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(
        pipeline, X, y, scoring=scoring, n_repeats=n_repeats,
        random_state=random_state, n_jobs=-1
    )
    return pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)


def global_importance_shap(pipeline, X_background, X_sample, max_background=100, max_sample=200):
    """Preferred but UNTESTED-HERE path: mean |SHAP value| per engineered
    feature, using a model-agnostic Explainer around predict_proba.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap is not installed in this environment")
    import shap

    bg = X_background.sample(min(max_background, len(X_background)), random_state=42)
    sample = X_sample.sample(min(max_sample, len(X_sample)), random_state=42) if len(X_sample) > max_sample else X_sample

    def predict_fn(X_df):
        X_df = pd.DataFrame(X_df, columns=X_background.columns)
        return pipeline.predict_proba(X_df)[:, 1]

    explainer = shap.Explainer(predict_fn, bg)
    shap_values = explainer(sample)
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    return pd.Series(mean_abs, index=X_background.columns).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# LOCAL explanation (single prediction)
# ---------------------------------------------------------------------------
def reference_values_from(background_df):
    """Precompute the median/mode reference point used by the permutation
    explainer, so the app doesn't need to load the full training data at
    runtime - just this small dict (stored in model_metadata.json).
    """
    ref = {f: float(background_df[f].median()) for f in RAW_INPUT_FIELDS if f != "Type"}
    ref["Type"] = background_df["Type"].mode().iloc[0]
    return ref


def local_explanation_permutation(pipeline, raw_input_dict, background):
    """Tested fallback: occlusion-style local attribution on the RAW input
    fields the Streamlit UI exposes (not the engineered columns), so the
    explanation is directly readable by someone who just filled in a form.

    `background` may be either a DataFrame (median/mode computed on the fly,
    used during training/validation) or a plain dict of pre-computed
    reference values keyed by raw field name (used by the deployed app, via
    `reference_values_from` + model_metadata.json - avoids shipping/loading
    the full training set just to explain one prediction).

    For each raw field, substitute the reference's typical value, re-run
    feature engineering, and record how much the failure probability
    changes relative to that "typical machine" baseline. A positive
    contribution means this machine's actual value pushed the predicted
    risk UP versus typical; negative means it pushed risk DOWN.
    """
    if isinstance(background, dict):
        ref_values = background
    else:
        ref_values = reference_values_from(background)

    baseline_row = engineer_single_input(raw_input_dict)
    actual_proba = float(pipeline.predict_proba(baseline_row)[0, 1])

    contributions = {}
    for field in RAW_INPUT_FIELDS:
        ref_value = ref_values[field]
        if raw_input_dict[field] == ref_value:
            contributions[field] = 0.0
            continue

        modified = dict(raw_input_dict)
        modified[field] = ref_value
        modified_row = engineer_single_input(modified)
        modified_proba = float(pipeline.predict_proba(modified_row)[0, 1])
        contributions[field] = actual_proba - modified_proba

    ranked = pd.Series(contributions).sort_values(key=lambda s: s.abs(), ascending=False)
    return actual_proba, ranked


def local_explanation_shap(pipeline, raw_input_dict, X_background, max_background=100):
    """Preferred but UNTESTED-HERE path: SHAP values for one prediction, on
    the engineered feature space the model actually sees.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap is not installed in this environment")
    import shap

    row = engineer_single_input(raw_input_dict)
    bg = X_background.sample(min(max_background, len(X_background)), random_state=42)

    def predict_fn(X_df):
        X_df = pd.DataFrame(X_df, columns=X_background.columns)
        return pipeline.predict_proba(X_df)[:, 1]

    explainer = shap.Explainer(predict_fn, bg)
    shap_values = explainer(row)
    contributions = pd.Series(shap_values.values[0], index=FEATURE_ORDER)
    contributions = contributions.sort_values(key=lambda s: s.abs(), ascending=False)
    return float(pipeline.predict_proba(row)[0, 1]), contributions


def explain_local(pipeline, raw_input_dict, background):
    """Single entry point used by app.py and the test script: try SHAP,
    fall back to the tested permutation method, and report which one ran.

    `background` is a DataFrame (training data, used offline) or a dict of
    precomputed reference values (used by the deployed app). SHAP requires
    the DataFrame form; if only a dict is available, SHAP is skipped and
    the permutation fallback (which accepts either) is used directly.
    """
    if SHAP_AVAILABLE and not isinstance(background, dict):
        try:
            proba, contributions = local_explanation_shap(pipeline, raw_input_dict, background)
            return proba, contributions, "shap"
        except Exception:
            pass  # fall through to the tested fallback
    proba, contributions = local_explanation_permutation(pipeline, raw_input_dict, background)
    return proba, contributions, "permutation"
