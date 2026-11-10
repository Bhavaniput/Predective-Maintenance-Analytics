"""
Predictive Maintenance - Streamlit App
=======================================
Loads the FINAL trained pipeline (saved by
notebooks/validate_explain_finalize.py, Step 13) and predicts failure risk
for NEW equipment parameters entered through the UI.

The app never retrains and never hard-codes a prediction, probability, or
explanation - everything shown comes from the loaded pipeline, the loaded
threshold/metadata, and the shared common/ modules (so it stays in sync
with the notebook/scripts).

Run with:  streamlit run app.py
"""

import os
import json

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.data_prep import engineer_single_input, RAW_NUMERICAL_COLS
from common.explainability import explain_local, RAW_INPUT_FIELDS, SHAP_AVAILABLE
from common.maintenance import recommendation_for
from common.validation import validate_raw_input

# ---------------------------------------------------------------------------
# Setup / load artifacts
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "predictive_maintenance_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

st.set_page_config(page_title="Predictive Maintenance", page_icon="🛠️", layout="centered")


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        return None, None
    pipeline = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return pipeline, metadata


pipeline, metadata = load_artifacts()

st.title("Equipment Failure Risk Predictor")

if pipeline is None:
    st.error(
        "No trained model found at `models/predictive_maintenance_model.pkl`. "
        "Run `python notebooks/train_pipeline.py` and then "
        "`python notebooks/validate_explain_finalize.py` first to train and "
        "save the pipeline, then restart this app."
    )
    st.stop()

selected_threshold = metadata.get("selected_threshold", 0.5)
risk_bands = metadata.get("risk_bands", {"low_max": 0.30, "high_min": 0.60})
reference_values = metadata.get("reference_values")

st.caption(
    f"Model: **{metadata['best_model_name']}**  |  "
    f"Operating threshold: **{selected_threshold}**  |  "
    f"Trained on the AI4I 2020 Predictive Maintenance Dataset (synthetic)"
)
st.write(
    "Enter the current sensor readings for a piece of equipment below. "
    "The trained model will estimate the probability of a machine failure, "
    "translate it into a risk category, explain which inputs drove that "
    "specific prediction, and show an illustrative maintenance recommendation."
)

# ---------------------------------------------------------------------------
# Input widgets — match the RAW features the pipeline was trained on
# ---------------------------------------------------------------------------
st.subheader("Equipment Parameters")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        machine_type = st.selectbox("Product Type", options=["L", "M", "H"], index=0,
                                     help="L = Low, M = Medium, H = High quality variant")
        air_temp = st.number_input("Air temperature [K]", min_value=290.0, max_value=310.0,
                                    value=300.0, step=0.1, format="%.1f")
        process_temp = st.number_input("Process temperature [K]", min_value=295.0, max_value=320.0,
                                        value=310.0, step=0.1, format="%.1f")
    with col2:
        rpm = st.number_input("Rotational speed [rpm]", min_value=1000, max_value=3000,
                               value=1500, step=10)
        torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=100.0,
                                  value=40.0, step=0.1, format="%.1f")
        tool_wear = st.number_input("Tool wear [min]", min_value=0, max_value=260,
                                     value=100, step=1)
    submitted = st.form_submit_button("Predict Failure Risk", type="primary")

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if submitted:
    try:
        raw_input = {
            "Type": machine_type,
            "Air temperature [K]": float(air_temp),
            "Process temperature [K]": float(process_temp),
            "Rotational speed [rpm]": float(rpm),
            "Torque [Nm]": float(torque),
            "Tool wear [min]": float(tool_wear),
        }

        errors, warnings_list = validate_raw_input(raw_input)
        if errors:
            for e in errors:
                st.error(e)
            st.stop()
        for w in warnings_list:
            st.warning(w)

        model_input = engineer_single_input(raw_input)  # SAME feature engineering as training
        prediction = int(pipeline.predict(model_input)[0])
        probability = float(pipeline.predict_proba(model_input)[0, 1])  # P(failure)

        risk_label, recommendation_text = recommendation_for(
            probability, low_max=risk_bands["low_max"], high_min=risk_bands["high_min"]
        )
        risk_color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}[risk_label]

        st.subheader("Prediction Result")
        st.markdown(f"**Predicted Failure Risk:** :{risk_color}[{risk_label}]")
        st.markdown(f"**Failure Probability:** {probability * 100:.1f}%")
        st.progress(min(max(probability, 0.0), 1.0))

        if prediction == 1:
            st.error(f"Model prediction at threshold {selected_threshold}: Machine failure likely (class = 1).")
        else:
            st.success(f"Model prediction at threshold {selected_threshold}: No failure expected (class = 0).")

        st.subheader("Maintenance Recommendation")
        st.info(recommendation_text)
        st.caption(
            "Illustrative project output for an academic demo - not a validated "
            "industrial maintenance standard, and not based on real "
            "maintenance-cost data."
        )

        # --- Local explanation ---
        st.subheader("Why did the model say this?")
        try:
            _, contributions, method = explain_local(pipeline, raw_input, reference_values)
            method_label = "SHAP" if method == "shap" else "permutation-based (occlusion) attribution"
            st.caption(
                f"Method: {method_label}. Positive bars pushed the predicted risk UP "
                "relative to a typical machine in the training data; negative bars "
                "pushed it DOWN. This reflects association learned by the model, "
                "not a proven physical cause."
            )
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ordered = contributions.sort_values()
            colors = ["#C44E52" if v > 0 else "#4C72B0" for v in ordered.values]
            ax.barh(ordered.index, ordered.values, color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Contribution to failure probability")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not generate a local explanation for this input ({e}).")

        with st.expander("Show raw model input / details"):
            st.dataframe(model_input)
            st.write(f"Model used: {metadata['best_model_name']}")
            st.write(f"Operating threshold: {selected_threshold}")
            st.write("Test-set performance at this threshold:", metadata.get("test_metrics_selected_threshold"))

    except Exception as e:
        st.error(f"Could not generate a prediction for these inputs: {e}")

st.divider()

with st.expander("About this model, the dataset, and its limitations"):
    st.markdown(
        f"""
- **Dataset:** AI4I 2020 Predictive Maintenance Dataset - a **synthetic**
  dataset designed to mimic real milling-machine sensor behavior; it was not
  collected from a live production line.
- **Model:** {metadata['best_model_name']}, selected using a recall-weighted
  score (0.5×Recall + 0.3×F1 + 0.2×ROC-AUC) rather than raw accuracy, because
  a missed failure (false negative) is more costly than a false alarm here.
- **Cross-validated stability:** see `models/model_metadata.json` ->
  `cv_stability` for mean/std Recall, Precision, F1 and ROC-AUC across 5
  stratified folds on the training data.
- **Operating threshold ({selected_threshold}):** chosen using only
  cross-validated training-set predictions (never the test set), as the
  lowest threshold reaching at least 80% recall on that data.
- **Explanation method:** {"SHAP" if SHAP_AVAILABLE else "a permutation-based (occlusion) fallback, since the `shap` package was not available when this model was built"}.
- **Known limitations:** synthetic data may not reflect real-world sensor
  noise, failure dynamics, or confounding factors (operator behavior,
  environmental drift, long-term sensor degradation); each row is an
  independent snapshot, not a time series leading up to failure; the risk
  thresholds/categories are project-defined for this academic demo, not a
  validated industrial standard.
- This app is a decision-support tool for an academic project, not a
  certified safety system - predictions should not replace qualified
  maintenance judgment.
"""
    )
