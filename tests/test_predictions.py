"""
Step 15 - Reproducible prediction testing.

Run from the project root:  python tests/test_predictions.py

This checks the SAVED pipeline (models/predictive_maintenance_model.pkl)
with:
  - valid, clearly-labelled example equipment inputs
  - plausible boundary / unusual inputs
  - missing / invalid values (should raise/handle cleanly, not silently
    produce a nonsense number)
  - probability-output and prediction consistency (same input -> same
    output, predicted class consistent with the probability vs threshold)
  - compatibility between the saved pipeline's expected columns and the
    columns app.py actually builds from the UI inputs

No expected pass/fail probability is hard-coded for the "valid" cases -
this script reports what the model actually outputs; it does not assert
a specific probability, only that the mechanics are sound.
"""

import os
import sys
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd

from common.data_prep import engineer_single_input, FEATURE_ORDER
from common.explainability import explain_local
from common.validation import validate_raw_input

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "predictive_maintenance_model.pkl")
METADATA_PATH = os.path.join(PROJECT_DIR, "models", "model_metadata.json")

passed = 0
failed = 0


def check(label, condition, detail=""):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    if condition:
        passed += 1
    else:
        failed += 1
    print(f"[{status}] {label}" + (f" - {detail}" if detail else ""))


def main():
    print("=" * 70)
    print("Loading saved pipeline + metadata")
    print("=" * 70)
    assert os.path.exists(MODEL_PATH), f"Missing {MODEL_PATH} - run the training scripts first."
    pipeline = joblib.load(MODEL_PATH)
    metadata = json.load(open(METADATA_PATH))
    threshold = metadata["selected_threshold"]
    reference_values = metadata["reference_values"]
    print(f"Model: {metadata['best_model_name']}  |  threshold: {threshold}")

    # -----------------------------------------------------------------
    # 1) Compatibility: saved pipeline's expected columns vs what
    #    engineer_single_input() (used by app.py) actually produces.
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1) Pipeline <-> app feature compatibility")
    print("=" * 70)
    sample_raw = {
        "Type": "L", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500, "Torque [Nm]": 40.0, "Tool wear [min]": 100,
    }
    model_input = engineer_single_input(sample_raw)
    check("engineer_single_input produces expected FEATURE_ORDER columns",
          list(model_input.columns) == FEATURE_ORDER, f"got {list(model_input.columns)}")
    try:
        pipeline.predict(model_input)
        check("Saved pipeline accepts app-engineered feature row without error", True)
    except Exception as e:
        check("Saved pipeline accepts app-engineered feature row without error", False, str(e))

    # -----------------------------------------------------------------
    # 2) Valid, clearly-labelled example inputs
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2) Valid example inputs (labelled; outputs reported, not assumed)")
    print("=" * 70)
    valid_examples = {
        "typical/reference machine": dict(reference_values),
        "high torque + high tool wear (expected higher risk per Step 5 trend analysis)": {
            "Type": "L", "Air temperature [K]": 302.5, "Process temperature [K]": 311.0,
            "Rotational speed [rpm]": 1400, "Torque [Nm]": 65.0, "Tool wear [min]": 220,
        },
        "low torque + low tool wear (expected lower risk per Step 5 trend analysis)": {
            "Type": "H", "Air temperature [K]": 298.0, "Process temperature [K]": 308.5,
            "Rotational speed [rpm]": 1550, "Torque [Nm]": 35.0, "Tool wear [min]": 20,
        },
    }
    probas = {}
    for label, raw in valid_examples.items():
        row = engineer_single_input(raw)
        pred = int(pipeline.predict(row)[0])
        proba = float(pipeline.predict_proba(row)[0, 1])
        probas[label] = proba
        expected_pred = int(proba >= threshold)
        check(f"predict() consistent with predict_proba() @ threshold -- {label}",
              pred == expected_pred, f"pred={pred}, proba={proba:.4f}, threshold={threshold}")
        print(f"    -> {label}: proba={proba:.4f}, predicted class={pred}")

    high_key = "high torque + high tool wear (expected higher risk per Step 5 trend analysis)"
    low_key = "low torque + low tool wear (expected lower risk per Step 5 trend analysis)"
    check("High-stress example scores higher probability than low-stress example",
          probas[high_key] > probas[low_key],
          f"high={probas[high_key]:.4f} vs low={probas[low_key]:.4f}")

    # -----------------------------------------------------------------
    # 3) Boundary / unusual (but technically valid) inputs
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3) Boundary / unusual inputs")
    print("=" * 70)
    boundary_examples = {
        "min of observed sensor ranges": {
            "Type": "L", "Air temperature [K]": 295.3, "Process temperature [K]": 305.7,
            "Rotational speed [rpm]": 1168, "Torque [Nm]": 3.8, "Tool wear [min]": 0,
        },
        "max of observed sensor ranges": {
            "Type": "H", "Air temperature [K]": 304.5, "Process temperature [K]": 313.8,
            "Rotational speed [rpm]": 2886, "Torque [Nm]": 76.6, "Tool wear [min]": 253,
        },
        "zero tool wear, high torque": {
            "Type": "L", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1400, "Torque [Nm]": 70.0, "Tool wear [min]": 0,
        },
        "max tool wear, low torque": {
            "Type": "L", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500, "Torque [Nm]": 10.0, "Tool wear [min]": 253,
        },
    }
    for label, raw in boundary_examples.items():
        try:
            row = engineer_single_input(raw)
            pred = int(pipeline.predict(row)[0])
            proba = float(pipeline.predict_proba(row)[0, 1])
            check(f"Boundary case handled without error -- {label}", True)
            check(f"Boundary case probability is a valid [0,1] float -- {label}",
                  0.0 <= proba <= 1.0, f"proba={proba}")
            print(f"    -> {label}: proba={proba:.4f}, predicted class={pred}")
        except Exception as e:
            check(f"Boundary case handled without error -- {label}", False, str(e))

    # -----------------------------------------------------------------
    # 4) Missing / invalid values -> app.py runs validate_raw_input() on
    #    every input BEFORE calling the pipeline (see common/validation.py).
    #    This step mirrors that real flow and checks it actually blocks bad
    #    inputs, rather than relying on the model to fail safely.
    #
    #    Real finding from an earlier version of this test: the selected
    #    model, HistGradientBoostingClassifier, NATIVELY treats NaN as an
    #    encoded "missing value" and returns a normal-looking probability
    #    instead of erroring - it does NOT fail on bad input by itself.
    #    That is why validate_raw_input() exists and is checked explicitly
    #    here rather than trusting the pipeline to reject NaN on its own.
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4) Missing / invalid values -> must be caught by validate_raw_input()")
    print("=" * 70)
    invalid_cases = {
        "missing 'Torque [Nm]' key": {
            "Type": "L", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500, "Tool wear [min]": 100,
        },
        "NaN value (the HistGB-silently-accepts-this case)": {
            "Type": "L", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500, "Torque [Nm]": float("nan"), "Tool wear [min]": 100,
        },
        "unseen category 'X' for Type": {
            "Type": "X", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500, "Torque [Nm]": 40.0, "Tool wear [min]": 100,
        },
        "string in a numeric field": {
            "Type": "L", "Air temperature [K]": "not a number", "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500, "Torque [Nm]": 40.0, "Tool wear [min]": 100,
        },
        "None value": {
            "Type": "L", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500, "Torque [Nm]": None, "Tool wear [min]": 100,
        },
    }
    for label, raw in invalid_cases.items():
        errors, _ = validate_raw_input(raw)
        check(f"validate_raw_input() flags this as an error (prediction is blocked, as in app.py) -- {label}",
              len(errors) > 0, f"errors={errors}")

    # Confirm the HistGB-silently-accepts-NaN finding is still real, i.e.
    # validate_raw_input() is load-bearing and not just defensive clutter.
    nan_case = invalid_cases["NaN value (the HistGB-silently-accepts-this case)"]
    try:
        row = engineer_single_input(nan_case)
        proba = float(pipeline.predict_proba(row)[0, 1])
        pipeline_accepts_nan_silently = not np.isnan(proba)
    except Exception:
        pipeline_accepts_nan_silently = False
    check("Documented finding still holds: the raw pipeline alone does NOT reject NaN "
          "(this is exactly why validate_raw_input() must run before every prediction)",
          pipeline_accepts_nan_silently,
          "pipeline.predict_proba returned a normal number for NaN input" if pipeline_accepts_nan_silently
          else "pipeline no longer silently accepts NaN - re-check this note if the model changed")

    # Valid inputs must NOT be blocked
    for label, raw in valid_examples.items():
        errors, _ = validate_raw_input(raw)
        check(f"validate_raw_input() does NOT block a valid input -- {label}", len(errors) == 0, f"errors={errors}")

    # -----------------------------------------------------------------
    # 5) Determinism / consistency: same input -> same output
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5) Prediction consistency (determinism)")
    print("=" * 70)
    raw = valid_examples[high_key]
    row = engineer_single_input(raw)
    proba1 = float(pipeline.predict_proba(row)[0, 1])
    proba2 = float(pipeline.predict_proba(engineer_single_input(raw))[0, 1])
    check("Repeated predict_proba() on identical input returns identical probability",
          proba1 == proba2, f"{proba1} vs {proba2}")

    pipeline_reloaded = joblib.load(MODEL_PATH)
    proba3 = float(pipeline_reloaded.predict_proba(row)[0, 1])
    check("Freshly reloaded pipeline matches in-memory pipeline on same input",
          abs(proba1 - proba3) < 1e-9, f"{proba1} vs {proba3}")

    # -----------------------------------------------------------------
    # 6) Local explanation sanity (Step 11 <-> app integration)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("6) Local explanation sanity check")
    print("=" * 70)
    try:
        proba, contributions, method = explain_local(pipeline, valid_examples[high_key], reference_values)
        check("explain_local() returns a contribution for every raw UI field",
              set(contributions.index) == {"Type", "Air temperature [K]", "Process temperature [K]",
                                            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"},
              f"got {list(contributions.index)}")
        check("explain_local() reports the same probability as predict_proba()",
              abs(proba - probas[high_key]) < 1e-9, f"{proba} vs {probas[high_key]}")
        print(f"    method used: {method}")
    except Exception as e:
        check("explain_local() runs without error on a valid input", False, traceback.format_exc())

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 70)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
