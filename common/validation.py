"""
Shared input validation, used by both app.py and tests/test_predictions.py.

Centralizing this matters because of a real finding from Step 15 testing:
`HistGradientBoostingClassifier` (the selected model) NATIVELY supports NaN
as an encoded "missing value" and will happily return a normal-looking
probability for a NaN input instead of raising an error. The Streamlit
number widgets can't produce NaN through the UI, but a defensive, explicit
check is still needed for any other caller (the test script, a future API
wrapper, a CSV-batch mode, etc.) - silently predicting on missing data is
exactly the kind of thing an input-validation step exists to catch.
"""

import math

RAW_FIELD_SPECS = {
    "Type": {"kind": "categorical", "allowed": ["L", "M", "H"]},
    "Air temperature [K]": {"kind": "numeric", "min": 290.0, "max": 310.0},
    "Process temperature [K]": {"kind": "numeric", "min": 295.0, "max": 320.0},
    "Rotational speed [rpm]": {"kind": "numeric", "min": 1000, "max": 3000},
    "Torque [Nm]": {"kind": "numeric", "min": 0.0, "max": 100.0},
    "Tool wear [min]": {"kind": "numeric", "min": 0, "max": 260},
}


def validate_raw_input(raw_dict):
    """Validate a raw UI input dict before it reaches the pipeline.

    Returns (errors, warnings):
      - errors: problems severe enough that a prediction should NOT be made
        (missing field, NaN/None, wrong type, unknown category).
      - warnings: the input is usable but unusual (out of the observed
        training range, or a physically odd combination) - the prediction
        is still made, but flagged as a lower-confidence extrapolation.
    """
    errors = []
    warnings = []

    for field, spec in RAW_FIELD_SPECS.items():
        if field not in raw_dict:
            errors.append(f"Missing required field: '{field}'.")
            continue
        value = raw_dict[field]

        if spec["kind"] == "categorical":
            if value not in spec["allowed"]:
                errors.append(f"'{field}'={value!r} is not one of {spec['allowed']}.")
            continue

        # numeric field
        if value is None:
            errors.append(f"'{field}' is missing (None).")
            continue
        if isinstance(value, str):
            errors.append(f"'{field}'={value!r} is not numeric.")
            continue
        try:
            fvalue = float(value)
        except (TypeError, ValueError):
            errors.append(f"'{field}'={value!r} could not be interpreted as a number.")
            continue
        if math.isnan(fvalue) or math.isinf(fvalue):
            errors.append(f"'{field}' is NaN/infinite, which is not a valid sensor reading.")
            continue

        lo, hi = spec["min"], spec["max"]
        if fvalue < lo or fvalue > hi:
            warnings.append(
                f"'{field}'={fvalue} is outside the training data's observed range "
                f"[{lo}, {hi}]. This prediction is an extrapolation and may be unreliable."
            )

    if not errors:
        try:
            if raw_dict["Process temperature [K]"] <= raw_dict["Air temperature [K]"]:
                warnings.append(
                    "Process temperature is not above air temperature - physically "
                    "unusual for this process; the engineered 'Temp_diff_K' feature "
                    "will be zero or negative."
                )
            if float(raw_dict["Torque [Nm]"]) == 0 and float(raw_dict["Rotational speed [rpm]"]) > 0:
                warnings.append(
                    "Torque is 0 while the machine is spinning - the model has not "
                    "seen this combination during training."
                )
        except (KeyError, TypeError, ValueError):
            pass  # already caught above as an error

    return errors, warnings
