# Predictive Maintenance — AI4I 2020 Dataset

**Predict Equipment Failure Risks to Reduce Downtime and Maintenance Costs in Production Environments**

Individual academic ML project. Implements the full 17-step workflow below on the
**AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository) — a
**synthetic** dataset that mimics milling-machine sensor behavior.

## Project structure

```text
predictive-maintenance/
│
├── data/
│   └── ai4i2020.csv                           # raw dataset (10,000 rows, 14 columns)
│
├── common/                                     # single source of truth, shared by every script below
│   ├── data_prep.py                            # loading, feature engineering, train/test split, preprocessor
│   ├── explainability.py                       # Step 11: SHAP + tested permutation fallback
│   ├── maintenance.py                          # Step 12: risk categories + recommendations
│   └── validation.py                           # Step 15: input validation used by app.py and tests
│
├── notebooks/
│   ├── predictive_maintenance_eda_model.ipynb  # Steps 1–13, narrated notebook (run top to bottom)
│   ├── train_pipeline.py                       # Steps 1–8 as a plain script
│   └── validate_explain_finalize.py            # Steps 9–13 as a plain script (run after train_pipeline.py)
│
├── tests/
│   └── test_predictions.py                     # Step 15: reproducible prediction test suite
│
├── models/
│   ├── predictive_maintenance_model.pkl        # final saved preprocessing+model pipeline (joblib)
│   └── model_metadata.json                     # feature order, CV results, threshold, risk bands, reference values
│
├── figures/                                     # every EDA / evaluation / explainability plot, numbered by step
│
├── app.py                                       # Step 14: Streamlit prediction app (loads the saved pipeline only)
├── requirements.txt                             # pinned to the exact versions that built the current .pkl
└── README.md
```

`common/` exists so the notebook, the two scripts, the Streamlit app, and the test suite all
engineer features, split data, and classify risk in **exactly one place** — the single
biggest real-world cause of a model that behaves differently in an app than in training.

## How to run

```bash
pip install -r requirements.txt

# 1. Steps 1–8: EDA, cleaning, feature engineering, train/compare/select a model
cd notebooks
python train_pipeline.py

# 2. Steps 9–13: cross-validation, threshold selection, explainability, final save
python validate_explain_finalize.py

# (equivalently, open predictive_maintenance_eda_model.ipynb and run all cells —
#  it contains the same Steps 1–13 logic, narrated)

# 3. Step 15: run the reproducible test suite
cd ..
python tests/test_predictions.py

# 4. Step 14: launch the prediction app
streamlit run app.py
```

---

## Step 1 — Project objective

**Business problem.** Unplanned equipment failure in a production environment causes
downtime, emergency repair costs, scrapped work-in-progress, and safety risk. Purely
reactive maintenance and blind time-based maintenance are both inefficient.

**ML problem.** Binary classification: given current sensor readings, predict whether the
machine will experience a failure.

**Prediction target.** `Machine failure` (1 = failure, 0 = no failure) — the dataset's only
general, overall failure indicator. The five sub-flags `TWF`, `HDF`, `PWF`, `OSF`, `RNF`
describe specific failure *mechanisms* and are excluded from the model inputs (see Step 2).

**How this reduces downtime/cost.** A live failure-risk score lets a maintenance team
prioritize inspection of genuinely high-risk machines and avoid unnecessary servicing of
healthy ones, catching problems before an unplanned stoppage occurs.

## Step 2 — Dataset understanding (verified programmatically, not assumed)

- 10,000 rows, 14 columns, **no missing values**, **no duplicate rows**.
- Columns: `UDI`, `Product ID` (identifiers) · `Type` (categorical: L/M/H) ·
  `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`,
  `Torque [Nm]`, `Tool wear [min]` (numerical sensor features) ·
  `Machine failure` (target) · `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (failure-mode sub-flags).
- **Target is imbalanced:** 339/10,000 rows (3.39%) are failures.
- **Leakage found and handled:** the five sub-flags almost perfectly determine
  `Machine failure` (only ~27 of 10,000 rows disagree) — dropped from the feature set, as
  are `UDI`/`Product ID` (identifiers with no generalizable signal).

## Steps 3–5 — EDA and failure trend analysis (see `figures/01`–`07`)

Findings actually supported by the data:
- Failed machines show clearly **higher average torque and tool wear** than healthy ones.
- Process temperature is only **slightly** higher on average for failed machines.
- Rotational speed alone shows a **weak** association with failure; it matters more combined
  with torque (mechanical power).
- Failure rate varies mildly by product `Type`.

## Step 4 — Data cleaning

- No missing values, no duplicates, no invalid/negative ranges — verified, not assumed.
- **Outliers are retained.** An unusually high torque or tool-wear reading is frequently the
  genuine high-risk condition this model exists to catch, not sensor noise.
- Dropped before modeling: `UDI`, `Product ID` (identifiers) and
  `TWF`/`HDF`/`PWF`/`OSF`/`RNF` (leakage — see Step 2).

## Step 6 — Feature engineering

| Feature | Formula | Rationale | Leakage risk |
|---|---|---|---|
| `Temp_diff_K` | `Process temp − Air temp` | Thermal stress / heat build-up in the process | None — raw sensors only |
| `Mechanical_power_W` | `Torque × (rpm × 2π/60)` | Approximates real mechanical power delivered; abnormal power is a known stress indicator | None — raw sensors only |
| `ToolWear_x_Torque` | `Tool wear × Torque` | Approximates cumulative mechanical overstrain on a worn tool | None — raw sensors only |

None of these use the target column, row identifiers, or the failure-mode flags.

## Step 7 — Train/test split

80/20 split, `random_state=42`, **stratified** on the target (imbalanced), X/y clearly
separated, and all scaling/encoding fit only on the training fold via a scikit-learn
`Pipeline` (prevents test-set leakage into preprocessing statistics).

## Step 8 — Model training & selection

Three models trained and compared: Logistic Regression (baseline), Random Forest, and
`HistGradientBoostingClassifier` (scikit-learn's native gradient-boosting model — used
instead of an external XGBoost dependency for portability/reproducibility; see
"Assumptions" below).

Test-set results from the current run (`figures/model_comparison.csv`):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.859 | 0.179 | **0.882** | 0.298 | 0.939 |
| Random Forest | 0.992 | 0.933 | 0.824 | 0.875 | 0.969 |
| Gradient Boosting (HistGB) | 0.992 | 0.919 | 0.838 | 0.877 | 0.979 |

**Selection rule:** `score = 0.5×Recall + 0.3×F1 + 0.2×ROC-AUC`. Gradient Boosting was
selected: it keeps most of Logistic Regression's high recall while avoiding its excessive
false-alarm rate (precision 0.18). **The model with the highest raw accuracy was not
automatically chosen**, since accuracy is dominated by the majority class on this 3.4%-positive
target. The pipeline supports `predict_proba()`, used directly by the app.

## Step 9 — Validate model stability (`figures/cv_stability_results.json`)

5-fold **stratified** cross-validation, **on the training split only** — each fold refits
the entire `Pipeline` (scaler + one-hot-encoder + model) from scratch, so no fold's
preprocessing statistics ever see another fold's held-out rows, and the test set is never
touched in this step.

| Model | Recall (mean ± std) | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.841 ± 0.046 | 0.162 ± 0.008 | 0.271 ± 0.011 | 0.923 ± 0.013 |
| Random Forest | 0.815 ± 0.048 | 0.909 ± 0.013 | 0.859 ± 0.032 | 0.976 ± 0.016 |
| Gradient Boosting (HistGB) | 0.823 ± 0.031 | 0.895 ± 0.043 | 0.856 ± 0.018 | 0.972 ± 0.009 |

**Overfitting check:** each model's CV mean is close to its single-split test score from
Step 8, with modest standard deviation across folds — indicating the Step 8 result was not a
lucky/unlucky split for any of the three models.

## Step 10 — Recall/error evaluation + threshold selection

Threshold selection used **only** cross-validated out-of-fold probabilities on the
**training set** (`cross_val_predict`, never the test set) — swept over thresholds
0.05–0.95 (`figures/threshold_analysis_train_oof.csv`). **Selection rule:** the lowest
threshold still reaching ≥80% recall on that training out-of-fold sweep, tie-broken by F1.
This run selected **threshold = 0.85**.

The test set was then touched exactly once, to report this threshold's honest performance:

| | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Default threshold (0.5) | 0.919 | 0.838 | 0.877 | 0.979 |
| **Selected threshold (0.85)** | **0.983** | **0.838** | **0.905** | 0.979 |

Confusion matrix at the selected threshold: **11 false negatives** (missed failures) out of
68 actual failures, **1 false positive** out of 1,932 healthy machines. Recall is prioritized
because a false negative is an unplanned stoppage — the event this project exists to
prevent — while a false positive only costs one avoidable inspection. See
`figures/11_confusion_matrix_selected_threshold.png`, `12_roc_curve_test.png`,
`13_precision_recall_curve_test.png`.

## Step 11 — Explainability

**SHAP was not installed in the sandbox this project was built in** (no outbound PyPI
access) — `common/explainability.py` detects this automatically (`SHAP_AVAILABLE`) and
falls back to a **tested, permutation-based occlusion attribution**:
- *Global:* scikit-learn's `permutation_importance` on the test set, scored by the drop in
  Recall when each engineered feature is shuffled (`figures/14_global_importance.png`).
- *Local:* for one prediction, each raw input field is swapped for its typical
  (median/mode) training value, one at a time, and the resulting shift in predicted
  probability is reported as that field's contribution (`figures/15_local_explanation_examples.png`).

Both methods describe **association**, not causation — the model's learned relationship
between an input and predicted risk, not a proven physical cause (AI4I is observational
data, not a controlled experiment).

`common/explainability.py` also contains a fully-written SHAP code path
(`global_importance_shap`, `local_explanation_shap`) that activates automatically the moment
`shap` is installed — **this path is untested in this project's build environment** and
should be verified once you `pip install shap` locally (see "What remains untested" below).

## Step 12 — Maintenance scheduling insights

Risk bands used by the app (`common/maintenance.py`), with the High-risk boundary aligned to
the Step 10 validated threshold:

| Probability | Category | Recommendation |
|---|---|---|
| < 30% | **LOW** | Continue routine monitoring. |
| 30%–85% | **MEDIUM** | Consider closer monitoring and a scheduled inspection in the near term. |
| ≥ 85% | **HIGH** | Prioritize inspection and a maintenance assessment. |

**These are illustrative project outputs for an academic demo** — not a validated
industrial maintenance standard, and not derived from real maintenance-cost data. No
specific maintenance schedule or cost saving is claimed.

## Step 13 — Save the final model

`validate_explain_finalize.py` saves the complete `Pipeline` (scaler + one-hot-encoder +
classifier) plus `model_metadata.json` — feature order, CV stability, selected threshold,
risk bands, both default- and selected-threshold test metrics, and reference values used for
local explanations. It then **reloads the saved file and asserts** its prediction on a test
row matches the in-memory pipeline bit-for-bit before declaring success (see the script's
final section) — this was re-verified in this sandbox and passed.

## Step 14 — Streamlit application (`app.py`)

- Loads the saved pipeline and metadata **once**, cached with `st.cache_resource` — it never
  retrains.
- Collects the 6 raw UI inputs, runs them through `common/validation.py` (blocks missing/NaN/
  out-of-vocabulary inputs before they ever reach the model — see Step 15 for why this check
  is load-bearing, not just defensive), then `common/data_prep.engineer_single_input` (the
  *same* function used in training) before calling `pipeline.predict`/`predict_proba`.
- Displays: Failure/No-Failure prediction at the validated threshold, failure probability,
  Low/Medium/High risk category, a local explanation chart (SHAP if installed, else the
  tested permutation fallback), an illustrative maintenance recommendation, and an "About"
  panel stating the dataset is synthetic and listing the model's limitations.
- Nothing is hard-coded: every number shown comes from the loaded pipeline/metadata.

## Step 15 — Test predictions (`tests/test_predictions.py`)

27 checks, all currently passing in this environment (`python tests/test_predictions.py`):
pipeline↔app feature-order compatibility; three labelled, clearly-described valid inputs
(a "typical" machine, a high-torque/high-wear case, a low-torque/low-wear case — the
high-stress case is asserted to score a higher probability than the low-stress case, not a
fabricated number); four boundary/extreme-but-valid sensor combinations; five invalid-input
cases (missing key, NaN, unseen category, string-in-numeric-field, `None`) each confirmed to
be **blocked by `validate_raw_input`**; a confirmation that the raw pipeline (HistGB) does
**not** reject NaN on its own — `predict_proba` happily returns a normal-looking number for
NaN input, which is exactly why the explicit validation step exists and is checked here
rather than assumed; determinism checks (same input twice, and a freshly reloaded pipeline,
both match bit-for-bit); and a local-explanation sanity check. No pass/fail probability is
hard-coded anywhere — the script reports what the model actually outputs.

## Step 16 — Preparing for Streamlit Community Cloud deployment

**Not deployed by this project** — deployment is the user's own step, per the brief. What's
prepared:
- `requirements.txt` pinned to the exact library versions verified in this environment
  (see "Dependency pinning" below).
- All file paths in `app.py` and `common/` are relative to the project root via
  `os.path.dirname(os.path.abspath(__file__))` — no absolute/local paths.
- `app.py` only loads a pre-trained pickle; it does not need the full training pipeline or
  internet access to run.

**Deployment steps (do this yourself):**
1. Push this whole `predictive-maintenance/` folder to a GitHub repo (including
   `models/predictive_maintenance_model.pkl` and `models/model_metadata.json` — Streamlit
   Cloud does not run your training scripts for you).
2. On [streamlit.io/cloud](https://streamlit.io/cloud), create a new app pointing at that
   repo, branch, and `app.py` as the entry point.
3. Streamlit Cloud reads `requirements.txt` automatically and installs those exact versions
   during the build — this matters because of the dependency-pinning note below.
4. If Streamlit Cloud's available Python version differs from `3.13` (used to build the
   current pickle), see "Resolving dependency/model-loading errors" below — the safe fix is
   always to retrain in the same environment you deploy from.

### Dependency pinning — why, and what it fixes

`requirements.txt` pins **exact** versions (`pandas==3.0.5`, `numpy==2.5.3`,
`matplotlib==3.11.2`, `seaborn==0.13.2`, `scikit-learn==1.9.1`, `joblib==1.6.0`) rather than
`>=` ranges, deliberately. `models/predictive_maintenance_model.pkl` is a pickled
scikit-learn `Pipeline`, and **scikit-learn's own documented policy is that a pickle is only
guaranteed to load with the same scikit-learn minor version that created it** — loading it
under a different version can fail outright or silently misbehave.

### Resolving common dependency / model-loading / deployment errors

These are the two actual errors hit while building/running this project, and how to fix each:

**1. `ModuleNotFoundError: No module named 'joblib'`**
`joblib` was in `requirements.txt` but not installed in the Python environment actually
running `streamlit`. Cause: usually multiple Python installs, or `pip install` run against a
different interpreter/venv than the one `streamlit run` uses.
```bash
# confirm which python/streamlit you're using, then:
python -m pip install -r requirements.txt
# if using a venv, activate it BEFORE installing AND before running:
venv\Scripts\activate   # Windows, or: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

**2. `ModuleNotFoundError: No module named '_loss'` when `joblib.load(...)` runs**
This is a **scikit-learn version mismatch**: the `.pkl` was built with one scikit-learn
version, and the environment loading it has a different version whose internal module
layout (used by `HistGradientBoostingClassifier`) doesn't match. Two fixes, in order of
reliability:
```bash
# Preferred: retrain in YOUR environment, so the pickle matches your installed
# library versions exactly (also proves the pipeline is genuinely reproducible):
cd notebooks
python train_pipeline.py
python validate_explain_finalize.py
cd ..
streamlit run app.py

# Alternative: install the exact pinned versions from requirements.txt instead
# (only works if your Python version can satisfy them):
pip install -r requirements.txt
```

**General rule for any future dependency/model-loading error:** check `python --version`
and the installed versions (`pip show scikit-learn pandas numpy`) against
`requirements.txt`'s pinned versions first; if they differ and you can't/don't want to match
them exactly, retrain locally instead of chasing version compatibility.

## Step 17 — Downtime reduction benefits (documentation, not a financial claim)

**Potential business benefits, and how they could be measured** (illustrative, not promised):
- Fewer unplanned stoppages, measurable as a reduction in unscheduled-downtime hours per
  machine per month, compared to a baseline period, after deployment on a real production
  line.
- Reduced unnecessary preventive servicing, measurable as a reduction in maintenance labor
  hours spent on machines the model would have flagged as low-risk.
- Earlier intervention on genuine high-risk conditions, measurable as the lead time between
  a HIGH-risk flag and an eventual real failure or maintenance action, once real failure
  timestamps are available.

**None of these numbers are claimed here.** AI4I 2020 is a **synthetic** dataset generated
to mimic milling-machine behavior; it contains no cost data, no maintenance-labor data, and
no real downtime records. Any actual downtime-reduction or cost-saving figure requires
validating this approach against real production and maintenance-cost data from an actual
deployment — which has not happened in this academic project.

---

## Explanation for academic defense

**1. What each major section does.** Step 2 loads/profiles the raw data; Step 3 visualizes
failure patterns; Step 4 cleans/justifies column removal; Step 5 quantifies trend findings;
Step 6 builds three physically grounded features; Step 7 splits data safely; Step 8
trains/compares/selects a model; Step 9 checks that selection is stable via cross-validation;
Step 10 picks an operating threshold without touching the test set, then reports test
performance once; Step 11 explains predictions; Step 12 attaches illustrative
recommendations; Step 13 saves and verifies the final artifact; Step 14 serves it; Step 15
tests it; Steps 16–17 prepare deployment and document limitations/benefits honestly.

**2. Why each preprocessing technique was chosen.** `StandardScaler` on numeric sensor
columns because Logistic Regression is scale-sensitive; `OneHotEncoder` on `Type` because
it's an unordered categorical variable; both wrapped in a `ColumnTransformer` inside a
`Pipeline`, fit **only** on the training fold each time (Step 7, re-verified per-fold in
Step 9), preventing leakage of test-set or other-folds' statistics.

**3. Why each model was selected.** Logistic Regression as an interpretable linear baseline;
Random Forest as a bagging ensemble robust to feature scale/outliers; Gradient Boosting
(HistGB) as a boosting model that handles the class imbalance and feature interactions
(torque × tool wear) best of the three on this data, confirmed stable by Step 9's
cross-validation rather than trusted from a single split.

**4. Why recall is important here.** A false negative — predicting "no failure" for a
machine that actually fails — means the failure happens unexpectedly, the exact costly event
this project exists to prevent. A false positive only costs one avoidable inspection. This
asymmetry drove both model selection (Step 8) and threshold selection (Step 10).

**5. How the final model predicts a NEW equipment instance.** New raw sensor values go
through `common.data_prep.engineer_single_input`, which computes the identical three
engineered features used in training, in the identical column order
(`model_metadata.json["feature_order"]`); the saved `Pipeline` then applies its fitted
scaler/encoder and the trained classifier's `predict_proba()`, compared against the Step 10
validated threshold (0.85, not the default 0.5) to produce the Failure/No-Failure label.

**6. How the Streamlit app communicates with the trained model.** `app.py` loads
`models/predictive_maintenance_model.pkl` once (`st.cache_resource`), builds the same raw+
engineered feature row via the shared `common/data_prep.py`, and calls
`pipeline.predict()`/`predict_proba()` directly — there is no separate or hard-coded
prediction logic duplicated in the app.

**7. Assumptions made.**
- `Machine failure` is the correct general target; the sub-flags are diagnostic detail, not
  alternative targets.
- `HistGradientBoostingClassifier` is an acceptable substitute for XGBoost as "a suitable
  gradient-boosting classifier" (per the brief's own wording), chosen for
  dependency-free reproducibility in a sandboxed, no-internet build environment.
- The project-defined risk thresholds (Low <30%, Medium 30–85%, High ≥85%) use the Step 10
  validated decision point for the High boundary, but are otherwise a reasonable starting
  point for demonstration, not a validated business policy.

**8. Limitations from the AI4I dataset being synthetic.** Failure rates, thresholds, and
feature relationships may not transfer directly to a real factory; there is no time-series/
sequential ordering (each row is an independent snapshot, not a sensor stream leading up to
failure); there are likely fewer confounding real-world factors (operator behavior,
environmental drift, sensor degradation) than a genuine deployment would encounter. Any
production use would need validation on real operational data before trusting the risk
scores or the Step 17 benefit estimates.

## What remains untested

Run and verified in this build environment: Steps 1–13 (notebook and both scripts, full
output matches what's documented above), Step 15's full 27-check test suite (all passing),
and all Python files' syntax (`py_compile`, Step 14 included).

**Not run here, because this sandbox has no outbound network access to PyPI:**
- `streamlit run app.py` itself — `streamlit` could not be installed in this environment.
  The app's logic was verified indirectly: every function it calls
  (`engineer_single_input`, `validate_raw_input`, `explain_local`, `recommendation_for`,
  `pipeline.predict`/`predict_proba`) is exercised directly by `tests/test_predictions.py`,
  which passes. Run `streamlit run app.py` yourself after `pip install -r requirements.txt`
  to see the actual UI before your defense.
- The SHAP code path in `common/explainability.py` (`global_importance_shap`,
  `local_explanation_shap`) — `shap` could not be installed here either. The code follows
  SHAP's documented `Explainer(predict_fn, background)` API and is wired to activate
  automatically the moment `shap` is importable, with no app/script changes needed, but it
  has not actually been executed. If you install `shap` locally, re-run
  `validate_explain_finalize.py` and check `model_metadata.json["global_explainability_method"]`
  reads `"shap"` instead of `"permutation"` to confirm it worked.
- Deployment to Streamlit Community Cloud itself (Step 16) — prepared, not performed; you
  deploy it yourself as stated in the brief.
