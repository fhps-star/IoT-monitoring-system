# Trained models

The Isolation Forest is written here as `isolation_forest.joblib` after running
`python -m backend.train_model --csv PATH`. It is intentionally not committed.

The artifact contains the model, exact feature order, causal preprocessing
configuration, training dataset identity, evaluation summary, model parameters,
and training timestamp. Training uses only `NORMAL` rows, preserves the input
CSV, writes a separate cleaned copy/audit report, and atomically replaces the
prior model only after fitting succeeds. The default minimum is 30 training
rows; configure it with `MINIMUM_TRAINING_SAMPLES`.

The multiclass Random Forest trainer is run with
`python -m backend.train_random_forest`. It writes its combined dataset,
preprocessing pipeline, model, feature list, classification report, confusion
matrix, and metrics under `backend/models/random_forest/`; generated artifacts
are intentionally not committed. The four collected CSVs do not contain
`gyro_x/y/z` or `estimated_vibration`, so the current classifier uses only the
seven available or derived features. Its test split is chronological within
each class file with a time embargo; because there is no independent run per
class, the evaluation is not cross-session validation.

The selected provisional real-time classifier is stored as XGBoost JSON under
`backend/models/xgboost/`. Load it with `xgboost.XGBClassifier.load_model`, and
use `xgboost_preprocessing.pkl` plus
`xgboost_class_labels.json` for the trained median imputer, seven-feature order,
and encoded class order. `backend.assess_xgboost_readiness` checks those
artifacts and evaluates saved held-out records without retraining. The
deployment-readiness report calls out the same-capture split limitation and
does not claim that `predict_proba()` values are calibrated.
