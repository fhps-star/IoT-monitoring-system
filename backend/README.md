# Industrial IoT Motor Monitoring Backend

The backend subscribes to ESP32 MQTT telemetry, serves live readings received
by the current backend process, and stores received JSON in Firebase Realtime
Database for history and training. The dashboard reads only the live MQTT
buffer; Firebase history is not used as a fallback or as a source of displayed
sensor readings. It never publishes synthetic telemetry. Sensor field names are
deliberately not assumed: configure the field paths after inspecting an actual
ESP32 message.

## Setup

From the project root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r backend/requirements.txt
```

Edit `backend/.env`:

Set `GOOGLE_APPLICATION_CREDENTIALS` to the Firebase service-account JSON file.
Relative paths resolve from the `backend/` directory, so
`GOOGLE_APPLICATION_CREDENTIALS=./serviceAccountKey.json` points to
`backend/serviceAccountKey.json` even when you start the backend from the
project root. Absolute paths are also supported. Keep service-account files
out of Git. Alternatively, configure Google Application Default Credentials.

The Firebase Realtime Database URL is already set to the project URL. The
database rules and service account must allow the backend to read and write.
The configured HiveMQ public broker is unauthenticated; use it for non-sensitive
demo telemetry only. Use a private authenticated broker for production.

The ESP32 publisher uses `temperature_c`, `estimated_vibration`, and
`motor_current_a`; these exact field paths are configured in `backend/.env`. Its
`device_id` and `motor_id` are preserved in stored telemetry and predictions.
The Firebase collection path is keyed by configured `DEVICE_ID` (`motor_01`),
not by the payload `device_id`; if omitted, the payload `motor_id` defaults to
that same configured value. If the ESP32 payload schema changes, inspect a
newly received JSON message and update the three sensor field mappings before
restarting the backend.

## Run

From the project root:

```bash
python -m backend.main
```

Or start FastAPI with:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The process subscribes to the configured MQTT topic and logs received JSON.
Malformed messages are logged and ignored. Paho retries a lost broker
connection automatically. Each MQTT payload is validated and written to
Firebase before it is exposed as a live reading. The persisted Firebase key is
included as the telemetry record ID. The dashboard polls `/api/v1/live` every
second and Firebase history every 30 seconds; data older than
`DATA_STALE_TIMEOUT_SECONDS` is marked stale, not live. The default freshness
timeout is 15 seconds, longer than the ESP32's 5-second publish interval.
Firebase write failures do not expose unpersisted readings or trigger AI
predictions.

For MQTT diagnostics, inspect the backend's structured logs for each received
message's topic and raw payload, followed by any JSON or sensor validation
errors. `/health` reports `mqtt_status.messages_received` for all broker
messages and `mqtt_status.payloads_processed` for JSON-object payloads passed
to telemetry validation. The configured `DEVICE_ID` selects the Firebase
collection path; telemetry uses Firebase `push()` keys, so each successful
write creates a separate record rather than replacing the collection.

The same MQTT subscriber also reads ESP32 network fields (`ip_address`,
`wifi_connected`, `mqtt_connected`, `mqtt_rtt_ms`, `sequence`, and
`timestamp_ms`). Network state is saved independently at
`network_metrics/{device_id}/latest`; the four network status/RTT fields are
not copied into persisted sensor telemetry. MQTT RTT average is cumulative
across received samples. Estimated packet loss is based on forward sequence
gaps; a sequence decrease starts a new device sequence epoch and is not
counted as packet loss. Device health is `HEALTHY` when the required network
fields are present, both links are connected, RTT is at most 500 ms, and
estimated loss is at most 1%; it is `WARNING` when connected data is incomplete
or exceeds those thresholds, `STALE` when the last update exceeds
`DATA_STALE_TIMEOUT_SECONDS`, and `OFFLINE` before the first update or when
Wi-Fi/MQTT is explicitly disconnected.

## Audit, train, and evaluate the model

Train from a labelled CSV. The input file is never edited. The pipeline detects
CSV delimiters, validates timezone-aware timestamps and finite sensor values,
removes invalid rows and duplicates into a separate cleaned CSV, and flags
extreme values with an IQR report without automatically deleting them or
calling them faults.

```bash
.venv/bin/python -m backend.train_model \
  --csv data/wokwi_normal_motor_telemetry.csv
```

By default this writes `data/wokwi_normal_motor_telemetry_cleaned.csv`,
`data/wokwi_normal_motor_telemetry_audit.json`, and
`data/wokwi_normal_motor_telemetry_training_report.json`. Isolation Forest is fit on the earliest 60% of time-ordered feature rows that
are labeled `NORMAL`; the next 20% calibrates the anomaly-score threshold, and
the latest 20% is a separate time-based evaluation split. These fractions can
be adjusted with `--calibration-fraction` and `--validation-fraction`. The
deployed model uses the same fit as evaluation so the calibrated threshold
remains comparable. The default minimum training count is 30.

The feature order saved in the joblib artifact is:
`vibration_x`, `vibration_y`, `vibration_z`, `temperature_c`, `current_a`,
`motor_speed`, `acceleration_magnitude`, `vibration_rms`, `vibration_peak`,
`vibration_std`, `temperature_rate`, `current_rate`.
The trailing time window defaults to 10 seconds, requires at least three
samples, and resets after gaps over 2.5 seconds; use `--window-seconds`,
`--max-gap-seconds`, or `--min-window-samples` to configure it. Rates use the
actual elapsed time and only current/past readings. Vibration RMS/peak/std are
low-rate, detrended time-window statistics—not frequency-domain vibration
features. Do not infer vibration frequencies from these 1–2 second MQTT
samples.

Isolation Forest uses `contamination="auto"` to fit; deployment threshold is
calibrated from the median and median absolute deviation (MAD) of a separate
normal calibration split rather than assuming a fixed anomaly percentage.
With normal-only data there are no labeled faults for tuning fault sensitivity.
The report includes held-out score summaries and the observed proportion of
held-out `NORMAL` records flagged as an anomaly, as a false-alarm proxy. It does
not calculate accuracy, precision, recall, or F1 without labeled anomalies.
An anomaly means an unusual pattern, not a particular diagnosed fault.

The saved model artifact includes the exact feature order, preprocessing
configuration, model parameters, training dataset SHA-256/path/counts, and
training timestamp. A successful training run atomically replaces the previous
model artifact. Restart the backend so it loads the new model.

At runtime, MQTT payloads with the six raw model inputs are processed through
the same causal feature engineer. The first predictions wait until the
time-window features have enough recent samples. Valid raw telemetry is stored
at `telemetry/motor_01/`; predictions and anomaly alerts remain in their
separate Firebase collections.

## Real-time XGBoost dashboard integration

The existing MQTT subscriber persists each valid sensor payload once under
`telemetry/{device_id}`. After Firebase returns that telemetry key, the saved
XGBoost JSON model is called only when all required training inputs are valid.
It uses the feature order and median imputer in
`backend/models/xgboost/xgboost_class_labels.json` and
`xgboost_preprocessing.pkl`; acceleration magnitude is derived from the three
measured axes. Missing inputs skip prediction. The probability is stored as a
model probability and is not described as calibrated confidence.

XGBoost results are written idempotently at
`xgboost_predictions/{device_id}/{telemetry_record_id}`. Non-NORMAL classes
also create an AI prediction alert at
`anomaly_alerts/{device_id}/{telemetry_record_id}`. These records never replace
the raw telemetry or the legacy Isolation Forest collections. AI alerts are
warnings for operator review, not confirmed physical failures. No sensor
threshold rules are configured by this project yet.

The React dashboard keeps its existing 1-second live and 30-second history
polling. It provides separate Live Motor Monitoring, AI Anomaly Detection, and
Motor Health & Alerts sections. Run the backend and frontend using the setup
steps in this README and `frontend/README.md`.

## Train a multiclass Random Forest

Train a supervised classifier from the four labelled Wokwi datasets:

```bash
.venv/bin/python -m backend.train_random_forest
```

The trainer checks the four expected files and their labels, detects CSV
delimiters, reports columns/types/missing values/duplicates, and writes a new
combined dataset without changing the source CSVs. It uses
`temperature_c`, acceleration axes (mapped from `vibration_x/y/z`), derived
`acceleration_magnitude`, `motor_speed`, and current (mapped from `current_a`).
The collected files do not contain `gyro_x/y/z` or `estimated_vibration`; those
requested features are reported as unavailable and are not fabricated or used.
Median imputation is fit on training rows only; the inspected datasets have no
missing sensor values.

The Random Forest uses 200 trees, `random_state=42`, and
`class_weight="balanced"`. The latest 20% of each class file is held out
chronologically, with a 30-second embargo before the test segment. No explicit
session/run IDs exist, and each class is in one capture file, so a whole-session
holdout would omit a class from training. This same-capture temporal evaluation
is not independent-run validation; collect additional runs for that stronger
check.

The model, imputer, feature list, combined dataset, per-class classification
report, confusion-matrix PNG, and JSON metrics are written under
`backend/models/random_forest/`. The serialized model is a scikit-learn
pipeline; inference should pass values in the exact order in
`feature_columns.json`. Test predictions and probability confidences are in
`training_metrics.json`.

## Compare with XGBoost

Train XGBoost and compare it with the saved Random Forest without fitting the
Random Forest again:

```bash
.venv/bin/python -m backend.train_xgboost
```

The runner reads the saved combined dataset, feature list, Random Forest model,
preprocessor, and split metadata. It reconstructs the chronological class-wise
split from source-file/row provenance and aborts unless its train/test/embargo
counts and saved Random Forest test accuracy match. The XGBoost objective is
`multi:softprob` to support four-class probabilities; it uses balanced sample
weights. Artifacts, including the split indices and comparison table, are
written under `backend/models/xgboost/`. Random Forest training time was not
recorded previously, and is reported as unavailable rather than retraining it.

Assess the saved XGBoost model for deployment without fitting it again:

```bash
.venv/bin/python -m backend.assess_xgboost_readiness
```

The assessment verifies the model, feature and class mappings, preprocessing
artifact, split provenance, and source hashes. It tests the held-out records,
checks `predict_proba()` output, reports Brier score/log loss/calibration bins,
and measures single-record latency including preprocessing. Probabilities are
not treated as calibrated confidence. JSON and Markdown reports are written to
`backend/models/xgboost/`.

## Collect normal-operation CSV data

The collector subscribes to the configured MQTT topic and writes each complete,
valid sensor message from the same payload to a CSV row. It maps
`acceleration_x/y/z` to `vibration_x/y/z`, and uses the configured temperature
and current field paths plus `motor_speed`. Missing, non-numeric, non-finite, or
wrong-device samples are logged and skipped; readings are never interpolated or
fabricated. Timestamps are UTC receive times. The ESP32 should publish fresh
sensor measurements every 1–2 seconds for that sampling cadence.

Run from the project root only after confirming the physical motor is operating
normally. Keep the ESP32 connected and vary speed/load only within verified
normal operating conditions:

```bash
.venv/bin/python -m backend.collect_data \
  --duration-minutes 30 \
  --normal-confirmed
```

Use `--duration-minutes 60` for an hour. Output defaults to
`data/normal_motor_telemetry.csv`; use `--output PATH` to choose another file.
The CSV includes `timestamp`, `vibration_x`, `vibration_y`, `vibration_z`,
`temperature_c`, `current_a`, `motor_speed`, and `label`. The collector appends
records, so choose a new output file if you want to keep collection sessions
separate. `--normal-confirmed` is required because the collector cannot verify
the motor's physical condition itself. Do not collect simulated Wokwi readings
as a physical-motor training dataset. To collect Wokwi messages separately,
pass `--source wokwi`; they are written to
`data/wokwi_normal_motor_telemetry.csv` and labeled `NORMAL` only to indicate
the simulator's asserted normal mode. They are simulated readings, not
physical-motor measurements.

## Verify

- `GET http://localhost:8000/health` reports MQTT, Firebase, and model status.
- `GET http://localhost:8000/api/v1/health` is the versioned health alias.
- `GET http://localhost:8000/api/v1/live` returns MQTT readings successfully
  written to Firebase during the current backend process, XGBoost results, and
  freshness state.
- `GET http://localhost:8000/api/v1/telemetry` returns raw stored telemetry.
- `GET http://localhost:8000/api/v1/predictions` returns legacy and XGBoost predictions.
- `GET http://localhost:8000/api/v1/anomalies` returns XGBoost anomaly-prediction alerts.
- `GET http://localhost:8000/api/v1/devices` returns configured device connectivity.
- `GET http://localhost:8000/api/v1/devices/motor_01/network` returns the device's latest network state and calculated metrics.
- `GET http://localhost:8000/api/v1/health` includes the device health state and latest network metrics.
- `GET http://localhost:8000/api/v1/model/status` returns XGBoost readiness and input mapping.
- `GET http://localhost:8000/api/v1/device-status` returns connection status.
- `GET http://localhost:8000/api/v1/alerts` returns anomaly alerts.

Firebase paths are `telemetry/motor_01/`, `predictions/motor_01/`,
`xgboost_predictions/motor_01/`, `anomaly_alerts/motor_01/`,
`network_metrics/motor_01/latest`, `device_status/motor_01/`, and legacy
`alerts/motor_01/`.
