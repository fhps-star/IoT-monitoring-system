"""FastAPI application for live MQTT telemetry and Firebase persistence."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import math
import pickle
from numbers import Number
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin.exceptions import FirebaseError
from google.auth.exceptions import GoogleAuthError
from requests import RequestException
from xgboost.core import XGBoostError

from backend.anomaly_detector import AnomalyDetector
from backend.ai_model import IsolationForestModel
from backend.data_processor import DataProcessor
from backend.data_processor import MODEL_SAMPLE_FIELDS
from backend.feature_engineering import FEATURE_NAMES, FeatureEngineer
from backend.firebase_config import FirebaseStore
from backend.live_data import LiveTelemetryBuffer
from backend.mqtt_handler import MQTTHandler
from backend.network_monitor import NETWORK_PAYLOAD_FIELDS, NetworkMonitor
from backend.settings import BACKEND_DIR, Settings
from backend.utils.logger import configure_logging
from backend.coap_handler import CoAPHandler
from backend.protocol_tracker import ProtocolTracker
from backend.random_forest_predictor import RandomForestPredictor
from backend.xgboost_predictor import XGBoostPredictor
from backend.ensemble_anomaly import build_anomaly_alert, select_model_prediction

logger = logging.getLogger(__name__)

RF_METRICS_PATH = BACKEND_DIR / "models" / "random_forest" / "training_metrics.json"
XGB_METRICS_PATH = BACKEND_DIR / "models" / "xgboost" / "xgboost_metrics.json"


def _load_offline_metrics() -> dict[str, Any]:
    """Load offline evaluation metrics for both models from saved JSON files."""
    metrics: dict[str, Any] = {"random_forest": {}, "xgboost": {}}

    rf_path = RF_METRICS_PATH
    if rf_path.is_file():
        try:
            data = json.loads(rf_path.read_text(encoding="utf-8"))
            metrics["random_forest"] = _extract_offline_metrics(data, "random_forest")
        except (OSError, ValueError, json.JSONDecodeError):
            logger.exception("Could not load Random Forest offline metrics")

    xgb_path = XGB_METRICS_PATH
    if xgb_path.is_file():
        try:
            data = json.loads(xgb_path.read_text(encoding="utf-8"))
            metrics["xgboost"] = _extract_offline_metrics(data, "xgboost")
            rf_section = data.get("random_forest", {})
            if isinstance(rf_section, dict):
                rf_metrics = metrics["random_forest"]
                if rf_metrics.get("inference_time_500_records") is None:
                    rf_batch = rf_section.get("prediction_time_seconds_per_test_batch_mean")
                    if isinstance(rf_batch, (int, float)) and math.isfinite(rf_batch):
                        rf_metrics["inference_time_500_records"] = rf_batch * 1000.0
                if rf_metrics.get("accuracy") is None:
                    rf_metrics["accuracy"] = rf_section.get("metrics", {}).get("accuracy")
                if rf_metrics.get("precision") is None:
                    rf_metrics["precision"] = rf_section.get("metrics", {}).get("precision")
                if rf_metrics.get("recall") is None:
                    rf_metrics["recall"] = rf_section.get("metrics", {}).get("recall")
                if rf_metrics.get("f1_score") is None:
                    rf_metrics["f1_score"] = rf_section.get("metrics", {}).get("f1")
        except (OSError, ValueError, json.JSONDecodeError):
            logger.exception("Could not load XGBoost offline metrics")

    return metrics


def _extract_offline_metrics(data: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Extract accuracy, precision, recall, F1 and inference time from metrics JSON."""
    result: dict[str, Any] = {}

    if model_name == "random_forest":
        result["accuracy"] = data.get("accuracy")
        report = data.get("classification_report", {})
        macro = report.get("macro avg", {}) if isinstance(report, dict) else {}
        result["precision"] = macro.get("precision")
        result["recall"] = macro.get("recall")
        result["f1_score"] = macro.get("f1-score")
        batch_time = data.get("prediction_time_seconds_per_test_batch_mean")
        if isinstance(batch_time, (int, float)) and math.isfinite(batch_time):
            result["inference_time_500_records"] = batch_time * 1000.0
        else:
            result["inference_time_500_records"] = None
    else:
        xgb_section = data.get("xgboost", {})
        if isinstance(xgb_section, dict):
            result["accuracy"] = xgb_section.get("metrics", {}).get("accuracy")
            report = xgb_section.get("metrics", {}).get("classification_report", {})
            macro = report.get("macro avg", {}) if isinstance(report, dict) else {}
            result["precision"] = macro.get("precision")
            result["recall"] = macro.get("recall")
            result["f1_score"] = macro.get("f1-score")
            batch_time = xgb_section.get("prediction_time_seconds_per_test_batch_mean")
            if isinstance(batch_time, (int, float)) and math.isfinite(batch_time):
                result["inference_time_500_records"] = batch_time * 1000.0
            else:
                result["inference_time_500_records"] = None
        else:
            result["accuracy"] = data.get("accuracy")
            result["precision"] = None
            result["recall"] = None
            result["f1_score"] = None
            result["inference_time_500_records"] = None

    return result


def _compute_live_metrics(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute live evaluation metrics from comparison records with ground truth."""
    rf_predictions: list[str | None] = []
    xgb_predictions: list[str | None] = []
    ground_truths: list[str | None] = []

    for record in comparisons:
        gt = record.get("ground_truth")
        rf = record.get("random_forest", {}).get("predicted_class") if record.get("random_forest") else None
        xgb = record.get("xgboost", {}).get("predicted_class") if record.get("xgboost") else None
        rf_predictions.append(rf)
        xgb_predictions.append(xgb)
        ground_truths.append(gt)

    rf_metrics = _calculate_metrics(rf_predictions, ground_truths)
    xgb_metrics = _calculate_metrics(xgb_predictions, ground_truths)

    return {
        "random_forest": rf_metrics,
        "xgboost": xgb_metrics,
        "has_ground_truth": any(gt is not None for gt in ground_truths),
        "total_records": len(comparisons),
    }


def _calculate_metrics(predictions: list[str | None], ground_truths: list[str | None]) -> dict[str, Any]:
    """Calculate accuracy, precision, recall and F1 from predictions and ground truth."""
    valid_indices = [
        i for i, gt in enumerate(ground_truths)
        if gt is not None and predictions[i] is not None
    ]

    if not valid_indices:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
        }

    classes = sorted({gt for i, gt in enumerate(ground_truths) if gt is not None})
    if not classes:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
        }

    correct = sum(1 for i in valid_indices if predictions[i] == ground_truths[i])
    accuracy = correct / len(valid_indices)

    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []

    for cls in classes:
        tp = sum(1 for i in valid_indices if predictions[i] == cls and ground_truths[i] == cls)
        fp = sum(1 for i in valid_indices if predictions[i] == cls and ground_truths[i] != cls)
        fn = sum(1 for i in valid_indices if predictions[i] != cls and ground_truths[i] == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1,
    }


def _extract_ground_truth(record: dict[str, Any]) -> str | None:
    candidates = (
        "ground_truth",
        "true_label",
        "actual_label",
        "expected_class",
        "true_class",
        "label",
    )
    for key in candidates:
        value = record.get(key)
        if isinstance(value, str) and value in {
            "NORMAL",
            "HIGH_VIBRATION",
            "HIGH_TEMPERATURE",
            "HIGH_CURRENT",
        }:
            return value
    raw_payload = record.get("raw_payload")
    if isinstance(raw_payload, dict):
        for key in candidates:
            value = raw_payload.get(key)
            if isinstance(value, str) and value in {
                "NORMAL",
                "HIGH_VIBRATION",
                "HIGH_TEMPERATURE",
                "HIGH_CURRENT",
            }:
                return value
    return None


def _has_valid_features(record: dict[str, Any]) -> bool:
    features = record.get("features")
    return bool(
        isinstance(features, list)
        and len(features) == 3
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in features
        )
    )


def _has_valid_sensor_input(record: dict[str, Any]) -> bool:
    if _has_valid_features(record):
        return True
    sample = record.get("model_sample")
    return bool(
        isinstance(sample, dict)
        and all(
            isinstance(sample.get(name), (int, float))
            and not isinstance(sample.get(name), bool)
            and math.isfinite(sample[name])
            for name in MODEL_SAMPLE_FIELDS
        )
    )


def _has_any_valid_sensor_input(record: dict[str, Any]) -> bool:
    sensor_values = record.get("sensor_values")
    if isinstance(sensor_values, dict) and any(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in sensor_values.values()
    ):
        return True
    payload = record.get("raw_payload")
    return bool(
        isinstance(payload, dict)
        and any(
            isinstance(payload.get(field), (int, float))
            and not isinstance(payload.get(field), bool)
            and math.isfinite(payload[field])
            for field in (
                "temperature_c",
                "estimated_vibration",
                "motor_current_a",
                "acceleration_x",
                "acceleration_y",
                "acceleration_z",
            )
        )
    )


def _add_raw_sensor_fields(
    telemetry: dict[str, Any], payload: dict[str, Any]
) -> None:
    sensor_values = telemetry.get("sensor_values", {})
    for output_name, sensor_name in (
        ("temperature_c", "temperature"),
        ("estimated_vibration", "vibration"),
        ("motor_current_a", "current"),
    ):
        value = sensor_values.get(sensor_name)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            telemetry[output_name] = value

    for field in (
        "acceleration_x",
        "acceleration_y",
        "acceleration_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "motor_speed",
        "motor_steps",
        "timestamp_ms",
    ):
        value = payload.get(field)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            telemetry[field] = value
    for field in (
        "device_id",
        "motor_status",
        "temperature_status",
        "vibration_status",
        "current_status",
        "wifi_status",
    ):
        if field in payload and isinstance(payload[field], (str, bool)):
            telemetry[field] = payload[field]
    sequence = payload.get("sequence")
    if isinstance(sequence, (int, str)) and not isinstance(sequence, bool):
        telemetry["sequence"] = sequence

    axes = [telemetry.get(f"acceleration_{axis}") for axis in "xyz"]
    if all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in axes
    ):
        telemetry["acceleration_magnitude"] = math.sqrt(
            sum(value * value for value in axes)
        )
    else:
        magnitude = payload.get("acceleration_magnitude")
        if (
            isinstance(magnitude, (int, float))
            and not isinstance(magnitude, bool)
            and math.isfinite(magnitude)
        ):
            telemetry["acceleration_magnitude"] = magnitude


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        settings = Settings.from_env()
        store = FirebaseStore(settings)
        model = IsolationForestModel(settings.model_path)
        random_forest_model = RandomForestPredictor()
        xgboost_model = XGBoostPredictor()
        live_data = LiveTelemetryBuffer(settings.telemetry_limit)
        network_monitor = NetworkMonitor(
            settings.device_id,
            settings.data_stale_timeout_seconds,
        )
        processor = DataProcessor(settings)
        feature_engineer = FeatureEngineer()
        try:
            store.initialize()
            logger.info("Firebase Realtime Database initialized")
        except (
            FirebaseError,
            GoogleAuthError,
            RequestException,
            OSError,
            ValueError,
            RuntimeError,
        ):
            store.last_error = "Firebase initialization failed; see backend logs"
            logger.exception("Firebase initialization failed")

        network_metrics_restored = False
        if store.ready:
            try:
                network_record = store.get_value(
                    f"network_metrics/{settings.device_id}/latest"
                )
                network_monitor.restore(network_record)
                network_metrics_restored = isinstance(network_record, dict)
            except RuntimeError:
                store.last_error = "Firebase network metrics read failed; see backend logs"
                logger.exception("Could not restore latest network metrics")

        if store.ready:
            try:
                latest_records = store.get_records(
                    f"telemetry/{settings.device_id}", 30
                )
                logger.info(
                    "Firebase telemetry read verified",
                    extra={
                        "firebase_path": f"telemetry/{settings.device_id}",
                        "records_found": len(latest_records),
                    },
                )
                if latest_records:
                    latest_records.sort(
                        key=lambda record: str(record.get("received_at", ""))
                    )
                    latest_record = latest_records[-1]
                    received_at = latest_record.get("received_at")
                    if (
                        latest_record.get("source") == "mqtt"
                        and isinstance(received_at, str)
                    ):
                        live_data.restore_last_message(received_at)
                    replayed_network_metrics = False
                    for record in latest_records:
                        payload = record.get("raw_payload")
                        timestamp = record.get("received_at")
                        if not isinstance(payload, dict) or not isinstance(timestamp, str):
                            continue
                        if not network_metrics_restored:
                            replayed = network_monitor.record(payload, timestamp)
                            replayed_network_metrics |= replayed is not None
                        try:
                            processed = processor.process(payload, timestamp)
                            if processed["model_sample"] is None:
                                feature_engineer.reset()
                                continue
                            feature_engineer.process(
                                processed["model_sample"], timestamp
                            )
                        except ValueError:
                            feature_engineer.reset()
                            logger.warning(
                                "Skipping stored telemetry while warming feature window",
                                extra={"received_at": timestamp},
                                exc_info=True,
                            )
                    if replayed_network_metrics:
                        network_path = f"network_metrics/{settings.device_id}"
                        if (
                            store.set_record(
                                network_path,
                                "latest",
                                network_monitor.snapshot(),
                            )
                            is None
                        ):
                            logger.error(
                                "Could not persist migrated device network metrics",
                                extra={"firebase_path": f"{network_path}/latest"},
                            )
            except RuntimeError:
                store.last_error = "Firebase telemetry history read failed; see backend logs"
                logger.exception("Could not restore last MQTT message timestamp")

        try:
            model.load()
            if model.is_trained and model.feature_names != FEATURE_NAMES:
                logger.error(
                    "Saved model feature schema is incompatible with live telemetry",
                    extra={
                        "model_features": list(model.feature_names),
                        "live_features": list(FEATURE_NAMES),
                    },
                )
        except (OSError, ValueError, EOFError):
            logger.exception("Could not load saved AI model; inference is disabled")
        detector = AnomalyDetector(model)
        try:
            random_forest_model.load()
        except (OSError, ValueError, EOFError, pickle.UnpicklingError):
            logger.exception("Could not load Random Forest model; multiclass inference is disabled")
        try:
            xgboost_model.load()
        except (OSError, ValueError, EOFError, pickle.UnpicklingError, XGBoostError):
            logger.exception("Could not load XGBoost model; multiclass inference is disabled")
        protocol_tracker = ProtocolTracker(settings.device_id)
        app.state.settings = settings
        app.state.firebase = store
        app.state.model = model
        app.state.random_forest_model = random_forest_model
        app.state.xgboost_model = xgboost_model
        app.state.live_data = live_data
        app.state.network_monitor = network_monitor
        app.state.protocol_tracker = protocol_tracker
        app.state.feature_engineer = feature_engineer
        app.state.mqtt = None
        app.state.coap = None

        def persist_telemetry(
            payload: dict[str, Any],
            raw_bytes: int | None = None,
            source: str = "mqtt",
        ) -> None:
            received_at = datetime.now(timezone.utc).isoformat()
            proto_str = str(payload.get("protocol") or source or "mqtt")
            protocol_tracker.record_telemetry(
                protocol=proto_str,
                payload=payload,
                raw_bytes=raw_bytes,
                received_at=received_at,
            )
            if store.ready:
                try:
                    store.update_value(
                        f"protocol_comparison/{settings.device_id}",
                        protocol_tracker.get_comparison(),
                    )
                except Exception:
                    logger.debug("Failed to sync protocol comparison to Firebase")

            live_data.note_message(received_at)
            network_metrics = network_monitor.record(payload, received_at)
            if network_metrics is not None:
                network_path = f"network_metrics/{settings.device_id}"
                if (
                    store.set_record(
                        network_path,
                        "latest",
                        network_metrics,
                    )
                    is None
                ):
                    logger.error(
                        "Device network metrics were not persisted",
                        extra={
                            "firebase_path": f"{network_path}/latest",
                            "device_id": settings.device_id,
                        },
                    )
            telemetry = processor.process(payload, received_at)
            telemetry["raw_payload"] = {
                field: value
                for field, value in payload.items()
                if field not in NETWORK_PAYLOAD_FIELDS
            }
            _add_raw_sensor_fields(telemetry, payload)
            if not _has_any_valid_sensor_input(telemetry):
                logger.warning(
                    "Ignoring MQTT telemetry with no valid sensor readings",
                    extra={
                        "topic": settings.mqtt_topic,
                        "payload": payload,
                        "device_id": telemetry["device_id"],
                        "validation_errors": telemetry["validation_errors"],
                    },
                )
                return
            telemetry["model_feature_names"] = list(FEATURE_NAMES)
            if telemetry["model_sample"] is not None:
                engineered = feature_engineer.process(
                    telemetry["model_sample"], received_at
                )
                if engineered is not None:
                    telemetry["derived_features"] = {
                        name: engineered[name]
                        for name in FEATURE_NAMES[6:]
                    }
                    telemetry["model_features"] = [
                        engineered[name] for name in FEATURE_NAMES
                    ]
                    telemetry["sensor_values"]["vibration"] = engineered[
                        "vibration_rms"
                    ]
                    telemetry["features"] = [
                        telemetry["sensor_values"][name]
                        for name in ("temperature", "vibration", "current")
                    ]
                    telemetry["validation_errors"].pop("vibration", None)
            else:
                feature_engineer.reset()
                logger.warning(
                    "Skipping anomaly inference because model input readings are invalid",
                    extra={
                        "topic": settings.mqtt_topic,
                        "device_id": telemetry["device_id"],
                        "model_sample_errors": telemetry["model_sample_errors"],
                    },
                )
            telemetry["source"] = str(source).lower()
            telemetry["protocol"] = proto_str
            telemetry_path = f"telemetry/{settings.device_id}"
            logger.info(
                "Persisting validated MQTT telemetry",
                extra={
                    "firebase_path": telemetry_path,
                    "device_id": telemetry["device_id"],
                    "motor_id": telemetry["motor_id"],
                },
            )
            telemetry_id = store.push(telemetry_path, telemetry)
            if telemetry_id is None:
                logger.error(
                    "MQTT telemetry was not cached because Firebase write failed",
                    extra={
                        "firebase_path": telemetry_path,
                        "device_id": telemetry["device_id"],
                        "motor_id": telemetry["motor_id"],
                    },
                )
                return
            telemetry["id"] = telemetry_id
            logger.info(
                "MQTT telemetry persisted; starting AI inference",
                extra={
                    "firebase_path": telemetry_path,
                    "firebase_key": telemetry_id,
                    "device_id": telemetry["device_id"],
                    "motor_id": telemetry["motor_id"],
                },
            )
            prediction = detector.predict(telemetry)
            if prediction is not None:
                prediction_id = store.push(
                    f"predictions/{settings.device_id}", prediction
                )
                if prediction_id is None:
                    prediction = None
                else:
                    prediction["id"] = prediction_id
            alert = None
            if prediction is not None and prediction.get("prediction") == "ANOMALY":
                s_values = telemetry.get("sensor_values", {})
                t_val = s_values.get("temperature")
                v_val = s_values.get("vibration")
                c_val = s_values.get("current")
                abnormal_sensors = []
                if isinstance(t_val, (int, float)) and t_val >= 48.0:
                    abnormal_sensors.append("Temperature Sensor")
                if isinstance(v_val, (int, float)) and v_val >= 0.35:
                    abnormal_sensors.append("Vibration Sensor")
                if isinstance(c_val, (int, float)) and c_val >= 1.2:
                    abnormal_sensors.append("Current Sensor")
                sensor_name = ", ".join(abnormal_sensors) if abnormal_sensors else "Vibration Sensor"
                alert = {
                    **prediction,
                    "created_at": telemetry["received_at"],
                    "status": "OPEN",
                    "sensor": sensor_name,
                    "prediction_record_id": prediction["id"],
                }
                alert_id = store.push(f"alerts/{settings.device_id}", alert)
                if alert_id is None:
                    alert = None
                else:
                    alert["id"] = alert_id
            random_forest_prediction = None
            random_forest_latency_ms = None
            if random_forest_model.is_ready:
                try:
                    started = perf_counter()
                    random_forest_prediction = random_forest_model.predict(telemetry)
                    random_forest_latency_ms = (perf_counter() - started) * 1000.0
                except (ValueError, RuntimeError, TypeError):
                    logger.exception(
                        "Random Forest prediction failed",
                        extra={"telemetry_record_id": telemetry_id},
                    )
                if random_forest_prediction is not None:
                    random_forest_prediction["inference_latency_ms"] = random_forest_latency_ms
                    random_forest_prediction["inference_time_ms"] = random_forest_latency_ms
                    rf_prediction_path = f"random_forest_predictions/{settings.device_id}"
                    random_forest_prediction_id = store.set_record(
                        rf_prediction_path,
                        telemetry_id,
                        random_forest_prediction,
                    )
                    if random_forest_prediction_id is None:
                        random_forest_prediction = None
                    else:
                        random_forest_prediction["id"] = random_forest_prediction_id
            xgboost_prediction = None
            xgboost_latency_ms = None
            anomaly = None
            if xgboost_model.is_ready:
                try:
                    started = perf_counter()
                    xgboost_prediction = xgboost_model.predict(telemetry)
                    xgboost_latency_ms = (perf_counter() - started) * 1000.0
                except (ValueError, RuntimeError, TypeError, XGBoostError):
                    logger.exception(
                        "XGBoost prediction failed",
                        extra={"telemetry_record_id": telemetry_id},
                    )
                if xgboost_prediction is not None:
                    xgboost_prediction["inference_latency_ms"] = xgboost_latency_ms
                    xgboost_prediction["inference_time_ms"] = xgboost_latency_ms
                    prediction_path = f"xgboost_predictions/{settings.device_id}"
                    xgboost_prediction_id = store.set_record(
                        prediction_path,
                        telemetry_id,
                        xgboost_prediction,
                    )
                    if xgboost_prediction_id is None:
                        xgboost_prediction = None
                    else:
                        xgboost_prediction["id"] = xgboost_prediction_id

            # Evaluate both models and select final anomaly prediction for Alerts
            anomaly = None
            selected_pred, selected_model, is_disagreement = select_model_prediction(
                random_forest_prediction, xgboost_prediction
            )
            if selected_pred is not None and not is_disagreement:
                built_alert = build_anomaly_alert(
                    telemetry=telemetry,
                    telemetry_id=telemetry_id,
                    selected_prediction=selected_pred,
                    selected_model_name=selected_model,
                    rf_prediction=random_forest_prediction,
                    xgb_prediction=xgboost_prediction,
                )
                if built_alert is not None:
                    anomaly_id = store.set_record(
                        f"anomaly_alerts/{settings.device_id}",
                        telemetry_id,
                        built_alert,
                    )
                    if anomaly_id is not None:
                        built_alert["id"] = anomaly_id
                        anomaly = built_alert

            ground_truth = _extract_ground_truth(telemetry)
            comparison_record = None
            if random_forest_prediction is not None or xgboost_prediction is not None:
                rf_class = (
                    random_forest_prediction["predicted_class"]
                    if random_forest_prediction is not None
                    else None
                )
                xgb_class = (
                    xgboost_prediction["predicted_class"]
                    if xgboost_prediction is not None
                    else None
                )
                comparison_record = {
                    "device_id": telemetry["device_id"],
                    "telemetry_record_id": telemetry_id,
                    "timestamp": telemetry["received_at"],
                    "prediction_timestamp": telemetry["received_at"],
                    "random_forest": (
                        {
                            "predicted_class": rf_class,
                            "confidence": random_forest_prediction["confidence"],
                            "inference_latency_ms": random_forest_prediction.get(
                                "inference_latency_ms"
                            ),
                        }
                        if random_forest_prediction is not None
                        else None
                    ),
                    "xgboost": (
                        {
                            "predicted_class": xgb_class,
                            "confidence": xgboost_prediction["confidence"],
                            "inference_latency_ms": xgboost_prediction.get(
                                "inference_latency_ms"
                            ),
                        }
                        if xgboost_prediction is not None
                        else None
                    ),
                    "agreement": bool(
                        rf_class is not None
                        and xgb_class is not None
                        and rf_class == xgb_class
                    ),
                    "ground_truth": ground_truth,
                }
                try:
                    store.set_record(
                        f"model_comparisons/{settings.device_id}",
                        telemetry_id,
                        comparison_record,
                    )
                except RuntimeError:
                    logger.exception(
                        "Model comparison persistence failed",
                        extra={"telemetry_record_id": telemetry_id},
                    )

            live_data.add(
                telemetry,
                prediction,
                alert,
                xgboost_prediction,
                anomaly,
                random_forest_prediction=random_forest_prediction,
                comparison=comparison_record,
                ground_truth=ground_truth,
            )
            if prediction is not None:
                logger.info(
                    "AI prediction processed",
                    extra={
                        "device_id": telemetry["device_id"],
                        "telemetry_record_id": telemetry_id,
                        "prediction": prediction["prediction"],
                        "anomaly_score": prediction["anomaly_score"],
                        "model_status": prediction["model_status"],
                        "sensor_values": telemetry["sensor_values"],
                    },
                )
            if xgboost_prediction is not None:
                logger.info(
                    "XGBoost prediction persisted",
                    extra={
                        "telemetry_record_id": telemetry_id,
                        "prediction": xgboost_prediction["predicted_class"],
                        "confidence": xgboost_prediction["confidence"],
                    },
                )

        app.state.persist_telemetry = persist_telemetry

        def update_device_status(status: str, error: str | None) -> None:
            if not store.initialized:
                return
            is_connected = status.lower() == "connected"
            value: dict[str, Any] = {
                "motor_id": settings.device_id,
                "status": "ONLINE" if is_connected else "OFFLINE",
                "mqtt_status": status.upper(),
                "mqtt_topic": settings.mqtt_topic,
            }
            if error:
                value["last_error"] = error
            store.update_value(f"device_status/{settings.device_id}", value)

        mqtt_handler = MQTTHandler(
            settings,
            on_telemetry=persist_telemetry,
            on_status=update_device_status,
        )
        app.state.mqtt = mqtt_handler
        try:
            mqtt_handler.start()
        except (OSError, ValueError, RuntimeError):
            logger.exception("Could not start MQTT subscriber")

        def on_coap_telemetry(payload: dict[str, Any], raw_bytes: int, src: str) -> None:
            persist_telemetry(payload, raw_bytes=raw_bytes, source=src)

        coap_handler = CoAPHandler(
            host="0.0.0.0",
            port=5683,
            on_telemetry=on_coap_telemetry,
        )
        try:
            coap_handler.start()
        except Exception:
            logger.warning("Could not start CoAP handler", exc_info=True)
        app.state.coap = coap_handler

        try:
            yield
        finally:
            mqtt_handler.stop()
            coap_handler.stop()
            update_device_status("disconnected", "Backend shutting down")

    app = FastAPI(
        title="Industrial IoT Motor Monitoring API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.protocol_tracker = ProtocolTracker()
    app.state.persist_telemetry = None
    app.state.settings = Settings.from_env()
    app.state.firebase = FirebaseStore(app.state.settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(Settings.from_env().cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    @app.get("/api/v1/health")
    def health(request: Request) -> dict[str, Any]:
        store: FirebaseStore | None = getattr(request.app.state, "firebase", None)
        mqtt: MQTTHandler | None = getattr(request.app.state, "mqtt", None)
        model: IsolationForestModel | None = getattr(request.app.state, "model", None)
        xgboost_model: XGBoostPredictor | None = getattr(
            request.app.state, "xgboost_model", None
        )
        settings: Settings | None = getattr(request.app.state, "settings", None)
        live_data: LiveTelemetryBuffer | None = getattr(
            request.app.state, "live_data", None
        )
        network_monitor: NetworkMonitor | None = getattr(
            request.app.state, "network_monitor", None
        )
        network_metrics = network_monitor.snapshot() if network_monitor else None
        live_snapshot = live_data.snapshot(1) if live_data else None
        latest = live_snapshot["telemetry"] if live_snapshot else []
        latest_record = latest[0] if latest else None
        last_message_at = live_snapshot["last_message_at"] if live_snapshot else None
        message_age_seconds = None
        if last_message_at:
            try:
                message_time = datetime.fromisoformat(
                    last_message_at.replace("Z", "+00:00")
                )
                if message_time.tzinfo is None:
                    message_time = message_time.replace(tzinfo=timezone.utc)
                message_age_seconds = max(
                    0.0,
                    (datetime.now(timezone.utc) - message_time).total_seconds(),
                )
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring invalid persisted MQTT message timestamp",
                    extra={"last_message_at": last_message_at},
                )
                last_message_at = None
        model_status = "untrained"
        if model and model.is_trained:
            model_status = (
                "trained" if model.feature_names == FEATURE_NAMES else "incompatible"
            )
        components = {
            "mqtt": "connected" if mqtt and mqtt.connected else "disconnected",
            "firebase": "ready" if store and store.ready else "unavailable",
            "model": model_status,
            "xgboost": "trained" if xgboost_model and xgboost_model.is_ready else "unavailable",
        }
        mqtt_ready = bool(
            mqtt
            and mqtt.connected
            and mqtt.subscription_status == "subscribed"
        )
        current_record_is_valid = bool(
            latest_record
            and latest_record.get("source") in {"mqtt", "http", "coap"}
            and latest_record.get("id")
            and latest_record.get("received_at") == last_message_at
        )
        if not mqtt or not mqtt.connected:
            data_state = "mqtt_disconnected"
        elif mqtt.subscription_status not in {"subscribed", "subscribing"}:
            data_state = "mqtt_disconnected"
        elif not store or not store.ready:
            data_state = "firebase_disconnected"
        elif last_message_at is None:
            data_state = "waiting"
        elif (
            message_age_seconds is not None
            and settings is not None
            and message_age_seconds > settings.data_stale_timeout_seconds
        ):
            data_state = "stale"
        elif not current_record_is_valid:
            data_state = "waiting"
        else:
            data_state = "live"
        healthy = mqtt_ready and components["firebase"] == "ready"
        return {
            "status": "ok" if healthy else "degraded",
            "components": components,
            "firebase_error": (
                "See backend logs for Firebase configuration or write failures"
                if store and not store.ready
                else None
            ),
            "mqtt_status": {
                "subscription": (
                    mqtt.subscription_status if mqtt else "disconnected"
                ),
                "subscription_error": mqtt.subscription_error if mqtt else None,
                "broker": settings.mqtt_broker if settings else None,
                "port": settings.mqtt_port if settings else None,
                "topic": settings.mqtt_topic if settings else None,
                "messages_received": mqtt.messages_received if mqtt else 0,
                "payloads_processed": (
                    live_snapshot["total_received"] if live_snapshot else 0
                ),
                "last_message_at": last_message_at,
                "data_age_seconds": message_age_seconds,
                "stale_timeout_seconds": (
                    settings.data_stale_timeout_seconds if settings else None
                ),
            },
            "data_state": data_state,
            "device_health": (
                network_metrics["status"] if network_metrics else "OFFLINE"
            ),
            "network": network_metrics,
        }

    @app.get("/api/v1/live")
    def get_live_data(
        request: Request,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        live_data: LiveTelemetryBuffer = request.app.state.live_data
        data = live_data.snapshot(limit)
        latest = data["telemetry"][0] if data["telemetry"] else None
        data_health = health(request)
        state = data_health["data_state"]
        mqtt: MQTTHandler | None = getattr(request.app.state, "mqtt", None)
        if latest is None:
            device_status = None
        else:
            connected = state == "live" and bool(mqtt and mqtt.connected)
            device_status = {
                "device_id": latest["device_id"],
                "motor_id": latest["motor_id"],
                "status": (
                    "ONLINE"
                    if connected
                    else "STALE"
                    if state == "stale"
                    else "OFFLINE"
                ),
                "online": connected,
                "last_seen": latest["received_at"],
                "mqtt_topic": request.app.state.settings.mqtt_topic,
            }
        return {
            **data,
            "device_status": device_status,
            "health": data_health,
        }

    @app.get("/api/v1/model/status")
    def get_model_status(request: Request) -> dict[str, Any]:
        model: XGBoostPredictor = request.app.state.xgboost_model
        return {
            "model_name": "XGBoost",
            "status": "ready" if model.is_ready else "unavailable",
            "feature_columns": list(model.feature_columns),
            "class_labels_by_probability_column": list(model.class_labels),
            "confidence_calibrated": False,
            "load_error": model.load_error,
        }

    def require_firebase(request: Request) -> FirebaseStore:
        store: FirebaseStore | None = getattr(request.app.state, "firebase", None)
        if store is None or not store.ready:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Firebase is unavailable; configure credentials and check "
                    "backend logs"
                ),
            )
        return store

    def read_records(
        request: Request,
        collection: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        store = require_firebase(request)
        device_id = request.app.state.settings.device_id
        try:
            return store.get_records(f"{collection}/{device_id}", limit)
        except RuntimeError as exc:
            logger.exception("Firebase API read failed", extra={"collection": collection})
            raise HTTPException(
                status_code=503,
                detail="Firebase read failed; see backend logs",
            ) from exc

    @app.get("/api/v1/telemetry")
    def get_telemetry(
        request: Request,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict[str, Any]]:
        records = read_records(request, "telemetry", limit)
        return [
            record
            for record in records
            if record.get("source") == "mqtt"
            and record.get("id")
            and isinstance(record.get("raw_payload"), dict)
        ]

    @app.get("/api/v1/predictions")
    def get_predictions(
        request: Request,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict[str, Any]]:
        records = read_records(request, "predictions", limit)
        legacy_records = [
            record
            for record in records
            if record.get("model_status") == "TRAINED"
            and record.get("prediction") in {"NORMAL", "ANOMALY"}
            and record.get("telemetry_record_id")
            and record.get("id")
        ]
        xgboost_records = read_records(request, "xgboost_predictions", limit)
        xgboost_records = [
            record
            for record in xgboost_records
            if record.get("model_name") == "XGBoost"
            and record.get("predicted_class")
            in {"NORMAL", "HIGH_VIBRATION", "HIGH_TEMPERATURE", "HIGH_CURRENT"}
            and record.get("telemetry_record_id")
            and record.get("id")
        ]
        return [*legacy_records, *xgboost_records]

    @app.get("/api/v1/anomalies")
    def get_anomalies(
        request: Request,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict[str, Any]]:
        records = read_records(request, "anomaly_alerts", limit)
        return [
            record
            for record in records
            if (
                record.get("model_name") in {"XGBoost", "Random Forest"}
                or record.get("selected_model_name") in {"XGBoost", "Random Forest"}
            )
            and record.get("telemetry_record_id")
            and record.get("anomaly_type")
            and record.get("id")
        ]

    @app.get("/api/v1/devices")
    def get_devices(request: Request) -> list[dict[str, Any]]:
        settings: Settings = request.app.state.settings
        live_data: LiveTelemetryBuffer = request.app.state.live_data
        network_monitor: NetworkMonitor = request.app.state.network_monitor
        health_data = health(request)
        snapshot = live_data.snapshot(1)
        latest = snapshot["telemetry"][0] if snapshot["telemetry"] else None
        state = health_data["data_state"]
        network = network_monitor.snapshot()
        status = network["status"]
        return [
            {
                "device_id": settings.device_id,
                "status": status,
                "online": status in {"HEALTHY", "WARNING"},
                "last_telemetry_at": (
                    latest.get("received_at")
                    if latest
                    else snapshot["last_message_at"]
                ),
                "mqtt_status": health_data["components"]["mqtt"],
                "data_state": state,
                **{
                    key: value
                    for key, value in network.items()
                    if key not in {"device_id", "status"}
                },
            }
        ]

    @app.get("/api/v1/devices/{device_id}/network")
    def get_device_network(device_id: str, request: Request) -> dict[str, Any]:
        settings: Settings = request.app.state.settings
        if device_id != settings.device_id:
            raise HTTPException(
                status_code=404,
                detail=f"Device '{device_id}' is not configured",
            )
        network_monitor: NetworkMonitor = request.app.state.network_monitor
        return network_monitor.snapshot()

    @app.get("/api/v1/device-status")
    def get_device_status(request: Request) -> Any:
        store = require_firebase(request)
        device_id = request.app.state.settings.device_id
        try:
            return store.get_value(f"device_status/{device_id}")
        except RuntimeError as exc:
            logger.exception("Firebase API read failed", extra={"collection": "device_status"})
            raise HTTPException(
                status_code=503,
                detail="Firebase read failed; see backend logs",
            ) from exc

    @app.get("/api/v1/alerts")
    def get_alerts(
        request: Request,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict[str, Any]]:
        records = read_records(request, "alerts", limit)
        legacy_records = [
            record
            for record in records
            if record.get("telemetry_record_id")
            and record.get("prediction_record_id")
            and record.get("id")
        ]
        anomaly_records = read_records(request, "anomaly_alerts", limit)
        return [
            *legacy_records,
            *[
                record
                for record in anomaly_records
                if record.get("telemetry_record_id")
                and record.get("prediction_record_id")
                and record.get("id")
            ],
        ]

    @app.get("/api/v1/model/evaluation")
    def get_model_evaluation(request: Request) -> dict[str, Any]:
        """Return offline evaluation metrics for Random Forest and XGBoost."""
        return _load_offline_metrics()

    @app.get("/api/v1/model/live-comparison")
    def get_live_comparison(request: Request) -> dict[str, Any]:
        """Return the latest live comparison data between Random Forest and XGBoost."""
        live_data: LiveTelemetryBuffer = request.app.state.live_data
        return live_data.live_stats()

    @app.get("/api/v1/model/live-metrics")
    def get_live_metrics(request: Request) -> dict[str, Any]:
        """Return live evaluation metrics when ground-truth labels are available."""
        live_data: LiveTelemetryBuffer = request.app.state.live_data
        snapshot = live_data.snapshot(500)
        comparisons = snapshot.get("comparisons", [])
        return _compute_live_metrics(comparisons)

    @app.post("/api/v1/telemetry")
    @app.post("/api/telemetry")
    @app.post("/telemetry")
    async def receive_http_telemetry(request: Request) -> dict[str, Any]:
        """Receive HTTP telemetry from ESP32 or simulation client."""
        try:
            body_bytes = await request.body()
            payload = json.loads(body_bytes.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Payload must be a JSON object")

        payload.setdefault("protocol", "HTTP")
        persist_fn = getattr(request.app.state, "persist_telemetry", None)
        if persist_fn:
            persist_fn(payload, raw_bytes=len(body_bytes), source="http")
        tracker: ProtocolTracker = request.app.state.protocol_tracker
        return {
            "status": "ok",
            "protocol": "http",
            "received_bytes": len(body_bytes),
            "metrics": tracker.get_comparison().get("http"),
        }

    @app.get("/api/v1/protocols/comparison")
    @app.get("/api/protocols/comparison")
    def get_protocol_comparison(request: Request) -> dict[str, Any]:
        """Return aggregated protocol comparison metrics for MQTT, HTTP, and CoAP."""
        tracker: ProtocolTracker = request.app.state.protocol_tracker
        comparison = tracker.get_comparison()
        store: FirebaseStore = request.app.state.firebase
        status = tracker.get_status()
        total_msgs = sum(p["total_messages"] for p in status["active_protocols"].values())
        if total_msgs == 0 and store.ready:
            try:
                fb_data = store.get_value(f"protocol_comparison/{request.app.state.settings.device_id}")
                if isinstance(fb_data, dict):
                    return fb_data
            except Exception:
                logger.debug("Could not read protocol comparison from Firebase")
        return comparison

    @app.get("/api/v1/protocols/status")
    @app.get("/api/protocols/status")
    def get_protocol_status(request: Request) -> dict[str, Any]:
        """Return live protocol connection and message counts."""
        tracker: ProtocolTracker = request.app.state.protocol_tracker
        return tracker.get_status()

    @app.get("/api/v1/protocols/live")
    @app.get("/api/protocols/live")
    def get_protocol_live(request: Request) -> dict[str, Any]:
        """Return the latest telemetry and metrics for each protocol."""
        tracker: ProtocolTracker = request.app.state.protocol_tracker
        return tracker.get_live()

    @app.get("/api/v1/protocols/history")
    @app.get("/api/protocols/history")
    def get_protocol_history(request: Request) -> list[dict[str, Any]]:
        """Return historical completed experiments."""
        tracker: ProtocolTracker = request.app.state.protocol_tracker
        return tracker.get_history()

    @app.post("/api/v1/protocols/reset")
    @app.post("/api/protocols/reset")
    def reset_protocol_experiment(
        request: Request,
        protocol: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Reset the current experiment for a protocol or all protocols."""
        tracker: ProtocolTracker = request.app.state.protocol_tracker
        return tracker.reset_experiment(protocol)

    @app.post("/api/v1/admin/clear-telemetry")
    @app.post("/api/admin/clear-telemetry")
    async def clear_telemetry(request: Request) -> dict[str, Any]:
        """Admin endpoint to delete all old telemetry, predictions, alerts, and experiments."""
        settings: Settings = request.app.state.settings

        # 1. Authentication Check
        auth_key = request.headers.get("x-admin-key")
        if not auth_key:
            auth_header = request.headers.get("authorization", "")
            if auth_header.lower().startswith("bearer "):
                auth_key = auth_header[7:].strip()

        body: dict[str, Any] = {}
        try:
            raw_body = await request.body()
            if raw_body:
                body = json.loads(raw_body.decode("utf-8"))
        except Exception:
            pass

        if not auth_key and isinstance(body, dict):
            auth_key = body.get("admin_key")
        if not auth_key:
            auth_key = request.query_params.get("admin_key")

        configured_key = getattr(settings, "admin_api_key", "admin-secret-key-iiot")
        if not auth_key or auth_key.strip() != configured_key:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Invalid or missing admin authentication key",
            )

        # 2. Confirmation Check
        is_confirmed = (
            isinstance(body, dict)
            and (
                body.get("confirm") is True
                or body.get("confirm_delete") is True
                or body.get("action") == "CLEAR_TELEMETRY"
            )
            or str(request.query_params.get("confirm", "")).lower() == "true"
        )
        if not is_confirmed:
            raise HTTPException(
                status_code=400,
                detail="Confirmation required. Please set 'confirm': true in request body to permanently clear telemetry.",
            )

        # 3. Determine device_id
        target_device_id = (
            (body.get("device_id") if isinstance(body, dict) else None)
            or request.query_params.get("device_id")
            or settings.device_id
        )

        # 4. Clear Firebase Database Records
        store: FirebaseStore = request.app.state.firebase
        cleanup_result: dict[str, Any] = {
            "deleted_counts": {},
            "preserved": [
                "device_status",
                "network_metrics",
                "device_configurations",
                "application_settings",
                "credentials",
            ],
            "device_id": target_device_id,
        }
        if store and store.initialized:
            try:
                cleanup_result = store.clear_telemetry_records(device_id=target_device_id)
            except Exception as exc:
                logger.warning("Firebase cleanup exception: %s", exc)

        counts = cleanup_result.get("deleted_counts", {})
        if "total" not in counts:
            counts["total"] = sum(v for v in counts.values() if isinstance(v, (int, float)))
        cleanup_result["deleted_counts"] = counts

        # 5. Clear In-Memory Buffers
        live_data: LiveTelemetryBuffer = request.app.state.live_data
        live_data.clear()

        protocol_tracker: ProtocolTracker | None = getattr(request.app.state, "protocol_tracker", None)
        if protocol_tracker:
            protocol_tracker.reset_experiment()

        feature_engineer = getattr(request.app.state, "feature_engineer", None)
        if feature_engineer:
            feature_engineer.reset()

        logger.info(
            "Admin cleared telemetry data for device %s. Deleted counts: %s",
            target_device_id,
            cleanup_result.get("deleted_counts"),
        )

        return {
            "status": "success",
            "message": "All historical telemetry, AI predictions, anomaly records, and experiment history have been cleared.",
            "deleted_counts": cleanup_result.get("deleted_counts", {}),
            "preserved": cleanup_result.get("preserved", []),
            "device_id": target_device_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
