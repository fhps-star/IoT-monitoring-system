"""Ensemble anomaly detection selector integrating Random Forest and XGBoost predictions."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

SENSOR_MAP = {
    "HIGH_TEMPERATURE": "Temperature Sensor",
    "HIGH_VIBRATION": "Vibration Sensor",
    "HIGH_CURRENT": "Current Sensor",
}


def select_model_prediction(
    rf_prediction: dict[str, Any] | None,
    xgb_prediction: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None, bool]:
    """Compare Random Forest and XGBoost predictions and select the winning or combined prediction.

    Rules:
    - If both models predict the same anomaly on the same sensor, select that prediction
      (picking the model with higher confidence score).
    - If both models detect anomalies on different sensors (e.g. RF detects HIGH_CURRENT
      and XGBoost detects HIGH_VIBRATION), both sensors are in an abnormal state.
      The arbiter preserves both anomaly labels together in a multi-sensor alert.
    - If one model predicts an anomaly and the other predicts NORMAL:
      - If the anomaly model confidence >= NORMAL confidence, select the anomaly.
      - If NORMAL confidence is higher with equal confidence on anomaly vs normal,
        mark as a model disagreement.
    - If both models predict NORMAL, no alert is generated.

    Returns:
        tuple of (selected_prediction_dict, selected_model_name, is_disagreement)
    """
    if rf_prediction is None and xgb_prediction is None:
        return None, None, False

    if rf_prediction is None:
        return xgb_prediction, "XGBoost", False

    if xgb_prediction is None:
        return rf_prediction, "Random Forest", False

    rf_class = rf_prediction.get("predicted_class")
    rf_conf = float(rf_prediction.get("confidence") or 0.0)
    xgb_class = xgb_prediction.get("predicted_class")
    xgb_conf = float(xgb_prediction.get("confidence") or 0.0)

    is_rf_anomaly = bool(rf_class and rf_class != "NORMAL")
    is_xgb_anomaly = bool(xgb_class and xgb_class != "NORMAL")

    # Case 1: Both models predict NORMAL
    if not is_rf_anomaly and not is_xgb_anomaly:
        higher_model = "Random Forest" if rf_conf >= xgb_conf else "XGBoost"
        higher_pred = rf_prediction if rf_conf >= xgb_conf else xgb_prediction
        return higher_pred, higher_model, False

    # Case 2: Both models predict the SAME anomaly class on the SAME sensor
    if is_rf_anomaly and is_xgb_anomaly and rf_class == xgb_class:
        if rf_conf >= xgb_conf:
            return rf_prediction, "Random Forest", False
        else:
            return xgb_prediction, "XGBoost", False

    # Case 3: Both models detect DIFFERENT anomalies on DIFFERENT sensors
    # e.g. RF -> HIGH_CURRENT (85%) & XGBoost -> HIGH_VIBRATION (85%)
    # or RF -> HIGH_TEMPERATURE (92%) & XGBoost -> HIGH_VIBRATION (70%)
    # Both anomalies are real and must be included together in the alert!
    if is_rf_anomaly and is_xgb_anomaly and rf_class != xgb_class:
        # Build composite prediction reference
        primary_pred = rf_prediction if rf_conf >= xgb_conf else xgb_prediction
        primary_model = "Random Forest" if rf_conf >= xgb_conf else "XGBoost"
        model_name = "Random Forest & XGBoost"
        return primary_pred, model_name, False

    # Case 4: One model predicts an anomaly and the other predicts NORMAL
    if is_rf_anomaly and not is_xgb_anomaly:
        diff = round(rf_conf - xgb_conf, 6)
        if diff >= 0:
            return rf_prediction, "Random Forest", False
        else:
            # RF anomaly has lower confidence than XGBoost NORMAL
            return None, None, True

    if is_xgb_anomaly and not is_rf_anomaly:
        diff = round(xgb_conf - rf_conf, 6)
        if diff >= 0:
            return xgb_prediction, "XGBoost", False
        else:
            # XGB anomaly has lower confidence than RF NORMAL
            return None, None, True

    return None, None, False


def build_anomaly_alert(
    telemetry: dict[str, Any],
    telemetry_id: str,
    selected_prediction: dict[str, Any],
    selected_model_name: str,
    rf_prediction: dict[str, Any] | None,
    xgb_prediction: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build an anomaly alert record supporting single and multi-sensor anomalies.

    - Single Sensor Anomaly: When both agree on the same anomaly, outputs that anomaly label and sensor.
    - Multi-Sensor Anomaly: When both models detect different anomalies (e.g. HIGH_CURRENT + HIGH_VIBRATION),
      both anomaly labels and both sensor names are included together in the alert.
    - Suppresses NORMAL predictions (generates alerts strictly for anomalies).
    """
    rf_class = rf_prediction.get("predicted_class") if rf_prediction else None
    rf_conf = float(rf_prediction.get("confidence") or 0.0) if rf_prediction else 0.0
    xgb_class = xgb_prediction.get("predicted_class") if xgb_prediction else None
    xgb_conf = float(xgb_prediction.get("confidence") or 0.0) if xgb_prediction else 0.0

    is_rf_anomaly = bool(rf_class and rf_class != "NORMAL")
    is_xgb_anomaly = bool(xgb_class and xgb_class != "NORMAL")

    # If neither model detected an anomaly, do not generate an alert
    if not is_rf_anomaly and not is_xgb_anomaly:
        return None

    timestamp = (
        selected_prediction.get("prediction_timestamp")
        or telemetry.get("received_at")
    )
    pred_id = selected_prediction.get("id") or telemetry_id

    both_models_predictions = {
        "random_forest": (
            {
                "predicted_class": rf_class,
                "confidence": rf_conf,
                "inference_latency_ms": rf_prediction.get("inference_latency_ms"),
                "prediction_id": rf_prediction.get("id"),
            }
            if rf_prediction is not None
            else None
        ),
        "xgboost": (
            {
                "predicted_class": xgb_class,
                "confidence": xgb_conf,
                "inference_latency_ms": xgb_prediction.get("inference_latency_ms"),
                "prediction_id": xgb_prediction.get("id"),
            }
            if xgb_prediction is not None
            else None
        ),
    }

    # Case A: Both models detect DIFFERENT anomalies on DIFFERENT sensors (Multi-Sensor Alert)
    if is_rf_anomaly and is_xgb_anomaly and rf_class != xgb_class:
        combined_classes = [rf_class, xgb_class]
        anomaly_type = ", ".join(combined_classes)
        sensor_list = [
            SENSOR_MAP.get(rf_class, "Sensor"),
            SENSOR_MAP.get(xgb_class, "Sensor"),
        ]
        sensor_name = ", ".join(sensor_list)
        winning_conf = max(rf_conf, xgb_conf)
        description = (
            f"Dual sensor anomaly detected: {rf_class} (Random Forest: {rf_conf*100:.1f}%) "
            f"and {xgb_class} (XGBoost: {xgb_conf*100:.1f}%)"
        )
        return {
            "device_id": telemetry["device_id"],
            "telemetry_record_id": telemetry_id,
            "prediction_record_id": pred_id,
            "anomaly_type": anomaly_type,
            "predicted_class": anomaly_type,
            "severity": "Warning",
            "sensor": sensor_name,
            "detected_timestamp": timestamp,
            "timestamp": timestamp,
            "related_sensor_values": (
                selected_prediction.get("sensor_values")
                or telemetry.get("sensor_values", {})
            ),
            "confidence": winning_conf,
            "confidence_score": winning_conf,
            "model_name": selected_model_name,
            "selected_model_name": selected_model_name,
            "alert_status": "OPEN",
            "description": description,
            "agreement": False,
            "multi_sensor": True,
            "both_models_predictions": both_models_predictions,
            "models_predictions": both_models_predictions,
            "random_forest_prediction": both_models_predictions["random_forest"],
            "xgboost_prediction": both_models_predictions["xgboost"],
        }

    # Case B: Both models agree on the SAME anomaly (Single Sensor Alert)
    if is_rf_anomaly and is_xgb_anomaly and rf_class == xgb_class:
        anomaly_type = rf_class
        sensor_name = SENSOR_MAP.get(rf_class, "Motor Sensor")
        winning_conf = max(rf_conf, xgb_conf)
        return {
            "device_id": telemetry["device_id"],
            "telemetry_record_id": telemetry_id,
            "prediction_record_id": pred_id,
            "anomaly_type": anomaly_type,
            "predicted_class": anomaly_type,
            "severity": "Warning",
            "sensor": sensor_name,
            "detected_timestamp": timestamp,
            "timestamp": timestamp,
            "related_sensor_values": (
                selected_prediction.get("sensor_values")
                or telemetry.get("sensor_values", {})
            ),
            "confidence": winning_conf,
            "confidence_score": winning_conf,
            "model_name": selected_model_name,
            "selected_model_name": selected_model_name,
            "alert_status": "OPEN",
            "description": "AI anomaly prediction; this is not a confirmed physical failure.",
            "agreement": True,
            "multi_sensor": False,
            "both_models_predictions": both_models_predictions,
            "models_predictions": both_models_predictions,
            "random_forest_prediction": both_models_predictions["random_forest"],
            "xgboost_prediction": both_models_predictions["xgboost"],
        }

    # Case C: Only ONE model predicted an anomaly (Single Sensor Alert)
    active_anomaly_pred = rf_prediction if is_rf_anomaly else xgb_prediction
    active_class = rf_class if is_rf_anomaly else xgb_class
    active_conf = rf_conf if is_rf_anomaly else xgb_conf
    sensor_name = SENSOR_MAP.get(active_class, "Motor Sensor")

    return {
        "device_id": telemetry["device_id"],
        "telemetry_record_id": telemetry_id,
        "prediction_record_id": pred_id,
        "anomaly_type": active_class,
        "predicted_class": active_class,
        "severity": "Warning",
        "sensor": sensor_name,
        "detected_timestamp": timestamp,
        "timestamp": timestamp,
        "related_sensor_values": (
            active_anomaly_pred.get("sensor_values")
            or telemetry.get("sensor_values", {})
        ),
        "confidence": active_conf,
        "confidence_score": active_conf,
        "model_name": selected_model_name,
        "selected_model_name": selected_model_name,
        "alert_status": "OPEN",
        "description": "AI anomaly prediction; this is not a confirmed physical failure.",
        "agreement": False,
        "multi_sensor": False,
        "both_models_predictions": both_models_predictions,
        "models_predictions": both_models_predictions,
        "random_forest_prediction": both_models_predictions["random_forest"],
        "xgboost_prediction": both_models_predictions["xgboost"],
    }
