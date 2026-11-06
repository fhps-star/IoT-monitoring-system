"""Translate model output into persisted prediction and alert records."""

from typing import Any

from backend.ai_model import IsolationForestModel
from backend.feature_engineering import FEATURE_NAMES


class AnomalyDetector:
    def __init__(self, model: IsolationForestModel) -> None:
        self._model = model

    def predict(self, telemetry: dict[str, Any]) -> dict[str, Any] | None:
        features = telemetry.get("model_features")
        if (
            not self._model.is_trained
            or not isinstance(features, list)
            or self._model.feature_count != len(features)
            or self._model.feature_names != FEATURE_NAMES
            or telemetry.get("model_feature_names") != list(FEATURE_NAMES)
        ):
            return None
        result: dict[str, Any] = {
            "device_id": telemetry["device_id"],
            "motor_id": telemetry["motor_id"],
            "telemetry_record_id": telemetry["id"],
            "predicted_at": telemetry["received_at"],
            "timestamp": telemetry["received_at"],
            "model": "IsolationForest",
            "anomaly_score": None,
        }
        threshold = self._model.artifact_metadata.get("anomaly_score_threshold")
        if isinstance(threshold, (int, float)):
            result["anomaly_score_threshold"] = threshold

        label, score = self._model.predict(features)
        result.update(
            {
                "prediction": label,
                "model_status": "TRAINED",
                "anomaly_score": score,
            }
        )
        return result
