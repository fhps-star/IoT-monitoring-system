"""In-memory buffer for telemetry received during this backend process."""

from collections import deque
from threading import Lock
from typing import Any


class LiveTelemetryBuffer:
    def __init__(self, capacity: int) -> None:
        self._telemetry: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._predictions: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._alerts: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = Lock()
        self._total_received = 0
        self._last_message_at: str | None = None
        self._xgboost_predictions: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._anomalies: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._random_forest_predictions: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._comparisons: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._ground_truths: deque[str | None] = deque(maxlen=capacity)
        self._rf_total = 0
        self._rf_normal = 0
        self._rf_anomaly = 0
        self._xgb_total = 0
        self._xgb_normal = 0
        self._xgb_anomaly = 0
        self._agreements = 0
        self._total_compared = 0

    def note_message(self, received_at: str) -> None:
        with self._lock:
            self._total_received += 1
            self._last_message_at = received_at

    def restore_last_message(self, received_at: str) -> None:
        with self._lock:
            self._last_message_at = received_at

    def clear(self) -> None:
        """Clear all in-memory telemetry, predictions, alerts, and counters."""
        with self._lock:
            self._telemetry.clear()
            self._predictions.clear()
            self._alerts.clear()
            self._xgboost_predictions.clear()
            self._anomalies.clear()
            self._random_forest_predictions.clear()
            self._comparisons.clear()
            self._ground_truths.clear()
            self._rf_total = 0
            self._rf_normal = 0
            self._rf_anomaly = 0
            self._xgb_total = 0
            self._xgb_normal = 0
            self._xgb_anomaly = 0
            self._agreements = 0
            self._total_compared = 0
            self._total_received = 0
            self._last_message_at = None

    def add(
        self,
        telemetry: dict[str, Any],
        prediction: dict[str, Any] | None = None,
        alert: dict[str, Any] | None = None,
        xgboost_prediction: dict[str, Any] | None = None,
        anomaly: dict[str, Any] | None = None,
        random_forest_prediction: dict[str, Any] | None = None,
        comparison: dict[str, Any] | None = None,
        ground_truth: str | None = None,
    ) -> None:
        with self._lock:
            self._telemetry.appendleft(telemetry)
            if prediction is not None:
                self._predictions.appendleft(prediction)
            if xgboost_prediction is not None:
                self._xgboost_predictions.appendleft(xgboost_prediction)
                self._xgb_total += 1
                if xgboost_prediction.get("predicted_class") == "NORMAL":
                    self._xgb_normal += 1
                else:
                    self._xgb_anomaly += 1
            if random_forest_prediction is not None:
                self._random_forest_predictions.appendleft(random_forest_prediction)
                self._rf_total += 1
                if random_forest_prediction.get("predicted_class") == "NORMAL":
                    self._rf_normal += 1
                else:
                    self._rf_anomaly += 1
            if alert is not None:
                self._alerts.appendleft(alert)
            if anomaly is not None:
                self._anomalies.appendleft(anomaly)
            if comparison is not None:
                self._comparisons.appendleft(comparison)
                if (
                    comparison.get("random_forest") is not None
                    and comparison.get("xgboost") is not None
                ):
                    self._total_compared += 1
                    if comparison.get("agreement") is True:
                        self._agreements += 1
            if ground_truth is not None:
                self._ground_truths.appendleft(ground_truth)

    def live_stats(self) -> dict[str, Any]:
        with self._lock:
            latest_comparison = self._comparisons[0] if self._comparisons else None
            rf_latest = (
                self._random_forest_predictions[0]
                if self._random_forest_predictions
                else None
            )
            xgb_latest = (
                self._xgboost_predictions[0]
                if self._xgboost_predictions
                else None
            )
            agreement_pct = (
                round((self._agreements / self._total_compared * 100.0), 2)
                if self._total_compared > 0
                else None
            )
            return {
                "latest_comparison": latest_comparison,
                "random_forest": {
                    "latest_prediction": rf_latest.get("predicted_class") if rf_latest else None,
                    "latest_confidence": rf_latest.get("confidence") if rf_latest else None,
                    "inference_latency_ms": rf_latest.get("inference_latency_ms") if rf_latest else None,
                    "total_predictions": self._rf_total,
                    "normal_predictions": self._rf_normal,
                    "anomaly_predictions": self._rf_anomaly,
                },
                "xgboost": {
                    "latest_prediction": xgb_latest.get("predicted_class") if xgb_latest else None,
                    "latest_confidence": xgb_latest.get("confidence") if xgb_latest else None,
                    "inference_latency_ms": xgb_latest.get("inference_latency_ms") if xgb_latest else None,
                    "total_predictions": self._xgb_total,
                    "normal_predictions": self._xgb_normal,
                    "anomaly_predictions": self._xgb_anomaly,
                },
                "agreement_percentage": agreement_pct,
                "agreement_count": self._agreements,
                "total_compared": self._total_compared,
                "last_updated": (
                    latest_comparison.get("timestamp")
                    if latest_comparison
                    else self._last_message_at
                ),
            }

    def snapshot(self, limit: int) -> dict[str, Any]:
        with self._lock:
            return {
                "telemetry": list(self._telemetry)[:limit],
                "predictions": list(self._predictions)[:limit],
                "xgboost_predictions": list(self._xgboost_predictions)[:limit],
                "random_forest_predictions": list(self._random_forest_predictions)[:limit],
                "comparisons": list(self._comparisons)[:limit],
                "ground_truths": list(self._ground_truths)[:limit],
                "alerts": list(self._alerts)[:limit],
                "anomalies": list(self._anomalies)[:limit],
                "total_received": self._total_received,
                "last_message_at": self._last_message_at,
            }
