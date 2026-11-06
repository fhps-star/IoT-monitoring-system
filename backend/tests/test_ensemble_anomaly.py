"""Tests for ensemble anomaly detection and alerts integration."""

from datetime import datetime, timezone
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

from fastapi.testclient import TestClient

from backend.ensemble_anomaly import build_anomaly_alert, select_model_prediction
from backend.firebase_config import FirebaseStore
from backend.main import create_app
from backend.mqtt_handler import MQTTHandler
from backend.random_forest_predictor import RandomForestPredictor
from backend.xgboost_predictor import XGBoostPredictor


class TestEnsembleAnomalySelection(unittest.TestCase):
    def test_both_models_agree_same_class_same_sensor_selects_prediction(self):
        # Example 1: Same sensor & same prediction -> single alert, higher confidence model selected
        rf_pred = {
            "predicted_class": "HIGH_VIBRATION",
            "confidence": 0.95,
            "model_name": "Random Forest",
            "id": "rf-1",
        }
        xgb_pred = {
            "predicted_class": "HIGH_VIBRATION",
            "confidence": 0.92,
            "model_name": "XGBoost",
            "id": "xgb-1",
        }
        selected, model_name, is_disagree = select_model_prediction(rf_pred, xgb_pred)
        self.assertFalse(is_disagree)
        self.assertEqual(model_name, "Random Forest")
        self.assertEqual(selected["predicted_class"], "HIGH_VIBRATION")
        self.assertEqual(selected["confidence"], 0.95)

        alert = build_anomaly_alert(
            telemetry={"device_id": "motor_01", "received_at": "2026-11-06T00:00:00Z"},
            telemetry_id="tel-1",
            selected_prediction=selected,
            selected_model_name=model_name,
            rf_prediction=rf_pred,
            xgb_prediction=xgb_pred,
        )
        self.assertIsNotNone(alert)
        self.assertEqual(alert["anomaly_type"], "HIGH_VIBRATION")
        self.assertEqual(alert["sensor"], "Vibration Sensor")
        self.assertEqual(alert["selected_model_name"], "Random Forest")
        self.assertFalse(alert["multi_sensor"])

    def test_both_models_detect_different_sensors_combines_both_labels_equal_confidence(self):
        # Example 3: RF -> HIGH_CURRENT (85%), XGBoost -> HIGH_VIBRATION (85%)
        # Two different sensors -> Both labels included together in the alert!
        rf_pred = {
            "predicted_class": "HIGH_CURRENT",
            "confidence": 0.85,
            "model_name": "Random Forest",
            "id": "rf-1",
        }
        xgb_pred = {
            "predicted_class": "HIGH_VIBRATION",
            "confidence": 0.85,
            "model_name": "XGBoost",
            "id": "xgb-1",
        }
        selected, model_name, is_disagree = select_model_prediction(rf_pred, xgb_pred)
        self.assertFalse(is_disagree)
        self.assertEqual(model_name, "Random Forest & XGBoost")

        alert = build_anomaly_alert(
            telemetry={"device_id": "motor_01", "received_at": "2026-11-06T00:00:00Z"},
            telemetry_id="tel-3",
            selected_prediction=selected,
            selected_model_name=model_name,
            rf_prediction=rf_pred,
            xgb_prediction=xgb_pred,
        )
        self.assertIsNotNone(alert)
        self.assertIn("HIGH_CURRENT", alert["anomaly_type"])
        self.assertIn("HIGH_VIBRATION", alert["anomaly_type"])
        self.assertIn("Current Sensor", alert["sensor"])
        self.assertIn("Vibration Sensor", alert["sensor"])
        self.assertTrue(alert["multi_sensor"])
        self.assertEqual(alert["confidence_score"], 0.85)

    def test_both_models_detect_different_sensors_combines_both_labels_differing_confidence(self):
        # Example 2: RF -> HIGH_TEMPERATURE (92%), XGBoost -> HIGH_VIBRATION (70%)
        # Two different sensors -> Both labels included together!
        rf_pred = {
            "predicted_class": "HIGH_TEMPERATURE",
            "confidence": 0.92,
            "model_name": "Random Forest",
            "id": "rf-1",
        }
        xgb_pred = {
            "predicted_class": "HIGH_VIBRATION",
            "confidence": 0.70,
            "model_name": "XGBoost",
            "id": "xgb-1",
        }
        selected, model_name, is_disagree = select_model_prediction(rf_pred, xgb_pred)
        self.assertFalse(is_disagree)
        self.assertEqual(model_name, "Random Forest & XGBoost")

        alert = build_anomaly_alert(
            telemetry={"device_id": "motor_01", "received_at": "2026-11-06T00:00:00Z"},
            telemetry_id="tel-2",
            selected_prediction=selected,
            selected_model_name=model_name,
            rf_prediction=rf_pred,
            xgb_prediction=xgb_pred,
        )
        self.assertIsNotNone(alert)
        self.assertIn("HIGH_TEMPERATURE", alert["anomaly_type"])
        self.assertIn("HIGH_VIBRATION", alert["anomaly_type"])
        self.assertIn("Temperature Sensor", alert["sensor"])
        self.assertIn("Vibration Sensor", alert["sensor"])
        self.assertTrue(alert["multi_sensor"])
        self.assertEqual(alert["confidence_score"], 0.92)

    def test_one_model_anomaly_one_model_normal_selects_anomaly(self):
        rf_pred = {
            "predicted_class": "NORMAL",
            "confidence": 0.65,
            "model_name": "Random Forest",
            "id": "rf-1",
        }
        xgb_pred = {
            "predicted_class": "HIGH_CURRENT",
            "confidence": 0.89,
            "model_name": "XGBoost",
            "id": "xgb-1",
        }
        selected, model_name, is_disagree = select_model_prediction(rf_pred, xgb_pred)
        self.assertFalse(is_disagree)
        self.assertEqual(model_name, "XGBoost")
        self.assertEqual(selected["predicted_class"], "HIGH_CURRENT")

        alert = build_anomaly_alert(
            telemetry={"device_id": "motor_01", "received_at": "2026-11-06T00:00:00Z"},
            telemetry_id="tel-4",
            selected_prediction=selected,
            selected_model_name=model_name,
            rf_prediction=rf_pred,
            xgb_prediction=xgb_pred,
        )
        self.assertIsNotNone(alert)
        self.assertEqual(alert["anomaly_type"], "HIGH_CURRENT")
        self.assertEqual(alert["sensor"], "Current Sensor")

    def test_both_models_normal_no_alert(self):
        rf_pred = {"predicted_class": "NORMAL", "confidence": 0.95}
        xgb_pred = {"predicted_class": "NORMAL", "confidence": 0.92}
        selected, model_name, is_disagree = select_model_prediction(rf_pred, xgb_pred)
        alert = build_anomaly_alert(
            telemetry={"device_id": "motor_01", "received_at": "2026-11-06T00:00:00Z"},
            telemetry_id="tel-5",
            selected_prediction=selected,
            selected_model_name=model_name,
            rf_prediction=rf_pred,
            xgb_prediction=xgb_pred,
        )
        self.assertIsNone(alert)

    def test_single_model_fallback(self):
        xgb_pred = {
            "predicted_class": "HIGH_VIBRATION",
            "confidence": 0.92,
            "model_name": "XGBoost",
            "id": "xgb-1",
        }
        selected, model_name, is_disagree = select_model_prediction(None, xgb_pred)
        self.assertFalse(is_disagree)
        self.assertEqual(model_name, "XGBoost")
        self.assertEqual(selected["predicted_class"], "HIGH_VIBRATION")

        alert = build_anomaly_alert(
            telemetry={"device_id": "motor_01", "received_at": "2026-11-06T00:00:00Z"},
            telemetry_id="tel-6",
            selected_prediction=selected,
            selected_model_name=model_name,
            rf_prediction=None,
            xgb_prediction=xgb_pred,
        )
        self.assertIsNotNone(alert)
        self.assertEqual(alert["anomaly_type"], "HIGH_VIBRATION")
        self.assertEqual(alert["sensor"], "Vibration Sensor")


class TestEnsembleAnomalyPipeline(unittest.TestCase):
    def test_mqtt_telemetry_ensemble_agreement_creates_single_alert(self):
        telemetry_payload = {
            "device_id": "motor_01",
            "motor_id": "motor_01",
            "temperature_c": 52.0,
            "vibration_x": 0.5,
            "vibration_y": 0.4,
            "vibration_z": 9.8,
            "motor_speed": 60.0,
            "current_a": 0.8,
            "sequence": 1,
        }

        stored_records = {}

        def mock_set_record(path, key, value):
            stored_records[f"{path}/{key}"] = value
            return key

        mock_rf_pred = {
            "telemetry_record_id": "tel-1",
            "predicted_class": "HIGH_VIBRATION",
            "confidence": 0.88,
            "model_name": "Random Forest",
            "prediction_timestamp": "2026-11-06T00:00:00Z",
            "sensor_values": {"temperature_c": 52.0},
        }
        mock_xgb_pred = {
            "telemetry_record_id": "tel-1",
            "predicted_class": "HIGH_VIBRATION",
            "confidence": 0.95,
            "model_name": "XGBoost",
            "prediction_timestamp": "2026-11-06T00:00:00Z",
            "sensor_values": {"temperature_c": 52.0},
        }

        with (
            patch.object(FirebaseStore, "initialize", autospec=True, side_effect=lambda s: setattr(s, "_database", object())),
            patch.object(FirebaseStore, "get_value", return_value=None),
            patch.object(FirebaseStore, "get_records", return_value=[]),
            patch.object(FirebaseStore, "push", return_value="tel-1"),
            patch.object(FirebaseStore, "set_record", side_effect=mock_set_record) as fb_set,
            patch.object(FirebaseStore, "update_value", return_value=True),
            patch.object(RandomForestPredictor, "is_ready", new_callable=PropertyMock, return_value=True),
            patch.object(RandomForestPredictor, "predict", return_value=mock_rf_pred),
            patch.object(XGBoostPredictor, "is_ready", new_callable=PropertyMock, return_value=True),
            patch.object(XGBoostPredictor, "predict", return_value=mock_xgb_pred),
            patch.object(MQTTHandler, "start", autospec=True, side_effect=lambda h: h._on_telemetry(telemetry_payload)),
            patch.object(MQTTHandler, "connected", new_callable=PropertyMock, return_value=True),
            patch.object(MQTTHandler, "subscription_status", new_callable=PropertyMock, return_value="subscribed"),
            patch.object(MQTTHandler, "stop"),
            TestClient(create_app()) as client,
        ):
            live = client.get("/api/v1/live")
            anomalies = live.json().get("anomalies", [])

        self.assertIn("anomaly_alerts/motor_01/tel-1", stored_records)
        alert_record = stored_records["anomaly_alerts/motor_01/tel-1"]
        self.assertEqual(alert_record["selected_model_name"], "XGBoost")
        self.assertEqual(alert_record["predicted_class"], "HIGH_VIBRATION")
        self.assertEqual(alert_record["confidence_score"], 0.95)
        self.assertEqual(len(anomalies), 1)

    def test_mqtt_telemetry_multi_sensor_anomaly_creates_combined_alert(self):
        # RF detects HIGH_CURRENT, XGBoost detects HIGH_VIBRATION -> Creates a combined alert with both
        telemetry_payload = {
            "device_id": "motor_01",
            "motor_id": "motor_01",
            "temperature_c": 52.0,
            "vibration_x": 0.5,
            "vibration_y": 0.4,
            "vibration_z": 9.8,
            "motor_speed": 60.0,
            "current_a": 0.8,
            "sequence": 1,
        }

        stored_records = {}

        def mock_set_record(path, key, value):
            stored_records[f"{path}/{key}"] = value
            return key

        mock_rf_pred = {
            "telemetry_record_id": "tel-2",
            "predicted_class": "HIGH_CURRENT",
            "confidence": 0.85,
            "model_name": "Random Forest",
            "prediction_timestamp": "2026-11-06T00:00:00Z",
            "sensor_values": {"temperature_c": 52.0},
        }
        mock_xgb_pred = {
            "telemetry_record_id": "tel-2",
            "predicted_class": "HIGH_VIBRATION",
            "confidence": 0.85,
            "model_name": "XGBoost",
            "prediction_timestamp": "2026-11-06T00:00:00Z",
            "sensor_values": {"temperature_c": 52.0},
        }

        with (
            patch.object(FirebaseStore, "initialize", autospec=True, side_effect=lambda s: setattr(s, "_database", object())),
            patch.object(FirebaseStore, "get_value", return_value=None),
            patch.object(FirebaseStore, "get_records", return_value=[]),
            patch.object(FirebaseStore, "push", return_value="tel-2"),
            patch.object(FirebaseStore, "set_record", side_effect=mock_set_record),
            patch.object(FirebaseStore, "update_value", return_value=True),
            patch.object(RandomForestPredictor, "is_ready", new_callable=PropertyMock, return_value=True),
            patch.object(RandomForestPredictor, "predict", return_value=mock_rf_pred),
            patch.object(XGBoostPredictor, "is_ready", new_callable=PropertyMock, return_value=True),
            patch.object(XGBoostPredictor, "predict", return_value=mock_xgb_pred),
            patch.object(MQTTHandler, "start", autospec=True, side_effect=lambda h: h._on_telemetry(telemetry_payload)),
            patch.object(MQTTHandler, "connected", new_callable=PropertyMock, return_value=True),
            patch.object(MQTTHandler, "subscription_status", new_callable=PropertyMock, return_value="subscribed"),
            patch.object(MQTTHandler, "stop"),
            TestClient(create_app()) as client,
        ):
            live = client.get("/api/v1/live")
            anomalies = live.json().get("anomalies", [])

        self.assertIn("anomaly_alerts/motor_01/tel-2", stored_records)
        alert_record = stored_records["anomaly_alerts/motor_01/tel-2"]
        self.assertIn("HIGH_CURRENT", alert_record["anomaly_type"])
        self.assertIn("HIGH_VIBRATION", alert_record["anomaly_type"])
        self.assertIn("Current Sensor", alert_record["sensor"])
        self.assertIn("Vibration Sensor", alert_record["sensor"])
        self.assertTrue(alert_record["multi_sensor"])
        self.assertEqual(len(anomalies), 1)


if __name__ == "__main__":
    unittest.main()
