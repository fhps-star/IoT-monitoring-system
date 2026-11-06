import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

from backend.ai_model import IsolationForestModel
from backend.anomaly_detector import AnomalyDetector
from backend.data_processor import DataProcessor
from backend.feature_engineering import FEATURE_NAMES
from backend.settings import Settings


class DataProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings.from_env()

    def test_unconfigured_fields_do_not_guess_sensor_names(self) -> None:
        settings = replace(
            self.settings,
            temperature_field=None,
            vibration_field=None,
            current_field=None,
        )
        processor = DataProcessor(settings)
        payload = {"temperature": 42, "vibration": 1.2, "current": 0.8}

        record = processor.process(payload)

        self.assertEqual(record["raw_payload"], payload)
        self.assertIsNone(record["features"])
        self.assertEqual(len(record["validation_errors"]), 3)

    def test_configured_nested_fields_are_extracted(self) -> None:
        settings = replace(
            self.settings,
            temperature_field="sensors.temp",
            vibration_field="sensors.vibration",
            current_field="electrical.current",
        )
        processor = DataProcessor(settings)

        record = processor.process(
            {
                "sensors": {"temp": "42.5", "vibration": 1.2},
                "electrical": {"current": 0.8},
            },
            received_at="2026-01-01T00:00:00+00:00",
        )

        self.assertEqual(record["features"], [42.5, 1.2, 0.8])
        self.assertEqual(record["received_at"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(record["validation_errors"], {})

    def test_inspected_esp32_payload_maps_fields_and_identifiers(self) -> None:
        settings = replace(
            self.settings,
            temperature_field="temperature",
            vibration_field="vibration",
            current_field="current",
        )
        payload = {
            "device_id": "motor-node-01",
            "motor_id": "motor_01",
            "temperature": 73.25,
            "acceleration_x": 0.12,
            "acceleration_y": 0.08,
            "acceleration_z": 9.81,
            "vibration": 0.015,
            "current": 0.85,
            "motor_speed": 180,
            "motor_status": "ON",
            "wifi_status": "CONNECTED",
        }

        record = DataProcessor(settings).process(payload)

        self.assertEqual(record["features"], [73.25, 0.015, 0.85])
        self.assertEqual(record["device_id"], "motor-node-01")
        self.assertEqual(record["motor_id"], "motor_01")
        self.assertEqual(record["raw_payload"], payload)

    def test_wokwi_firmware_field_names_are_mapped(self) -> None:
        processor = DataProcessor(self.settings)
        payload = {
            "device_id": "motor_01",
            "sequence": 9,
            "timestamp_ms": 24145,
            "temperature_c": 53.27,
            "acceleration_x": 0.440,
            "acceleration_y": 0.002,
            "acceleration_z": 9.499,
            "gyro_x": 2.729,
            "gyro_y": -2.403,
            "gyro_z": 4.648,
            "acceleration_magnitude": 9.509,
            "estimated_vibration": 0.187,
            "motor_speed": 60.00,
            "motor_steps": 4553,
            "motor_current_a": 1.337,
            "motor_status": "RUNNING",
            "wifi_connected": True,
        }

        record = processor.process(payload)

        self.assertEqual(record["features"], [53.27, 0.187, 1.337])
        self.assertEqual(record["device_id"], "motor_01")
        self.assertEqual(record["motor_id"], self.settings.device_id)
        self.assertEqual(record["raw_payload"], payload)

    def test_invalid_sensor_values_are_not_used_as_features(self) -> None:
        settings = replace(
            self.settings,
            temperature_field="temp",
            vibration_field="vibration",
            current_field="current",
        )
        processor = DataProcessor(settings)

        record = processor.process(
            {"temp": float("nan"), "vibration": True, "current": 0.8}
        )

        self.assertIsNone(record["features"])
        self.assertEqual(record["sensor_values"]["temperature"], None)
        self.assertEqual(record["sensor_values"]["vibration"], None)
        self.assertEqual(set(record["validation_errors"]), {"temperature", "vibration"})

    def test_inference_reports_untrained_model_without_synthetic_data(self) -> None:
        model = IsolationForestModel(Path("/nonexistent/model.joblib"))
        detector = AnomalyDetector(model)

        prediction = detector.predict(
            {
                "device_id": "motor_01",
                "motor_id": "motor_01",
                "received_at": "2026-01-01T00:00:00+00:00",
                "features": [42.5, 1.2, 0.8],
            }
        )

        self.assertIsNone(prediction)

    def test_trained_prediction_links_to_telemetry_record(self) -> None:
        model = Mock(spec=IsolationForestModel)
        model.is_trained = True
        model.feature_count = len(FEATURE_NAMES)
        model.feature_names = FEATURE_NAMES
        model.predict.return_value = ("NORMAL", 0.123)
        detector = AnomalyDetector(model)

        prediction = detector.predict(
            {
                "id": "firebase-telemetry-id",
                "device_id": "motor-node-01",
                "motor_id": "motor_01",
                "received_at": "2026-01-01T00:00:00+00:00",
                "model_features": [1.0] * len(FEATURE_NAMES),
                "model_feature_names": list(FEATURE_NAMES),
            }
        )

        self.assertIsNotNone(prediction)
        self.assertEqual(
            prediction["telemetry_record_id"],
            "firebase-telemetry-id",
        )
        self.assertEqual(prediction["prediction"], "NORMAL")
        self.assertEqual(prediction["timestamp"], "2026-01-01T00:00:00+00:00")

    def test_invalid_features_do_not_generate_prediction(self) -> None:
        model = Mock(spec=IsolationForestModel)
        model.is_trained = True
        prediction = AnomalyDetector(model).predict(
            {"id": "record-id", "features": None}
        )

        self.assertIsNone(prediction)
        model.predict.assert_not_called()

    def test_training_rejects_insufficient_real_records(self) -> None:
        model = IsolationForestModel(Path("/nonexistent/model.joblib"))

        with self.assertRaisesRegex(ValueError, "at least 3 valid real telemetry"):
            model.fit([], minimum_samples=3)

        self.assertFalse(model.is_trained)


if __name__ == "__main__":
    unittest.main()
