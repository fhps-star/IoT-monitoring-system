import unittest
from unittest.mock import Mock

import numpy as np

from backend.xgboost_predictor import XGBoostPredictor


class XGBoostPredictorTests(unittest.TestCase):
    def make_predictor(self, probabilities: list[float]) -> XGBoostPredictor:
        predictor = XGBoostPredictor()
        predictor._feature_columns = (
            "temperature_c",
            "acceleration_x",
            "acceleration_y",
            "acceleration_z",
            "acceleration_magnitude",
            "motor_speed",
            "motor_current_a",
        )
        predictor._class_labels = (
            "HIGH_CURRENT",
            "HIGH_TEMPERATURE",
            "HIGH_VIBRATION",
            "NORMAL",
        )
        predictor._preprocessor = Mock()
        predictor._preprocessor.transform.side_effect = lambda value: value
        predictor._model = Mock()
        predictor._model.predict_proba.return_value = np.asarray([probabilities])
        return predictor

    def make_telemetry(self) -> dict:
        return {
            "id": "firebase-telemetry-1",
            "device_id": "motor_01",
            "received_at": "2026-11-03T12:00:00+00:00",
            "model_sample": {
                "temperature_c": 81.2,
                "vibration_x": 1.2,
                "vibration_y": 1.1,
                "vibration_z": 9.5,
                "motor_speed": 80.0,
                "current_a": 1.4,
            },
            "raw_payload": {"sequence": 45, "simulation_mode": "fault"},
        }

    def test_builds_exact_ordered_features_and_maps_probability_class(self) -> None:
        predictor = self.make_predictor([0.01, 0.02, 0.94, 0.03])

        prediction = predictor.predict(self.make_telemetry())

        self.assertEqual(prediction["predicted_class"], "HIGH_VIBRATION")
        self.assertEqual(prediction["confidence"], 0.94)
        self.assertEqual(prediction["model_name"], "XGBoost")
        self.assertFalse(prediction["confidence_calibrated"])
        self.assertEqual(prediction["telemetry_sequence"], 45)
        supplied = predictor._preprocessor.transform.call_args.args[0]
        self.assertEqual(supplied.shape, (1, 7))
        self.assertAlmostEqual(supplied[0, 4], (1.2**2 + 1.1**2 + 9.5**2) ** 0.5)

    def test_missing_required_feature_skips_prediction(self) -> None:
        predictor = self.make_predictor([0.01, 0.02, 0.03, 0.94])
        telemetry = self.make_telemetry()
        telemetry["model_sample"].pop("current_a")

        self.assertIsNone(predictor.predict(telemetry))
        predictor._model.predict_proba.assert_not_called()

    def test_invalid_probability_distribution_is_rejected(self) -> None:
        predictor = self.make_predictor([0.1, 0.2, 0.3, 0.2])

        with self.assertRaisesRegex(ValueError, "invalid class probabilities"):
            predictor.predict(self.make_telemetry())


if __name__ == "__main__":
    unittest.main()