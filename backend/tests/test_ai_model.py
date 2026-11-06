import tempfile
from pathlib import Path
import unittest

from backend.ai_model import IsolationForestModel


class IsolationForestModelTests(unittest.TestCase):
    def test_saves_and_reloads_model_with_feature_order_and_metadata(self) -> None:
        rows = [
            [float(index), float(index + 1), float(index + 2)]
            for index in range(10)
        ]
        feature_names = ("temperature_c", "current_a", "motor_speed")
        metadata = {"training_dataset": {"path": "normal.csv"}}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            trained = IsolationForestModel(path)
            trained.fit(
                rows,
                3,
                feature_names=feature_names,
                metadata=metadata,
                score_threshold=2.0,
            )

            loaded = IsolationForestModel(path)
            self.assertTrue(loaded.load())
            self.assertEqual(loaded.feature_names, feature_names)
            self.assertEqual(
                loaded.artifact_metadata["training_dataset"],
                metadata["training_dataset"],
            )
            self.assertEqual(loaded.artifact_metadata["anomaly_score_threshold"], 2.0)
            self.assertEqual(loaded.predict(rows[-1])[0], "NORMAL")

    def test_rejects_prediction_with_wrong_feature_width(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = IsolationForestModel(Path(directory) / "model.joblib")
            model.fit([[float(i), float(i + 1)] for i in range(5)], 3)

            with self.assertRaisesRegex(ValueError, "expects 2 features"):
                model.predict([1.0])


if __name__ == "__main__":
    unittest.main()
