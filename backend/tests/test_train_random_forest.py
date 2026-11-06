import unittest
from datetime import datetime, timezone

from backend.train_random_forest import _source_field, chronological_split


class RandomForestTrainingTests(unittest.TestCase):
    def test_maps_existing_csv_aliases_and_derives_acceleration_magnitude(self) -> None:
        headers = {"vibration_x", "vibration_y", "vibration_z", "current_a"}

        self.assertEqual(_source_field("acceleration_x", headers), "vibration_x")
        self.assertEqual(_source_field("motor_current_a", headers), "current_a")
        self.assertEqual(
            _source_field("acceleration_magnitude", headers),
            "__derived_from_acceleration_axes__",
        )
        self.assertIsNone(_source_field("gyro_x", headers))
        self.assertIsNone(_source_field("estimated_vibration", headers))

    def test_splits_each_class_chronologically_with_an_embargo(self) -> None:
        labels = ("NORMAL", "HIGH_VIBRATION", "HIGH_TEMPERATURE", "HIGH_CURRENT")
        records = [
            {
                "label": label,
                "timestamp": datetime.fromtimestamp(second, timezone.utc),
                "source_row": second + 2,
            }
            for label in labels
            for second in range(10)
        ]

        training, testing, counts = chronological_split(
            records,
            test_fraction=0.2,
            embargo_seconds=2,
        )

        self.assertEqual(len(training), 28)
        self.assertEqual(len(testing), 8)
        self.assertEqual(counts["NORMAL"], {"training": 7, "testing": 2, "embargoed": 1})
        for label in labels:
            train_end = max(row["timestamp"] for row in training if row["label"] == label)
            test_start = min(row["timestamp"] for row in testing if row["label"] == label)
            self.assertGreaterEqual((test_start - train_end).total_seconds(), 2)


if __name__ == "__main__":
    unittest.main()