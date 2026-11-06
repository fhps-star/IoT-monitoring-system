import unittest
from datetime import datetime, timezone

from backend.train_xgboost import CLASS_LABELS, _reconstruct_rf_split


class XGBoostTrainingTests(unittest.TestCase):
    def test_reconstructs_saved_per_class_time_split_with_embargo(self) -> None:
        records = [
            {
                "label": label,
                "timestamp": datetime.fromtimestamp(second, timezone.utc),
                "source_file": f"{label}.csv",
                "source_row": second + 2,
            }
            for label in CLASS_LABELS
            for second in range(12)
        ]
        metrics = {
            "split": {
                "method": "Per-class chronological holdout: earliest rows train, latest 25% test.",
                "test_fraction_per_class": 0.25,
                "embargo_seconds": 2.0,
            },
            "training_records_per_class": {label: 8 for label in CLASS_LABELS},
            "testing_records_per_class": {label: 3 for label in CLASS_LABELS},
            "embargoed_records_per_class": {label: 1 for label in CLASS_LABELS},
        }

        training, testing, split = _reconstruct_rf_split(records, metrics)

        self.assertEqual(len(training), 32)
        self.assertEqual(len(testing), 12)
        self.assertEqual(split["per_class_counts"]["HIGH_CURRENT"]["embargoed"], 1)
        self.assertEqual(
            len(split["indices"]["HIGH_CURRENT"]["testing"]),
            3,
        )

    def test_refuses_a_split_that_does_not_match_saved_rf_counts(self) -> None:
        records = [
            {
                "label": label,
                "timestamp": datetime.fromtimestamp(second, timezone.utc),
                "source_file": f"{label}.csv",
                "source_row": second + 2,
            }
            for label in CLASS_LABELS
            for second in range(12)
        ]
        metrics = {
            "split": {
                "method": "Per-class chronological holdout",
                "test_fraction_per_class": 0.25,
                "embargo_seconds": 2.0,
            },
            "training_records_per_class": {label: 7 for label in CLASS_LABELS},
            "testing_records_per_class": {label: 3 for label in CLASS_LABELS},
            "embargoed_records_per_class": {label: 1 for label in CLASS_LABELS},
        }

        with self.assertRaisesRegex(ValueError, "count differs"):
            _reconstruct_rf_split(records, metrics)


if __name__ == "__main__":
    unittest.main()