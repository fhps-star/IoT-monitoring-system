import csv
import json
from pathlib import Path
import tempfile
import unittest

from backend.dataset import engineer_dataset, inspect_and_clean_csv
from backend.feature_engineering import FEATURE_NAMES


class DatasetTests(unittest.TestCase):
    def test_audits_tab_delimited_input_and_preserves_outliers_in_clean_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "motor.tsv"
            original = (
                "timestamp\tvibration_x\tvibration_y\tvibration_z\t"
                "temperature_c\tcurrent_a\tmotor_speed\tlabel\n"
                "2026-01-01T00:00:00+00:00\t0\t0\t9.81\t20\t1\t30\tNORMAL\n"
                "2026-01-01T00:00:01+00:00\t0\t0\t9.81\t20\t1\t30\tNORMAL\n"
                "2026-01-01T00:00:02+00:00\t0\t0\t9.81\t20\t1\t30\tNORMAL\n"
                "2026-01-01T00:00:03+00:00\t0\t0\t9.81\t20\t1\t30\tNORMAL\n"
                "2026-01-01T00:00:04+00:00\t100\t0\t9.81\t20\t1\t30\tNORMAL\n"
                "2026-01-01T00:00:04+00:00\t100\t0\t9.81\t20\t1\t30\tNORMAL\n"
                "invalid\t0\t0\t9.81\t20\t1\t30\tNORMAL\n"
            )
            source.write_text(original, encoding="utf-8")
            source_hash = source.read_bytes()
            cleaned = root / "cleaned.csv"
            report_path = root / "report.json"

            audit = inspect_and_clean_csv(source, cleaned, report_path)

            self.assertEqual(source.read_bytes(), source_hash)
            self.assertEqual(audit.report["valid_unique_rows"], 5)
            self.assertEqual(audit.report["invalid_rows_removed"], 1)
            self.assertEqual(audit.report["duplicate_records_removed"], 1)
            self.assertEqual(audit.report["outlier_row_count"], 1)
            with cleaned.open(newline="", encoding="utf-8") as file:
                cleaned_rows = list(csv.DictReader(file))
            self.assertEqual(len(cleaned_rows), 5)
            self.assertEqual(cleaned_rows[-1]["vibration_x"], "100.0")
            self.assertEqual(json.loads(report_path.read_text())["invalid_rows_removed"], 1)

    def test_engineers_sorted_rows_without_future_samples(self) -> None:
        rows = [
            {
                "timestamp": f"2026-01-01T00:00:0{second}+00:00",
                "vibration_x": float(second),
                "vibration_y": 0.0,
                "vibration_z": 9.81,
                "temperature_c": 20.0 + second,
                "current_a": 1.0 + second,
                "motor_speed": 30.0,
                "label": "NORMAL",
            }
            for second in range(5)
        ]

        transformed, report = engineer_dataset(rows)

        self.assertEqual(len(transformed), 3)
        self.assertEqual(report["rows_without_causal_window_features"], 2)
        self.assertEqual(len(transformed[0]["features"]), len(FEATURE_NAMES))
        self.assertEqual(transformed[0]["timestamp"], rows[2]["timestamp"])
        self.assertAlmostEqual(
            transformed[0]["features"][FEATURE_NAMES.index("temperature_rate")],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
