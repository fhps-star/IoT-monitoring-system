import unittest

from backend.collect_data import (
    CSV_FIELDS,
    _default_output_path,
    sample_row,
)


class CollectDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = {
            "device_id": "motor_01",
            "acceleration_x": 0.44,
            "acceleration_y": 0.002,
            "acceleration_z": 9.499,
            "temperature_c": 53.27,
            "motor_current_a": 1.337,
            "motor_speed": 60.0,
        }

    def test_maps_one_synchronized_mqtt_payload_to_csv_schema(self) -> None:
        row = sample_row(
            self.payload,
            device_id="motor_01",
            temperature_field="temperature_c",
            current_field="motor_current_a",
            label="NORMAL",
            timestamp="2026-11-02T13:00:00+00:00",
        )

        self.assertEqual(tuple(row), CSV_FIELDS)
        self.assertEqual(row["timestamp"], "2026-11-02T13:00:00+00:00")
        self.assertEqual(
            [row[field] for field in CSV_FIELDS[1:7]],
            [0.44, 0.002, 9.499, 53.27, 1.337, 60.0],
        )
        self.assertEqual(row["label"], "NORMAL")

    def test_high_vibration_uses_a_separate_wokwi_csv(self) -> None:
        anomaly_path = _default_output_path("wokwi", "HIGH_VIBRATION")
        normal_path = _default_output_path("wokwi", "NORMAL")

        self.assertEqual(anomaly_path.name, "wokwi_high_vibration_telemetry.csv")
        self.assertNotEqual(anomaly_path, normal_path)

    def test_maps_high_vibration_label(self) -> None:
        row = sample_row(
            self.payload,
            device_id="motor_01",
            temperature_field="temperature_c",
            current_field="motor_current_a",
            label="HIGH_VIBRATION",
            timestamp="2026-11-02T13:00:00+00:00",
        )

        self.assertEqual(row["label"], "HIGH_VIBRATION")

    def test_high_temperature_uses_a_separate_wokwi_csv_and_label(self) -> None:
        output_path = _default_output_path("wokwi", "HIGH_TEMPERATURE")
        row = sample_row(
            self.payload,
            device_id="motor_01",
            temperature_field="temperature_c",
            current_field="motor_current_a",
            label="HIGH_TEMPERATURE",
            timestamp="2026-11-02T13:00:00+00:00",
        )

        self.assertEqual(output_path.name, "wokwi_high_temperature_telemetry.csv")
        self.assertEqual(row["label"], "HIGH_TEMPERATURE")

    def test_high_current_uses_a_separate_wokwi_csv_and_label(self) -> None:
        output_path = _default_output_path("wokwi", "HIGH_CURRENT")
        row = sample_row(
            self.payload,
            device_id="motor_01",
            temperature_field="temperature_c",
            current_field="motor_current_a",
            label="HIGH_CURRENT",
            timestamp="2026-11-02T13:00:00+00:00",
        )

        self.assertEqual(output_path.name, "wokwi_high_current_telemetry.csv")
        self.assertEqual(row["current_a"], 1.337)
        self.assertEqual(row["label"], "HIGH_CURRENT")

    def test_rejects_missing_or_non_finite_sensor_readings(self) -> None:
        for value in (None, True, float("nan"), float("inf"), "not-a-number"):
            with self.subTest(value=value):
                payload = {**self.payload, "acceleration_y": value}
                with self.assertRaisesRegex(ValueError, "acceleration_y"):
                    sample_row(
                        payload,
                        device_id="motor_01",
                        temperature_field="temperature_c",
                        current_field="motor_current_a",
                        label="NORMAL",
                        timestamp="2026-11-02T13:00:00+00:00",
                    )

    def test_rejects_payload_from_another_device(self) -> None:
        with self.assertRaisesRegex(ValueError, "device_id"):
            sample_row(
                self.payload,
                device_id="different_motor",
                temperature_field="temperature_c",
                current_field="motor_current_a",
                label="NORMAL",
                timestamp="2026-11-02T13:00:00+00:00",
            )


if __name__ == "__main__":
    unittest.main()
