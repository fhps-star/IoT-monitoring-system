import unittest

from backend.feature_engineering import (
    FEATURE_NAMES,
    FeatureEngineer,
    parse_timestamp,
)


class FeatureEngineeringTests(unittest.TestCase):
    @staticmethod
    def _sample(
        *,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 9.81,
        temperature: float = 20.0,
        current: float = 1.0,
        speed: float = 30.0,
    ) -> dict[str, float]:
        return {
            "vibration_x": x,
            "vibration_y": y,
            "vibration_z": z,
            "temperature_c": temperature,
            "current_a": current,
            "motor_speed": speed,
        }

    def test_extracts_time_window_features_using_only_current_and_past_samples(self) -> None:
        engineer = FeatureEngineer(window_seconds=10, max_gap_seconds=2.5)
        first = engineer.process(
            self._sample(temperature=20, current=1),
            "2026-01-01T00:00:00+00:00",
        )
        second = engineer.process(
            self._sample(x=1, temperature=22, current=1.5),
            "2026-01-01T00:00:01+00:00",
        )
        third = engineer.process(
            self._sample(x=0, temperature=26, current=2),
            "2026-01-01T00:00:02+00:00",
        )

        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertIsNotNone(third)
        assert third is not None
        self.assertEqual(tuple(third), FEATURE_NAMES)
        self.assertAlmostEqual(third["acceleration_magnitude"], 9.81)
        self.assertAlmostEqual(third["temperature_rate"], 4.0)
        self.assertAlmostEqual(third["current_rate"], 0.5)
        self.assertGreater(third["vibration_rms"], 0)

    def test_resets_time_window_after_large_gaps_or_invalid_samples(self) -> None:
        engineer = FeatureEngineer()
        for second in range(3):
            engineer.process(self._sample(), f"2026-01-01T00:00:0{second}+00:00")
        self.assertIsNone(
            engineer.process(self._sample(), "2026-01-01T00:00:10+00:00")
        )
        with self.assertRaisesRegex(ValueError, "vibration_x"):
            engineer.process(
                self._sample(x=float("nan")),
                "2026-01-01T00:00:11+00:00",
            )

    def test_parses_iso_and_wokwi_timestamp_formats_with_timezones(self) -> None:
        self.assertEqual(
            parse_timestamp(
                "Mon Nov 02 2026 21:00:19 GMT+0530 (India Standard Time)"
            ).utcoffset().total_seconds(),
            19800,
        )
        with self.assertRaisesRegex(ValueError, "must include a timezone"):
            parse_timestamp("2026-11-02T21:00:19")


if __name__ == "__main__":
    unittest.main()
