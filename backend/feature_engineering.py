"""Causal time-window feature extraction shared by training and inference."""

from collections import deque
from datetime import datetime
import math
from statistics import pstdev
from typing import Any

FEATURE_NAMES = (
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "temperature_c",
    "current_a",
    "motor_speed",
    "acceleration_magnitude",
    "vibration_rms",
    "vibration_peak",
    "vibration_std",
    "temperature_rate",
    "current_rate",
)
RAW_FEATURE_NAMES = FEATURE_NAMES[:6]
DERIVED_FEATURE_NAMES = FEATURE_NAMES[6:]
DEFAULT_WINDOW_SECONDS = 10.0
DEFAULT_MAX_GAP_SECONDS = 2.5
DEFAULT_MIN_WINDOW_SAMPLES = 3


def parse_timestamp(value: str) -> datetime:
    text = value.strip()
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        without_timezone_name = text.rsplit(" (", 1)[0].removesuffix(")")
        try:
            timestamp = datetime.strptime(
                without_timezone_name,
                "%a %b %d %Y %H:%M:%S GMT%z",
            )
        except ValueError as exc:
            raise ValueError(f"invalid timestamp '{value}'") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"timestamp must include a timezone: '{value}'")
    return timestamp


def _finite_number(sample: dict[str, Any], name: str) -> float:
    value = sample.get(name)
    if isinstance(value, bool) or value is None:
        raise ValueError(f"missing or invalid '{name}'")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"missing or invalid '{name}'") from exc
    if not math.isfinite(number):
        raise ValueError(f"missing or invalid '{name}'")
    return number


class FeatureEngineer:
    def __init__(
        self,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
        min_window_samples: int = DEFAULT_MIN_WINDOW_SAMPLES,
    ) -> None:
        if window_seconds <= 0 or max_gap_seconds <= 0 or min_window_samples < 2:
            raise ValueError("Feature window settings must be positive")
        self.window_seconds = window_seconds
        self.max_gap_seconds = max_gap_seconds
        self.min_window_samples = min_window_samples
        self._window: deque[tuple[float, tuple[float, float, float], float, float]] = (
            deque()
        )
        self._previous: tuple[float, float, float] | None = None

    @property
    def preprocessing_config(self) -> dict[str, float | int | str]:
        return {
            "window_seconds": self.window_seconds,
            "max_gap_seconds": self.max_gap_seconds,
            "min_window_samples": self.min_window_samples,
            "vibration_definition": (
                "detrended acceleration vector within the trailing time window; "
                "no frequency-domain features"
            ),
            "acceleration_magnitude": "sqrt(x^2 + y^2 + z^2) per sample",
            "vibration_rms": (
                "RMS of the per-sample 3-axis vector after subtracting each "
                "axis's trailing-window mean"
            ),
            "vibration_peak": (
                "maximum norm of the same detrended 3-axis vectors in the window"
            ),
            "vibration_std": (
                "population standard deviation of acceleration magnitude in the window"
            ),
            "rates": "backward difference divided by elapsed seconds",
        }

    def reset(self) -> None:
        self._window.clear()
        self._previous = None

    def process(
        self,
        sample: dict[str, Any],
        timestamp: str | datetime,
    ) -> dict[str, float] | None:
        when = parse_timestamp(timestamp) if isinstance(timestamp, str) else timestamp
        if when.tzinfo is None or when.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        epoch = when.timestamp()
        values = {name: _finite_number(sample, name) for name in RAW_FEATURE_NAMES}
        axes = (values["vibration_x"], values["vibration_y"], values["vibration_z"])
        acceleration_magnitude = math.sqrt(sum(value * value for value in axes))

        if self._window:
            elapsed = epoch - self._window[-1][0]
            if elapsed <= 0:
                self.reset()
            elif elapsed > self.max_gap_seconds:
                self.reset()

        previous = self._previous
        self._window.append((epoch, axes, values["temperature_c"], values["current_a"]))
        while self._window and epoch - self._window[0][0] > self.window_seconds:
            self._window.popleft()

        self._previous = (
            epoch,
            values["temperature_c"],
            values["current_a"],
        )
        if len(self._window) < self.min_window_samples or previous is None:
            return None

        elapsed = epoch - previous[0]
        if elapsed <= 0 or elapsed > self.max_gap_seconds:
            return None

        axis_means = tuple(
            sum(point[1][axis] for point in self._window) / len(self._window)
            for axis in range(3)
        )
        deviations = [
            tuple(point[1][axis] - axis_means[axis] for axis in range(3))
            for point in self._window
        ]
        deviation_magnitudes = [
            math.sqrt(sum(value * value for value in deviation))
            for deviation in deviations
        ]
        magnitudes = [
            math.sqrt(sum(value * value for value in point[1]))
            for point in self._window
        ]
        vibration_rms = math.sqrt(
            sum(value * value for value in deviation_magnitudes)
            / len(deviation_magnitudes)
        )
        vibration_peak = max(deviation_magnitudes)
        vibration_std = pstdev(magnitudes)

        features = {
            **values,
            "acceleration_magnitude": acceleration_magnitude,
            "vibration_rms": vibration_rms,
            "vibration_peak": vibration_peak,
            "vibration_std": vibration_std,
            "temperature_rate": (values["temperature_c"] - previous[1]) / elapsed,
            "current_rate": (values["current_a"] - previous[2]) / elapsed,
        }
        return features
