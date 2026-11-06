"""Validate MQTT telemetry and extract configured sensor features."""

from datetime import datetime, timezone
import math
from typing import Any

from backend.settings import Settings

FEATURE_FIELDS = ("temperature", "vibration", "current")
MODEL_SAMPLE_FIELDS = (
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "temperature_c",
    "current_a",
    "motor_speed",
)


def _read_path(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


class DataProcessor:
    def __init__(self, settings: Settings) -> None:
        self._field_paths = {
            "temperature": settings.temperature_field,
            "vibration": settings.vibration_field,
            "current": settings.current_field,
        }
        self._device_id = settings.device_id

    def process(
        self, payload: dict[str, Any], received_at: str | None = None
    ) -> dict[str, Any]:
        values: dict[str, float | None] = {}
        errors: dict[str, str] = {}
        for feature, path in self._field_paths.items():
            if path is None:
                errors[feature] = "field path is not configured"
                values[feature] = None
                continue
            value = _number(_read_path(payload, path))
            if value is None:
                fallbacks = {
                    "temperature": ("temperature_c", "temperature", "temp_c"),
                    "vibration": ("estimated_vibration", "vibration_rms", "vibration"),
                    "current": ("motor_current_a", "current_a", "current"),
                }.get(feature, ())
                for fb in fallbacks:
                    val_fb = _number(_read_path(payload, fb))
                    if val_fb is not None:
                        value = val_fb
                        break
            if value is None:
                errors[feature] = f"missing or invalid value at '{path}'"
            values[feature] = value

        features = (
            [values[name] for name in FEATURE_FIELDS]
            if all(values[name] is not None for name in FEATURE_FIELDS)
            else None
        )
        temp_val = _number(
            _read_path(payload, self._field_paths["temperature"])
            if self._field_paths["temperature"] is not None
            else None
        )
        if temp_val is None:
            temp_val = values.get("temperature")

        curr_val = _number(
            _read_path(payload, self._field_paths["current"])
            if self._field_paths["current"] is not None
            else None
        )
        if curr_val is None:
            curr_val = values.get("current")

        speed_val = _number(payload.get("motor_speed"))
        if speed_val is None:
            speed_val = _number(
                payload.get("rpm")
                or payload.get("motor_speed_rpm")
                or payload.get("target_rpm")
            )

        model_sample_values = {
            "vibration_x": _number(payload.get("acceleration_x")),
            "vibration_y": _number(payload.get("acceleration_y")),
            "vibration_z": _number(payload.get("acceleration_z")),
            "temperature_c": temp_val,
            "current_a": curr_val,
            "motor_speed": speed_val,
        }
        model_sample_errors = [
            f"missing or invalid '{name}'"
            for name in MODEL_SAMPLE_FIELDS
            if model_sample_values[name] is None
        ]
        model_sample = (
            {
                name: model_sample_values[name]
                for name in MODEL_SAMPLE_FIELDS
            }
            if not model_sample_errors
            else None
        )
        device_id = payload.get("device_id")
        motor_id = payload.get("motor_id")
        return {
            "device_id": (
                device_id
                if isinstance(device_id, str) and device_id.strip()
                else self._device_id
            ),
            "motor_id": (
                motor_id
                if isinstance(motor_id, str) and motor_id.strip()
                else self._device_id
            ),
            "received_at": received_at
            or datetime.now(timezone.utc).isoformat(),
            "raw_payload": payload,
            "sensor_values": values,
            "features": features,
            "validation_errors": errors,
            "model_sample": model_sample,
            "model_sample_errors": model_sample_errors,
        }
