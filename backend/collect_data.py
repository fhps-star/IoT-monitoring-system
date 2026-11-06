"""Collect validated, synchronized MQTT readings into a normal-operation CSV."""

import argparse
import csv
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any

import paho.mqtt.client as mqtt

from backend.settings import BACKEND_DIR, Settings
from backend.utils.logger import configure_logging

logger = logging.getLogger(__name__)
CSV_FIELDS = (
    "timestamp",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "temperature_c",
    "current_a",
    "motor_speed",
    "label",
)


def _finite_number(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or value is None:
        raise ValueError(f"missing or invalid '{field}'")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"missing or invalid '{field}'") from exc
    if not math.isfinite(number):
        raise ValueError(f"missing or invalid '{field}'")
    return number


def sample_row(
    payload: dict[str, Any],
    *,
    device_id: str,
    temperature_field: str,
    current_field: str,
    label: str,
    timestamp: str,
) -> dict[str, str | float]:
    if payload.get("device_id") != device_id:
        raise ValueError(f"device_id does not match configured device '{device_id}'")
    return {
        "timestamp": timestamp,
        "vibration_x": _finite_number(payload, "acceleration_x"),
        "vibration_y": _finite_number(payload, "acceleration_y"),
        "vibration_z": _finite_number(payload, "acceleration_z"),
        "temperature_c": _finite_number(payload, temperature_field),
        "current_a": _finite_number(payload, current_field),
        "motor_speed": _finite_number(payload, "motor_speed"),
        "label": label,
    }


def _prepare_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(CSV_FIELDS)
        return
    with path.open("r", newline="", encoding="utf-8") as file:
        existing_fields = next(csv.reader(file), [])
    if tuple(existing_fields) != CSV_FIELDS:
        raise ValueError(f"CSV at {path} has a different header; choose another file")


def _default_output_path(source: str, label: str) -> Path:
    filenames = {
        ("physical", "NORMAL"): "normal_motor_telemetry.csv",
        ("physical", "HIGH_VIBRATION"): "high_vibration_motor_telemetry.csv",
        ("physical", "HIGH_TEMPERATURE"): "high_temperature_motor_telemetry.csv",
        ("physical", "HIGH_CURRENT"): "high_current_motor_telemetry.csv",
        ("wokwi", "NORMAL"): "wokwi_normal_motor_telemetry.csv",
        ("wokwi", "HIGH_VIBRATION"): "wokwi_high_vibration_telemetry.csv",
        ("wokwi", "HIGH_TEMPERATURE"): "wokwi_high_temperature_telemetry.csv",
        ("wokwi", "HIGH_CURRENT"): "wokwi_high_current_telemetry.csv",
    }
    return BACKEND_DIR.parent / "data" / filenames[(source, label)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect real MQTT sensor messages into a CSV file."
    )
    parser.add_argument(
        "--duration-minutes",
        type=int,
        default=30,
        help="Collection duration from 30 to 60 minutes (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path (defaults to a source-specific file in data/)",
    )
    parser.add_argument(
        "--source",
        choices=("physical", "wokwi"),
        default="physical",
        help="Data source, used to keep physical and simulated CSV files separate",
    )
    parser.add_argument(
        "--normal-confirmed",
        action="store_true",
        help="Confirm the motor/simulation is running in its normal operating mode",
    )
    parser.add_argument(
        "--label",
        choices=("NORMAL", "HIGH_VIBRATION", "HIGH_TEMPERATURE", "HIGH_CURRENT"),
        default="NORMAL",
        help="Label for collected readings (default: NORMAL)",
    )
    args = parser.parse_args()
    if not 30 <= args.duration_minutes <= 60:
        parser.error("--duration-minutes must be between 30 and 60")
    if args.label == "NORMAL" and not args.normal_confirmed:
        parser.error(
            "verify the motor/simulation is operating normally, then pass "
            "--normal-confirmed to label rows NORMAL"
        )

    configure_logging()
    settings = Settings.from_env()
    output_path = args.output or _default_output_path(args.source, args.label)
    try:
        _prepare_csv(output_path)
    except (OSError, ValueError):
        logger.exception("Could not prepare CSV output", extra={"path": str(output_path)})
        raise

    file_lock = threading.Lock()
    fatal_error = threading.Event()
    fatal_reason: list[str] = []
    counts = {"valid": 0, "invalid": 0}
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"iiot-csv-{int(time.time())}",
        protocol=mqtt.MQTTv311,
    )
    client.reconnect_delay_set(min_delay=1, max_delay=60)

    def abort_collection(reason: str) -> None:
        fatal_reason.append(reason)
        fatal_error.set()
        logger.error("CSV collection cannot continue", extra={"reason": reason})

    def on_connect(
        connected_client: mqtt.Client,
        _userdata: Any,
        _flags: mqtt.ConnectFlags,
        reason_code: mqtt.ReasonCode,
        _properties: mqtt.Properties | None,
    ) -> None:
        if reason_code.is_failure:
            logger.error("MQTT connection rejected", extra={"reason": str(reason_code)})
            return
        result, message_id = connected_client.subscribe(settings.mqtt_topic, qos=0)
        if result != mqtt.MQTTErrorCode.MQTT_ERR_SUCCESS:
            abort_collection(
                f"MQTT subscription request failed ({int(result)})"
            )

    def on_subscribe(
        _client: mqtt.Client,
        _userdata: Any,
        _message_id: int,
        reason_codes: list[mqtt.ReasonCode],
        _properties: mqtt.Properties | None,
    ) -> None:
        failures = [str(reason) for reason in reason_codes if reason.is_failure]
        if failures:
            abort_collection(f"MQTT subscription rejected: {', '.join(failures)}")
            return
        logger.info(
            "CSV collector subscription acknowledged",
            extra={"topic": settings.mqtt_topic},
        )

    def on_message(
        _client: mqtt.Client,
        _userdata: Any,
        message: mqtt.MQTTMessage,
    ) -> None:
        if message.topic != settings.mqtt_topic:
            return
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON payload must be an object")
            row = sample_row(
                payload,
                device_id=settings.device_id,
                temperature_field=settings.temperature_field or "temperature_c",
                current_field=settings.current_field or "motor_current_a",
                label=args.label,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            counts["invalid"] += 1
            logger.warning(
                "Skipping invalid MQTT sample",
                extra={"topic": message.topic, "reason": str(exc)},
            )
            return

        try:
            with file_lock, output_path.open(
                "a", newline="", encoding="utf-8"
            ) as file:
                csv.DictWriter(file, fieldnames=CSV_FIELDS).writerow(row)
                file.flush()
        except OSError:
            abort_collection("CSV write failed")
            logger.exception("Could not append sample to CSV", extra={"path": str(output_path)})
            return
        counts["valid"] += 1
        if counts["valid"] == 1 or counts["valid"] % 30 == 0:
            logger.info(
                "CSV collection progress",
                extra={
                    "valid_samples": counts["valid"],
                    "invalid_samples": counts["invalid"],
                    "output": str(output_path),
                },
            )

    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    deadline = time.monotonic() + args.duration_minutes * 60
    logger.info(
        "Starting telemetry data collection",
        extra={
            "duration_minutes": args.duration_minutes,
            "output": str(output_path),
            "source": args.source,
            "topic": settings.mqtt_topic,
            "device_id": settings.device_id,
            "label": args.label,
        },
    )
    try:
        client.connect_async(
            settings.mqtt_broker,
            settings.mqtt_port,
            keepalive=60,
        )
        client.loop_start()
        while time.monotonic() < deadline and not fatal_error.wait(1):
            pass
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
    except (OSError, ValueError, RuntimeError):
        logger.exception("MQTT collector failed")
        raise
    finally:
        client.disconnect()
        client.loop_stop()
    logger.info(
        "Data collection finished",
        extra={
            "valid_samples": counts["valid"],
            "invalid_samples": counts["invalid"],
            "output": str(output_path),
        },
    )
    if fatal_error.is_set():
        raise RuntimeError(f"{fatal_reason[0]}; see collector logs")


if __name__ == "__main__":
    main()
