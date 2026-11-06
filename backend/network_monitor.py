"""Track the configured device's connectivity and MQTT delivery metrics."""

from datetime import datetime, timezone
import math
from threading import Lock
from typing import Any


NETWORK_FIELDS = (
    "ip_address",
    "wifi_connected",
    "mqtt_connected",
    "mqtt_rtt_ms",
    "rtt_ms",
)
NETWORK_PAYLOAD_FIELDS = frozenset(NETWORK_FIELDS)
RTT_WARNING_THRESHOLD_MS = 500.0
PACKET_LOSS_WARNING_THRESHOLD_PERCENT = 1.0


class NetworkMonitor:
    def __init__(self, device_id: str, stale_timeout_seconds: int) -> None:
        self._device_id = device_id
        self._stale_timeout_seconds = stale_timeout_seconds
        self._lock = Lock()
        self._ip_address: str | None = None
        self._wifi_connected: bool | None = None
        self._mqtt_connected: bool | None = None
        self._latest_mqtt_rtt_ms: float | None = None
        self._rtt_total_ms = 0.0
        self._rtt_samples = 0
        self._last_sequence: int | None = None
        self._packets_received = 0
        self._packets_expected = 0
        self._packets_lost = 0
        self._device_restarts = 0
        self._timestamp_ms: int | float | None = None
        self._last_seen_at: str | None = None

    def record(
        self, payload: dict[str, Any], received_at: str
    ) -> dict[str, Any] | None:
        if not any(field in payload for field in NETWORK_FIELDS):
            return None

        with self._lock:
            if "ip_address" in payload:
                ip_address = payload["ip_address"]
                self._ip_address = (
                    ip_address.strip()
                    if isinstance(ip_address, str) and ip_address.strip()
                    else None
                )

            for field in ("wifi_connected", "mqtt_connected"):
                if field in payload:
                    value = payload[field]
                    connected = value if isinstance(value, bool) else None
                    if field == "wifi_connected":
                        self._wifi_connected = connected
                    else:
                        self._mqtt_connected = connected

            rtt_raw = payload.get("mqtt_rtt_ms") if "mqtt_rtt_ms" in payload else payload.get("rtt_ms")
            if rtt_raw is not None:
                rtt = self._finite_number(rtt_raw)
                self._latest_mqtt_rtt_ms = rtt if rtt is not None and rtt >= 0 else None
                if self._latest_mqtt_rtt_ms is not None:
                    self._rtt_total_ms += self._latest_mqtt_rtt_ms
                    self._rtt_samples += 1

            if "sequence" in payload:
                sequence = payload["sequence"]
                if (
                    isinstance(sequence, int)
                    and not isinstance(sequence, bool)
                    and sequence >= 0
                ):
                    self._record_sequence(sequence)
                else:
                    self._last_sequence = None

            if "timestamp_ms" in payload:
                timestamp_ms = payload["timestamp_ms"]
                if (
                    isinstance(timestamp_ms, int)
                    and not isinstance(timestamp_ms, bool)
                    and timestamp_ms >= 0
                ):
                    self._timestamp_ms = timestamp_ms
                elif (
                    isinstance(timestamp_ms, float)
                    and math.isfinite(timestamp_ms)
                    and timestamp_ms >= 0
                ):
                    self._timestamp_ms = timestamp_ms
                else:
                    self._timestamp_ms = None

            self._last_seen_at = received_at
            return self._snapshot_locked(datetime.now(timezone.utc))

    def restore(self, record: Any) -> None:
        if not isinstance(record, dict):
            return

        with self._lock:
            ip_address = record.get("ip_address")
            self._ip_address = (
                ip_address.strip()
                if isinstance(ip_address, str) and ip_address.strip()
                else None
            )
            self._wifi_connected = self._optional_bool(record.get("wifi_connected"))
            self._mqtt_connected = self._optional_bool(record.get("mqtt_connected"))
            self._latest_mqtt_rtt_ms = self._nonnegative_number(
                record.get("latest_mqtt_rtt_ms")
            )
            rtt_samples = self._nonnegative_int(record.get("rtt_samples"))
            self._rtt_samples = rtt_samples or 0
            average_rtt = self._nonnegative_number(
                record.get("average_mqtt_rtt_ms")
            )
            self._rtt_total_ms = (
                average_rtt * self._rtt_samples
                if average_rtt is not None
                else 0.0
            )
            self._last_sequence = self._nonnegative_int(record.get("sequence"))
            self._packets_received = self._nonnegative_int(
                record.get("packets_received")
            ) or 0
            self._packets_expected = self._nonnegative_int(
                record.get("packets_expected")
            ) or 0
            self._packets_lost = self._nonnegative_int(record.get("packets_lost")) or 0
            self._device_restarts = self._nonnegative_int(
                record.get("device_restarts")
            ) or 0
            timestamp_ms = record.get("timestamp_ms")
            if (
                isinstance(timestamp_ms, int)
                and not isinstance(timestamp_ms, bool)
                and timestamp_ms >= 0
            ):
                self._timestamp_ms = timestamp_ms
            elif (
                isinstance(timestamp_ms, float)
                and math.isfinite(timestamp_ms)
                and timestamp_ms >= 0
            ):
                self._timestamp_ms = timestamp_ms
            else:
                self._timestamp_ms = None
            last_seen_at = record.get("last_seen_at")
            self._last_seen_at = (
                last_seen_at if isinstance(last_seen_at, str) else None
            )

    def snapshot(self, now: datetime | None = None) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked(now or datetime.now(timezone.utc))

    def _record_sequence(self, sequence: int) -> None:
        if self._last_sequence is None:
            self._packets_received += 1
            self._packets_expected += 1
        elif sequence > self._last_sequence:
            delta = sequence - self._last_sequence
            self._packets_received += 1
            self._packets_expected += delta
            self._packets_lost += delta - 1
        elif sequence < self._last_sequence:
            self._device_restarts += 1
            self._packets_received += 1
            self._packets_expected += 1
        else:
            return
        self._last_sequence = sequence

    def _snapshot_locked(self, now: datetime) -> dict[str, Any]:
        average_rtt = (
            self._rtt_total_ms / self._rtt_samples if self._rtt_samples else None
        )
        packet_loss = (
            self._packets_lost / self._packets_expected * 100
            if self._packets_expected
            else None
        )
        return {
            "device_id": self._device_id,
            "ip_address": self._ip_address,
            "wifi_connected": self._wifi_connected,
            "mqtt_connected": self._mqtt_connected,
            "latest_mqtt_rtt_ms": self._latest_mqtt_rtt_ms,
            "average_mqtt_rtt_ms": average_rtt,
            "rtt_samples": self._rtt_samples,
            "sequence": self._last_sequence,
            "timestamp_ms": self._timestamp_ms,
            "packets_received": self._packets_received,
            "packets_expected": self._packets_expected,
            "packets_lost": self._packets_lost,
            "estimated_packet_loss_percent": packet_loss,
            "device_restarts": self._device_restarts,
            "last_seen_at": self._last_seen_at,
            "status": self._health_status(now),
        }

    def _health_status(self, now: datetime) -> str:
        if self._last_seen_at is None:
            return "OFFLINE"
        try:
            last_seen = datetime.fromisoformat(self._last_seen_at.replace("Z", "+00:00"))
            if last_seen.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=timezone.utc)
        except ValueError:
            return "STALE"
        if (
            now.astimezone(timezone.utc) - last_seen.astimezone(timezone.utc)
        ).total_seconds() > self._stale_timeout_seconds:
            return "STALE"
        if self._wifi_connected is False or self._mqtt_connected is False:
            return "OFFLINE"
        packet_loss = (
            self._packets_lost / self._packets_expected * 100
            if self._packets_expected
            else None
        )
        if (
            not self._ip_address
            or self._wifi_connected is not True
            or self._mqtt_connected is not True
            or self._latest_mqtt_rtt_ms is None
            or self._last_sequence is None
            or self._timestamp_ms is None
            or packet_loss is None
            or self._latest_mqtt_rtt_ms > RTT_WARNING_THRESHOLD_MS
            or packet_loss > PACKET_LOSS_WARNING_THRESHOLD_PERCENT
        ):
            return "WARNING"
        return "HEALTHY"

    @staticmethod
    def _finite_number(value: Any) -> float | None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except OverflowError:
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _nonnegative_number(value: Any) -> float | None:
        number = NetworkMonitor._finite_number(value)
        return number if number is not None and number >= 0 else None

    @staticmethod
    def _nonnegative_int(value: Any) -> int | None:
        return (
            value
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            else None
        )

    @staticmethod
    def _optional_bool(value: Any) -> bool | None:
        return value if isinstance(value, bool) else None
