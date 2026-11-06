"""Protocol-wise metrics tracking, experiment handling, and comparison aggregator."""

from datetime import datetime, timezone
import logging
import math
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Protocol-specific fixed/variable communication overhead (IP + Transport + Protocol headers)
# MQTT: 20B IPv4 + 20B TCP + 2B Fixed + 2B Var Topic Len + 23B Topic ("iiot/motor_01/telemetry") = 67 bytes
# HTTP: 20B IPv4 + 20B TCP + ~145B HTTP POST headers = 185 bytes
# CoAP: 20B IPv4 + 8B UDP + 4B CoAP Header + 10B Uri-Path ("telemetry") + 2B Content-Format + 1B Marker = 45 bytes
PROTOCOL_OVERHEAD_BYTES: dict[str, int] = {
    "mqtt": 67,
    "http": 185,
    "coap": 45,
}

TRANSPORT_PROTOCOLS: dict[str, str] = {
    "mqtt": "TCP",
    "http": "TCP",
    "coap": "UDP",
}


class ProtocolExperiment:
    """Tracks a single continuous experiment for one protocol."""

    def __init__(self, protocol: str, device_id: str, experiment_id: str | None = None) -> None:
        self.protocol = protocol.lower()
        self.device_id = device_id
        self.experiment_id = experiment_id or f"exp_{self.protocol}_{uuid.uuid4().hex[:8]}"
        self.started_at: datetime = datetime.now(timezone.utc)
        self.ended_at: datetime | None = None
        self.is_active = True

        self.sequences_received: set[int] = set()
        self.min_sequence: int | None = None
        self.max_sequence: int | None = None
        self.total_messages = 0
        self.total_bytes = 0
        self.rtt_measurements: list[float] = []
        self.latest_telemetry: dict[str, Any] | None = None
        self.latest_timestamp_ms: int | None = None
        self.first_timestamp_ms: int | None = None
        self.memory_samples: list[int] = []
        self.reconnection_time_ms: float | None = None

    def record(self, payload: dict[str, Any], raw_bytes: int | None = None) -> None:
        self.total_messages += 1
        self.latest_telemetry = payload

        # Sequence tracking for packet loss calculation
        seq = payload.get("sequence")
        if isinstance(seq, int) and seq >= 0:
            self.sequences_received.add(seq)
            if self.min_sequence is None or seq < self.min_sequence:
                self.min_sequence = seq
            if self.max_sequence is None or seq > self.max_sequence:
                self.max_sequence = seq

        # Timestamp tracking for throughput calculation
        ts_ms = payload.get("timestamp_ms")
        if isinstance(ts_ms, (int, float)) and math.isfinite(ts_ms):
            ts_int = int(ts_ms)
            if self.first_timestamp_ms is None:
                self.first_timestamp_ms = ts_int
            self.latest_timestamp_ms = ts_int

        # Round-trip latency tracking
        rtt = payload.get("rtt_ms")
        if isinstance(rtt, (int, float)) and not isinstance(rtt, bool) and math.isfinite(rtt) and rtt >= 0:
            self.rtt_measurements.append(float(rtt))

        # Bandwidth tracking: application bytes + protocol overhead
        overhead = PROTOCOL_OVERHEAD_BYTES.get(self.protocol, 50)
        app_bytes = raw_bytes if raw_bytes and raw_bytes > 0 else len(str(payload).encode("utf-8"))
        self.total_bytes += app_bytes + overhead

        # Memory usage tracking from ESP32 reported free heap
        heap = (
            payload.get("free_heap")
            or payload.get("memory_usage")
            or payload.get("free_memory")
            or payload.get("heap_bytes")
        )
        if isinstance(heap, (int, float)) and not isinstance(heap, bool) and math.isfinite(heap) and heap > 0:
            self.memory_samples.append(int(heap))

        # Reconnection time if measured
        recon = payload.get("reconnection_time_ms") or payload.get("reconnect_ms")
        if isinstance(recon, (int, float)) and not isinstance(recon, bool) and math.isfinite(recon) and recon >= 0:
            self.reconnection_time_ms = float(recon)

    def close(self) -> None:
        self.is_active = False
        self.ended_at = datetime.now(timezone.utc)

    def compute_metrics(self) -> dict[str, Any]:
        """Compute all protocol comparison metrics for this experiment."""
        transport = TRANSPORT_PROTOCOLS.get(self.protocol, "TCP")

        # Latency calculations
        if self.rtt_measurements:
            avg_latency = round(sum(self.rtt_measurements) / len(self.rtt_measurements), 2)
            min_latency = round(min(self.rtt_measurements), 2)
            max_latency = round(max(self.rtt_measurements), 2)
        else:
            avg_latency = None
            min_latency = None
            max_latency = None

        # Packet Loss calculation based on sequence numbers
        if self.min_sequence is not None and self.max_sequence is not None:
            expected_count = self.max_sequence - self.min_sequence + 1
            received_count = len(self.sequences_received)
            if expected_count > 0:
                missing = max(0, expected_count - received_count)
                packet_loss = round((missing / expected_count) * 100.0, 2)
            else:
                packet_loss = 0.0
        else:
            packet_loss = 0.0 if self.total_messages > 0 else None

        # Throughput (msg/sec)
        duration_s: float = 0.0
        if self.first_timestamp_ms is not None and self.latest_timestamp_ms is not None:
            diff_ms = self.latest_timestamp_ms - self.first_timestamp_ms
            if diff_ms > 0:
                duration_s = diff_ms / 1000.0
        elif self.started_at:
            now = self.ended_at or datetime.now(timezone.utc)
            duration_s = max(0.0, (now - self.started_at).total_seconds())

        if self.total_messages > 0:
            if duration_s >= 1.0:
                throughput = round(self.total_messages / duration_s, 2)
            else:
                throughput = 1.0
        else:
            throughput = None

        # Bandwidth Usage (KB)
        bandwidth_usage = round(self.total_bytes / 1024.0, 2) if self.total_bytes > 0 else (0.0 if self.total_messages > 0 else None)

        # Message overhead
        message_overhead = PROTOCOL_OVERHEAD_BYTES.get(self.protocol, 50)

        # Memory usage (reported free heap)
        memory_usage = self.memory_samples[-1] if self.memory_samples else None

        # CPU usage (not measurable in Wokwi simulation -> null)
        cpu_usage = None

        return {
            "transport_protocol": transport,
            "average_latency": avg_latency,
            "minimum_latency": min_latency,
            "maximum_latency": max_latency,
            "packet_loss": packet_loss,
            "throughput": throughput,
            "bandwidth_usage": bandwidth_usage,
            "message_overhead": message_overhead,
            "memory_usage": memory_usage,
            "cpu_usage": cpu_usage,
            "reconnection_time": self.reconnection_time_ms,
            "total_messages": self.total_messages,
            "experiment_id": self.experiment_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
        }


class ProtocolTracker:
    """Manages protocol-wise data collection, separation, and experiment tracking."""

    def __init__(self, default_device_id: str = "motor_01") -> None:
        self._default_device_id = default_device_id
        # Current active experiment per protocol: {"mqtt": exp, "http": exp, "coap": exp}
        self._active_experiments: dict[str, ProtocolExperiment] = {}
        # Historical completed experiments
        self._history: list[dict[str, Any]] = []
        # Current active protocol reported by ESP32
        self._current_protocol: str = "mqtt"
        self._last_received_at: str | None = None

        # Initialize fresh experiments for all 3 protocols
        for proto in ("mqtt", "http", "coap"):
            self._active_experiments[proto] = ProtocolExperiment(proto, default_device_id)

    def normalize_protocol(self, proto: Any) -> str:
        if not isinstance(proto, str):
            return "mqtt"
        p = proto.strip().lower()
        if "http" in p:
            return "http"
        if "coap" in p:
            return "coap"
        return "mqtt"

    def record_telemetry(
        self,
        protocol: str,
        payload: dict[str, Any],
        raw_bytes: int | None = None,
        received_at: str | None = None,
    ) -> dict[str, Any]:
        """Record incoming telemetry under its respective protocol experiment."""
        proto = self.normalize_protocol(protocol or payload.get("protocol"))
        device_id = str(payload.get("device_id") or self._default_device_id)
        now_iso = received_at or datetime.now(timezone.utc).isoformat()
        self._last_received_at = now_iso
        self._current_protocol = proto

        exp = self._active_experiments.get(proto)
        if exp is None or not exp.is_active:
            exp = ProtocolExperiment(proto, device_id)
            self._active_experiments[proto] = exp

        exp.record(payload, raw_bytes)
        return exp.compute_metrics()

    def get_comparison(self) -> dict[str, Any]:
        """Return the comparison matrix for MQTT, HTTP, and CoAP formatted for the frontend."""
        by_proto = {
            "mqtt": self._active_experiments["mqtt"].compute_metrics(),
            "http": self._active_experiments["http"].compute_metrics(),
            "coap": self._active_experiments["coap"].compute_metrics(),
        }
        res: dict[str, Any] = {
            "protocols": by_proto,
            "mqtt": by_proto["mqtt"],
            "http": by_proto["http"],
            "coap": by_proto["coap"],
            # Key-indexed format matching frontend COMPARISON_ROWS
            "transport": {p: by_proto[p]["transport_protocol"] for p in ("mqtt", "http", "coap")},
            "avg_latency": {p: by_proto[p]["average_latency"] for p in ("mqtt", "http", "coap")},
            "min_latency": {p: by_proto[p]["minimum_latency"] for p in ("mqtt", "http", "coap")},
            "max_latency": {p: by_proto[p]["maximum_latency"] for p in ("mqtt", "http", "coap")},
            "packet_loss": {p: by_proto[p]["packet_loss"] for p in ("mqtt", "http", "coap")},
            "throughput": {p: by_proto[p]["throughput"] for p in ("mqtt", "http", "coap")},
            "bandwidth": {p: by_proto[p]["bandwidth_usage"] for p in ("mqtt", "http", "coap")},
            "overhead": {p: by_proto[p]["message_overhead"] for p in ("mqtt", "http", "coap")},
            "memory": {p: by_proto[p]["memory_usage"] for p in ("mqtt", "http", "coap")},
            "cpu": {p: by_proto[p]["cpu_usage"] for p in ("mqtt", "http", "coap")},
            "reconnect_time": {p: by_proto[p]["reconnection_time"] for p in ("mqtt", "http", "coap")},
        }
        return res

    def get_status(self) -> dict[str, Any]:
        """Return live connection and protocol status."""
        return {
            "current_protocol": self._current_protocol,
            "last_message_at": self._last_received_at,
            "active_protocols": {
                proto: {
                    "active": exp.total_messages > 0,
                    "total_messages": exp.total_messages,
                    "transport": TRANSPORT_PROTOCOLS.get(proto),
                    "experiment_id": exp.experiment_id,
                }
                for proto, exp in self._active_experiments.items()
            },
        }

    def get_live(self) -> dict[str, Any]:
        """Return the latest telemetry payload and metrics for each protocol."""
        return {
            proto: {
                "latest_telemetry": exp.latest_telemetry,
                "metrics": exp.compute_metrics(),
            }
            for proto, exp in self._active_experiments.items()
        }

    def get_history(self) -> list[dict[str, Any]]:
        """Return historical completed experiments."""
        return list(self._history)

    def reset_experiment(self, protocol: str | None = None) -> dict[str, Any]:
        """Archive current experiment(s) and start a fresh one without deleting history."""
        protos_to_reset = [self.normalize_protocol(protocol)] if protocol else ["mqtt", "http", "coap"]

        for p in protos_to_reset:
            current = self._active_experiments.get(p)
            if current and current.total_messages > 0:
                current.close()
                self._history.append({
                    "protocol": p,
                    "device_id": current.device_id,
                    "experiment_id": current.experiment_id,
                    "metrics": current.compute_metrics(),
                })
            self._active_experiments[p] = ProtocolExperiment(p, self._default_device_id)

        return self.get_comparison()
