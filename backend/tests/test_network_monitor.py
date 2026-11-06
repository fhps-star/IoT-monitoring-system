import unittest
from datetime import datetime, timedelta, timezone

from backend.network_monitor import NetworkMonitor


class NetworkMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.monitor = NetworkMonitor("motor_01", stale_timeout_seconds=15)
        self.now = datetime.now(timezone.utc)

    def _record(self, **overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "ip_address": "192.168.1.42",
            "wifi_connected": True,
            "mqtt_connected": True,
            "mqtt_rtt_ms": 20,
            "sequence": 1,
            "timestamp_ms": 5000,
        }
        payload.update(overrides)
        result = self.monitor.record(payload, self.now.isoformat())
        self.assertIsNotNone(result)
        return result or {}

    def test_computes_rtt_average_and_sequence_loss(self) -> None:
        self._record(mqtt_rtt_ms=20, sequence=10)
        self._record(mqtt_rtt_ms=30, sequence=11)
        snapshot = self._record(mqtt_rtt_ms=40, sequence=14)

        self.assertEqual(snapshot["latest_mqtt_rtt_ms"], 40.0)
        self.assertEqual(snapshot["average_mqtt_rtt_ms"], 30.0)
        self.assertEqual(snapshot["rtt_samples"], 3)
        self.assertEqual(snapshot["packets_received"], 3)
        self.assertEqual(snapshot["packets_expected"], 5)
        self.assertEqual(snapshot["packets_lost"], 2)
        self.assertEqual(snapshot["estimated_packet_loss_percent"], 40.0)
        self.assertEqual(snapshot["status"], "WARNING")

    def test_sequence_reset_is_counted_as_restart_not_packet_loss(self) -> None:
        self._record(sequence=10)
        self._record(sequence=11)
        self._record(sequence=0)
        snapshot = self._record(sequence=1)

        self.assertEqual(snapshot["device_restarts"], 1)
        self.assertEqual(snapshot["packets_lost"], 0)
        self.assertEqual(snapshot["packets_expected"], 4)
        self.assertEqual(snapshot["estimated_packet_loss_percent"], 0.0)
        self.assertEqual(snapshot["status"], "HEALTHY")

    def test_duplicate_sequence_does_not_change_loss_counters(self) -> None:
        self._record(sequence=4)
        snapshot = self._record(sequence=4)

        self.assertEqual(snapshot["packets_received"], 1)
        self.assertEqual(snapshot["packets_expected"], 1)
        self.assertEqual(snapshot["packets_lost"], 0)

    def test_device_health_distinguishes_offline_and_stale(self) -> None:
        self.assertEqual(self.monitor.snapshot(self.now)["status"], "OFFLINE")

        self._record(wifi_connected=False)
        self.assertEqual(self.monitor.snapshot(self.now)["status"], "OFFLINE")

        self._record(wifi_connected=True)
        later = self.now + timedelta(seconds=16)
        self.assertEqual(self.monitor.snapshot(later)["status"], "STALE")

    def test_restore_keeps_cumulative_average_and_sequence_state(self) -> None:
        self._record(mqtt_rtt_ms=20, sequence=8)
        saved = self._record(mqtt_rtt_ms=40, sequence=9)
        restored = NetworkMonitor("motor_01", stale_timeout_seconds=15)
        restored.restore(saved)

        next_sample = restored.record(
            {
                "ip_address": "192.168.1.42",
                "wifi_connected": True,
                "mqtt_connected": True,
                "mqtt_rtt_ms": 60,
                "sequence": 10,
                "timestamp_ms": 6000,
            },
            self.now.isoformat(),
        )

        self.assertIsNotNone(next_sample)
        self.assertEqual(next_sample["average_mqtt_rtt_ms"], 40.0)
        self.assertEqual(next_sample["rtt_samples"], 3)
        self.assertEqual(next_sample["packets_lost"], 0)

    def test_unrelated_or_invalid_network_values_do_not_create_metrics(self) -> None:
        self.assertIsNone(self.monitor.record({"sequence": 1}, self.now.isoformat()))
        snapshot = self.monitor.record(
            {
                "ip_address": "",
                "wifi_connected": "yes",
                "mqtt_connected": 1,
                "mqtt_rtt_ms": float("nan"),
            },
            self.now.isoformat(),
        )

        self.assertIsNotNone(snapshot)
        self.assertIsNone(snapshot["ip_address"])
        self.assertIsNone(snapshot["wifi_connected"])
        self.assertIsNone(snapshot["mqtt_connected"])
        self.assertIsNone(snapshot["latest_mqtt_rtt_ms"])
        self.assertEqual(snapshot["status"], "WARNING")

    def test_invalid_latest_values_are_not_replaced_with_previous_metrics(self) -> None:
        self._record()
        snapshot = self.monitor.record(
            {"mqtt_rtt_ms": "unavailable"},
            self.now.isoformat(),
        )

        self.assertIsNotNone(snapshot)
        self.assertIsNone(snapshot["latest_mqtt_rtt_ms"])
        self.assertEqual(snapshot["average_mqtt_rtt_ms"], 20.0)
        self.assertEqual(snapshot["status"], "WARNING")


if __name__ == "__main__":
    unittest.main()
