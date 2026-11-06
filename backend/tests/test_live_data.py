import unittest

from backend.live_data import LiveTelemetryBuffer


class LiveTelemetryBufferTests(unittest.TestCase):
    def test_restored_timestamp_does_not_count_as_a_new_message(self) -> None:
        buffer = LiveTelemetryBuffer(5)

        buffer.restore_last_message("2026-10-31T12:00:00+00:00")
        snapshot = buffer.snapshot(5)

        self.assertEqual(snapshot["last_message_at"], "2026-10-31T12:00:00+00:00")
        self.assertEqual(snapshot["total_received"], 0)
        self.assertEqual(snapshot["telemetry"], [])

    def test_received_mqtt_message_updates_count_and_timestamp(self) -> None:
        buffer = LiveTelemetryBuffer(5)

        buffer.note_message("2026-10-31T12:00:00+00:00")
        snapshot = buffer.snapshot(5)

        self.assertEqual(snapshot["last_message_at"], "2026-10-31T12:00:00+00:00")
        self.assertEqual(snapshot["total_received"], 1)


if __name__ == "__main__":
    unittest.main()
