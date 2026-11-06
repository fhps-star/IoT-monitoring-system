import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.mqtt_handler import MQTTHandler
from backend.settings import Settings
from backend.mqtt_handler import logger as mqtt_logger


class MQTTHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.on_telemetry = Mock()
        self.handler = MQTTHandler(
            Settings.from_env(),
            on_telemetry=self.on_telemetry,
            on_status=Mock(),
        )

    @staticmethod
    def _message(payload: bytes) -> SimpleNamespace:
        return SimpleNamespace(
            topic="iiot/motor_01/telemetry",
            payload=payload,
        )

    def test_logs_raw_mqtt_message_and_passes_json_object_to_pipeline(self) -> None:
        payload = {
            "device_id": "motor-node-01",
            "temperature_c": 41.25,
            "estimated_vibration": 0.103,
            "motor_current_a": 0.42,
        }

        with patch.object(mqtt_logger, "info") as info_log:
            self.handler._handle_message(
                None,
                None,
                self._message(json.dumps(payload).encode()),
            )

        self.assertEqual(self.handler.messages_received, 1)
        self.on_telemetry.assert_called_once_with(payload)
        info_log.assert_called_once_with(
            "Received MQTT message",
            extra={
                "topic": "iiot/motor_01/telemetry",
                "payload": json.dumps(payload),
            },
        )

    def test_counts_and_logs_malformed_json_without_calling_pipeline(self) -> None:
        raw_payload = b'{"temperature_c":'

        with (
            patch.object(mqtt_logger, "info") as info_log,
            patch.object(mqtt_logger, "warning") as warning_log,
        ):
            self.handler._handle_message(
                None,
                None,
                self._message(raw_payload),
            )

        self.assertEqual(self.handler.messages_received, 1)
        self.on_telemetry.assert_not_called()
        info_log.assert_called_once_with(
            "Received MQTT message",
            extra={
                "topic": "iiot/motor_01/telemetry",
                "payload": raw_payload.decode(),
            },
        )
        warning_log.assert_called_once()
        self.assertEqual(warning_log.call_args.kwargs["extra"]["payload"], raw_payload.decode())

    def test_counts_non_object_json_and_does_not_call_pipeline(self) -> None:
        with (
            patch.object(mqtt_logger, "info"),
            patch.object(mqtt_logger, "warning") as warning_log,
        ):
            self.handler._handle_message(
                None,
                None,
                self._message(b"[]"),
            )

        self.assertEqual(self.handler.messages_received, 1)
        self.on_telemetry.assert_not_called()
        warning_log.assert_called_once_with(
            "Ignoring MQTT JSON that is not an object",
            extra={"topic": "iiot/motor_01/telemetry", "payload": "[]"},
        )


if __name__ == "__main__":
    unittest.main()
