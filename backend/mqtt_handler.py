"""MQTT subscriber with connection recovery and JSON payload validation."""

import json
import logging
from collections.abc import Callable
from threading import Lock
from typing import Any
from uuid import uuid4

import paho.mqtt.client as mqtt

from backend.settings import Settings

logger = logging.getLogger(__name__)
TelemetryCallback = Callable[[dict[str, Any]], None]
StatusCallback = Callable[[str, str | None], None]


class MQTTHandler:
    def __init__(
        self,
        settings: Settings,
        on_telemetry: TelemetryCallback,
        on_status: StatusCallback,
    ) -> None:
        self._settings = settings
        self._on_telemetry = on_telemetry
        self._on_status = on_status
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"iiot{uuid4().hex[:19]}",
            protocol=mqtt.MQTTv311,
        )
        self._client.on_connect = self._handle_connect
        self._client.on_subscribe = self._handle_subscribe
        self._client.on_disconnect = self._handle_disconnect
        self._client.on_message = self._handle_message
        self._client.reconnect_delay_set(min_delay=1, max_delay=60)
        self._started = False
        self._stopping = False
        self._subscribed = False
        self._subscription_error: str | None = None
        self._messages_received = 0
        self._messages_lock = Lock()

    @property
    def connected(self) -> bool:
        return self._client.is_connected()

    @property
    def subscription_status(self) -> str:
        if not self.connected:
            return "disconnected"
        if self._subscription_error:
            return "failed"
        return "subscribed" if self._subscribed else "subscribing"

    @property
    def subscription_error(self) -> str | None:
        return self._subscription_error

    @property
    def messages_received(self) -> int:
        with self._messages_lock:
            return self._messages_received

    def start(self) -> None:
        if self._started:
            return
        self._stopping = False
        logger.info(
            "Starting MQTT subscriber",
            extra={
                "broker": self._settings.mqtt_broker,
                "port": self._settings.mqtt_port,
                "topic": self._settings.mqtt_topic,
            },
        )
        self._client.connect_async(
            self._settings.mqtt_broker,
            self._settings.mqtt_port,
            keepalive=60,
        )
        self._client.loop_start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._stopping = True
        self._client.disconnect()
        self._client.loop_stop()
        self._started = False

    def _handle_connect(
        self,
        client: mqtt.Client,
        _userdata: Any,
        _flags: mqtt.ConnectFlags,
        reason_code: mqtt.ReasonCode,
        _properties: mqtt.Properties | None,
    ) -> None:
        if reason_code.is_failure:
            logger.error(
                "MQTT connection rejected",
                extra={"reason_code": str(reason_code)},
            )
            self._on_status("disconnected", str(reason_code))
            return
        self._subscribed = False
        self._subscription_error = None
        logger.info(
            "Connected to MQTT broker",
            extra={
                "broker": self._settings.mqtt_broker,
                "topic": self._settings.mqtt_topic,
            },
        )
        result, message_id = client.subscribe(self._settings.mqtt_topic, qos=0)
        if result != mqtt.MQTTErrorCode.MQTT_ERR_SUCCESS:
            self._subscription_error = f"Subscribe request failed ({int(result)})"
            logger.error(
                "MQTT subscription request failed",
                extra={"result": int(result), "topic": self._settings.mqtt_topic},
            )
            self._on_status("disconnected", self._subscription_error)
            return
        logger.info(
            "Subscribed to MQTT topic",
            extra={"topic": self._settings.mqtt_topic, "message_id": message_id},
        )

    def _handle_subscribe(
        self,
        _client: mqtt.Client,
        _userdata: Any,
        _message_id: int,
        reason_codes: list[mqtt.ReasonCode],
        _properties: mqtt.Properties | None,
    ) -> None:
        failures = [str(reason) for reason in reason_codes if reason.is_failure]
        if failures:
            self._subscription_error = ", ".join(failures)
            logger.error(
                "MQTT subscription rejected",
                extra={
                    "topic": self._settings.mqtt_topic,
                    "reason_codes": failures,
                },
            )
            self._on_status("disconnected", self._subscription_error)
            return
        self._subscribed = True
        self._subscription_error = None
        logger.info(
            "MQTT subscription acknowledged",
            extra={"topic": self._settings.mqtt_topic},
        )
        self._on_status("connected", None)

    def _handle_disconnect(
        self,
        _client: mqtt.Client,
        _userdata: Any,
        _disconnect_flags: mqtt.DisconnectFlags,
        reason_code: mqtt.ReasonCode,
        _properties: mqtt.Properties | None,
    ) -> None:
        self._subscribed = False
        if self._stopping:
            logger.info("MQTT subscriber stopped")
        else:
            logger.warning(
                "MQTT connection lost; client will reconnect",
                extra={"reason_code": str(reason_code)},
            )
        self._on_status("disconnected", str(reason_code))

    def _handle_message(
        self, _client: mqtt.Client, _userdata: Any, message: mqtt.MQTTMessage
    ) -> None:
        with self._messages_lock:
            self._messages_received += 1

        try:
            text = message.payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raw_payload = message.payload.decode("utf-8", errors="replace")
            logger.warning(
                "Ignoring MQTT message with invalid UTF-8",
                extra={
                    "topic": message.topic,
                    "payload": raw_payload,
                    "error": str(exc),
                },
            )
            return

        logger.info(
            "Received MQTT message",
            extra={"topic": message.topic, "payload": text},
        )
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Ignoring malformed MQTT JSON",
                extra={
                    "topic": message.topic,
                    "payload": text,
                    "error": str(exc),
                },
            )
            return

        if not isinstance(payload, dict):
            logger.warning(
                "Ignoring MQTT JSON that is not an object",
                extra={"topic": message.topic, "payload": text},
            )
            return

        try:
            self._on_telemetry(payload)
        except Exception:
            logger.exception(
                "Telemetry processing callback failed",
                extra={"topic": message.topic},
            )
