"""Unit tests for ProtocolTracker, CoAPHandler, and protocol endpoints."""

import json
import socket
import time
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.coap_handler import CoAPHandler
from backend.main import create_app
from backend.protocol_tracker import ProtocolTracker, ProtocolExperiment


class TestProtocolTracker(unittest.TestCase):
    def test_protocol_separation_and_metrics(self) -> None:
        tracker = ProtocolTracker(default_device_id="motor_01")

        # Record MQTT messages
        tracker.record_telemetry("mqtt", {"sequence": 1, "rtt_ms": 40.0, "free_heap": 200000, "timestamp_ms": 1000})
        tracker.record_telemetry("mqtt", {"sequence": 2, "rtt_ms": 50.0, "free_heap": 199000, "timestamp_ms": 2000})
        tracker.record_telemetry("mqtt", {"sequence": 4, "rtt_ms": 60.0, "free_heap": 198000, "timestamp_ms": 4000})  # seq 3 missing

        # Record HTTP messages
        tracker.record_telemetry("http", {"sequence": 10, "rtt_ms": 120.0, "timestamp_ms": 10000})
        tracker.record_telemetry("http", {"sequence": 11, "rtt_ms": 140.0, "timestamp_ms": 11000})

        comparison = tracker.get_comparison()

        # MQTT checks
        mqtt = comparison["mqtt"]
        self.assertEqual(mqtt["transport_protocol"], "TCP")
        self.assertEqual(mqtt["total_messages"], 3)
        self.assertAlmostEqual(mqtt["average_latency"], 50.0)
        self.assertAlmostEqual(mqtt["minimum_latency"], 40.0)
        self.assertAlmostEqual(mqtt["maximum_latency"], 60.0)
        # Expected sequences: 1..4 = 4 total, 3 received -> 1 missing -> 25% loss
        self.assertEqual(mqtt["packet_loss"], 25.0)
        self.assertEqual(mqtt["memory_usage"], 198000)
        self.assertEqual(mqtt["message_overhead"], 67)
        self.assertIsNone(mqtt["cpu_usage"])

        # HTTP checks
        http = comparison["http"]
        self.assertEqual(http["transport_protocol"], "TCP")
        self.assertEqual(http["total_messages"], 2)
        self.assertAlmostEqual(http["average_latency"], 130.0)
        self.assertEqual(http["packet_loss"], 0.0)
        self.assertEqual(http["message_overhead"], 185)

        # CoAP should have 0 messages and null latency
        coap = comparison["coap"]
        self.assertEqual(coap["transport_protocol"], "UDP")
        self.assertEqual(coap["total_messages"], 0)
        self.assertIsNone(coap["average_latency"])
        self.assertIsNone(coap["throughput"])
        self.assertEqual(coap["message_overhead"], 45)

        # Check key-indexed mapping
        self.assertIn("avg_latency", comparison)
        self.assertEqual(comparison["avg_latency"]["mqtt"], 50.0)
        self.assertEqual(comparison["avg_latency"]["http"], 130.0)
        self.assertIsNone(comparison["avg_latency"]["coap"])

    def test_experiment_reset(self) -> None:
        tracker = ProtocolTracker()
        tracker.record_telemetry("mqtt", {"sequence": 1, "rtt_ms": 45.0})
        self.assertEqual(tracker.get_status()["active_protocols"]["mqtt"]["total_messages"], 1)

        tracker.reset_experiment("mqtt")
        self.assertEqual(tracker.get_status()["active_protocols"]["mqtt"]["total_messages"], 0)
        self.assertEqual(len(tracker.get_history()), 1)


class TestCoAPHandler(unittest.TestCase):
    def test_coap_con_and_ack(self) -> None:
        received = []

        def callback(payload: dict, raw_bytes: int, src: str) -> None:
            received.append((payload, raw_bytes, src))

        # Start handler on dynamic port
        handler = CoAPHandler(host="127.0.0.1", port=0, on_telemetry=callback)
        # Manually bind socket to test datagram processing
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        _, port = sock.getsockname()
        sock.close()

        handler.port = port
        self.assertTrue(handler.start())

        try:
            client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            client_sock.settimeout(2.0)

            # Build CON POST packet
            msg_id = 0x1234
            header = bytearray([0x40, 0x02, 0x12, 0x34])
            payload = json.dumps({"sequence": 1, "temperature": 42.0}).encode("utf-8")
            packet = bytes(header + b"\xFF" + payload)

            client_sock.sendto(packet, ("127.0.0.1", port))
            ack, _ = client_sock.recvfrom(1024)

            # Check ACK packet: Type = 2 (ACK), Code = 0x44 (2.04 Changed), MsgID = 0x1234
            self.assertEqual((ack[0] >> 4) & 0x03, 2)  # Type ACK
            self.assertEqual(ack[1], 0x44)             # Code 2.04 Changed
            self.assertEqual(ack[2:4], b"\x12\x34")   # MsgID

            time.sleep(0.1)
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0][0]["sequence"], 1)
            self.assertEqual(received[0][2], "coap")
            client_sock.close()
        finally:
            handler.stop()


class TestProtocolEndpoints(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_http_telemetry_post_and_comparison_get(self) -> None:
        with self.client as client:
            resp = client.post(
                "/api/v1/telemetry",
                json={
                    "device_id": "motor_01",
                    "sequence": 1,
                    "rtt_ms": 115.5,
                    "temperature": 40.0,
                    "vibration": 0.22,
                    "current": 0.48,
                    "free_heap": 210000,
                },
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["protocol"], "http")

            # GET comparison
            comp_resp = client.get("/api/v1/protocols/comparison")
            self.assertEqual(comp_resp.status_code, 200)
            comp = comp_resp.json()

            self.assertEqual(comp["http"]["total_messages"], 1)
            self.assertEqual(comp["http"]["average_latency"], 115.5)
            self.assertEqual(comp["http"]["transport_protocol"], "TCP")

            # GET status
            status_resp = client.get("/api/v1/protocols/status")
            self.assertEqual(status_resp.status_code, 200)
            status = status_resp.json()
            self.assertEqual(status["active_protocols"]["http"]["total_messages"], 1)


if __name__ == "__main__":
    unittest.main()

