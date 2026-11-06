"""Utility script to simulate ESP32 sending MQTT, HTTP, and CoAP telemetry."""

import argparse
import json
import logging
import random
import socket
import sys
import time
from datetime import datetime, timezone
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProtocolSimulator")


def generate_telemetry_payload(seq: int, protocol: str, rtt_ms: float, anomaly_mode: str = "none") -> dict:
    temp = round(38.0 + random.uniform(-1.5, 2.5), 2)
    accel_x = round(random.uniform(-0.15, 0.15), 3)
    accel_y = round(random.uniform(-0.15, 0.15), 3)
    accel_z = round(0.98 + random.uniform(-0.05, 0.05), 3)
    vib_rms = round(random.uniform(0.12, 0.35), 3)
    current_a = round(0.45 + random.uniform(-0.05, 0.08), 3)

    if anomaly_mode == "temperature":
        temp = round(52.0 + random.uniform(1.0, 5.0), 2)
    elif anomaly_mode == "vibration":
        vib_rms = round(0.75 + random.uniform(0.1, 0.4), 3)
        accel_x = round(0.65 + random.uniform(0.1, 0.2), 3)
    elif anomaly_mode == "current":
        current_a = round(1.8 + random.uniform(0.2, 0.6), 3)

    heap_bytes = 214560 - (seq * 16) % 4096

    return {
        "device_id": "motor_01",
        "motor_id": "motor_01",
        "sequence": seq,
        "protocol": protocol.upper(),
        "timestamp_ms": int(time.time() * 1000),
        "rtt_ms": round(rtt_ms, 2),
        "free_heap": heap_bytes,
        "temperature": temp,
        "temperature_c": temp,
        "acceleration_x": accel_x,
        "acceleration_y": accel_y,
        "acceleration_z": accel_z,
        "vibration_rms": vib_rms,
        "current": current_a,
        "current_a": current_a,
        "rpm": 60,
    }


def send_http(url: str, seq: int, anomaly: str = "none") -> bool:
    start = time.perf_counter()
    rtt_sim = random.uniform(85.0, 140.0)
    payload = generate_telemetry_payload(seq, "HTTP", rtt_sim, anomaly)
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "ESP32-HTTPClient/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.info("HTTP POST seq=%d status=%d elapsed=%.1fms", seq, resp.status, elapsed_ms)
            return True
    except Exception as err:
        logger.error("HTTP POST seq=%d failed: %s", seq, err)
        return False


def send_coap(host: str, port: int, seq: int, anomaly: str = "none") -> bool:
    rtt_sim = random.uniform(22.0, 48.0)
    payload = generate_telemetry_payload(seq, "CoAP", rtt_sim, anomaly)
    body = json.dumps(payload).encode("utf-8")

    # Build RFC 7252 CoAP CON POST packet
    # Ver=1, Type=CON(0), TKL=0 -> 0x40
    # Code = 0.02 POST -> 0x02
    msg_id = (seq & 0xFFFF)
    header = bytearray([0x40, 0x02, (msg_id >> 8) & 0xFF, msg_id & 0xFF])
    # Uri-Path option (Option 11): Delta=11, Len=9, Value="telemetry"
    opt_header = bytearray([(11 << 4) | len("telemetry")])
    opt_uri = b"telemetry"
    # Payload marker 0xFF
    packet = bytes(header + opt_header + opt_uri + b"\xFF" + body)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    start = time.perf_counter()
    try:
        sock.sendto(packet, (host, port))
        resp, _ = sock.recvfrom(1024)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        # Check ACK code (0x44 = 2.04 Changed)
        code = resp[1] if len(resp) >= 2 else 0
        logger.info("CoAP CON seq=%d ACK code=0x%02x RTT=%.1fms", seq, code, elapsed_ms)
        return True
    except socket.timeout:
        logger.warning("CoAP seq=%d timeout waiting for ACK", seq)
        return False
    except Exception as err:
        logger.error("CoAP seq=%d error: %s", seq, err)
        return False
    finally:
        sock.close()


def send_mqtt(broker: str, port: int, topic: str, seq: int, anomaly: str = "none") -> bool:
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        logger.warning("paho-mqtt not installed; skipping MQTT simulation")
        return False

    rtt_sim = random.uniform(35.0, 65.0)
    payload = generate_telemetry_payload(seq, "MQTT", rtt_sim, anomaly)
    data = json.dumps(payload)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    try:
        client.connect(broker, port, keepalive=10)
        client.publish(topic, data, qos=0)
        logger.info("MQTT publish seq=%d to %s", seq, topic)
        client.disconnect()
        return True
    except Exception as err:
        logger.error("MQTT publish seq=%d error: %s", seq, err)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate protocol telemetry for Protocol Comparison")
    parser.add_argument("--protocol", choices=["http", "coap", "mqtt", "all"], default="all")
    parser.add_argument("--count", type=int, default=10, help="Number of messages to send per protocol")
    parser.add_argument("--anomaly", choices=["none", "temperature", "vibration", "current"], default="none")
    parser.add_argument("--http-url", default="http://127.0.0.1:8000/api/telemetry")
    parser.add_argument("--coap-host", default="127.0.0.1")
    parser.add_argument("--coap-port", type=int, default=5683)
    parser.add_argument("--mqtt-broker", default="broker.hivemq.com")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-topic", default="iiot/motor_01/telemetry")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    protocols = ["http", "coap", "mqtt"] if args.protocol == "all" else [args.protocol]

    for proto in protocols:
        logger.info("=== Simulating protocol: %s (%d messages, anomaly=%s) ===", proto.upper(), args.count, args.anomaly)
        for seq in range(1, args.count + 1):
            if proto == "http":
                send_http(args.http_url, seq, args.anomaly)
            elif proto == "coap":
                send_coap(args.coap_host, args.coap_port, seq, args.anomaly)
            elif proto == "mqtt":
                send_mqtt(args.mqtt_broker, args.mqtt_port, args.mqtt_topic, seq, args.anomaly)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()

