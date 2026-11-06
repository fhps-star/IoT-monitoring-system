"""CoAP UDP receiver and acknowledgement responder (RFC 7252)."""

import json
import logging
import select
import socket
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CoAPHandler:
    """Listens for CoAP messages on UDP and sends acknowledgements."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5683,
        on_telemetry: Callable[[dict[str, Any], int, str], None] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.on_telemetry = on_telemetry
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> bool:
        """Start the CoAP UDP receiver loop in a daemon thread."""
        if self._running:
            return True

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((self.host, self.port))
            self._socket.setblocking(False)
            self._running = True
            self._thread = threading.Thread(target=self._listen_loop, daemon=True, name="CoAP-Server")
            self._thread.start()
            logger.info("CoAP UDP server listening on %s:%d", self.host, self.port)
            return True
        except (OSError, socket.error) as err:
            logger.warning("Could not bind CoAP server on %s:%d: %s", self.host, self.port, err)
            self._running = False
            if self._socket:
                try:
                    self._socket.close()
                except OSError:
                    pass
                self._socket = None
            return False

    def stop(self) -> None:
        """Stop the CoAP server gracefully."""
        self._running = False
        sock = self._socket
        if sock:
            try:
                sock.close()
            except OSError:
                pass
            self._socket = None
        if self._thread and self._thread.is_alive() and self._thread != threading.current_thread():
            self._thread.join(timeout=1.0)
        logger.info("CoAP UDP server stopped")

    def _listen_loop(self) -> None:
        while self._running:
            sock = self._socket
            if not sock:
                break
            try:
                rlist, _, _ = select.select([sock], [], [], 0.5)
                if not rlist or not self._running:
                    continue

                try:
                    data, addr = sock.recvfrom(4096)
                except OSError:
                    if not self._running:
                        break
                    continue

                if not data or len(data) < 4:
                    continue

                self._handle_datagram(data, addr)
            except (OSError, ValueError):
                if not self._running:
                    break
                logger.debug("Error in CoAP listen loop", exc_info=True)

    def _handle_datagram(self, data: bytes, addr: tuple[str, int]) -> None:
        """Parse CoAP packet, send ACK if Confirmable (CON), and extract telemetry."""
        # Header: Ver (2b), Type (2b), TKL (4b), Code (8b), Message ID (16b)
        byte0 = data[0]
        ver = (byte0 >> 6) & 0x03
        msg_type = (byte0 >> 4) & 0x03
        tkl = byte0 & 0x0F
        code = data[1]
        msg_id_bytes = data[2:4]

        # Extract token
        token = data[4 : 4 + tkl] if len(data) >= 4 + tkl else b""

        # CoAP ACK response for Confirmable (CON, msg_type == 0) messages:
        # Send ACK (type = 2) with Code = 2.04 Changed (0x44)
        if msg_type == 0 and self._socket:
            try:
                ack_header = bytearray([
                    (1 << 6) | (2 << 4) | (tkl & 0x0F),  # Ver=1, Type=ACK (2), TKL
                    0x44,                                # Code 2.04 Changed
                    msg_id_bytes[0],                     # Message ID MSB
                    msg_id_bytes[1],                     # Message ID LSB
                ])
                ack_packet = bytes(ack_header) + token
                self._socket.sendto(ack_packet, addr)
                logger.debug("Sent CoAP ACK to %s (msg_id=%s)", addr, msg_id_bytes.hex())
            except OSError:
                logger.debug("Failed to send CoAP ACK to %s", addr, exc_info=True)

        # Extract payload: look for 0xFF payload marker
        marker_pos = data.find(b"\xFF", 4 + tkl)
        if marker_pos == -1 or marker_pos >= len(data) - 1:
            return

        payload_bytes = data[marker_pos + 1 :]
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
            if isinstance(payload, dict):
                payload.setdefault("protocol", "CoAP")
                if self.on_telemetry:
                    self.on_telemetry(payload, len(data), "coap")
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.debug("CoAP payload is not valid JSON from %s", addr)
