import socket
import struct
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "openclaw/monitor"))
import egress_guard


class EgressSecurityTests(unittest.TestCase):
    def test_external_dns_cannot_echo_or_forward_patient_canary(self):
        query = struct.pack("!HHHHHH",123,0x0100,1,0,0,0)+b"PATIENT_CANARY.attacker.invalid"
        with patch.object(egress_guard.socket,"create_connection") as upstream:
            response = egress_guard.dns_refusal(query)
            self.assertEqual(len(response),12)
            self.assertEqual(struct.unpack("!HHHHHH",response)[1]&15,5)
            self.assertNotIn(b"PATIENT_CANARY",response)
            upstream.assert_not_called()

    def test_destination_and_protocol_bypasses_are_denied(self):
        allowed = egress_guard.POLICIES[3128]
        for authority in ("attacker.invalid:443", "api.telegram.org.attacker.invalid:443", "api.telegram.org:80", "user@api.telegram.org:443", "127.0.0.1:443", "api.telegram.org.:443", "api.telegram.org:443 HTTP/1.1\r\nCONNECT attacker.invalid:443"):
            request = f"CONNECT {authority} HTTP/1.1\r\nHost: api.telegram.org\r\n\r\n".encode()
            with self.assertRaises(ValueError):
                egress_guard.connect_target(request, allowed)
        self.assertEqual(egress_guard.connect_target(b"CONNECT api.telegram.org:443 HTTP/1.1\r\n\r\n", allowed), ("api.telegram.org", 443))
        with self.assertRaises(ValueError):
            egress_guard.connect_target(b"GET https://api.telegram.org/x HTTP/1.1\r\n\r\n", allowed)

    def test_denied_canary_never_reaches_upstream_or_error_response(self):
        server = egress_guard.Server(("127.0.0.1", 0), egress_guard.Relay)
        server.allowed = egress_guard.POLICIES[3128]
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with patch.object(egress_guard.socket, "create_connection") as upstream:
                with socket.socket() as client:
                    client.connect(server.server_address)
                    client.sendall(b"CONNECT PATIENT_CANARY.attacker.invalid:443 HTTP/1.1\r\n\r\n")
                    result = client.recv(1024)
                self.assertIn(b"403", result)
                self.assertNotIn(b"PATIENT_CANARY", result)
                upstream.assert_not_called()
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    unittest.main()
