"""TLS CONNECT relay with fixed host allowlists and no payload logging.

TLS stays end-to-end. This relay never decrypts patient traffic. Client-side
firewall rules additionally prevent bypass through a different proxy or port.
"""
import re
import select
import socket
import socketserver
import struct
import threading

UPSTREAM = ("singbox", 1080)
POLICIES = {
    3128: frozenset({"api.telegram.org"}),
    3129: frozenset({"naykalab.ru", "tc.naykalab.ru"}),
    3130: frozenset({"api.telegram.org"}),
}
SLOTS = threading.BoundedSemaphore(96)


def connect_target(header: bytes, allowed):
    if len(header) > 8192 or b"\r\n\r\n" not in header:
        raise ValueError("invalid_proxy_request")
    first = header.split(b"\r\n", 1)[0].decode("ascii", "strict")
    for line in header.split(b"\r\n")[1:-2]:
        if not re.fullmatch(rb"[A-Za-z0-9-]+:[\x20-\x7e]*", line):
            raise ValueError("invalid_proxy_header")
    match = re.fullmatch(r"CONNECT ([a-zA-Z0-9.-]+):443 HTTP/1\.[01]", first)
    if not match or match[1].lower() not in allowed:
        raise ValueError("egress_destination_denied")
    return match[1].lower(), 443


def read_header(connection):
    result = b""
    while b"\r\n\r\n" not in result:
        byte = connection.recv(1)
        if not byte or len(result) >= 8192:
            raise ValueError("invalid_proxy_header")
        result += byte
    return result


class Relay(socketserver.BaseRequestHandler):
    def handle(self):
        client = self.request
        if not SLOTS.acquire(blocking=False):
            client.sendall(b"HTTP/1.1 503 Busy\r\nContent-Length: 0\r\n\r\n")
            return
        upstream = None
        established = False
        try:
            client.settimeout(10)
            host, port = connect_target(read_header(client), self.server.allowed)
            upstream = socket.create_connection(UPSTREAM, timeout=10)
            upstream.sendall(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode("ascii"))
            response = read_header(upstream)
            if not re.match(rb"HTTP/1\.[01] 200(?: |\r)", response):
                raise OSError("upstream_unavailable")
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            established = True
            while True:
                readable, _, _ = select.select([client, upstream], [], [], 120)
                if not readable:
                    break
                for source in readable:
                    block = source.recv(65536)
                    if not block:
                        return
                    (upstream if source is client else client).sendall(block)
        except (OSError, ValueError, UnicodeError):
            # Never include the URL, header, TLS bytes or exception text.
            if not established:
                try:
                    client.sendall(b"HTTP/1.1 403 Egress denied\r\nContent-Length: 0\r\n\r\n")
                except OSError:
                    pass
        finally:
            if upstream:
                upstream.close()
            SLOTS.release()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def dns_refusal(request):
    # No recursion or query logging. Prevent Docker's external DNS forwarding
    # from becoming an outbound channel outside the CONNECT host allowlist.
    if len(request) < 12:
        return b""
    identifier, flags = struct.unpack("!HH", request[:4])
    return struct.pack("!HHHHHH", identifier, 0x8085 | (flags & 0x0100), 0, 0, 0, 0)


class DNSDatagram(socketserver.BaseRequestHandler):
    def handle(self):
        request, connection = self.request
        response = dns_refusal(request)
        if response:
            connection.sendto(response, self.client_address)


class DNSStream(socketserver.BaseRequestHandler):
    def handle(self):
        connection = self.request
        connection.settimeout(2)
        try:
            length_bytes = connection.recv(2)
            if len(length_bytes) != 2:
                return
            length = struct.unpack("!H", length_bytes)[0]
            if not 12 <= length <= 4096:
                return
            request = b""
            while len(request) < length:
                part = connection.recv(length-len(request))
                if not part:
                    return
                request += part
            response = dns_refusal(request)
            connection.sendall(struct.pack("!H",len(response))+response)
        except OSError:
            pass


class DNSUDPServer(socketserver.UDPServer):
    allow_reuse_address = True
    max_packet_size = 4096


def main():
    servers = []
    for port, allowed in POLICIES.items():
        server = Server(("0.0.0.0", port), Relay)
        server.allowed = allowed
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
    for cls, handler in ((DNSUDPServer, DNSDatagram), (Server, DNSStream)):
        server = cls(("0.0.0.0",53),handler)
        servers.append(server)
        threading.Thread(target=server.serve_forever,daemon=True).start()
    threading.Event().wait()


if __name__ == "__main__":
    main()
