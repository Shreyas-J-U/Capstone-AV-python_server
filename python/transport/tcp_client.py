import socket


class TCPClient:

    def __init__(
        self,
        host: str,
        port: int,
        timeout: float = 10.0,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None

    def connect(self):
        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        self.sock.settimeout(self.timeout)

        self.sock.connect(
            (self.host, self.port)
        )

    def send(self, data: bytes):
        if self.sock is None:
            raise ConnectionError("Not connected")

        self.sock.sendall(data)

    def receive_exact(self, size: int):
        data = bytearray()

        while len(data) < size:
            chunk = self.sock.recv(size - len(data))

            if not chunk:
                raise ConnectionError(
                    "Peer disconnected"
                )

            data.extend(chunk)

        return bytes(data)

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None