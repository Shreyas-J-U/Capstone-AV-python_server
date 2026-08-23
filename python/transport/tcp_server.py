import socket


class TCPServer:

    def __init__(
        self,
        host: str,
        port: int,
        timeout: float = 10.0,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout

        self.server_socket = None
        self.client_socket = None

    def start(self):
        self.server_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        self.server_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        self.server_socket.bind(
            (self.host, self.port)
        )

        self.server_socket.listen(1)

        print(
            f"Server listening on "
            f"{self.host}:{self.port}"
        )

    def accept(self):
        print("Waiting for Unreal client...")

        self.client_socket, address = (
            self.server_socket.accept()
        )

        self.client_socket.settimeout(
            self.timeout
        )

        print(
            f"Unreal connected from {address}"
        )

    def send(self, data: bytes):
        if self.client_socket is None:
            raise ConnectionError(
                "No Unreal client connected"
            )

        self.client_socket.sendall(data)

    def receive_exact(self, size: int) -> bytes:
        if self.client_socket is None:
            raise ConnectionError(
                "No Unreal client connected"
            )

        data = bytearray()

        while len(data) < size:

            chunk = self.client_socket.recv(
                size - len(data)
            )

            if not chunk:
                raise ConnectionError(
                    "Unreal disconnected"
                )

            data.extend(chunk)

        return bytes(data)

    def close(self):

        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None

        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None