class TCPSocketTransport:

    def __init__(self, sock):
        self.sock = sock

    def send(self, data: bytes):

        self.sock.sendall(data)

    def receive_exact(
        self,
        size: int,
    ):

        data = bytearray()

        while len(data) < size:

            chunk = self.sock.recv(
                size - len(data)
            )

            if not chunk:

                raise ConnectionError(
                    "Peer disconnected"
                )

            data.extend(chunk)

        return bytes(data)

    def close(self):

        self.sock.close()