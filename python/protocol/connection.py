from protocol.framing import (
    encode_message,
    decode_header,
)


class ProtocolConnection:

    def __init__(self, transport):
        self.transport = transport

    def send_message(
        self,
        message_type,
        payload: bytes,
    ):
        packet = encode_message(
            message_type,
            payload,
        )

        self.transport.send(packet)

    def receive_message(self):

        header = self.transport.receive_exact(12)

        message_type, payload_size = (
            decode_header(header)
        )

        payload = self.transport.receive_exact(
            payload_size
        )

        return message_type, payload

    def close(self):
        if hasattr(self.transport, "close"):
            self.transport.close()