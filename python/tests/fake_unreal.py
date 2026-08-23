import argparse
import socket

from protocol.constants import MessageType
from protocol.framing import (
    HEADER_SIZE,
    decode_header,
    encode_message,
)


def receive_message(sock):
    """
    Receive one complete framed protocol message.

    TCP is a byte stream, so we cannot assume that
    one recv() call gives us the complete message.
    """

    # -----------------------------------------
    # Receive header
    # -----------------------------------------

    header = bytearray()

    while len(header) < HEADER_SIZE:

        chunk = sock.recv(
            HEADER_SIZE - len(header)
        )

        if not chunk:
            raise ConnectionError(
                "Python server disconnected while "
                "receiving header"
            )

        header.extend(chunk)

    # -----------------------------------------
    # Decode header
    # -----------------------------------------

    message_type, payload_size = decode_header(
        bytes(header)
    )

    # -----------------------------------------
    # Receive payload
    # -----------------------------------------

    payload = bytearray()

    while len(payload) < payload_size:

        chunk = sock.recv(
            payload_size - len(payload)
        )

        if not chunk:
            raise ConnectionError(
                "Python server disconnected while "
                "receiving payload"
            )

        payload.extend(chunk)

    return message_type, bytes(payload)


def main():

    parser = argparse.ArgumentParser(
        description="Fake Unreal TCP client"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Python server address",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Python server port",
    )

    args = parser.parse_args()

    print(
        f"Connecting to Python server "
        f"at {args.host}:{args.port}"
    )

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    )

    try:

        # -----------------------------------------
        # Connect
        # -----------------------------------------

        sock.connect(
            (args.host, args.port)
        )

        print(
            "Connected to Python server."
        )

        # -----------------------------------------
        # Send HELLO
        # -----------------------------------------

        payload = b"Hello from fake Unreal!"

        packet = encode_message(
            MessageType.HELLO,
            payload,
        )

        sock.sendall(packet)

        print(
            f"Sent HELLO: {len(packet)} bytes"
        )

        print(
            f"Packet: {packet.hex(' ')}"
        )

        # -----------------------------------------
        # Wait for Python HELLO_ACK
        # -----------------------------------------

        print(
            "Waiting for HELLO_ACK..."
        )

        message_type, response_payload = (
            receive_message(sock)
        )

        print(
            f"Received message:"
            f" type={message_type},"
            f" payload_size={len(response_payload)}"
        )

        print(
            f"Payload: {response_payload!r}"
        )

        # -----------------------------------------
        # Validate response
        # -----------------------------------------

        if message_type != MessageType.HELLO_ACK:

            print(
                "ERROR: Expected HELLO_ACK "
                f"but received type={message_type}"
            )

        else:

            print(
                "HELLO/HELLO_ACK handshake "
                "successful."
            )

    except ConnectionError as e:

        print(
            f"Connection error: {e}"
        )

    except Exception as e:

        print(
            f"Unexpected error:"
            f" {type(e).__name__}: {e}"
        )

    finally:

        sock.close()

        print(
            "Fake Unreal stopped."
        )


if __name__ == "__main__":
    main()