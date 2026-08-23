import argparse
import socket
import time

from protocol.constants import MessageType
from protocol.framing import (
    encode_message,
    HEADER_SIZE,
    decode_header,
)

from protocol.observation_serializer import (
    serialize_observation,
)

from protocol.action_deserializer import (
    deserialize_action,
)

from protocol.models import (
    Observation,
    Sensor,
    Termination,
)


def receive_exact(sock, size):

    data = bytearray()

    while len(data) < size:

        chunk = sock.recv(
            size - len(data)
        )

        if not chunk:

            raise ConnectionError(
                "Python server disconnected"
            )

        data.extend(chunk)

    return bytes(data)


def receive_message(sock):

    header = receive_exact(
        sock,
        HEADER_SIZE
    )

    message_type, payload_size = (
        decode_header(header)
    )

    payload = receive_exact(
        sock,
        payload_size
    )

    return message_type, payload


def make_observation(
    episode_id,
    step_id,
    frame_id,
):

    return Observation(

        episode_id=episode_id,

        step_id=step_id,

        frame_id=frame_id,

        simulation_time=step_id * 0.2,

        image=None,

        sensors=[
            Sensor(
                type=1,
                format=1,
                value=b"\x00\x00\x48\x41",
            ),
        ],

        reward=0.5,

        termination=Termination(
            terminated=False,
            reason="",
        ),
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=9000,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=5,
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

        sock.connect(
            (args.host, args.port)
        )

        print(
            "Connected to Python server."
        )

        # ==================================================
        # HELLO
        # ==================================================

        hello_payload = (
            b"Hello from fake Unreal!"
        )

        sock.sendall(
            encode_message(
                MessageType.HELLO,
                hello_payload,
            )
        )

        print(
            "Sent HELLO"
        )

        # ==================================================
        # HELLO ACK
        # ==================================================

        print(
            "Waiting for HELLO_ACK..."
        )

        message_type, payload = (
            receive_message(sock)
        )

        print(
            f"Received message: "
            f"type={message_type}, "
            f"payload_size={len(payload)}"
        )

        print(
            f"Payload: {payload!r}"
        )

        if message_type != MessageType.HELLO_ACK:

            raise RuntimeError(
                "Expected HELLO_ACK"
            )

        print(
            "HELLO/HELLO_ACK handshake successful."
        )

        # ==================================================
        # OBSERVATION → ACTION LOOP
        # ==================================================

        episode_id = 1

        for step_id in range(args.steps):

            frame_id = 1000 + step_id

            # ----------------------------------------------
            # Create observation
            # ----------------------------------------------

            observation = make_observation(
                episode_id=episode_id,
                step_id=step_id,
                frame_id=frame_id,
            )

            payload = serialize_observation(
                observation
            )

            packet = encode_message(
                MessageType.OBSERVATION,
                payload,
            )

            # ----------------------------------------------
            # Send observation
            # ----------------------------------------------

            sock.sendall(packet)

            print()
            print(
                f"Sent OBSERVATION: "
                f"episode={episode_id}, "
                f"step={step_id}, "
                f"frame={frame_id}"
            )

            # ----------------------------------------------
            # Wait for action
            # ----------------------------------------------

            message_type, action_payload = (
                receive_message(sock)
            )

            if message_type != MessageType.ACTION:

                raise RuntimeError(
                    "Expected ACTION, "
                    f"received type={message_type}"
                )

            # ----------------------------------------------
            # Decode action
            # ----------------------------------------------

            action = deserialize_action(
                action_payload
            )

            print(
                "Received ACTION:"
            )

            print(
                f"  Episode : "
                f"{action.episode_id}"
            )

            print(
                f"  Step    : "
                f"{action.step_id}"
            )

            print(
                f"  Throttle: "
                f"{action.throttle}"
            )

            print(
                f"  Steering: "
                f"{action.steering}"
            )

            print(
                f"  Brake   : "
                f"{action.brake}"
            )

            # ----------------------------------------------
            # Validate response
            # ----------------------------------------------

            if action.episode_id != episode_id:

                raise RuntimeError(
                    "Episode ID mismatch: "
                    f"expected {episode_id}, "
                    f"received {action.episode_id}"
                )

            if action.step_id != step_id:

                raise RuntimeError(
                    "Step ID mismatch: "
                    f"expected {step_id}, "
                    f"received {action.step_id}"
                )

            if not -1.0 <= action.throttle <= 1.0:

                raise RuntimeError(
                    "Throttle outside [-1, 1]"
                )

            if not -1.0 <= action.steering <= 1.0:

                raise RuntimeError(
                    "Steering outside [-1, 1]"
                )

            print(
                "ACTION validation: PASS"
            )

            time.sleep(0.2)

        print()
        print(
            "=========================================="
        )

        print(
            "END-TO-END TEST PASSED"
        )

        print(
            f"Completed {args.steps} steps."
        )

        print(
            "Observation → Python → Agent → Action"
        )

        print(
            "=========================================="
        )

    finally:

        sock.close()

        print(
            "Fake Unreal stopped."
        )


if __name__ == "__main__":
    main()