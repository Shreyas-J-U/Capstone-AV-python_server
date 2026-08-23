import argparse
import socket
import struct
import time

from protocol.constants import (
    MessageType,
    SensorType,
    SensorFormat,
)

from protocol.framing import (
    encode_message,
    decode_header,
    HEADER_SIZE,
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


def recv_exact(
    sock: socket.socket,
    size: int,
) -> bytes:

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


def receive_message(
    sock: socket.socket,
):

    # ----------------------------------------
    # Receive fixed-size protocol header
    # ----------------------------------------

    header = recv_exact(
        sock,
        HEADER_SIZE,
    )

    # ----------------------------------------
    # Decode header
    # ----------------------------------------

    message_type, payload_size = (
        decode_header(header)
    )

    # ----------------------------------------
    # Receive payload
    # ----------------------------------------

    payload = recv_exact(
        sock,
        payload_size,
    )

    return message_type, payload


def send_message(
    sock: socket.socket,
    message_type,
    payload: bytes,
):

    packet = encode_message(
        message_type,
        payload,
    )

    sock.sendall(packet)


def create_observation(
    episode_id: int,
    step_id: int,
    frame_id: int,
) -> Observation:

    speed = Sensor(
        type=SensorType.SPEED,
        format=SensorFormat.FLOAT32,
        value=struct.pack(
            "<f",
            10.0 + step_id,
        ),
    )

    collision = Sensor(
        type=SensorType.COLLISION,
        format=SensorFormat.BOOL,
        value=struct.pack(
            "<B",
            0,
        ),
    )

    return Observation(

        episode_id=episode_id,

        step_id=step_id,

        frame_id=frame_id,

        simulation_time=step_id * 0.2,

        image=None,

        sensors=[
            speed,
            collision,
        ],

        reward=0.5,

        termination=Termination(
            terminated=False,
            reason="",
        ),
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Fake Unreal client for "
            "Python RL pipeline testing"
        )
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=9000,
    )

    args = parser.parse_args()

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    )

    try:

        # ==================================================
        # Connect
        # ==================================================

        print(
            f"Connecting to Python server "
            f"at {args.host}:{args.port}"
        )

        sock.connect(
            (args.host, args.port)
        )

        print(
            "Connected to Python server."
        )

        # ==================================================
        # HELLO
        # ==================================================

        send_message(
            sock,
            MessageType.HELLO,
            b"Hello from fake Unreal!",
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

        if message_type != MessageType.HELLO_ACK:

            raise RuntimeError(
                "Expected HELLO_ACK"
            )

        print(
            f"Payload: {payload!r}"
        )

        print(
            "HELLO/HELLO_ACK handshake successful."
        )

        # Wait for RESET from UERLEnvironment before sending initial observation
        message_type, payload = receive_message(sock)
        if message_type == MessageType.RESET:
            print("Received RESET message from Python server.")
        else:
            raise RuntimeError(f"Expected RESET message, received type={message_type}")

        # ==================================================
        # Observation -> Action loop
        # ==================================================

        episode_id = 1

        number_of_steps = 5

        for step_id in range(
            number_of_steps
        ):

            frame_id = 1000 + step_id

            # ----------------------------------------------
            # Create observation
            # ----------------------------------------------

            observation = create_observation(
                episode_id=episode_id,
                step_id=step_id,
                frame_id=frame_id,
            )

            observation_payload = (
                serialize_observation(
                    observation
                )
            )

            # ----------------------------------------------
            # Send observation
            # ----------------------------------------------

            send_message(
                sock,
                MessageType.OBSERVATION,
                observation_payload,
            )

            print()
            print(
                f"Sent OBSERVATION:"
                f" step={step_id},"
                f" frame={frame_id},"
                f" payload={len(observation_payload)} bytes"
            )

            # ----------------------------------------------
            # Wait for Python action
            # ----------------------------------------------

            print(
                "Waiting for ACTION..."
            )

            message_type, payload = (
                receive_message(sock)
            )

            if message_type != MessageType.ACTION:

                raise RuntimeError(
                    f"Expected ACTION, "
                    f"received type={message_type}"
                )

            # ----------------------------------------------
            # Deserialize action
            # ----------------------------------------------

            action = (
                deserialize_action(
                    payload
                )
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
            # Validate EpisodeID
            # ----------------------------------------------

            if action.episode_id != episode_id:

                raise RuntimeError(
                    "Episode ID mismatch: "
                    f"expected {episode_id}, "
                    f"received {action.episode_id}"
                )

            # ----------------------------------------------
            # Validate StepID
            # ----------------------------------------------

            if action.step_id != step_id:

                raise RuntimeError(
                    "Step ID mismatch: "
                    f"expected {step_id}, "
                    f"received {action.step_id}"
                )

            # ----------------------------------------------
            # Validate action ranges
            # ----------------------------------------------

            if not (
                -1.0
                <= action.throttle
                <= 1.0
            ):

                raise RuntimeError(
                    f"Invalid throttle: "
                    f"{action.throttle}"
                )

            if not (
                -1.0
                <= action.steering
                <= 1.0
            ):

                raise RuntimeError(
                    f"Invalid steering: "
                    f"{action.steering}"
                )

            if not isinstance(
                action.brake,
                bool,
            ):

                raise RuntimeError(
                    "Brake must be bool"
                )

            print(
                "ACTION validation passed."
            )

            # ----------------------------------------------
            # Simulate Unreal doing work
            # ----------------------------------------------

            time.sleep(0.2)

        # ==================================================
        # Success
        # ==================================================

        print()
        print(
            "========================================"
        )

        print(
            "FULL TCP PIPELINE TEST PASSED"
        )

        print(
            "========================================"
        )

    except ConnectionRefusedError:

        print(
            "Connection refused."
        )

        print(
            "Make sure Python server is running first."
        )

    except ConnectionError as e:

        print(
            f"Connection error: {e}"
        )

    finally:

        sock.close()

        print(
            "Fake Unreal stopped."
        )


if __name__ == "__main__":
    main()