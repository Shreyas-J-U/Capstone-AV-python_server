from python import transport
import argparse
import socket

from protocol.constants import MessageType

from protocol.framing import (
    encode_message,
    decode_header,
)

from protocol.connection import (
    ProtocolConnection,
)

from transport.tcp_socket import (
    TCPSocketTransport,
)

from protocol.observation_serializer import (
    serialize_observation,
)

from protocol.observation_deserializer import (
    deserialize_observation,
)

from protocol.action_deserializer import (
    deserialize_action,
)


def create_test_observation(
    episode_id,
    step_id,
    frame_id,
):

    from protocol.models import (
        Observation,
        Sensor,
        Termination,
    )

    sensors = [
        Sensor(
            type=1,
            format=1,
            value=(42.0).hex().encode(),
        ),

        Sensor(
            type=2,
            format=2,
            value=b"\x00",
        ),
    ]

    return Observation(
        episode_id=episode_id,
        step_id=step_id,
        frame_id=frame_id,
        simulation_time=step_id * 0.2,
        image=None,
        sensors=sensors,
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

        # =================================================
        # Protocol connection
        # =================================================

        transport = TCPSocketTransport(
            sock
        )

        connection = ProtocolConnection(
            transport
        )

        # =================================================
        # HELLO
        # =================================================

        connection.send_message(
            MessageType.HELLO,
            b"Hello from fake Unreal!",
        )

        print(
            "Sent HELLO"
        )

        # =================================================
        # HELLO ACK
        # =================================================

        print(
            "Waiting for HELLO_ACK..."
        )

        message_type, payload = (
            connection.receive_message()
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

        # =================================================
        # Observation → Action loop
        # =================================================

        episode_id = 1

        for step_id in range(
            args.steps
        ):

            frame_id = 1000 + step_id

            observation = (
                create_test_observation(
                    episode_id=episode_id,
                    step_id=step_id,
                    frame_id=frame_id,
                )
            )

            observation_payload = (
                serialize_observation(
                    observation
                )
            )

            # ---------------------------------------------
            # Send observation
            # ---------------------------------------------

            connection.send_message(
                MessageType.OBSERVATION,
                observation_payload,
            )

            print()
            print(
                f"Sent OBSERVATION: "
                f"episode={episode_id}, "
                f"step={step_id}, "
                f"frame={frame_id}"
            )

            # ---------------------------------------------
            # Wait for action
            # ---------------------------------------------

            print(
                "Waiting for ACTION..."
            )

            message_type, payload = (
                connection.receive_message()
            )

            if message_type != MessageType.ACTION:

                raise RuntimeError(
                    "Expected ACTION, "
                    f"received type={message_type}"
                )

            # ---------------------------------------------
            # Decode action
            # ---------------------------------------------

            action = (
                deserialize_action(
                    payload
                )
            )

            print()
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

            # ---------------------------------------------
            # Validate action corresponds to observation
            # ---------------------------------------------

            if (
                action.episode_id
                != episode_id
            ):

                raise RuntimeError(
                    "Episode ID mismatch: "
                    f"expected {episode_id}, "
                    f"got {action.episode_id}"
                )

            if (
                action.step_id
                != step_id
            ):

                raise RuntimeError(
                    "Step ID mismatch: "
                    f"expected {step_id}, "
                    f"got {action.step_id}"
                )

        print()
        print(
            "===================================="
        )

        print(
            "FULL OBSERVATION/ACTION TEST PASSED"
        )

        print(
            "===================================="
        )

    except ConnectionRefusedError:

        print(
            "Connection refused. "
            "Start Python server first."
        )

    except Exception as e:

        print()
        print(
            f"Fake Unreal error: "
            f"{type(e).__name__}: {e}"
        )

    finally:

        sock.close()

        print(
            "Fake Unreal stopped."
        )


if __name__ == "__main__":
    main()