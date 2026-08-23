import argparse

from transport.tcp_server import TCPServer
from protocol.connection import ProtocolConnection
from protocol.constants import MessageType

from protocol.observation_deserializer import (
    deserialize_observation,
)

from protocol.action_serializer import (
    serialize_action,
)

from agents.test_agent import MyAgent


def main():

    parser = argparse.ArgumentParser(
        description="RL TCP server for Unreal"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    server = TCPServer(
        host=args.host,
        port=args.port,
    )

    try:

        # ==================================================
        # START TCP SERVER
        # ==================================================

        server.start()

        server.accept()

        print(
            "Unreal connection established."
        )

        connection = ProtocolConnection(
            server
        )

        # ==================================================
        # CREATE TEST AGENT
        # ==================================================

        agent = MyAgent()

        # ==================================================
        # HELLO
        # ==================================================

        print(
            "Waiting for HELLO..."
        )

        message_type, payload = (
            connection.receive_message()
        )

        if message_type != MessageType.HELLO:

            raise ConnectionError(
                "Expected HELLO message, "
                f"received type={message_type}"
            )

        print(
            f"Received HELLO "
            f"({len(payload)} bytes)"
        )

        print(
            f"HELLO payload: {payload!r}"
        )

        # ==================================================
        # HELLO ACK
        # ==================================================

        connection.send_message(
            MessageType.HELLO_ACK,
            b"Hello from Python!",
        )

        print(
            "Sent HELLO_ACK"
        )

        print()
        print(
            "Handshake successful."
        )

        # ==================================================
        # OBSERVATION → AGENT → ACTION LOOP
        # ==================================================

        print(
            "Waiting for observations..."
        )

        while True:

            message_type, payload = (
                connection.receive_message()
            )

            # ----------------------------------------------
            # OBSERVATION
            # ----------------------------------------------

            if message_type == MessageType.OBSERVATION:

                print()
                print(
                    f"Received OBSERVATION "
                    f"({len(payload)} bytes)"
                )

                # Deserialize Unreal payload
                observation = (
                    deserialize_observation(
                        payload
                    )
                )

                # ------------------------------------------
                # Display observation
                # ------------------------------------------

                print()
                print(
                    "========== OBSERVATION =========="
                )

                print(
                    f"Episode ID       : "
                    f"{observation.episode_id}"
                )

                print(
                    f"Step ID          : "
                    f"{observation.step_id}"
                )

                print(
                    f"Frame ID         : "
                    f"{observation.frame_id}"
                )

                print(
                    f"Simulation Time  : "
                    f"{observation.simulation_time}"
                )

                print(
                    f"Reward           : "
                    f"{observation.reward}"
                )

                print(
                    f"Terminated       : "
                    f"{observation.termination.terminated}"
                )

                print(
                    f"Termination Reason: "
                    f"{observation.termination.reason}"
                )

                print(
                    f"Image            : "
                    f"{'present' if observation.image else 'none'}"
                )

                print(
                    f"Sensor Count     : "
                    f"{len(observation.sensors)}"
                )

                for index, sensor in enumerate(
                    observation.sensors
                ):

                    print(
                        f"  Sensor {index}: "
                        f"type={sensor.type}, "
                        f"format={sensor.format}, "
                        f"data_size={len(sensor.value)}"
                    )

                print(
                    "================================="
                )

                # ------------------------------------------
                # Agent inference
                # ------------------------------------------

                action = agent.act(
                    observation
                )

                # ------------------------------------------
                # Serialize action
                # ------------------------------------------

                action_payload = (
                    serialize_action(
                        action
                    )
                )

                # ------------------------------------------
                # Send action to Unreal
                # ------------------------------------------

                connection.send_message(
                    MessageType.ACTION,
                    action_payload,
                )

                print()
                print(
                    f"Sent ACTION "
                    f"(step={action.step_id}, "
                    f"{len(action_payload)} bytes)"
                )

            # ----------------------------------------------
            # Unexpected message
            # ----------------------------------------------

            else:

                print(
                    f"Received unexpected "
                    f"message type={message_type}"
                )

    except KeyboardInterrupt:

        print(
            "\nStopping server..."
        )

    except ConnectionError as e:

        print(
            f"Connection error: {e}"
        )

    except Exception as e:

        print(
            f"Unexpected error: "
            f"{type(e).__name__}: {e}"
        )

    finally:

        server.close()

        print(
            "Server closed."
        )


if __name__ == "__main__":
    main()