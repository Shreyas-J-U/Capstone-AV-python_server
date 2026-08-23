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


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Python server disconnected")
        data.extend(chunk)
    return bytes(data)


def receive_message(sock: socket.socket):
    header = recv_exact(sock, HEADER_SIZE)
    message_type, payload_size = decode_header(header)
    payload = recv_exact(sock, payload_size)
    return message_type, payload


def send_message(sock: socket.socket, message_type: MessageType, payload: bytes):
    packet = encode_message(message_type, payload)
    sock.sendall(packet)


def create_observation(
    episode_id: int,
    step_id: int,
    frame_id: int,
    terminated: bool = False,
    reason: str = "",
) -> Observation:
    speed = Sensor(
        type=SensorType.SPEED,
        format=SensorFormat.FLOAT32,
        value=struct.pack("<f", 10.0 + step_id * 1.5),
    )

    collision = Sensor(
        type=SensorType.COLLISION,
        format=SensorFormat.BOOL,
        value=struct.pack("<B", 1 if terminated else 0),
    )

    return Observation(
        episode_id=episode_id,
        step_id=step_id,
        frame_id=frame_id,
        simulation_time=step_id * 0.2,
        image=None,
        sensors=[speed, collision],
        reward=-10.0 if terminated else 0.5,
        termination=Termination(
            terminated=terminated,
            reason=reason,
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fake Unreal multi-episode client for testing RESET & lifecycle."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps-per-episode", type=int, default=5)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        print(f"[FakeUE] Connecting to Python server at {args.host}:{args.port}...")
        sock.connect((args.host, args.port))
        print("[FakeUE] Connected.")

        # ----------------------------------------------------
        # Handshake: Send HELLO, receive HELLO_ACK
        # ----------------------------------------------------
        send_message(sock, MessageType.HELLO, b"Hello from Fake Unreal multi-episode client!")
        print("[FakeUE] Sent HELLO. Waiting for HELLO_ACK...")

        msg_type, payload = receive_message(sock)
        if msg_type != MessageType.HELLO_ACK:
            raise RuntimeError(f"Expected HELLO_ACK, received {msg_type}")
        print(f"[FakeUE] Handshake OK. Server reply: {payload!r}")

        # ----------------------------------------------------
        # Multi-Episode Loop
        # ----------------------------------------------------
        for ep_index in range(1, args.episodes + 1):
            print(f"\n================ EPISODE {ep_index} START ================")
            print("[FakeUE] Waiting for RESET message from Python environment...")

            msg_type, reset_payload = receive_message(sock)
            if msg_type != MessageType.RESET:
                raise RuntimeError(f"Expected RESET message, received {msg_type}")

            episode_id = ep_index
            step_id = 0
            frame_id = 1000 * episode_id + step_id

            print(f"[FakeUE] Received RESET! Initializing episode_id={episode_id}, step_id={step_id}")

            # Send initial observation (step 0)
            initial_obs = create_observation(episode_id, step_id, frame_id, terminated=False)
            send_message(sock, MessageType.OBSERVATION, serialize_observation(initial_obs))
            print(f"[FakeUE] Sent Initial OBSERVATION (step=0, frame={frame_id})")

            # Step loop for episode
            for step in range(args.steps_per_episode):
                print(f"[FakeUE] Waiting for ACTION (expected ep={episode_id}, step={step_id})...")
                msg_type, action_bytes = receive_message(sock)
                if msg_type != MessageType.ACTION:
                    raise RuntimeError(f"Expected ACTION, received {msg_type}")

                action = deserialize_action(action_bytes)
                print(
                    f"[FakeUE] Received ACTION: ep={action.episode_id}, step={action.step_id}, "
                    f"throttle={action.throttle:.2f}, steering={action.steering:.2f}, brake={action.brake}"
                )

                # Validate episode & step match
                if action.episode_id != episode_id or action.step_id != step_id:
                    raise RuntimeError(
                        f"Action sequence mismatch! Expected ep={episode_id}, step={step_id}; "
                        f"got ep={action.episode_id}, step={action.step_id}"
                    )

                # Advance simulator step
                step_id += 1
                frame_id += 1
                is_last_step = (step == args.steps_per_episode - 1)
                reason = "Goal reached / Max steps" if is_last_step else ""

                obs = create_observation(
                    episode_id=episode_id,
                    step_id=step_id,
                    frame_id=frame_id,
                    terminated=is_last_step,
                    reason=reason,
                )

                send_message(sock, MessageType.OBSERVATION, serialize_observation(obs))
                print(
                    f"[FakeUE] Sent OBSERVATION: step={step_id}, frame={frame_id}, "
                    f"terminated={is_last_step}"
                )

                if is_last_step:
                    print(f"[FakeUE] Episode {episode_id} terminated: {reason}")
                    break

                time.sleep(0.05)

        print("\n==========================================")
        print("ALL EPISODES COMPLETED SUCCESSFULLY IN FAKE UNREAL!")
        print("==========================================")

    except Exception as e:
        print(f"[FakeUE] Error: {e}")
        raise
    finally:
        sock.close()
        print("[FakeUE] Disconnected.")


if __name__ == "__main__":
    main()
