from typing import Tuple, Dict, Any

from protocol.connection import ProtocolConnection
from protocol.constants import MessageType
from protocol.models import Observation, Action
from protocol.observation_deserializer import deserialize_observation
from protocol.action_serializer import serialize_action
from environment.validation import validate_observation, validate_action


class UERLEnvironment:
    """
    Reinforcement Learning Environment wrapper over TCP ProtocolConnection.

    Provides a standard Gym/Gymnasium style API:
      - reset() -> Observation
      - step(action: Action) -> (Observation, reward, terminated, info)
      - close()
    """

    def __init__(self, connection: ProtocolConnection):
        self.connection = connection
        self.current_episode_id: int = 0
        self.current_step_id: int = 0
        self.is_connected: bool = True
        self.is_episode_active: bool = False

    def perform_handshake(self) -> None:
        """
        Executes the initial HELLO / HELLO_ACK handshake with Unreal client.
        """
        message_type, payload = self.connection.receive_message()
        if message_type != MessageType.HELLO:
            raise ConnectionError(
                f"Handshake failed: expected HELLO ({MessageType.HELLO}), "
                f"received type={message_type}"
            )

        self.connection.send_message(
            MessageType.HELLO_ACK,
            b"Hello from Python RL Environment!",
        )

    def reset(self, reset_payload: bytes = b"") -> Observation:
        """
        Resets the environment for a new episode.

        Sends a RESET message to Unreal, increments episode_id, resets step_id to 0,
        and returns the initial Observation.
        """
        self.current_episode_id += 1
        self.current_step_id = 0

        # Send RESET message to Unreal
        self.connection.send_message(
            MessageType.RESET,
            reset_payload,
        )

        # Receive initial Observation for new episode
        message_type, payload = self.connection.receive_message()
        if message_type != MessageType.OBSERVATION:
            raise RuntimeError(
                f"Expected initial OBSERVATION after RESET, "
                f"received type={message_type}"
            )

        initial_obs = deserialize_observation(payload)

        # Validate initial observation
        validate_observation(
            initial_obs,
            expected_episode_id=self.current_episode_id,
            expected_step_id=0,
        )

        self.is_episode_active = True
        return initial_obs

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Executes one control step in the environment.

        1. Validates Action sequence and bounds
        2. Serializes and sends Action to Unreal
        3. Waits for next Observation from Unreal
        4. Validates Observation sequence
        5. Returns (observation, reward, terminated, info)
        """
        if not self.is_episode_active:
            raise RuntimeError(
                "Cannot call step() on an inactive episode. Call reset() first."
            )

        # 1. Validate outgoing action
        validate_action(
            action,
            expected_episode_id=self.current_episode_id,
            expected_step_id=self.current_step_id,
        )

        # 2. Serialize and send Action
        action_bytes = serialize_action(action)
        self.connection.send_message(
            MessageType.ACTION,
            action_bytes,
        )

        # Advance internal step counter
        self.current_step_id += 1

        # 3. Receive next Observation
        message_type, payload = self.connection.receive_message()
        if message_type != MessageType.OBSERVATION:
            raise RuntimeError(
                f"Expected OBSERVATION after ACTION, received type={message_type}"
            )

        obs = deserialize_observation(payload)

        # 4. Validate incoming observation
        validate_observation(
            obs,
            expected_episode_id=self.current_episode_id,
            expected_step_id=self.current_step_id,
        )

        terminated = obs.termination.terminated
        if terminated:
            self.is_episode_active = False

        info = {
            "reason": obs.termination.reason,
            "frame_id": obs.frame_id,
            "simulation_time": obs.simulation_time,
        }

        return obs, obs.reward, terminated, info

    def close(self) -> None:
        """
        Closes the underlying transport connection.
        """
        if self.is_connected:
            self.connection.close()
            self.is_connected = False
            self.is_episode_active = False