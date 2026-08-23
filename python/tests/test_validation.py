import unittest

from protocol.models import Observation, Action, Termination
from environment.validation import (
    validate_observation,
    validate_action,
    ValidationError,
    EpisodeMismatchError,
    StepMismatchError,
    InvalidActionError,
)


class TestValidation(unittest.TestCase):

    def test_validate_observation_valid(self):
        obs = Observation(
            episode_id=1,
            step_id=0,
            frame_id=100,
            simulation_time=0.0,
            image=None,
            sensors=[],
            reward=0.0,
            termination=Termination(terminated=False, reason=""),
        )
        # Should not raise
        validate_observation(obs, expected_episode_id=1, expected_step_id=0)

    def test_validate_observation_episode_mismatch(self):
        obs = Observation(
            episode_id=2,
            step_id=0,
            frame_id=100,
            simulation_time=0.0,
            image=None,
            sensors=[],
            reward=0.0,
            termination=Termination(terminated=False, reason=""),
        )
        with self.assertRaises(EpisodeMismatchError):
            validate_observation(obs, expected_episode_id=1, expected_step_id=0)

    def test_validate_observation_step_mismatch(self):
        obs = Observation(
            episode_id=1,
            step_id=5,
            frame_id=100,
            simulation_time=0.0,
            image=None,
            sensors=[],
            reward=0.0,
            termination=Termination(terminated=False, reason=""),
        )
        with self.assertRaises(StepMismatchError):
            validate_observation(obs, expected_episode_id=1, expected_step_id=0)

    def test_validate_action_valid(self):
        act = Action(
            episode_id=1,
            step_id=0,
            throttle=0.5,
            steering=-0.2,
            brake=False,
        )
        # Should not raise
        validate_action(act, expected_episode_id=1, expected_step_id=0)

    def test_validate_action_out_of_bounds_throttle(self):
        act = Action(
            episode_id=1,
            step_id=0,
            throttle=1.5,
            steering=0.0,
            brake=False,
        )
        with self.assertRaises(InvalidActionError):
            validate_action(act, expected_episode_id=1, expected_step_id=0)

    def test_validate_action_out_of_bounds_steering(self):
        act = Action(
            episode_id=1,
            step_id=0,
            throttle=0.5,
            steering=-2.0,
            brake=False,
        )
        with self.assertRaises(InvalidActionError):
            validate_action(act, expected_episode_id=1, expected_step_id=0)


if __name__ == "__main__":
    unittest.main()
