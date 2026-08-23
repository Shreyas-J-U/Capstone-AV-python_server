from protocol.models import Observation, Action


class ValidationError(Exception):
    """Base exception for protocol and state validation errors."""
    pass


class EpisodeMismatchError(ValidationError):
    """Raised when an incoming episode ID does not match the expected state."""
    pass


class StepMismatchError(ValidationError):
    """Raised when an incoming step ID does not match the expected state."""
    pass


class InvalidActionError(ValidationError):
    """Raised when an action contains invalid out-of-bound control values."""
    pass


def validate_observation(
    observation: Observation,
    expected_episode_id: int,
    expected_step_id: int,
) -> None:
    """
    Validates observation sequence and simulation bounds.
    """
    if observation.episode_id != expected_episode_id:
        raise EpisodeMismatchError(
            f"Episode ID mismatch in Observation: "
            f"expected {expected_episode_id}, got {observation.episode_id}"
        )

    if observation.step_id != expected_step_id:
        raise StepMismatchError(
            f"Step ID mismatch in Observation: "
            f"expected {expected_step_id}, got {observation.step_id}"
        )

    if observation.simulation_time < 0.0:
        raise ValidationError(
            f"Invalid simulation_time in Observation: {observation.simulation_time}"
        )


def validate_action(
    action: Action,
    expected_episode_id: int,
    expected_step_id: int,
) -> None:
    """
    Validates action sequence and control ranges before sending to Unreal.
    """
    if action.episode_id != expected_episode_id:
        raise EpisodeMismatchError(
            f"Episode ID mismatch in Action: "
            f"expected {expected_episode_id}, got {action.episode_id}"
        )

    if action.step_id != expected_step_id:
        raise StepMismatchError(
            f"Step ID mismatch in Action: "
            f"expected {expected_step_id}, got {action.step_id}"
        )

    if not (-1.0 <= action.throttle <= 1.0):
        raise InvalidActionError(
            f"Throttle out of bounds [-1.0, 1.0]: {action.throttle}"
        )

    if not (-1.0 <= action.steering <= 1.0):
        raise InvalidActionError(
            f"Steering out of bounds [-1.0, 1.0]: {action.steering}"
        )

    if not isinstance(action.brake, bool):
        raise InvalidActionError(
            f"Brake must be a boolean, got type {type(action.brake).__name__}"
        )
