from dataclasses import dataclass


@dataclass
class Action:
    episode_id: int
    step_id: int

    throttle: float
    steering: float
    brake: bool

    def validate(self):

        if not -1.0 <= self.throttle <= 1.0:
            raise ValueError(
                f"Invalid throttle: {self.throttle}"
            )

        if not -1.0 <= self.steering <= 1.0:
            raise ValueError(
                f"Invalid steering: {self.steering}"
            )

        if not isinstance(self.brake, bool):
            raise TypeError(
                "Brake must be bool"
            )