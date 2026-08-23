from dataclasses import dataclass
from typing import Any


@dataclass
class Termination:
    terminated: bool
    reason: str = ""


@dataclass
class Observation:
    episode_id: int
    step_id: int
    frame_id: int
    simulation_time: float

    image: Any
    sensors: dict

    reward: float
    termination: Termination

    def get_sensor(self, name):
        return self.sensors.get(name)