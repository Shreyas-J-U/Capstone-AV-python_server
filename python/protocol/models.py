from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Termination:
    terminated: bool
    reason: str = ""


@dataclass
class ImageData:
    """
    Camera image contained in an Observation.

    width:
        Image width in pixels.

    height:
        Image height in pixels.

    channels:
        Number of image channels.

    data:
        Raw image bytes.
    """

    width: int
    height: int
    channels: int
    data: bytes


@dataclass
class Sensor:
    """
    Sensor data contained in an Observation.

    type:
        SensorType value.

    format:
        SensorFormat value.

    value:
        Raw serialized sensor bytes.
    """

    type: int
    format: int
    value: bytes


@dataclass
class Observation:
    episode_id: int
    step_id: int
    frame_id: int
    simulation_time: float

    image: Optional[ImageData]

    sensors: list[Sensor] = field(
        default_factory=list
    )

    reward: float = 0.0

    termination: Termination = field(
        default_factory=lambda: Termination(
            terminated=False
        )
    )


@dataclass
class Action:
    """
    Action sent from Python/RL to Unreal.

    Values:

        throttle: -1.0 .. 1.0
        steering: -1.0 .. 1.0
        brake: True / False
    """

    episode_id: int
    step_id: int

    throttle: float
    steering: float
    brake: bool