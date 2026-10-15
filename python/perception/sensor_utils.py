import math
import struct
from typing import Any, List, Optional

from protocol.constants import SensorType, SensorFormat
from protocol.models import Observation, Sensor


def extract_ego_speed(observation_or_sensors: Any) -> float:
    """
    Extract ego vehicle speed (in m/s) from an Observation or a list of Sensor objects.

    Supported formats:
    - SensorType.SPEED with SensorFormat.FLOAT32
    - SensorType.VELOCITY with SensorFormat.VECTOR3 (computes magnitude ||v||)
    - Dictionary with 'speed' or 'velocity' key
    - Float/int directly

    Returns:
        float: Speed in m/s (non-negative). Defaults to 0.0 if not found.
    """
    if observation_or_sensors is None:
        return 0.0

    if isinstance(observation_or_sensors, (int, float)):
        return max(0.0, float(observation_or_sensors))

    sensors: List[Sensor] = []

    if isinstance(observation_or_sensors, Observation):
        sensors = observation_or_sensors.sensors
    elif isinstance(observation_or_sensors, list):
        sensors = observation_or_sensors
    elif isinstance(observation_or_sensors, dict):
        if "speed" in observation_or_sensors:
            return max(0.0, float(observation_or_sensors["speed"]))
        if "sensors" in observation_or_sensors and isinstance(observation_or_sensors["sensors"], list):
            sensors = observation_or_sensors["sensors"]

    for sensor in sensors:
        if not hasattr(sensor, "type") or not hasattr(sensor, "format") or not hasattr(sensor, "value"):
            continue

        # Direct SPEED sensor
        if sensor.type == SensorType.SPEED:
            if sensor.format == SensorFormat.FLOAT32 and len(sensor.value) >= 4:
                speed_val = struct.unpack("<f", sensor.value[:4])[0]
                return max(0.0, float(speed_val))

        # VELOCITY vector (3D vector)
        elif sensor.type == SensorType.VELOCITY:
            if sensor.format == SensorFormat.VECTOR3 and len(sensor.value) >= 12:
                vx, vy, vz = struct.unpack("<fff", sensor.value[:12])
                speed_val = math.sqrt(vx * vx + vy * vy + vz * vz)
                return max(0.0, float(speed_val))

    return 0.0
