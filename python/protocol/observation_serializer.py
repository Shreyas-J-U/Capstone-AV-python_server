import struct

from protocol.models import (
    Observation,
    ImageData,
    Sensor,
    Termination,
)


UINT64 = struct.Struct("<Q")
DOUBLE = struct.Struct("<d")
FLOAT32 = struct.Struct("<f")
UINT32 = struct.Struct("<I")
UINT16 = struct.Struct("<H")
UINT8 = struct.Struct("<B")

def serialize_observation(
    observation: Observation,
) -> bytes:

    data = bytearray()

    # ----------------------------------------
    # Basic identifiers
    # ----------------------------------------

    data.extend(
        UINT64.pack(observation.episode_id)
    )

    data.extend(
        UINT64.pack(observation.step_id)
    )

    data.extend(
        UINT64.pack(observation.frame_id)
    )

    data.extend(
        DOUBLE.pack(observation.simulation_time)
    )

    data.extend(
        FLOAT32.pack(observation.reward)
    )

    # ----------------------------------------
    # Termination
    # ----------------------------------------

    data.extend(
        UINT8.pack(
            1 if observation.termination.terminated
            else 0
        )
    )

    reason = observation.termination.reason.encode(
        "utf-8"
    )

    data.extend(
        UINT32.pack(len(reason))
    )

    data.extend(reason)

    # ----------------------------------------
    # Image
    # ----------------------------------------

    if observation.image is None:

        data.extend(
            UINT32.pack(0)
        )

        data.extend(
            UINT32.pack(0)
        )

        data.extend(
            UINT16.pack(0)
        )

        data.extend(
            UINT32.pack(0)
        )

    else:

        image = observation.image

        data.extend(
            UINT32.pack(image.width)
        )

        data.extend(
            UINT32.pack(image.height)
        )

        data.extend(
            UINT16.pack(image.channels)
        )

        data.extend(
            UINT32.pack(len(image.data))
        )

        data.extend(image.data)

    # ----------------------------------------
    # Sensors
    # ----------------------------------------

    data.extend(
        UINT16.pack(len(observation.sensors))
    )

    for sensor in observation.sensors:

        data.extend(
            UINT16.pack(int(sensor.type))
        )

        data.extend(
            UINT16.pack(int(sensor.format))
        )

        sensor_data = sensor.value

        if isinstance(sensor_data, bytes):

            raw = sensor_data

        else:

            raise TypeError(
                "Sensor serialization currently "
                "expects bytes"
            )

        data.extend(
            UINT32.pack(len(raw))
        )

        data.extend(raw)

    return bytes(data)