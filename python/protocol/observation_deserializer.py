import struct

from protocol.models import (
    Observation,
    ImageData,
    Sensor,
    Termination,
)

from protocol.constants import (
    SensorType,
    SensorFormat,
)


UINT64 = struct.Struct("<Q")
DOUBLE = struct.Struct("<d")
FLOAT32 = struct.Struct("<f")
UINT32 = struct.Struct("<I")
UINT16 = struct.Struct("<H")
UINT8 = struct.Struct("<B")


class BufferReader:

    def __init__(self, data: bytes):

        self.data = data
        self.offset = 0

    def read(self, size: int) -> bytes:

        if self.offset + size > len(self.data):

            raise ValueError(
                "Unexpected end of payload"
            )

        result = self.data[
            self.offset:
            self.offset + size
        ]

        self.offset += size

        return result

    def uint64(self):

        return UINT64.unpack(
            self.read(UINT64.size)
        )[0]

    def uint32(self):

        return UINT32.unpack(
            self.read(UINT32.size)
        )[0]

    def uint16(self):

        return UINT16.unpack(
            self.read(UINT16.size)
        )[0]

    def uint8(self):

        return UINT8.unpack(
            self.read(UINT8.size)
        )[0]

    def float32(self):

        return FLOAT32.unpack(
            self.read(FLOAT32.size)
        )[0]

    def double(self):

        return DOUBLE.unpack(
            self.read(DOUBLE.size)
        )[0]

    def remaining(self):

        return len(self.data) - self.offset

    
def deserialize_observation(
    data: bytes,
) -> Observation:

    reader = BufferReader(data)

    # ----------------------------------------
    # IDs
    # ----------------------------------------

    episode_id = reader.uint64()
    step_id = reader.uint64()
    frame_id = reader.uint64()

    simulation_time = reader.double()

    reward = reader.float32()

    # ----------------------------------------
    # Termination
    # ----------------------------------------

    terminated = reader.uint8() != 0

    reason_size = reader.uint32()

    reason = reader.read(
        reason_size
    ).decode(
        "utf-8"
    )

    termination = Termination(
        terminated=terminated,
        reason=reason,
    )

    # ----------------------------------------
    # Image
    # ----------------------------------------

    width = reader.uint32()
    height = reader.uint32()
    channels = reader.uint16()
    image_size = reader.uint32()

    image = None

    if image_size > 0:

        image_data = reader.read(
            image_size
        )

        image = ImageData(
            width=width,
            height=height,
            channels=channels,
            data=image_data,
        )

    # ----------------------------------------
    # Sensors
    # ----------------------------------------

    sensor_count = reader.uint16()

    sensors = []

    for _ in range(sensor_count):

        sensor_type = reader.uint16()

        sensor_format = reader.uint16()

        data_size = reader.uint32()

        raw_data = reader.read(
            data_size
        )

        sensors.append(
            Sensor(
                type=sensor_type,
                format=sensor_format,
                value=raw_data,
            )
        )

    # ----------------------------------------
    # Ensure entire packet was consumed
    # ----------------------------------------

    if reader.remaining() != 0:

        raise ValueError(
            f"Unused bytes at end of observation: "
            f"{reader.remaining()}"
        )

    return Observation(
        episode_id=episode_id,
        step_id=step_id,
        frame_id=frame_id,
        simulation_time=simulation_time,
        image=image,
        sensors=sensors,
        reward=reward,
        termination=termination,
    )