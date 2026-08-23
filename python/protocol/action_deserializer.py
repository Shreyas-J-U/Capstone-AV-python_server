import struct

from protocol.models import Action


UINT64 = struct.Struct("<Q")
FLOAT32 = struct.Struct("<f")
UINT8 = struct.Struct("<B")


EXPECTED_SIZE = (
    UINT64.size
    + UINT64.size
    + FLOAT32.size
    + FLOAT32.size
    + UINT8.size
)


def deserialize_action(
    data: bytes,
) -> Action:

    if len(data) != EXPECTED_SIZE:

        raise ValueError(
            f"Invalid action size: "
            f"{len(data)}, "
            f"expected {EXPECTED_SIZE}"
        )

    offset = 0

    episode_id = UINT64.unpack_from(
        data,
        offset,
    )[0]

    offset += UINT64.size

    step_id = UINT64.unpack_from(
        data,
        offset,
    )[0]

    offset += UINT64.size

    throttle = FLOAT32.unpack_from(
        data,
        offset,
    )[0]

    offset += FLOAT32.size

    steering = FLOAT32.unpack_from(
        data,
        offset,
    )[0]

    offset += FLOAT32.size

    brake = (
        UINT8.unpack_from(
            data,
            offset,
        )[0]
        != 0
    )

    return Action(
        episode_id=episode_id,
        step_id=step_id,
        throttle=throttle,
        steering=steering,
        brake=brake,
    )