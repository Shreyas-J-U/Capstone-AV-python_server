import struct

from protocol.models import Action


UINT64 = struct.Struct("<Q")
FLOAT32 = struct.Struct("<f")
UINT8 = struct.Struct("<B")

ACTION_SIZE = UINT64.size * 2 + FLOAT32.size * 2 + UINT8.size  # 25 bytes


def serialize_action(action: Action) -> bytes:

    data = bytearray()

    # Episode ID
    data.extend(
        UINT64.pack(action.episode_id)
    )

    # Step ID
    data.extend(
        UINT64.pack(action.step_id)
    )

    # Throttle
    data.extend(
        FLOAT32.pack(action.throttle)
    )

    # Steering
    data.extend(
        FLOAT32.pack(action.steering)
    )

    # Brake
    data.extend(
        UINT8.pack(
            1 if action.brake else 0
        )
    )

    return bytes(data)