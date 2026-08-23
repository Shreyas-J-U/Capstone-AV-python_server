import struct

from environment.action import Action


ACTION_STRUCT = struct.Struct("<QQffB")


def encode_action(action: Action) -> bytes:

    action.validate()

    return ACTION_STRUCT.pack(
        action.episode_id,
        action.step_id,
        action.throttle,
        action.steering,
        1 if action.brake else 0,
    )