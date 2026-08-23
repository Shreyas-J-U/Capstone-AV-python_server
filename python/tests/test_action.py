from protocol.models import Action

from protocol.action_serializer import (
    serialize_action,
    ACTION_SIZE,
)

from protocol.action_deserializer import (
    deserialize_action,
)


def main():

    action = Action(
        episode_id=1,
        step_id=42,
        throttle=0.5,
        steering=-0.25,
        brake=False,
    )

    # -----------------------------------------
    # Serialize
    # -----------------------------------------

    data = serialize_action(
        action
    )

    print(
        f"Serialized size: {len(data)}"
    )

    print(
        f"Expected size: {ACTION_SIZE}"
    )

    print(
        f"Bytes: {data.hex(' ')}"
    )

    # -----------------------------------------
    # Deserialize
    # -----------------------------------------

    decoded = deserialize_action(
        data
    )

    print()
    print(
        "Decoded action:"
    )

    print(
        f"Episode: {decoded.episode_id}"
    )

    print(
        f"Step: {decoded.step_id}"
    )

    print(
        f"Throttle: {decoded.throttle}"
    )

    print(
        f"Steering: {decoded.steering}"
    )

    print(
        f"Brake: {decoded.brake}"
    )

    # -----------------------------------------
    # Assertions
    # -----------------------------------------

    assert decoded.episode_id == 1

    assert decoded.step_id == 42

    assert abs(
        decoded.throttle - 0.5
    ) < 0.000001

    assert abs(
        decoded.steering - (-0.25)
    ) < 0.000001

    assert decoded.brake is False

    assert len(data) == ACTION_SIZE

    print()
    print(
        "ACTION SERIALIZATION TEST PASSED"
    )


if __name__ == "__main__":
    main()