import struct

from protocol.constants import (
    SensorType,
    SensorFormat,
)

from protocol.models import (
    Observation,
    Sensor,
    Termination,
)

from protocol.observation_serializer import (
    serialize_observation,
)

from protocol.observation_deserializer import (
    deserialize_observation,
)


def main():

    observation = Observation(

        episode_id=1,

        step_id=42,

        frame_id=1234,

        simulation_time=8.4,

        image=None,

        sensors=[

            Sensor(
                type=SensorType.SPEED,
                format=SensorFormat.FLOAT32,
                value=struct.pack(
                    "<f",
                    12.5,
                ),
            ),

            Sensor(
                type=SensorType.COLLISION,
                format=SensorFormat.BOOL,
                value=struct.pack(
                    "<B",
                    0,
                ),
            ),

        ],

        reward=0.75,

        termination=Termination(
            terminated=False,
            reason="",
        ),
    )

    # Serialize

    encoded = serialize_observation(
        observation
    )

    print(
        f"Serialized size: {len(encoded)}"
    )

    print(
        f"Bytes: {encoded.hex(' ')}"
    )

    # Deserialize

    decoded = deserialize_observation(
        encoded
    )

    print()
    print("Decoded observation:")
    print(
        "Episode:",
        decoded.episode_id
    )

    print(
        "Step:",
        decoded.step_id
    )

    print(
        "Frame:",
        decoded.frame_id
    )

    print(
        "Simulation time:",
        decoded.simulation_time
    )

    print(
        "Reward:",
        decoded.reward
    )

    print(
        "Terminated:",
        decoded.termination.terminated
    )

    print(
        "Sensors:",
        len(decoded.sensors)
    )


if __name__ == "__main__":
    main()