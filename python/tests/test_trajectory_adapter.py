from prediction.trajectory_adapter import (
    track_history_to_trajectories,
)


def main():

    tracks = {
        12: {
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.91,
            "bbox": [
                100,
                200,
                150,
                300,
            ],
            "history": [
                (125.0, 250.0),
                (128.0, 253.0),
                (132.0, 257.0),
                (137.0, 262.0),
                (143.0, 268.0),
            ],
        }
    }

    trajectories = (
        track_history_to_trajectories(
            tracks,
            start_timestamp=0.0,
            dt=0.1,
        )
    )

    assert len(trajectories) == 1

    trajectory = trajectories[0]

    assert trajectory.track_id == 12
    assert trajectory.class_name == "person"

    assert trajectory.length == 5

    assert trajectory.positions[0] == (
        125.0,
        250.0,
    )

    assert trajectory.positions[-1] == (
        143.0,
        268.0,
    )

    assert trajectory.points[0].timestamp == 0.0
    assert trajectory.points[-1].timestamp == 0.4

    print()
    print("=" * 60)
    print("TRAJECTORY ADAPTER TEST")
    print("=" * 60)

    print()
    print(
        f"Track ID : {trajectory.track_id}"
    )

    print(
        f"Class    : {trajectory.class_name}"
    )

    print(
        f"Points   : {trajectory.length}"
    )

    print(
        f"Positions: {trajectory.positions}"
    )

    print(
        f"Times    : "
        f"{[p.timestamp for p in trajectory.points]}"
    )

    print()
    print("=" * 60)
    print("TRAJECTORY ADAPTER TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()