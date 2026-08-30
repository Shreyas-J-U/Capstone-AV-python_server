from typing import Dict, List

from prediction.trajectory_data import (
    ObjectTrajectory,
)


def track_history_to_trajectories(
    tracks: Dict[int, dict],
    start_timestamp: float = 0.0,
    dt: float = 0.1,
) -> List[ObjectTrajectory]:
    """
    Convert TrackHistory output into trajectory objects.

    Parameters
    ----------
    tracks:
        Output of TrackHistory.get_all_tracks().

    start_timestamp:
        Timestamp assigned to the first observation.

    dt:
        Time difference between consecutive observations.

    Returns
    -------
    List[ObjectTrajectory]
    """

    trajectories = []

    for track_id, data in tracks.items():

        trajectory = ObjectTrajectory(
            track_id=int(track_id),
            class_id=int(
                data.get("class_id", -1)
            ),
            class_name=str(
                data.get("class_name", "unknown")
            ),
        )

        history = data.get(
            "history",
            []
        )

        for index, position in enumerate(history):

            x, y = position

            timestamp = (
                start_timestamp
                + index * dt
            )

            trajectory.add_point(
                x=x,
                y=y,
                timestamp=timestamp,
            )

        trajectories.append(
            trajectory
        )

    return trajectories