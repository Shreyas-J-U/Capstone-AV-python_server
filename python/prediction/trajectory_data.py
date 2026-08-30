from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TrajectoryPoint:
    """
    A single observed position of a tracked object.

    Coordinates are currently in image/pixel space.
    """

    x: float
    y: float
    timestamp: float


@dataclass
class ObjectTrajectory:
    """
    Temporal trajectory of one tracked object.

    This is the interface between:
        ByteTrack + TrackHistory
                    ↓
             Trajectory Prediction
    """

    track_id: int
    class_id: int
    class_name: str

    points: List[TrajectoryPoint] = field(
        default_factory=list
    )

    @property
    def length(self) -> int:
        """Number of observed trajectory points."""
        return len(self.points)

    @property
    def positions(self) -> List[Tuple[float, float]]:
        """Return only (x, y) positions."""
        return [
            (point.x, point.y)
            for point in self.points
        ]

    def add_point(
        self,
        x: float,
        y: float,
        timestamp: float,
    ) -> None:
        """Add one observation to the trajectory."""

        self.points.append(
            TrajectoryPoint(
                x=float(x),
                y=float(y),
                timestamp=float(timestamp),
            )
        )

    def is_sufficient(
        self,
        minimum_points: int,
    ) -> bool:
        """
        Check whether enough observations exist
        for trajectory prediction.
        """

        return self.length >= minimum_points