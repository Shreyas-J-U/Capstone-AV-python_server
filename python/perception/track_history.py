from collections import defaultdict, deque
from typing import Dict, List, Tuple


class TrackHistory:
    """
    Maintains the recent movement history of tracked objects.

    ByteTrack gives us persistent IDs.
    This class converts those tracked detections into
    temporal position histories.

    Example:

        track_id = 3

        [
            (350.0, 400.0),
            (356.0, 397.0),
            (363.0, 393.0),
        ]
    """

    def __init__(
        self,
        max_history: int = 20,
    ):
        self.max_history = max_history

        # track_id -> deque[(x, y)]
        self.positions: Dict[
            int,
            deque
        ] = defaultdict(
            lambda: deque(
                maxlen=self.max_history
            )
        )

        # track_id -> object metadata
        self.metadata: Dict[
            int,
            dict
        ] = {}

    @staticmethod
    def calculate_center(
        bbox: List[float],
    ) -> Tuple[float, float]:
        """
        Calculate the center point of a bounding box.

        bbox:
            [x1, y1, x2, y2]

        Returns:
            (center_x, center_y)
        """

        if len(bbox) != 4:
            raise ValueError(
                "Bounding box must contain "
                "[x1, y1, x2, y2]."
            )

        x1, y1, x2, y2 = bbox

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        return center_x, center_y

    def update(
        self,
        tracked_objects: List[dict],
    ) -> None:
        """
        Add the latest ByteTrack detections.

        Expected object format:

        {
            "track_id": 3,
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.91,
            "bbox": [x1, y1, x2, y2]
        }
        """

        for obj in tracked_objects:

            track_id = int(
                obj["track_id"]
            )

            bbox = obj["bbox"]

            center = self.calculate_center(
                bbox
            )

            self.positions[
                track_id
            ].append(center)

            self.metadata[
                track_id
            ] = {
                "class_id": int(
                    obj["class_id"]
                ),
                "class_name": obj[
                    "class_name"
                ],
                "confidence": float(
                    obj["confidence"]
                ),
                "bbox": list(bbox),
            }

    def get_history(
        self,
        track_id: int,
    ) -> List[Tuple[float, float]]:
        """
        Return position history for one track.
        """

        return list(
            self.positions.get(
                track_id,
                []
            )
        )

    def get_metadata(
        self,
        track_id: int,
    ) -> dict:
        """
        Return metadata for one tracked object.
        """

        return self.metadata.get(
            track_id,
            {}
        )

    def get_all_tracks(self) -> Dict[int, dict]:
        """
        Return all currently known tracks.

        Example:

        {
            3: {
                "class_name": "car",
                "confidence": 0.91,
                "bbox": [...],
                "history": [...]
            }
        }
        """

        tracks = {}

        for track_id in self.positions:

            tracks[track_id] = {
                **self.metadata.get(
                    track_id,
                    {}
                ),
                "history": self.get_history(
                    track_id
                ),
            }

        return tracks

    def clear(self) -> None:
        """
        Clear all tracking history.
        """

        self.positions.clear()
        self.metadata.clear()