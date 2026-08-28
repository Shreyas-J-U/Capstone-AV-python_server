from typing import Any

# pyrefly: ignore [missing-import]
from ultralytics.trackers.byte_tracker import BYTETracker


class ByteTracker:
    """
    ByteTrack wrapper for Ultralytics 8.4.x.

    Input:
        Ultralytics Results object.

    Output:
        List of dictionaries:

        {
            "track_id": 1,
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.91,
            "bbox": [x1, y1, x2, y2]
        }
    """

    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
    ):
        self.frame_id = 0
        self.frame_rate = frame_rate

        # --------------------------------------------------
        # Ultralytics 8.4.x tracker configuration
        # --------------------------------------------------

        tracker_args = type(
            "TrackerArgs",
            (),
            {
                "track_high_thresh": track_high_thresh,
                "track_low_thresh": track_low_thresh,
                "new_track_thresh": new_track_thresh,
                "track_buffer": track_buffer,
                "match_thresh": match_thresh,
                "fuse_score": True,
            },
        )()

        # Ultralytics 8.4.129:
        #
        # BYTETracker.__init__(self, args)
        #
        # There is NO frame_rate argument here.
        self.tracker = BYTETracker(
            args=tracker_args
        )

    def update(
        self,
        results: Any,
    ) -> list[dict]:
        """
        Update ByteTrack using YOLO results.

        The important detail for Ultralytics 8.4.129 is:

            results.boxes

        is passed to BYTETracker.update().

        Not:

            results
        """

        self.frame_id += 1

        # --------------------------------------------------
        # Validate YOLO Results
        # --------------------------------------------------

        if results is None:
            return []

        boxes = results.boxes

        if boxes is None:
            return []

        if len(boxes) == 0:
            return []

        # --------------------------------------------------
        # ByteTrack
        #
        # Ultralytics 8.4.129 expects an object with:
        #
        #   .conf
        #   .cls
        #   .xyxy
        #
        # Results.boxes provides these attributes.
        # --------------------------------------------------

        tracker_results = self.tracker.update(
            boxes
        )

        if tracker_results is None:
            return []

        if len(tracker_results) == 0:
            return []

        tracked_objects = []

        # --------------------------------------------------
        # Convert ByteTrack output into our own stable
        # project format.
        # --------------------------------------------------

        for track in tracker_results:

            x1 = float(track[0])
            y1 = float(track[1])
            x2 = float(track[2])
            y2 = float(track[3])

            track_id = int(track[4])
            confidence = float(track[5])
            class_id = int(track[6])

            class_name = results.names[class_id]

            tracked_objects.append(
                {
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                }
            )

        return tracked_objects