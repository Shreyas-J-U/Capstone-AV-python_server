from typing import Any, Optional

# pyrefly: ignore [missing-import]
from ultralytics.trackers.byte_tracker import BYTETracker

from perception.safe_distance import SafeDistanceCalculator


class ByteTracker:
    """
    ByteTrack wrapper for Ultralytics 8.4.x with optional safe distance evaluation.

    Input:
        Ultralytics Results object.

    Output:
        List of dictionaries with tracking IDs and optional safe distance metrics.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
        safe_distance_calculator: Optional[SafeDistanceCalculator] = None,
    ):
        self.frame_id = 0
        self.frame_rate = frame_rate
        self.safe_distance_calculator = safe_distance_calculator

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
        ego_speed_ms: Optional[float] = None,
    ) -> list[dict]:
        """
        Update ByteTrack using YOLO results.

        If safe_distance_calculator is configured and ego_speed_ms is provided,
        each tracked object is enriched with safe distance metrics.
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

            obj = {
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

            image_width = None
            if hasattr(results, "orig_shape") and results.orig_shape:
                image_width = int(results.orig_shape[1])

            if self.safe_distance_calculator is not None and ego_speed_ms is not None:
                obj = self.safe_distance_calculator.enrich_detection(
                    obj,
                    ego_speed_ms=ego_speed_ms,
                    image_width=image_width,
                )

            tracked_objects.append(obj)

        return tracked_objects