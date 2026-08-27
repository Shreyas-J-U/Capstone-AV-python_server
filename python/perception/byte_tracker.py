from typing import Any
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from ultralytics.trackers.byte_tracker import (
    BYTETracker,
)
# pyrefly: ignore [missing-import]
from ultralytics.engine.results import (
    Boxes,
)


class ByteTracker:
    """
    ByteTrack wrapper.

    Receives YOLO detections and assigns persistent
    track IDs across consecutive frames.
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

        # Ultralytics tracker configuration.
        self.tracker = BYTETracker(
            args=type(
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
            )(),
            frame_rate=frame_rate,
        )

    def update(
        self,
        result: Any,
    ) -> list[dict]:

        self.frame_id += 1

        if result.boxes is None:
            return []

        boxes = result.boxes

        if len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confidence = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()

        # ByteTrack expects:
        #
        # [x1, y1, x2, y2, confidence, class_id]
        detections = np.column_stack(
            (
                xyxy,
                confidence,
                class_ids,
            )
        )

        tracker_results = self.tracker.update(
            detections,
            (result.orig_shape[0], result.orig_shape[1]),
        )

        tracked_objects = []

        for track in tracker_results:

            # Ultralytics BYTETracker returns:
            #
            # x1, y1, x2, y2, track_id,
            # score, class_id, index

            x1 = float(track[0])
            y1 = float(track[1])
            x2 = float(track[2])
            y2 = float(track[3])

            track_id = int(track[4])
            score = float(track[5])
            class_id = int(track[6])

            class_name = result.names[class_id]

            tracked_objects.append(
                {
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": score,
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                }
            )

        return tracked_objects