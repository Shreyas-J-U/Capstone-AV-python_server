from pathlib import Path

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

from perception.byte_tracker import ByteTracker
from perception.track_history import TrackHistory
from prediction.trajectory_adapter import (
    track_history_to_trajectories,
)


FRAMES_DIR = (
    Path(__file__).parent
    / "assets"
    / "sampled_frames"
)

MODEL_NAME = "yolov8n.pt"

# Number of historical positions required before
# an object becomes a trajectory candidate.
MIN_HISTORY = 5

# Time between sampled frames.
#
# IMPORTANT:
# This is currently an assumed value for the test.
# We will later replace this with the actual
# Unreal simulation timestamp.
DT = 0.1


def main():

    print()
    print("=" * 70)
    print("REAL TRAJECTORY INPUT TEST")
    print("YOLOv8 → ByteTrack → TrackHistory → Trajectory Adapter")
    print("=" * 70)

    frame_paths = sorted(
        FRAMES_DIR.glob("*.png")
    )

    if not frame_paths:
        raise FileNotFoundError(
            f"No PNG frames found in: {FRAMES_DIR}"
        )

    print()
    print(f"Frames directory : {FRAMES_DIR}")
    print(f"Frames found     : {len(frame_paths)}")

    # --------------------------------------------------
    # Load models
    # --------------------------------------------------

    print()
    print("Loading YOLO model...")

    detector = YOLO(MODEL_NAME)

    print("YOLO model loaded.")

    tracker = ByteTracker()

    history = TrackHistory(
        max_history=20
    )

    # --------------------------------------------------
    # Process frames
    # --------------------------------------------------

    for frame_number, frame_path in enumerate(
        frame_paths,
        start=1,
    ):

        print()
        print("=" * 70)
        print(
            f"FRAME {frame_number:03d}: "
            f"{frame_path.name}"
        )
        print("=" * 70)

        frame = cv2.imread(
            str(frame_path)
        )

        if frame is None:
            print(
                f"WARNING: Could not read "
                f"{frame_path.name}"
            )
            continue

        height, width = frame.shape[:2]

        print(
            f"Image: {width}x{height}"
        )

        # --------------------------------------------------
        # YOLO
        # --------------------------------------------------

        results = detector(
            frame,
            verbose=False,
        )

        result = results[0]

        # --------------------------------------------------
        # ByteTrack
        # --------------------------------------------------

        tracked_objects = tracker.update(
            result
        )

        if not tracked_objects:

            print(
                "No tracked objects."
            )

            continue

        # --------------------------------------------------
        # TrackHistory
        # --------------------------------------------------

        history.update(
            tracked_objects
        )

        # Print currently tracked objects

        for obj in tracked_objects:

            track_id = obj["track_id"]
            class_name = obj["class_name"]
            confidence = obj["confidence"]

            center = history.calculate_center(
                obj["bbox"]
            )

            print(
                f"ID={track_id:<4} "
                f"{class_name:<15} "
                f"confidence={confidence:.3f} "
                f"center=({center[0]:.1f}, "
                f"{center[1]:.1f})"
            )

    # ==================================================
    # Convert TrackHistory → Trajectories
    # ==================================================

    print()
    print()
    print("=" * 70)
    print("TRAJECTORY CANDIDATES")
    print("=" * 70)

    all_tracks = history.get_all_tracks()

    trajectories = (
        track_history_to_trajectories(
            all_tracks,
            start_timestamp=0.0,
            dt=DT,
        )
    )

    valid_count = 0

    for trajectory in trajectories:

        print()
        print(
            f"Track ID : "
            f"{trajectory.track_id}"
        )

        print(
            f"Class    : "
            f"{trajectory.class_name}"
        )

        print(
            f"Points   : "
            f"{trajectory.length}"
        )

        if trajectory.is_sufficient(
            MIN_HISTORY
        ):

            valid_count += 1

            print(
                "Status   : VALID TRAJECTORY"
            )

            print(
                f"Positions: "
                f"{trajectory.positions}"
            )

        else:

            print(
                "Status   : INSUFFICIENT HISTORY"
            )

    # ==================================================
    # Summary
    # ==================================================

    print()
    print("=" * 70)
    print("TRAJECTORY INPUT SUMMARY")
    print("=" * 70)

    print(
        f"Total tracks found       : "
        f"{len(trajectories)}"
    )

    print(
        f"Valid trajectories       : "
        f"{valid_count}"
    )

    print(
        f"Minimum history required : "
        f"{MIN_HISTORY}"
    )

    print(
        f"Sampling interval (test) : "
        f"{DT} seconds"
    )

    print()
    print("=" * 70)
    print("REAL TRAJECTORY INPUT TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()