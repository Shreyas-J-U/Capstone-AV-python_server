from pathlib import Path

# pyrefly: ignore [missing-import]
import cv2

from protocol.models import ImageData
from perception.yolo_detector import YOLODetector
from perception.byte_tracker import ByteTracker
from perception.track_history import TrackHistory
from perception.trajectory_predictor import TrajectoryPredictor


FRAMES_DIR = (
    Path(__file__).parent
    / "assets"
    / "sampled_frames"
)

MIN_HISTORY = 5
PREDICTION_STEPS = 5


def main():

    print()
    print("=" * 68)
    print("REAL TRAJECTORY PREDICTION")
    print("YOLOv8 → ByteTrack → TrackHistory → Trajectory Predictor")
    print("=" * 68)

    # ==================================================
    # FIND FRAMES
    # ==================================================

    frame_paths = sorted(
        FRAMES_DIR.glob("frame_*.png")
    )

    print()
    print(f"Frames directory : {FRAMES_DIR}")
    print(f"Frames found     : {len(frame_paths)}")

    if not frame_paths:
        raise RuntimeError(
            "No sampled frames found."
        )

    # ==================================================
    # INITIALIZE PIPELINE
    # ==================================================

    print()
    print("Loading YOLO model...")

    detector = YOLODetector()
    tracker = ByteTracker()

    history = TrackHistory(
        max_history=20
    )

    predictor = TrajectoryPredictor(
        prediction_horizon=PREDICTION_STEPS
    )

    print("YOLO model loaded.")

    # ==================================================
    # PROCESS ALL FRAMES
    # ==================================================

    for frame_number, frame_path in enumerate(
        frame_paths,
        start=1,
    ):

        print()
        print("=" * 68)
        print(
            f"FRAME {frame_number:03d}: "
            f"{frame_path.name}"
        )
        print("=" * 68)

        # --------------------------------------------------
        # Read image
        # --------------------------------------------------

        image = cv2.imread(
            str(frame_path)
        )

        if image is None:
            print(
                "Could not read frame."
            )
            continue

        height, width, channels = image.shape

        print(
            f"Image: "
            f"{width}x{height} RGB"
        )

        # --------------------------------------------------
        # Convert OpenCV image to ImageData
        # --------------------------------------------------

        image_data = ImageData(
            width=width,
            height=height,
            channels=channels,
            data=image.tobytes(),
        )

        # --------------------------------------------------
        # YOLO DETECTION
        # --------------------------------------------------

        detections = detector.detect(
            image_data
        )

        # --------------------------------------------------
        # BYTE TRACK
        # --------------------------------------------------

        tracked_objects = tracker.update(
            detections
        )

        if not tracked_objects:
            print(
                "No tracked objects."
            )
            continue

        # --------------------------------------------------
        # TRACK HISTORY
        # --------------------------------------------------

        history.update(
            tracked_objects
        )

        # --------------------------------------------------
        # Print current tracked objects
        # --------------------------------------------------

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
    # TRAJECTORY PREDICTION
    # ==================================================

    print()
    print()
    print("=" * 68)
    print("REAL TRAJECTORY PREDICTIONS")
    print("=" * 68)

    all_tracks = history.get_all_tracks()

    valid_count = 0
    prediction_count = 0

    # ==================================================
    # PREDICT FOR EACH TRACK
    # ==================================================

    for track_id, data in all_tracks.items():

        observed = data["history"]

        print()
        print(
            f"Track ID : {track_id}"
        )

        print(
            f"Class    : "
            f"{data['class_name']}"
        )

        print(
            f"Points   : "
            f"{len(observed)}"
        )

        # --------------------------------------------------
        # Check minimum history
        # --------------------------------------------------

        if len(observed) < MIN_HISTORY:

            print(
                "Status   : "
                "INSUFFICIENT HISTORY"
            )

            continue

        valid_count += 1

        print(
            "Status   : "
            "VALID TRAJECTORY"
        )

        # --------------------------------------------------
        # Observed trajectory
        # --------------------------------------------------

        print()
        print(
            "Observed trajectory:"
        )

        for index, position in enumerate(
            observed
        ):

            print(
                f"  t{index:<2}: "
                f"({position[0]:.2f}, "
                f"{position[1]:.2f})"
            )

        # --------------------------------------------------
        # Future trajectory prediction
        # --------------------------------------------------

        predicted = predictor.predict(
            observed,
            steps=PREDICTION_STEPS,
        )

        # --------------------------------------------------
        # Validate prediction
        # --------------------------------------------------

        if len(predicted) != PREDICTION_STEPS:

            print()
            print(
                "Prediction status : FAILED"
            )

            print(
                f"Expected "
                f"{PREDICTION_STEPS} predictions, "
                f"got {len(predicted)}."
            )

            continue

        prediction_count += 1

        # --------------------------------------------------
        # Print predictions
        # --------------------------------------------------

        print()
        print(
            "Predicted future trajectory:"
        )

        for index, position in enumerate(
            predicted,
            start=1,
        ):

            print(
                f"  t+{index}: "
                f"({position[0]:.2f}, "
                f"{position[1]:.2f})"
            )

        print(
            "Prediction status : SUCCESS"
        )

    # ==================================================
    # SUMMARY
    # ==================================================

    print()
    print()
    print("=" * 68)
    print(
        "TRAJECTORY PREDICTION SUMMARY"
    )
    print("=" * 68)

    print(
        f"Total tracks found       : "
        f"{len(all_tracks)}"
    )

    print(
        f"Valid trajectories       : "
        f"{valid_count}"
    )

    print(
        f"Successful predictions   : "
        f"{prediction_count}"
    )

    print(
        f"Minimum history required : "
        f"{MIN_HISTORY}"
    )

    print(
        f"Prediction horizon       : "
        f"{PREDICTION_STEPS} steps"
    )

    print()
    print(
        "REAL TRAJECTORY PREDICTION "
        "TEST COMPLETE"
    )


if __name__ == "__main__":
    main()