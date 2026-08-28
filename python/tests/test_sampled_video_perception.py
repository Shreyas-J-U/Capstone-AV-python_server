from pathlib import Path

# pyrefly: ignore [missing-import]
from PIL import Image

from protocol.models import ImageData

from perception.yolo_detector import YOLODetector
from perception.byte_tracker import ByteTracker
from perception.track_history import TrackHistory


def load_image_data(
    image_path: Path,
) -> ImageData:
    """
    Load a PNG/JPG test frame and convert it into
    the same ImageData representation used by
    the Unreal → Python protocol.

    The resulting image contains:
        width
        height
        channels = 3
        raw RGB bytes
    """

    with Image.open(image_path) as image:

        # ----------------------------------------------------
        # Convert to RGB
        # ----------------------------------------------------

        image = image.convert("RGB")

        width, height = image.size

        # ----------------------------------------------------
        # Raw RGB bytes
        # ----------------------------------------------------

        raw_rgb = image.tobytes()

    return ImageData(
        width=width,
        height=height,
        channels=3,
        data=raw_rgb,
    )


def main():

    # ============================================================
    # PATHS
    # ============================================================

    base_dir = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    frames_dir = (
        base_dir
        / "tests"
        / "assets"
        / "sampled_frames"
    )

    if not frames_dir.exists():

        raise FileNotFoundError(
            f"Sampled frames directory not found: "
            f"{frames_dir}"
        )

    frame_paths = sorted(
        frames_dir.glob("frame_*.png")
    )

    if not frame_paths:

        raise FileNotFoundError(
            f"No sampled frames found in: "
            f"{frames_dir}"
        )

    # ============================================================
    # INITIALIZE
    # ============================================================

    print()
    print("=" * 70)
    print("SAMPLED VIDEO PERCEPTION PIPELINE")
    print("YOLOv8 → ByteTrack → TrackHistory")
    print("=" * 70)

    print()
    print(
        f"Frames directory : {frames_dir}"
    )

    print(
        f"Frames found     : {len(frame_paths)}"
    )

    print()
    print("Loading YOLO model...")

    detector = YOLODetector()

    print("YOLO model loaded.")

    tracker = ByteTracker()

    history = TrackHistory(
        max_history=20
    )

    # ============================================================
    # PROCESS FRAMES
    # ============================================================

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

        # --------------------------------------------------------
        # Load frame as ImageData
        # --------------------------------------------------------

        image_data = load_image_data(
            frame_path
        )

        print(
            f"Image: "
            f"{image_data.width}x"
            f"{image_data.height} "
            f"RGB"
        )

        # --------------------------------------------------------
        # YOLO
        # --------------------------------------------------------

        detections = detector.detect(
            image_data
        )

        if not detections:

            print(
                "YOLO: No detections."
            )

            continue

        # --------------------------------------------------------
        # ByteTrack
        # --------------------------------------------------------

        tracked_objects = tracker.update(
            detections
        )

        if not tracked_objects:

            print(
                "Detections found, "
                "but no tracking IDs."
            )

            continue

        # --------------------------------------------------------
        # TrackHistory
        # --------------------------------------------------------

        history.update(
            tracked_objects
        )

        # --------------------------------------------------------
        # Display tracked objects
        # --------------------------------------------------------

        for obj in tracked_objects:

            track_id = obj[
                "track_id"
            ]

            class_name = obj[
                "class_name"
            ]

            confidence = obj[
                "confidence"
            ]

            bbox = obj[
                "bbox"
            ]

            center_x, center_y = (
                history.calculate_center(
                    bbox
                )
            )

            print(
                f"ID={track_id:<4} "
                f"{class_name:<15} "
                f"confidence={confidence:.3f} "
                f"center=("
                f"{center_x:.1f}, "
                f"{center_y:.1f})"
            )

    # ============================================================
    # FINAL TRACK HISTORY
    # ============================================================

    print()
    print("=" * 70)
    print("FINAL TRACK HISTORY")
    print("=" * 70)

    tracks = history.get_all_tracks()

    if not tracks:

        print(
            "No tracks were created."
        )

    else:

        for track_id, data in tracks.items():

            print()
            print(
                f"Track ID: {track_id}"
            )

            print(
                f"Class: "
                f"{data['class_name']}"
            )

            print(
                f"History length: "
                f"{len(data['history'])}"
            )

            print("Positions:")

            for index, position in enumerate(
                data["history"]
            ):

                print(
                    f"  t{index}: "
                    f"({position[0]:.2f}, "
                    f"{position[1]:.2f})"
                )

    # ============================================================
    # SUMMARY
    # ============================================================

    print()
    print("=" * 70)
    print("PERCEPTION PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"Frames processed : "
        f"{len(frame_paths)}"
    )

    print(
        f"Tracks created   : "
        f"{len(tracks)}"
    )

    print()
    print("Pipeline:")
    print()
    print("Sampled Video")
    print("    ↓")
    print("PNG → Raw RGB ImageData")
    print("    ↓")
    print("YOLOv8")
    print("    ↓")
    print("ByteTrack")
    print("    ↓")
    print("TrackHistory")
    print("    ↓")
    print("Object Trajectories")

    print("=" * 70)


if __name__ == "__main__":
    main()