from pathlib import Path

# pyrefly: ignore [missing-import]
from ultralytics import YOLO

from perception.track_history import TrackHistory


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ASSETS_DIR = BASE_DIR / "assets"

MODEL_PATH = "yolov8n.pt"

FRAME_PATHS = [
    ASSETS_DIR / "frame1.png",
    ASSETS_DIR / "frame2.png",
    ASSETS_DIR / "frame3.png",
    ASSETS_DIR / "frame4.png",
    ASSETS_DIR / "frame5.png",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("TEMPORAL PERCEPTION PIPELINE TEST")
    print("YOLOv8 → ByteTrack → TrackHistory")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input frames
    # --------------------------------------------------------

    for frame_path in FRAME_PATHS:

        if not frame_path.exists():

            raise FileNotFoundError(
                f"Frame not found: {frame_path}"
            )

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    print()
    print("Loading YOLO model...")

    model = YOLO(MODEL_PATH)

    print("YOLO model loaded.")

    # --------------------------------------------------------
    # Track history
    # --------------------------------------------------------

    history = TrackHistory(
        max_history=20
    )

    # --------------------------------------------------------
    # Process frames sequentially
    # --------------------------------------------------------

    for frame_number, frame_path in enumerate(
        FRAME_PATHS,
        start=1
    ):

        print()
        print("=" * 70)
        print(
            f"FRAME {frame_number}: "
            f"{frame_path.name}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # YOLO + ByteTrack
        # ----------------------------------------------------

        results = model.track(
            source=str(frame_path),
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        if not results:

            print("No YOLO result.")

            continue

        result = results[0]

        # ----------------------------------------------------
        # Image information
        # ----------------------------------------------------

        if result.orig_img is not None:

            height, width = result.orig_img.shape[:2]

            print(
                f"Image shape: "
                f"({height}, {width}, 3)"
            )

        # ----------------------------------------------------
        # Check boxes
        # ----------------------------------------------------

        if result.boxes is None:

            print("No detections.")

            continue

        if len(result.boxes) == 0:

            print("No detections.")

            continue

        # ----------------------------------------------------
        # ByteTrack IDs
        # ----------------------------------------------------

        if result.boxes.id is None:

            print(
                "Detections found, "
                "but no tracking IDs."
            )

            continue

        boxes = result.boxes

        track_ids = (
            boxes.id
            .int()
            .cpu()
            .tolist()
        )

        class_ids = (
            boxes.cls
            .int()
            .cpu()
            .tolist()
        )

        confidences = (
            boxes.conf
            .cpu()
            .tolist()
        )

        bounding_boxes = (
            boxes.xyxy
            .cpu()
            .tolist()
        )

        # ----------------------------------------------------
        # Convert YOLO/ByteTrack output
        # into TrackHistory format
        # ----------------------------------------------------

        tracked_objects = []

        for (
            track_id,
            class_id,
            confidence,
            bbox,
        ) in zip(
            track_ids,
            class_ids,
            confidences,
            bounding_boxes,
        ):

            class_name = model.names[
                class_id
            ]

            tracked_object = {

                "track_id": int(
                    track_id
                ),

                "class_id": int(
                    class_id
                ),

                "class_name": class_name,

                "confidence": float(
                    confidence
                ),

                "bbox": [
                    float(value)
                    for value in bbox
                ],
            }

            tracked_objects.append(
                tracked_object
            )

        # ----------------------------------------------------
        # Update TrackHistory
        # ----------------------------------------------------

        history.update(
            tracked_objects
        )

        # ----------------------------------------------------
        # Display current frame
        # ----------------------------------------------------

        if not tracked_objects:

            print("No tracked objects.")

            continue

        for obj in tracked_objects:

            bbox = obj["bbox"]

            center_x, center_y = (
                history.calculate_center(
                    bbox
                )
            )

            print(
                f"ID={obj['track_id']:<3} "
                f"{obj['class_name']:<12} "
                f"confidence="
                f"{obj['confidence']:.3f} "
                f"center="
                f"({center_x:.1f}, "
                f"{center_y:.1f})"
            )

    # ========================================================
    # FINAL TRACK HISTORY
    # ========================================================

    print()
    print("=" * 70)
    print("TRACK HISTORY SUMMARY")
    print("=" * 70)

    tracks = history.get_all_tracks()

    if not tracks:

        print("No tracks were created.")

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

            for timestep, position in enumerate(
                data["history"]
            ):

                x, y = position

                print(
                    f"  t{timestep}: "
                    f"({x:.2f}, {y:.2f})"
                )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("TEMPORAL PERCEPTION TEST COMPLETE")
    print("=" * 70)

    print()
    print(
        "Pipeline:"
    )

    print(
        "Ordered Frames"
    )

    print(
        "    ↓"
    )

    print(
        "YOLOv8 Detection"
    )

    print(
        "    ↓"
    )

    print(
        "ByteTrack"
    )

    print(
        "    ↓"
    )

    print(
        "TrackHistory"
    )

    print(
        "    ↓"
    )

    print(
        "Object Trajectories"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()