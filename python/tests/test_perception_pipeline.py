from pathlib import Path

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

from perception.track_history import TrackHistory


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ASSET_DIR = Path("tests/assets")

MODEL_PATH = "yolov8n.pt"

CONFIDENCE = 0.4

DEVICE = "cpu"


def load_frame(path: Path) -> np.ndarray:
    """
    Load an image as raw RGB uint8 data.

    This mimics the format that will eventually come
    from Unreal ImageData.
    """

    image = Image.open(path).convert("RGB")

    frame = np.asarray(
        image,
        dtype=np.uint8,
    )

    return frame


def main():

    # --------------------------------------------------------
    # Find frames
    # --------------------------------------------------------

    frame_paths = [
        ASSET_DIR / "frame1.png",
        ASSET_DIR / "frame2.png",
        ASSET_DIR / "frame3.png",
        ASSET_DIR / "frame4.png",
        ASSET_DIR / "frame5.png",
    ]

    for path in frame_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Frame not found: {path}"
            )

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    model = YOLO(MODEL_PATH)

    # --------------------------------------------------------
    # Create TrackHistory
    # --------------------------------------------------------

    history = TrackHistory(
        max_history=20
    )

    print()
    print("=" * 70)
    print("PERCEPTION PIPELINE TEST")
    print("YOLOv8 → ByteTrack → TrackHistory")
    print("=" * 70)

    # --------------------------------------------------------
    # Process frames sequentially
    # --------------------------------------------------------

    for frame_number, frame_path in enumerate(
        frame_paths,
        start=1,
    ):

        print()
        print(
            f"FRAME {frame_number}: "
            f"{frame_path.name}"
        )

        print("-" * 70)

        # ----------------------------------------------------
        # Load RGB frame
        # ----------------------------------------------------

        frame = load_frame(
            frame_path
        )

        print(
            f"Image shape: {frame.shape}"
        )

        # ----------------------------------------------------
        # YOLO + ByteTrack
        # ----------------------------------------------------

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE,
            device=DEVICE,
            verbose=False,
        )

        result = results[0]

        # ----------------------------------------------------
        # Extract tracked objects
        # ----------------------------------------------------

        tracked_objects = []

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):

            boxes = result.boxes

            track_ids = (
                boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

            class_ids = (
                boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            confidences = (
                boxes.conf
                .cpu()
                .numpy()
            )

            bboxes = (
                boxes.xyxy
                .cpu()
                .numpy()
            )

            for i in range(
                len(track_ids)
            ):

                track_id = int(
                    track_ids[i]
                )

                class_id = int(
                    class_ids[i]
                )

                confidence = float(
                    confidences[i]
                )

                bbox = [
                    float(x)
                    for x in bboxes[i]
                ]

                class_name = result.names[
                    class_id
                ]

                tracked_objects.append(
                    {
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": bbox,
                    }
                )

        # ----------------------------------------------------
        # Update TrackHistory
        # ----------------------------------------------------

        history.update(
            tracked_objects
        )

        # ----------------------------------------------------
        # Print current detections
        # ----------------------------------------------------

        if not tracked_objects:

            print(
                "No tracked objects."
            )

            continue

        for obj in tracked_objects:

            track_id = obj[
                "track_id"
            ]

            center = (
                history.calculate_center(
                    obj["bbox"]
                )
            )

            print(
                f"ID={track_id:<3} "
                f"{obj['class_name']:<10} "
                f"confidence="
                f"{obj['confidence']:.3f} "
                f"center="
                f"({center[0]:.1f}, "
                f"{center[1]:.1f})"
            )

    # --------------------------------------------------------
    # Final trajectory summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRACK HISTORY SUMMARY")
    print("=" * 70)

    all_tracks = (
        history.get_all_tracks()
    )

    if not all_tracks:

        print(
            "No tracks were created."
        )

    else:

        for track_id, data in (
            all_tracks.items()
        ):

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

            print(
                "Positions:"
            )

            for index, position in enumerate(
                data["history"]
            ):

                print(
                    f"  t{index}: "
                    f"({position[0]:.2f}, "
                    f"{position[1]:.2f})"
                )

    print()
    print("=" * 70)
    print(
        "PERCEPTION PIPELINE TEST COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()