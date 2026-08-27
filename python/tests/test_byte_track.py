from pathlib import Path

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from ultralytics import YOLO


def main():

    image_path = Path(
        "tests/assets/test_scene.webp"
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Test image not found: {image_path}"
        )

    # ----------------------------------------
    # Load RGB image
    # ----------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (640, 480)
    )

    frame = np.asarray(
        image,
        dtype=np.uint8,
    )

    # ----------------------------------------
    # Load YOLO
    # ----------------------------------------

    model = YOLO("yolov8n.pt")

    print()
    print("=" * 60)
    print("YOLO + BYTE TRACK")
    print("=" * 60)

    # ----------------------------------------
    # Run the SAME frame multiple times
    #
    # This is only a tracker smoke test.
    # Later Unreal will provide different frames.
    # ----------------------------------------

    for frame_id in range(1, 6):

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.4,
            device="cpu",
            verbose=False,
        )

        result = results[0]

        print()
        print(f"FRAME {frame_id}")
        print("-" * 60)

        if result.boxes is None:
            print("No detections.")
            continue

        if result.boxes.id is None:
            print("Detections found, but no track IDs yet.")
            continue

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

        xyxy = (
            boxes.xyxy
            .cpu()
            .numpy()
        )

        for i in range(len(track_ids)):

            class_id = class_ids[i]

            class_name = result.names[
                class_id
            ]

            confidence = float(
                confidences[i]
            )

            bbox = [
                float(x)
                for x in xyxy[i]
            ]

            print(
                f"track_id={track_ids[i]:3d} "
                f"class={class_name:10} "
                f"confidence={confidence:.3f} "
                f"bbox={bbox}"
            )

    print()
    print("=" * 60)
    print("BYTE TRACK TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()