from pathlib import Path

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image

from protocol.models import ImageData
from perception.yolo_detector import YOLODetector


def main():

    image_path = Path(
        "tests/assets/test_scene.webp"
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Test image not found: {image_path}"
        )

    # ----------------------------------------
    # Load test image
    # ----------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (640, 480)
    )

    rgb = np.asarray(
        image,
        dtype=np.uint8,
    )

    # ----------------------------------------
    # Build ImageData exactly like Unreal
    # ----------------------------------------

    image_data = ImageData(
        width=640,
        height=480,
        channels=3,
        data=rgb.tobytes(),
    )

    # ----------------------------------------
    # Load YOLO
    # ----------------------------------------

    detector = YOLODetector(
        model_path="yolov8n.pt",
        confidence=0.4,
        device="cpu",
    )

    # ----------------------------------------
    # Run detection
    # ----------------------------------------

    detections = detector.detect_objects(
        image_data
    )

    # ----------------------------------------
    # Print results
    # ----------------------------------------

    print()
    print("=" * 60)
    print("YOLO DETECTIONS")
    print("=" * 60)

    if not detections:
        print("No objects detected.")

    for detection in detections:

        print(
            f"{detection['class_name']:15}"
            f" confidence={detection['confidence']:.3f}"
            f" bbox={detection['bbox']}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()