from typing import Any

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from ultralytics import YOLO


class YOLODetector:
    """
    YOLOv8 object detector for raw RGB ImageData
    received from Unreal Engine.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.4,
        device: str = "cpu",
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device

    @staticmethod
    def image_data_to_numpy(image: Any) -> np.ndarray:
        """
        Convert raw RGB ImageData into an
        H x W x C NumPy array.
        """

        if image is None:
            raise ValueError(
                "Observation does not contain an image."
            )

        if image.channels != 3:
            raise ValueError(
                f"Expected RGB image with 3 channels, "
                f"received {image.channels} channels."
            )

        expected_size = (
            image.width
            * image.height
            * image.channels
        )

        actual_size = len(image.data)

        if actual_size != expected_size:
            raise ValueError(
                f"Image byte size mismatch: "
                f"expected {expected_size} bytes, "
                f"received {actual_size} bytes."
            )

        frame = np.frombuffer(
            image.data,
            dtype=np.uint8,
        )

        frame = frame.reshape(
            image.height,
            image.width,
            image.channels,
        )

        return frame

    def detect(self, image: Any):
        """
        Run YOLO inference on an ImageData object.
        """

        frame = self.image_data_to_numpy(image)

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        return results[0]

    def detect_objects(self, image: Any) -> list[dict]:
        """
        Run YOLO detection and return simple
        Python dictionaries.
        """

        result = self.detect(image)

        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:

            xyxy = box.xyxy[0].cpu().numpy()

            confidence = float(
                box.conf[0].cpu().item()
            )

            class_id = int(
                box.cls[0].cpu().item()
            )

            class_name = result.names[class_id]

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [
                        float(xyxy[0]),
                        float(xyxy[1]),
                        float(xyxy[2]),
                        float(xyxy[3]),
                    ],
                }
            )

        return detections