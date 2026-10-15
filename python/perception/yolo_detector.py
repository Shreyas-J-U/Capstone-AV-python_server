from typing import Any, Optional

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

from perception.safe_distance import SafeDistanceCalculator


class YOLODetector:
    """
    YOLOv8 object detector for raw RGB ImageData
    received from Unreal Engine with dynamic safe distance evaluation.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.4,
        device: str = "cpu",
        safe_distance_calculator: Optional[SafeDistanceCalculator] = None,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device
        self.safe_distance_calculator = (
            safe_distance_calculator or SafeDistanceCalculator()
        )

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

    def detect_objects(
        self,
        image: Any,
        ego_speed_ms: Optional[float] = None,
    ) -> list[dict]:
        """
        Run YOLO detection and return simple Python dictionaries.

        If ego_speed_ms is provided, the detections are enriched with:
        - estimated_distance_m
        - safe_distance_m
        - distance_margin_m
        - size_category
        - safety_status ('SAFE', 'WARNING', 'CRITICAL')
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

            detection = {
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

            if ego_speed_ms is not None:
                detection = self.safe_distance_calculator.enrich_detection(
                    detection,
                    ego_speed_ms=ego_speed_ms,
                )

            detections.append(detection)

        return detections

    def detect_with_safe_distance(
        self,
        image: Any,
        ego_speed_ms: float = 0.0,
    ) -> list[dict]:
        """
        Run YOLO detection and always return enriched detections with
        safe distance calculations based on vehicle type and size.
        """
        return self.detect_objects(image, ego_speed_ms=ego_speed_ms)