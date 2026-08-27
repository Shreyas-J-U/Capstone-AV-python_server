# pyrefly: ignore [missing-import]
from ultralytics import YOLO


def main():
    print("Loading YOLOv8n...")

    model = YOLO("yolov8n.pt")

    print("YOLOv8n loaded successfully.")
    print(f"Model: {model.model}")


if __name__ == "__main__":
    main()