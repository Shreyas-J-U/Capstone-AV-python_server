import argparse
from pathlib import Path
from typing import List

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
from PIL import Image

from protocol.models import ImageData
from perception.yolo_detector import YOLODetector
from perception.byte_tracker import ByteTracker
from perception.track_history import TrackHistory
from perception.safe_distance import SafeDistanceCalculator


def load_image_data_from_path(frame_path: Path) -> ImageData:
    """Load image from disk and wrap in protocol ImageData."""
    with Image.open(frame_path) as img:
        img_rgb = img.convert("RGB")
        width, height = img_rgb.size
        raw_bytes = img_rgb.tobytes()

    return ImageData(
        width=width,
        height=height,
        channels=3,
        data=raw_bytes,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test Safe Distance Calculation directly from real video/sampled frames"
    )
    parser.add_argument(
        "--source",
        choices=["frames", "video"],
        default="frames",
        help="Input source: 'frames' (sampled PNGs) or 'video' (MP4 file)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=13.89,
        help="Ego vehicle speed in m/s (default: 13.89 m/s = 50 km/h)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=25,
        help="Maximum number of frames to process (default: 25)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent.parent
    config_path = repo_root / "configs" / "perception.yaml"

    print()
    print("=" * 80)
    print("REAL VIDEO SAFE DISTANCE EVALUATION")
    print("Real Video Frames → YOLOv8 → ByteTrack → Pinhole Metric Distance → Dynamic Safe Distance")
    print("=" * 80)
    print(f"Ego Vehicle Speed : {args.speed:.2f} m/s ({args.speed * 3.6:.1f} km/h)")
    print(f"Config File       : {config_path}")
    print(f"Max Frames        : {args.max_frames}")

    # Initialize Perception components
    safe_calc = SafeDistanceCalculator(config_path=config_path)
    detector = YOLODetector()
    tracker = ByteTracker(safe_distance_calculator=safe_calc)
    history = TrackHistory(max_history=50)

    # Prepare frames
    frames: List[ImageData] = []

    if args.source == "frames":
        frames_dir = base_dir / "assets" / "sampled_frames"
        frame_paths = sorted(frames_dir.glob("frame_*.png"))[: args.max_frames]
        if not frame_paths:
            raise RuntimeError(f"No sampled frames found in {frames_dir}")
        print(f"Processing {len(frame_paths)} frames from {frames_dir}...")
        for p in frame_paths:
            frames.append(load_image_data_from_path(p))
    else:
        video_path = base_dir.parent / "assets" / "test_video.mp4"
        if not video_path.exists():
            raise RuntimeError(f"Video file not found at {video_path}")
        print(f"Reading frames directly from video {video_path}...")
        cap = cv2.VideoCapture(str(video_path))
        count = 0
        while cap.isOpened() and count < args.max_frames:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, c = frame_rgb.shape
            frames.append(
                ImageData(width=w, height=h, channels=c, data=frame_rgb.tobytes())
            )
            count += 1
        cap.release()

    # Tracking metrics across video
    track_summary = {}

    # Process each real video frame
    for frame_idx, image_data in enumerate(frames, start=1):
        # 1. Run real YOLO detection on video frame
        yolo_results = detector.detect(image_data)

        # 2. Update ByteTrack & compute safe distance using real bboxes and detected vehicle classes
        tracked_objects = tracker.update(
            yolo_results,
            ego_speed_ms=args.speed,
        )

        # 3. Update track history
        history.update(tracked_objects)

        print()
        print("-" * 80)
        print(f"FRAME {frame_idx:03d} | Image: {image_data.width}x{image_data.height} | Detected & Tracked Objects: {len(tracked_objects)}")
        print("-" * 80)

        if not tracked_objects:
            print("  No objects tracked in this frame.")
            continue

        for obj in tracked_objects:
            t_id = obj["track_id"]
            c_name = obj["class_name"]
            conf = obj["confidence"]
            bbox = obj["bbox"]
            dist_m = obj["estimated_distance_m"]
            safe_m = obj["safe_distance_m"]
            margin_m = obj["distance_margin_m"]
            status = obj["safety_status"]
            size_cat = obj["size_category"]

            lat_m = obj.get("lateral_offset_m", 0.0)
            lane_pos = obj.get("lane_position", "IN_PATH")
            in_path = obj.get("in_ego_path", True)

            # Record stats for summary
            if t_id not in track_summary:
                track_summary[t_id] = {
                    "class_name": c_name,
                    "size_category": size_cat,
                    "lane_position": lane_pos,
                    "distances": [],
                    "safe_distances": [],
                    "worst_status": status,
                }
            track_summary[t_id]["distances"].append(dist_m)
            track_summary[t_id]["safe_distances"].append(safe_m)
            track_summary[t_id]["lane_position"] = lane_pos
            if "CRITICAL" in status:
                track_summary[t_id]["worst_status"] = "CRITICAL"
            elif "WARNING" in status and "CRITICAL" not in track_summary[t_id]["worst_status"]:
                track_summary[t_id]["worst_status"] = "WARNING"

            print(
                f"  ID={t_id:<3} | {c_name:<10} ({size_cat:<11}) | "
                f"Lane: {lane_pos:<14} | "
                f"X: {lat_m:+5.2f}m | "
                f"Dist: {dist_m:6.2f}m | "
                f"SafeDist: {safe_m:5.2f}m | "
                f"Margin: {margin_m:+6.2f}m | [{status}]"
            )

    # Final Overall Summary Table
    print()
    print("=" * 80)
    print("REAL VIDEO SAFE DISTANCE - VEHICLE TRACKING SUMMARY")
    print("=" * 80)
    print(f"{'ID':<4} | {'Class':<10} | {'Size':<11} | {'Lane Position':<14} | {'Frames':<6} | {'Min Dist':<10} | {'Max Dist':<10} | {'Status'}")
    print("-" * 80)

    for t_id, s in track_summary.items():
        min_d = min(s["distances"])
        max_d = max(s["distances"])
        avg_safe = s["safe_distances"][-1]
        print(
            f"{t_id:<4} | "
            f"{s['class_name']:<10} | "
            f"{s['size_category']:<11} | "
            f"{s['lane_position']:<14} | "
            f"{len(s['distances']):<6} | "
            f"{min_d:6.2f} m   | "
            f"{max_d:6.2f} m   | "
            f"[{s['worst_status']}]"
        )

    print("=" * 80)
    print("TEST COMPLETED: Safe distances dynamically evaluated from real video frames.")
    print("=" * 80)


if __name__ == "__main__":
    main()
