import json
from pathlib import Path

from perception.safe_distance import SafeDistanceCalculator


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    config_path = repo_root / "configs" / "perception.yaml"

    calc = SafeDistanceCalculator(config_path=config_path)

    print()
    print("=" * 80)
    print("AUTONOMOUS VEHICLE PERCEPTION - SAFE DISTANCE EVALUATION DEMO")
    print("Dynamic Safe Distance based on Vehicle Type, Physical Size & Ego Speed")
    print("=" * 80)

    # Simulated detections from camera at 1280x720
    test_scenarios = [
        {
            "name": "Scenario 1: Low Speed (30 km/h = 8.3 m/s) in Urban Traffic",
            "ego_speed_ms": 8.33,
            "detections": [
                {
                    "track_id": 1,
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.88,
                    "bbox": [600, 320, 640, 480],  # h = 160px -> ~6.8m
                },
                {
                    "track_id": 2,
                    "class_id": 2,
                    "class_name": "car",
                    "confidence": 0.94,
                    "bbox": [500, 300, 720, 420],  # h = 120px -> ~8.0m
                },
            ],
        },
        {
            "name": "Scenario 2: Highway Speed (72 km/h = 20.0 m/s) with Heavy Transport",
            "ego_speed_ms": 20.0,
            "detections": [
                {
                    "track_id": 3,
                    "class_id": 2,
                    "class_name": "car",
                    "confidence": 0.96,
                    "bbox": [560, 330, 680, 390],  # h = 60px -> ~16.0m
                },
                {
                    "track_id": 4,
                    "class_id": 7,
                    "class_name": "truck",
                    "confidence": 0.91,
                    "bbox": [480, 260, 780, 430],  # h = 170px -> ~12.8m (wide truck trailer)
                },
                {
                    "track_id": 5,
                    "class_id": 5,
                    "class_name": "bus",
                    "confidence": 0.93,
                    "bbox": [520, 300, 720, 410],  # h = 110px -> ~18.6m
                },
            ],
        },
    ]

    for scenario in test_scenarios:
        print()
        print("-" * 80)
        print(f"{scenario['name']} | Ego Speed: {scenario['ego_speed_ms']:.2f} m/s ({scenario['ego_speed_ms']*3.6:.1f} km/h)")
        print("-" * 80)

        for raw_det in scenario["detections"]:
            enriched = calc.enrich_detection(raw_det, ego_speed_ms=scenario["ego_speed_ms"])

            status_color = (
                "[CRITICAL / BRAKE]" if enriched["safety_status"] == "CRITICAL"
                else "[WARNING / SLOW]" if enriched["safety_status"] == "WARNING"
                else "[SAFE]"
            )

            print(f"Track #{enriched['track_id']} | Type: {enriched['class_name'].upper():<10} | Size Category: {enriched['size_category']}")
            print(f"  Confidence        : {enriched['confidence']:.2f}")
            print(f"  Bbox              : {enriched['bbox']}")
            print(f"  Estimated Dist    : {enriched['estimated_distance_m']:.2f} m")
            print(f"  Safe Dist (Req'd) : {enriched['safe_distance_m']:.2f} m  (THW: {enriched['safe_distance_thw_m']:.2f}m, RSS: {enriched['safe_distance_rss_m']:.2f}m)")
            print(f"  Distance Margin   : {enriched['distance_margin_m']:+.2f} m")
            print(f"  Status            : {status_color}")
            print()

    print("=" * 80)
    print("DEMO COMPLETE: All safe distance metrics computed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
