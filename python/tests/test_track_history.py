from perception.track_history import TrackHistory


def main():
    history = TrackHistory(max_history=5)

    # Simulated ByteTrack output.
    # In the real pipeline these will come
    # directly from YOLO + ByteTrack.

    frames = [
        [
            {
                "track_id": 1,
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.91,
                "bbox": [100, 200, 200, 300],
            }
        ],
        [
            {
                "track_id": 1,
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.92,
                "bbox": [110, 195, 210, 295],
            }
        ],
        [
            {
                "track_id": 1,
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.93,
                "bbox": [120, 190, 220, 290],
            }
        ],
        [
            {
                "track_id": 1,
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.94,
                "bbox": [130, 185, 230, 285],
            }
        ],
        [
            {
                "track_id": 1,
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.95,
                "bbox": [140, 180, 240, 280],
            }
        ],
    ]

    print()
    print("=" * 60)
    print("TRACK HISTORY TEST")
    print("=" * 60)

    for frame_id, tracked_objects in enumerate(frames, start=1):
        history.update(tracked_objects)

        print()
        print(f"FRAME {frame_id}")

        tracks = history.get_all_tracks()

        for track_id, data in tracks.items():
            print(f"track_id={track_id}")
            print(f"  class      : {data['class_name']}")
            print(f"  confidence : {data['confidence']:.3f}")
            print(f"  bbox       : {data['bbox']}")
            print(f"  history    : {data['history']}")

    print()
    print("=" * 60)

    # ----------------------------------------
    # Assertions
    # ----------------------------------------

    track = history.get_all_tracks()[1]

    assert track["class_name"] == "car"
    assert len(track["history"]) == 5

    assert track["history"][0] == (150.0, 250.0)
    assert track["history"][-1] == (190.0, 230.0)

    print("TRACK HISTORY TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()