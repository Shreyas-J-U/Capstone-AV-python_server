
# Perception & Object Tracking Pipeline

## Overview

The perception subsystem is responsible for extracting meaningful information about the surrounding environment from camera images.

The current pipeline consists of:

```text
Camera Image
     ↓
YOLOv8 Object Detection
     ↓
ByteTrack Object Tracking
     ↓
TrackHistory
     ↓
Temporal Object Trajectories
     ↓
Future Trajectory Prediction
````

The current implementation successfully integrates:

* Raw RGB image processing
* YOLOv8 object detection
* ByteTrack multi-object tracking
* Persistent track IDs
* Bounding-box extraction
* Object center calculation
* Temporal position history
* Consecutive-frame processing
* Integration with the existing Python RL environment architecture

---

# 1. Directory Structure

The perception components are organized as follows:

```text
Implementation/
│
├── README.md
├── WALKTHROUGH.md
├── UNREAL_GUIDE.md
│
└── python/
    │
    ├── main.py
    │
    ├── perception/
    │   ├── __init__.py
    │   ├── yolo_detector.py
    │   ├── byte_tracker.py
    │   └── track_history.py
    │
    ├── protocol/
    │   ├── constants.py
    │   ├── framing.py
    │   ├── models.py
    │   ├── connection.py
    │   ├── observation_serializer.py
    │   └── observation_deserializer.py
    │
    ├── environment/
    │   ├── validation.py
    │   ├── ue_environment.py
    │   ├── observation.py
    │   └── action.py
    │
    ├── agents/
    │   └── test_agent.py
    │
    └── tests/
        ├── assets/
        │   ├── test_scene.jpg
        │   └── sampled_frames/
        │       ├── frame_000001.png
        │       ├── frame_000002.png
        │       ├── ...
        │       └── frame_000050.png
        │
        ├── test_yolo.py
        ├── test_yolo_inference.py
        ├── test_byte_track.py
        ├── test_track_history.py
        ├── test_perception_pipeline.py
        └── test_sampled_video_perception.py
```

---

# 2. Perception Pipeline

The perception pipeline converts camera images into structured information about objects in the scene.

For every processed frame:

```text
RGB Image
   │
   ▼
YOLOv8
   │
   ├── Class
   ├── Confidence
   └── Bounding Box
          │
          ▼
      ByteTrack
          │
          └── Track ID
                 │
                 ▼
            TrackHistory
                 │
                 └── Position History
```

This provides the temporal information required for downstream trajectory prediction.

---

# 3. YOLOv8 Detection

## Purpose

YOLOv8 is used as the object detector.

For every image, YOLO produces detections containing:

```text
class
confidence
bounding box
```

The bounding box format is:

```text
[x1, y1, x2, y2]
```

where:

* `x1` = left
* `y1` = top
* `x2` = right
* `y2` = bottom

Example:

```text
class      : car
confidence : 0.861
bbox       : [100, 200, 300, 400]
```

---

# 4. YOLO Detection Test

The standalone YOLO inference test verifies that the detector can process an image and return valid detections.

Run:

```powershell
cd F:\Capstone\Implementation\python

python -m tests.test_yolo_inference
```

Example output:

```text
============================================================
YOLO DETECTIONS
============================================================

bicycle         confidence=0.905 bbox=[...]
person          confidence=0.583 bbox=[...]
car             confidence=0.451 bbox=[...]
person          confidence=0.414 bbox=[...]

============================================================
```

This confirms that:

* YOLO model loading works
* Image preprocessing works
* Object detection works
* Bounding boxes are being generated
* Confidence values are being returned
* Object classes are being identified

---

# 5. Raw RGB Image Representation

The perception system is designed to receive camera images as raw RGB data.

The image is represented using:

```text
width
height
channels
raw image bytes
```

The corresponding data structure is:

```python
@dataclass
class ImageData:

    width: int
    height: int
    channels: int
    data: bytes
```

For an RGB image:

```text
channels = 3
```

The raw byte buffer therefore contains:

```text
R G B R G B R G B ...
```

The serialized image data uses little-endian encoding for its numerical fields.

---

# 6. ByteTrack

## Purpose

YOLO detects objects independently in each frame.

Detection alone does not tell us whether:

```text
car in frame 1
```

is the same object as:

```text
car in frame 2
```

ByteTrack solves this problem by assigning persistent tracking IDs.

For example:

```text
Frame 1:
car → ID 6

Frame 2:
car → ID 6

Frame 3:
car → ID 6
```

This allows us to construct an object's temporal movement history.

---

# 7. ByteTrack Output

The tracking layer produces objects in the following format:

```python
{
    "track_id": 6,
    "class_id": 2,
    "class_name": "car",
    "confidence": 0.82,
    "bbox": [
        900,
        500,
        1000,
        600
    ]
}
```

The important addition compared with YOLO detection is:

```text
track_id
```

---

# 8. ByteTrack Test

Run:

```powershell
python -m tests.test_byte_track
```

Example:

```text
============================================================
YOLO + BYTE TRACK
============================================================

FRAME 1
------------------------------------------------------------
track_id=  1 class=bicycle
track_id=  2 class=person
track_id=  3 class=car
track_id=  4 class=person

FRAME 2
------------------------------------------------------------
track_id=  1 class=bicycle
track_id=  2 class=person
track_id=  3 class=car
track_id=  4 class=person

...

============================================================
BYTE TRACK TEST COMPLETE
============================================================
```

This verifies that detections can be passed into ByteTrack and assigned tracking IDs.

---

# 9. TrackHistory

ByteTrack gives us persistent object IDs.

TrackHistory stores the object's recent positions over time.

The center of every bounding box is calculated as:

```text
center_x = (x1 + x2) / 2

center_y = (y1 + y2) / 2
```

For example:

```text
Frame 1 → (150, 250)
Frame 2 → (160, 245)
Frame 3 → (170, 240)
Frame 4 → (180, 235)
Frame 5 → (190, 230)
```

This becomes:

```python
[
    (150, 250),
    (160, 245),
    (170, 240),
    (180, 235),
    (190, 230)
]
```

---

# 10. TrackHistory Storage

Track histories are stored using a bounded deque.

```python
deque(maxlen=max_history)
```

The default maximum history is:

```text
20 positions
```

This prevents the history from growing indefinitely.

For example, with:

```python
max_history = 5
```

after six observations:

```text
Oldest position
      ↓
      X  P2  P3  P4  P5  P6
```

the oldest point is automatically removed.

This keeps only the most recent temporal information.

---

# 11. TrackHistory Test

Run:

```powershell
python -m tests.test_track_history
```

Example:

```text
FRAME 1
track_id=1
  class      : car
  confidence : 0.910
  bbox       : [100, 200, 200, 300]
  history    : [(150.0, 250.0)]

FRAME 2
track_id=1
  history    : [
      (150.0, 250.0),
      (160.0, 245.0)
  ]

...

FRAME 5
track_id=1
  history    : [
      (150.0, 250.0),
      (160.0, 245.0),
      (170.0, 240.0),
      (180.0, 235.0),
      (190.0, 230.0)
  ]
```

The test ends with:

```text
TRACK HISTORY TEST PASSED
```

---

# 12. Temporal Perception

The individual components are then combined into one pipeline:

```text
Frame
  │
  ▼
YOLOv8
  │
  ▼
Detections
  │
  ▼
ByteTrack
  │
  ▼
Tracked Objects
  │
  ▼
TrackHistory
  │
  ▼
Temporal Positions
```

This is important because trajectory prediction requires more than the current position.

For example:

```text
t0 → (100, 500)
t1 → (110, 495)
t2 → (120, 490)
t3 → (130, 485)
```

From this sequence, downstream models can infer:

```text
direction
velocity
movement pattern
trajectory
```

---

# 13. Video Frame Sampling

Processing every frame of a video is unnecessary for the current testing pipeline.

Instead, frames are sampled from the video.

The sampling process extracts a limited number of frames:

```text
Video
  │
  ├── frame 1
  ├── frame 2
  ├── frame 3
  │     ...
  └── frame 295
          │
          ▼
     Sampling
          │
          ▼
     Selected Frames
          │
          ├── frame_000001.png
          ├── frame_000002.png
          ├── ...
          └── frame_000050.png
```

The current test extracted:

```text
Frames read   : 295
Frames saved  : 50
```

Output directory:

```text
tests/assets/sampled_frames/
```

---

# 14. Running the Video Sampler

Run the frame sampling test using the project's sampling script.

The resulting frames should be stored in:

```text
tests/assets/sampled_frames/
```

The files are named sequentially:

```text
frame_000001.png
frame_000002.png
frame_000003.png
...
frame_000050.png
```

The sequential naming is important because temporal tracking depends on processing frames in the correct order.

---

# 15. Sampled Video Perception Test

The complete perception pipeline can then be executed on the sampled frames.

Run:

```powershell
python -m tests.test_sampled_video_perception
```

The pipeline performs:

```text
Sampled Frame
     ↓
RGB Image
     ↓
YOLOv8
     ↓
Detections
     ↓
ByteTrack
     ↓
Track IDs
     ↓
TrackHistory
     ↓
Object Position History
```

---

# 16. Example Complete Pipeline Output

A successful run produces output similar to:

```text
======================================================================
SAMPLED VIDEO PERCEPTION PIPELINE
YOLOv8 → ByteTrack → TrackHistory
======================================================================

Frames directory : ...\tests\assets\sampled_frames
Frames found     : 50

Loading YOLO model...
YOLO model loaded.
```

For each frame:

```text
FRAME 001: frame_000001.png
======================================================================
Image: 1912x1070 RGB

ID=6   car
confidence=0.709
center=(952.7, 566.5)
```

The same track may then appear in subsequent frames:

```text
FRAME 002
ID=6 car
center=(952.7, 566.5)

FRAME 003
ID=6 car
center=(954.7, 575.6)
```

TrackHistory stores:

```text
Track ID: 6

t0: (970.62, 545.60)
t1: (952.74, 566.55)
t2: (954.69, 575.62)
```

---

# 17. What the Current Results Demonstrate

The current tests demonstrate that the following components are functioning:

### YOLOv8

```text
Image
 ↓
Object detections
```

The detector successfully identifies objects such as:

* cars
* trucks
* traffic lights
* bicycles
* pedestrians when detected

---

### ByteTrack

```text
Detections
 ↓
Persistent track IDs
```

Objects can be associated across consecutive frames when the tracker is able to match them.

---

### TrackHistory

```text
Track ID
 ↓
Position sequence
```

The system stores the temporal movement of tracked objects.

---

### Complete Perception Pipeline

```text
RGB Camera Frame
       ↓
    YOLOv8
       ↓
   Detections
       ↓
   ByteTrack
       ↓
 Tracked Objects
       ↓
  TrackHistory
       ↓
Object Trajectories
```

This provides the required intermediate representation for the next stage of the project.

---

# 18. Important Interpretation of Tracking IDs

Tracking IDs are not object identities in the real-world sense.

For example:

```text
ID=6
```

means:

> ByteTrack currently associates this detection with track 6.

It does **not** mean:

> This is permanently object number 6.

Track IDs can change when:

* an object disappears for too long
* detections become unreliable
* objects become heavily occluded
* objects enter/leave the scene
* tracking association fails

Therefore, downstream trajectory processing should account for possible track termination and re-identification.

---

# 19. Detection vs Tracking

These are separate stages.

### Detection

Answers:

> "What objects are visible in this frame?"

Example:

```text
car
person
traffic light
```

### Tracking

Answers:

> "Is this object the same object I saw in previous frames?"

Example:

```text
Frame 1 → car → ID 6
Frame 2 → car → ID 6
Frame 3 → car → ID 6
```

### TrackHistory

Answers:

> "Where has this tracked object been moving?"

Example:

```text
ID 6:

t0 → (970, 545)
t1 → (953, 567)
t2 → (955, 576)
```

---

# 20. Current Architecture

The perception subsystem currently fits into the larger project architecture as:

```text
                     UNREAL ENGINE
                           │
                           │ TCP
                           ▼
                  Python RL Environment
                           │
                           ▼
                    Observation
                           │
                           ▼
                    Raw RGB Image
                           │
                           ▼
                       YOLOv8
                           │
                           ▼
                     Detections
                           │
                           ▼
                      ByteTrack
                           │
                           ▼
                    Track IDs
                           │
                           ▼
                    TrackHistory
                           │
                           ▼
                Object Trajectories
                           │
                           ▼
              Trajectory Prediction
                           │
                           ▼
                 RL Decision Making
                           │
                           ▼
                        Action
                           │
                           │ TCP
                           ▼
                     UNREAL ENGINE
```

---

# 21. Current Status

| Component                           | Status         |
| ----------------------------------- | -------------- |
| Raw RGB image representation        | ✅              |
| Image serialization/deserialization | ✅              |
| YOLOv8 setup                        | ✅              |
| YOLO object detection               | ✅              |
| ByteTrack setup                     | ✅              |
| ByteTrack integration               | ✅              |
| Persistent tracking IDs             | ✅              |
| TrackHistory                        | ✅              |
| Bounding-box center calculation     | ✅              |
| Temporal position storage           | ✅              |
| Video frame sampling                | ✅              |
| Sampled-frame perception pipeline   | ✅              |
| Unreal TCP communication            | ✅              |
| Trajectory prediction               | 🔄 Next stage  |
| RL policy                           | 🔄 Later stage |

---

# 22. Tests Implemented

The current perception testing sequence is:

### Test 1 — YOLO Detection

```powershell
python -m tests.test_yolo_inference
```

Verifies YOLO inference.

---

### Test 2 — ByteTrack

```powershell
python -m tests.test_byte_track
```

Verifies object tracking and track IDs.

---

### Test 3 — TrackHistory

```powershell
python -m tests.test_track_history
```

Verifies temporal position storage.

---

### Test 4 — Complete Perception Pipeline

```powershell
python -m tests.test_perception_pipeline
```

Verifies:

```text
YOLOv8
   ↓
ByteTrack
   ↓
TrackHistory
```

using ordered image frames.

---

### Test 5 — Sampled Video Pipeline

```powershell
python -m tests.test_sampled_video_perception
```

Verifies the complete perception pipeline using frames sampled from a video.

---

# 23. Dependencies

The perception pipeline currently uses:

```text
Python
NumPy
OpenCV
Ultralytics
YOLOv8
ByteTrack
```

Check the installed Ultralytics version with:

```powershell
python -c "import ultralytics; print(ultralytics.__version__)"
```

The current implementation was tested with:

```text
Ultralytics 8.4.129
```

---

# 24. Development Notes

### Always run modules from the `python` directory

Correct:

```powershell
cd F:\Capstone\Implementation\python

python -m tests.test_yolo_inference
```

Incorrect:

```powershell
python tests/test_yolo_inference.py
```

The module-based approach ensures imports such as:

```python
from protocol.models import ImageData
```

resolve correctly.

---

# 25. Next Stage

The perception and tracking pipeline produces the following data:

```text
Track ID
Class
Confidence
Bounding Box
Current Center
Position History
```

The next stage is to convert this temporal information into features suitable for trajectory prediction.

Conceptually:

```text
TrackHistory
     │
     ▼
Position Sequence
     │
     ├── Δx
     ├── Δy
     ├── velocity
     ├── direction
     └── temporal context
          │
          ▼
   Trajectory Predictor
          │
          ▼
 Future Object Positions
```

These predicted trajectories will eventually become part of the state used by the decision-making / reinforcement-learning component.

---

# 26. Final Pipeline

The perception subsystem currently follows:

```text
                 CAMERA
                    │
                    ▼
             Raw RGB Image
                    │
                    ▼
                 YOLOv8
                    │
                    ▼
              Object Detection
                    │
                    ▼
                ByteTrack
                    │
                    ▼
              Persistent IDs
                    │
                    ▼
              TrackHistory
                    │
                    ▼
           Temporal Trajectories
                    │
                    ▼
        ┌───────────────────────┐
        │   NEXT DEVELOPMENT    │
        │                       │
        │ Trajectory Prediction │
        └───────────────────────┘
                    │
                    ▼
             Future Motion
                    │
                    ▼
              RL / Planning
                    │
                    ▼
                 Action
                    │
                    ▼
               UNREAL ENGINE
```

## Current Milestone

**Perception + Tracking pipeline successfully implemented and tested.**

The system can now take ordered camera frames, detect surrounding objects using YOLOv8, associate detections across frames using ByteTrack, and maintain temporal position histories using TrackHistory.

The next major implementation milestone is **trajectory prediction from these tracked histories**.

```

