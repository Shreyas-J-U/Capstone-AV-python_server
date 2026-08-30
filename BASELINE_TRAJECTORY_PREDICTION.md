# Baseline Trajectory Prediction

## 1. Overview

This milestone implements and validates the **baseline trajectory prediction pipeline** for the autonomous vehicle project.

The pipeline takes video frames, detects objects using **YOLOv8**, assigns persistent identities using **ByteTrack**, stores their movement history using **TrackHistory**, and predicts future positions using a **constant-velocity trajectory predictor**.

```text
Video / Camera Frames
        │
        ▼
     YOLOv8
 Object Detection
        │
        ▼
    ByteTrack
 Object Tracking
        │
        ▼
   TrackHistory
 Position History
        │
        ▼
TrajectoryPredictor
 Constant-Velocity
    Prediction
        │
        ▼
 Future Trajectory
```

This implementation serves as the **baseline prediction system** before integrating a learned trajectory prediction model such as **Trajectron++**.

---

## 2. Objective

The objective of this milestone is to verify that:

* Objects can be detected from real frames.
* Detected objects can be assigned persistent track IDs.
* Object positions can be accumulated over multiple frames.
* A sufficient observation history can be identified.
* Future object positions can be predicted.
* The entire perception-to-prediction pipeline works end-to-end.

The baseline provides a simple reference implementation that can later be compared against **Trajectron++**.

---

## 3. Components

### 3.1 YOLOv8

YOLOv8 performs object detection on each input frame.

The detector provides information such as:

* Object class
* Detection confidence
* Bounding box

Example:

```text
ID=27    car    confidence=0.849
```

The center of the bounding box is subsequently used as the object's position.

---

### 3.2 ByteTrack

ByteTrack associates detections across consecutive frames and assigns track IDs.

Example:

```text
FRAME 043
ID=27   car   center=(1234.3, 544.0)

FRAME 044
ID=27   car   center=(1250.2, 542.6)

FRAME 045
ID=27   car   center=(1270.8, 542.9)
```

The same object therefore maintains the same identity across frames.

This is important because trajectory prediction requires a sequence of positions belonging to the **same object**.

---

### 3.3 TrackHistory

`TrackHistory` stores the center position of each tracked object.

For example:

```text
Track ID 27

t0  → (1137.31, 539.19)
t1  → (1142.91, 539.53)
t2  → (1147.89, 542.77)
...
t14 → (1363.11, 557.07)
t15 → (1403.93, 563.04)
t16 → (1456.90, 569.93)
```

The history is limited using:

```python
max_history=20
```

This prevents unlimited growth of stored trajectory data.

---

## 4. Baseline Trajectory Predictor

The current predictor is intentionally simple.

It uses a **constant-velocity assumption**.

Given the final two observed positions:

```text
(x₁, y₁)
(x₂, y₂)
```

the estimated velocity is:

```text
vx = x₂ - x₁
vy = y₂ - y₁
```

The next position is then estimated as:

```text
x_next = x_current + vx
y_next = y_current + vy
```

This process is repeated for the configured prediction horizon.

### Implementation

The predictor is implemented in:

```text
Implementation/
└── python/
    └── perception/
        └── trajectory_predictor.py
```

Current configuration:

```python
TrajectoryPredictor(
    prediction_horizon=10
)
```

The real trajectory test requests:

```python
PREDICTION_STEPS = 5
```

and the predictor is configured accordingly in the tested implementation.

---

## 5. Prediction Input

The predictor receives a list of observed `(x, y)` positions.

Example:

```text
Observed trajectory:

(1137.31, 539.19)
(1142.91, 539.53)
(1147.89, 542.77)
(1157.65, 547.01)
(1166.81, 546.45)
...
(1403.93, 563.04)
(1456.90, 569.93)
```

The predictor uses the most recent motion to estimate future positions.

---

## 6. Minimum History Requirement

Trajectory prediction is only attempted when an object has enough observations.

Current configuration:

```python
MIN_HISTORY = 5
```

Therefore:

```text
history < 5
        │
        ▼
INSUFFICIENT HISTORY
```

while:

```text
history >= 5
        │
        ▼
VALID TRAJECTORY
        │
        ▼
Prediction
```

This prevents predictions from being generated from extremely short observation sequences.

---

## 7. Real-Frame Test

The complete pipeline was tested using:

```text
50 sampled frames
```

from:

```text
Implementation/
└── python/
    └── tests/
        └── assets/
            └── sampled_frames/
```

The frames were processed sequentially.

Test command:

```powershell
cd F:\Capstone\Implementation\python
python -m tests.test_real_trajectory_prediction
```

---

## 8. Test Pipeline

The test performs the following operations for every frame:

```text
Read image
   │
   ▼
Convert image → ImageData
   │
   ▼
YOLOv8 detection
   │
   ▼
ByteTrack update
   │
   ▼
TrackHistory update
   │
   ▼
Store object centers
```

After all frames are processed:

```text
TrackHistory
     │
     ▼
Retrieve all tracks
     │
     ▼
Check minimum history
     │
     ▼
TrajectoryPredictor
     │
     ▼
Predict future positions
```

---

## 9. Important Integration Fix

During the initial real-frame test, the YOLO detector expected an `ImageData` object but the test was passing a raw NumPy image.

This produced:

```text
AttributeError:
'numpy.ndarray' object has no attribute 'channels'
```

The test was corrected to construct:

```python
image_data = ImageData(
    width=width,
    height=height,
    channels=channels,
    data=image.tobytes(),
)
```

and pass that object to:

```python
detector.detect(image_data)
```

The `ImageData` structure is:

```text
(width, height, channels, data)
```

After this integration fix, YOLO detection and the complete downstream pipeline executed successfully.

---

## 10. Test Results

### Dataset / Input

```text
Frames processed : 50
Image resolution : 1912 × 1070
```

### Tracking / Prediction

```text
Total tracks found       : 15
Valid trajectories       : 12
Successful predictions   : 12
Minimum history required : 5
Prediction horizon       : 5 steps
```

### Final Result

```text
REAL TRAJECTORY PREDICTION TEST COMPLETE
```

Therefore, the complete baseline pipeline successfully executed on real sampled frames.

---

## 11. Example Prediction

### Track ID 27

Class:

```text
car
```

Observed positions near the end of the track:

```text
t13 : (1331.54, 549.96)
t14 : (1363.11, 557.07)
t15 : (1403.93, 563.04)
t16 : (1456.90, 569.93)
```

The baseline predicts:

```text
t+1 : (1509.86, 576.81)
t+2 : (1562.83, 583.70)
t+3 : (1615.80, 590.59)
t+4 : (1668.76, 597.48)
t+5 : (1721.73, 604.37)
```

The predicted points continue the object's recent direction of motion.

---

## 12. Another Example

### Track ID 5

Observed:

```text
t0 : (361.76, 541.33)
t1 : (325.03, 542.21)
t2 : (284.42, 544.10)
t3 : (240.34, 546.92)
t4 : (190.66, 551.22)
```

Predicted:

```text
t+1 : (140.98, 555.53)
t+2 : (91.30, 559.84)
t+3 : (41.63, 564.14)
t+4 : (-8.05, 568.45)
t+5 : (-57.73, 572.75)
```

The negative future coordinates demonstrate an important limitation of the current baseline: it performs **unconstrained mathematical extrapolation** and does not yet understand image boundaries, road geometry, obstacles, or vehicle dynamics.

---

## 13. Tracks With Insufficient History

Not every detected object had enough observations.

For example:

```text
Track ID : 4
Class    : car
Points   : 1
Status   : INSUFFICIENT HISTORY
```

and:

```text
Track ID : 6
Class    : car
Points   : 4
Status   : INSUFFICIENT HISTORY
```

These tracks were correctly excluded from prediction because:

```text
Points < MIN_HISTORY
```

This is expected behavior.

---

## 14. Current Limitations

The current predictor is a **baseline**, not the final trajectory prediction model.

### 14.1 Constant-velocity assumption

The predictor assumes that the object's recent velocity remains constant.

Real objects can:

* Accelerate
* Brake
* Turn
* Change lanes
* Stop
* Change direction

The current model does not explicitly account for these behaviors.

---

### 14.2 Only recent velocity is used

The current prediction is based on the latest two observations:

```text
Previous position
       +
Current position
       ↓
Velocity estimate
```

It does not learn from the complete historical trajectory.

Consequently, noisy detections can influence the predicted path.

---

### 14.3 No scene context

The baseline does not consider:

* Road layout
* Lane boundaries
* Traffic signals
* Other agents
* Pedestrian intent
* Vehicle dynamics
* Map information

It predicts motion purely from image-space coordinates.

---

### 14.4 Image-space prediction

The trajectory currently exists in pixel coordinates:

```text
(x, y)
```

rather than physical/world coordinates.

Therefore:

```text
pixel displacement ≠ physical displacement
```

without appropriate camera calibration, depth estimation, or coordinate transformation.

---

### 14.5 Track interruptions

Objects may temporarily disappear from detections.

For example, the test contained frames with:

```text
No tracked objects.
```

Temporary detection/tracking loss can cause fragmented trajectories and new track IDs.

This is an important issue to address when building the final system.

---

### 14.6 Predictions can leave the image

Because there is currently no spatial constraint, predicted positions can become:

```text
x < 0
```

or:

```text
x > image_width
```

This does not indicate that the predictor crashed; it indicates that the baseline has no scene or image-boundary constraints.

---

## 15. Why This Baseline Is Important

Although simple, this implementation establishes the complete interface required by the future prediction system.

The important separation is:

```text
Perception
    │
    ▼
Track History
    │
    ▼
Prediction Interface
```

The prediction implementation can therefore be replaced later without redesigning the YOLO or tracking components.

Current:

```text
TrackHistory
     │
     ▼
Constant-Velocity Predictor
```

Future:

```text
TrackHistory
     │
     ▼
Trajectron++
```

This makes the baseline useful as a **reference system** for evaluating improvements from the learned predictor.

---

## 16. Current Project Status

### Perception

```text
YOLOv8             ✅
ByteTrack          ✅
TrackHistory       ✅
```

### Prediction

```text
Baseline Predictor     ✅
Real-frame testing     ✅
5-step prediction      ✅
Trajectron++           ⏳
```

### Planning

```text
RL Planner             ⏳
```

### Vehicle Control

```text
Steering / Acceleration    ⏳
```

---

## 17. Milestone Completion

**Milestone: Baseline Trajectory Prediction**

```text
Status: COMPLETE ✅
```

The system has successfully demonstrated:

```text
Real Frames
     ↓
YOLOv8
     ↓
ByteTrack
     ↓
TrackHistory
     ↓
Trajectory Predictor
     ↓
Future Trajectory
```

The next major development step is to replace the constant-velocity baseline with a **learned trajectory prediction model**, primarily **Trajectron++**, while preserving the existing perception and tracking interfaces.

---

## 18. Quick Run Command

From the Python implementation directory:

```powershell
cd F:\Capstone\Implementation\python
```

Run the baseline trajectory test:

```powershell
python -m tests.test_real_trajectory_prediction
```

Expected final section:

```text
====================================================================
TRAJECTORY PREDICTION SUMMARY
====================================================================

Total tracks found       : 15
Valid trajectories       : 12
Successful predictions   : 12
Minimum history required : 5
Prediction horizon       : 5 steps

REAL TRAJECTORY PREDICTION TEST COMPLETE
```

---

## 19. Development Progress

```text
[✓] YOLOv8 object detection
[✓] ByteTrack object tracking
[✓] Track history accumulation
[✓] Baseline trajectory predictor
[✓] Real-frame integration
[✓] End-to-end trajectory prediction
[ ] Improve trajectory estimation
[ ] Trajectron++ integration
[ ] Pedestrian trajectory prediction
[ ] Pedestrian intent prediction
[ ] RL-based planning
[ ] Vehicle control integration
```

**Baseline trajectory prediction is now established as the reference prediction layer for the autonomous navigation pipeline.**
