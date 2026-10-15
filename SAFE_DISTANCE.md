# Dynamic Safe Distance Estimation & Perception Pipeline

## 1. Overview

This document provides a comprehensive technical reference for the **Dynamic Safe Distance Estimation** subsystem implemented within the autonomous vehicle perception pipeline.

The subsystem determines the required physical clearance between the ego-vehicle and surrounding objects dynamically in real time, taking into account:
1. **Ego-vehicle speed** ($v_{\text{ego}}$) streamed from Unreal Engine 5 sensors.
2. **Detected object class / type** (e.g., passenger car, heavy truck, transit bus, motorcycle, pedestrian).
3. **Physical and bounding dimensions** (aspect ratio, length, volume category).
4. **3D lateral lane corridor position** (differentiating in-lane vehicles from adjacent-lane or roadside objects).

```text
Camera / Video Frames + Ego Velocity (UE5 Sensor)
                    │
                    ▼
           YOLOv8 Detection
      (Bboxes, Classes, Confidence)
                    │
                    ▼
          ByteTrack Association
          (Persistent Track IDs)
                    │
                    ▼
       SafeDistanceCalculator (3D Pinhole + Corridor + Dynamics)
          ├── 1. Monocular 3D Position: Longitudinal (Z) & Lateral (X)
          ├── 2. Lane Corridor Filter: IN_PATH vs ADJACENT_LANE vs ROADSIDE
          ├── 3. Dynamic Safe Headway (THW) & Kinematic Braking (RSS)
          └── 4. Safety Status: SAFE / WARNING / CRITICAL
                    │
                    ▼
         Enriched Perception Outputs
       (Telemetry, Margins, Action Limits)
                    │
                    ▼
      Downstream RL Agent / Safety Monitor
```

---

## 2. Theoretical Formulation & Mathematical Models

### 2.1 Model A: Dynamic Time-Headway (THW) with Class & Size Scaling

The Time Headway model is the standard international benchmark (ISO 15623 / ISO 22179) for Forward Collision Warning (FCW) and Adaptive Cruise Control (ACC), extended to include class severity ($\alpha_{\text{type}}$) and physical dimension scaling ($\beta_{\text{size}}$):

$$d_{\text{safe,thw}} = d_{\text{standstill}} + \Big(v_{\text{ego}} \times t_{\text{hw}} \times \alpha_{\text{type}} \times \beta_{\text{size}}\Big)$$

Where:
* $d_{\text{standstill}}$: Standstill buffer when stationary ($2.5\text{ m}$).
* $v_{\text{ego}}$: Current ego speed in $\text{m/s}$ (extracted via `SensorType.SPEED` or `SensorType.VELOCITY`).
* $t_{\text{hw}}$: Nominal baseline time headway ($1.8\text{ s}$).
* $\alpha_{\text{type}}$: Class severity multiplier:
  * **Car / Van**: $\alpha = 1.00$ (Standard braking capability $\approx 7.0\text{ m/s}^2$).
  * **Truck / Commercial**: $\alpha = 1.45$ (High inertia, extended braking distance, rear-swing hazard).
  * **Bus / Transit**: $\alpha = 1.40$ (Passenger safety limits, large blind zones).
  * **Motorcycle / Bicycle**: $\alpha = 1.30 - 1.35$ (High agility, quick stop potential, stability variance).
  * **Pedestrian (VRU)**: $\alpha = 1.60$ (Maximum vulnerability, erratic motion cushion).
* $\beta_{\text{size}}$: Dimensional scaling factor based on physical length and bounding aspect ratio ($0.85$ for compacts up to $1.40$ for articulated long-chassis vehicles).

---

### 2.2 Model B: Responsibility-Sensitive Safety (RSS) Kinematic Braking Model

Based on Mobileye's Responsibility-Sensitive Safety (RSS) formal model, this calculates the worst-case physics stopping distance assuming the target vehicle executes an abrupt emergency brake:

$$d_{\text{safe,rss}} = \max\left(d_{\text{standstill}}, \; v_{\text{ego}}\cdot \tau_{\text{react}} + \frac{v_{\text{ego}}^2}{2 \cdot a_{\text{ego,decel}}} - \frac{v_{\text{target}}^2}{2 \cdot a_{\text{target,decel}}} + d_{\text{standstill}} + \Delta d_{\text{buffer}}\right)$$

Where:
* $\tau_{\text{react}}$: Autonomous system reaction and actuation latency ($0.5\text{ s}$).
* $a_{\text{ego,decel}}$: Ego comfortable maximum deceleration ($6.0\text{ m/s}^2$).
* $a_{\text{target,decel}}$: Prior maximum deceleration of the lead vehicle (Truck: $4.5\text{ m/s}^2$, Car: $7.0\text{ m/s}^2$, Pedestrian: $4.0\text{ m/s}^2$).
* $\Delta d_{\text{buffer}}$: Physical vehicle clearance offset ($\text{length} \times 0.2 \times \beta_{\text{size}}$ applied dynamically during motion).

The final safe distance returned is:
$$d_{\text{safe}} = \max\left(d_{\text{safe,thw}}, \; d_{\text{safe,rss}}\right)$$

---

### 2.3 Model C: Monocular Pinhole Metric Distance Estimation ($Z_{\text{est}}$)

To determine whether the vehicle currently complies with $d_{\text{safe}}$, the perception module converts 2D pixel bounding boxes into metric space:

$$Z_{\text{est}} = \frac{f_y \cdot H_{\text{prior}}}{h_{\text{bbox}}}$$

Where:
* $h_{\text{bbox}} = y_2 - y_1$: Bounding box height in pixels.
* $H_{\text{prior}}$: Standard real-world height prior for the detected class.
* $f_y$: Vertical focal length computed dynamically from the image resolution:
  $$f_y = \frac{\text{image\_width}}{2 \cdot \tan\left(\frac{\text{FOV}_h}{2}\right)}$$

---

### 2.4 Model D: 3D Lateral Position ($X$) & Lane Corridor Filtering

In camera coordinates, the lateral offset $X$ from the ego vehicle centerline is:

$$X = \frac{(x_{\text{center}} - c_x) \cdot Z_{\text{est}}}{f_x}$$

Where:
* $x_{\text{center}} = \frac{x_1 + x_2}{2}$
* $c_x = \frac{\text{image\_width}}{2}$ (principal optical center)

#### Corridor Decision Boundary
With standard road lane width $W_{\text{lane}} = 3.6\text{ m}$ and half-corridor $W_{\text{corridor}} = 1.85\text{ m}$:
* $|X| \le 1.85\text{ m}$: **`IN_PATH`** (Object is in our active driving lane $\rightarrow$ poses longitudinal collision hazard).
* $1.85\text{ m} < X \le 5.45\text{ m}$: **`RIGHT_LANE`** (Vehicle in adjacent right lane $\rightarrow$ safe to pass).
* $X > 5.45\text{ m}$: **`RIGHT_ROADSIDE`** (Sidewalk / parked / infrastructure).
* $-5.45\text{ m} \le X < -1.85\text{ m}$: **`LEFT_LANE`** (Adjacent left lane / oncoming).
* $X < -5.45\text{ m}$: **`LEFT_ROADSIDE`** (Parked vehicles / curb).

---

## 3. Physical Parameters & Priors Configuration

Configured in [configs/perception.yaml](file:///f:/Capstone/Implementation/configs/perception.yaml):

```yaml
camera:
  fov_horizontal_deg: 90.0
  image_width: 1280
  image_height: 720
  fov_vertical_deg: 59.0
  focal_length_px: 640.0

vehicle_priors:
  car:
    height_m: 1.50
    width_m: 1.80
    length_m: 4.50
    alpha: 1.00
    max_decel_ms2: 7.0
  truck:
    height_m: 3.40
    width_m: 2.50
    length_m: 10.00
    alpha: 1.45
    max_decel_ms2: 4.5
  bus:
    height_m: 3.20
    width_m: 2.55
    length_m: 12.00
    alpha: 1.40
    max_decel_ms2: 4.8
  motorcycle:
    height_m: 1.40
    width_m: 0.80
    length_m: 2.10
    alpha: 1.30
    max_decel_ms2: 5.5
  bicycle:
    height_m: 1.40
    width_m: 0.60
    length_m: 1.70
    alpha: 1.35
    max_decel_ms2: 5.0
  person:
    height_m: 1.70
    width_m: 0.50
    length_m: 0.30
    alpha: 1.60
    max_decel_ms2: 4.0

safety_params:
  standstill_distance_m: 2.5
  time_headway_s: 1.8
  reaction_time_s: 0.5
  ego_comfortable_decel_ms2: 6.0
  warning_margin_multiplier: 1.25
  critical_margin_multiplier: 1.00

corridor_params:
  lane_width_m: 3.6
  lateral_corridor_half_width_m: 1.85
```

---

## 4. Problem-Solving Journey & Bias Elimination

### 4.1 The Initial Naive Implementation
In the first iteration, safe distance was calculated using longitudinal distance $Z$ only:
```text
if estimated_distance < safe_distance:
    status = "CRITICAL"
```

### 4.2 The Biased Result (User Observation)
When evaluated against real video frames from CARLA/UE5 (`assets/sampled_frames`), the terminal displayed:
```text
FRAME 001:
  ID=1 | bus  | Dist:  6.58m | SafeDist: 46.25m | [CRITICAL / COLLISION RISK]
  ID=2 | car  | Dist:  9.70m | SafeDist: 27.50m | [CRITICAL / COLLISION RISK]
  ID=3 | car  | Dist: 12.46m | SafeDist: 27.50m | [CRITICAL / COLLISION RISK]
  ID=12| traffic light | Dist: 3.81m | SafeDist: 25.59m | [CRITICAL]
```
**User Feedback:**
> *"I don't think this is an unbiased result. Every outcome shows critical or collision risk, but the frames look safe."*

### 4.3 Root Cause Analysis
Inspecting `frame_000001.png` visually revealed three critical flaws:
1. **The Bus (`ID=1`) and Car (`ID=2`) were in the right lane** driving parallel to the ego vehicle. They were not directly in our path!
2. **The Traffic Light (`ID=12`) was an overhead signal**, not a road obstacle.
3. **Focal Length Mismatch**: The config used $f = 640\text{ px}$ (assumed 720p), but the actual sampled video was $1912 \times 1070$. The true focal length was $f = 956\text{ px}$, causing metric distances to be underestimated by $1.5\times$.

### 4.4 The Unbiased Solution
1. **Integrated 3D Lateral Position ($X$)**:
   * Objects with $|X| > 1.85\text{m}$ are identified as adjacent-lane or roadside objects.
   * Status for adjacent vehicles is mapped to **`[SAFE (RIGHT_LANE)]`** or **`[SAFE (LEFT_LANE)]`**.
2. **Infrastructure Signal Filtering**:
   * `traffic light` and `stop sign` detections are classified as **`[INFO (SIGNAL)]`** rather than collision risks.
3. **Resolution-Adaptive Focal Length**:
   * Focal length automatically adapts to the frame's width ($f_x = \frac{W}{2 \tan(\text{FOV}_h / 2)}$).
4. **Lane-Specific Longitudinal Warnings**:
   * Only vehicles in our driving corridor (`IN_PATH`) trigger following distance warnings.

---

## 5. Output Telemetry Specification

Every tracked detection now outputs the following enriched schema:

```json
{
  "track_id": 1,
  "class_id": 5,
  "class_name": "bus",
  "confidence": 0.89,
  "bbox": [1180.2, 305.1, 1590.4, 615.8],
  "bbox_dimensions": {
    "width_px": 410.2,
    "height_px": 310.7,
    "area_px": 127449.1
  },
  "size_category": "EXTRA_LARGE",
  "size_factor": 1.25,
  "estimated_distance_m": 9.83,
  "lateral_offset_m": 4.50,
  "lane_position": "RIGHT_LANE",
  "in_ego_path": false,
  "safe_distance_m": 46.25,
  "safe_distance_thw_m": 46.25,
  "safe_distance_rss_m": 35.12,
  "distance_margin_m": -36.42,
  "safety_status": "SAFE (RIGHT_LANE)",
  "time_to_collision_s": null
}
```

### Safety Status Definitions:
| Status | Condition | Control Recommendation |
| :--- | :--- | :--- |
| **`SAFE`** | In-path & $Z_{\text{est}} \ge 1.25 \times d_{\text{safe}}$ | Normal cruising speed |
| **`SAFE (RIGHT_LANE)`** | Outside path ($X > 1.85\text{m}$) | Normal cruise, maintain lane |
| **`SAFE (LEFT_ROADSIDE)`** | Outside path ($X < -1.85\text{m}$) | Normal cruise |
| **`INFO (SIGNAL)`** | Traffic signal / infrastructure | Observer mode |
| **`WARNING`** | In-path & $d_{\text{safe}} \le Z_{\text{est}} < 1.25 \times d_{\text{safe}}$ | Release throttle, coast |
| **`CRITICAL`** | In-path & $Z_{\text{est}} < d_{\text{safe}}$ | Engage brakes / AEB evasion |

---

## 6. How to Run & Evaluate

All commands should be executed from the `python/` directory:

```bash
cd f:\Capstone\Implementation\python
```

### 6.1 Unit Test Suite
Verifies mathematical models, dimension scaling, sensor extraction, and state thresholds:
```bash
python -m unittest tests.test_safe_distance
```

### 6.2 Master System Health Check
Verifies networking, gym environment loop, trajectory prediction, and safe distance simultaneously:
```bash
python -m tests.verify_all
```

### 6.3 Real Video Evaluation (Sampled Frames)
Processes real driving frames through YOLOv8, ByteTrack, and the safe distance engine:
```bash
python -m tests.test_real_safe_distance --max-frames 25
```

### 6.4 Real Video Evaluation (Direct MP4 File)
Processes directly from the video file (`assets/test_video.mp4`):
```bash
python -m tests.test_real_safe_distance --source video --max-frames 25
```

### 6.5 Speed Variation Testing
Test how safe distance scales dynamically at different driving speeds:
```bash
# City driving (30 km/h = 8.33 m/s)
python -m tests.test_real_safe_distance --speed 8.33 --max-frames 15

# Highway driving (72 km/h = 20.0 m/s)
python -m tests.test_real_safe_distance --speed 20.0 --max-frames 15
```

---

## 7. Real Video Experimental Outcomes

### Output Summary Table (Real Sampled Frames, 25 Frames Evaluated)

```text
================================================================================
REAL VIDEO SAFE DISTANCE - VEHICLE TRACKING SUMMARY
================================================================================
ID   | Class      | Size        | Lane Position  | Frames | Min Dist   | Max Dist   | Status
--------------------------------------------------------------------------------
1    | bus        | EXTRA_LARGE | RIGHT_LANE     | 13     |   4.13 m   |   9.83 m   | [SAFE (RIGHT_LANE)]
2    | car        | LARGE       | RIGHT_LANE     | 19     |  13.98 m   |  14.92 m   | [SAFE (RIGHT_LANE)]
3    | car        | LARGE       | IN_PATH        | 18     |  17.69 m   |  19.40 m   | [WARNING]
4    | car        | LARGE       | LEFT_ROADSIDE  | 1      |  15.00 m   |  15.00 m   | [SAFE (LEFT_ROADSIDE)]
5    | car        | LARGE       | LEFT_ROADSIDE  | 6      |   9.32 m   |  12.66 m   | [SAFE (LEFT_ROADSIDE)]
7    | car        | LARGE       | IN_PATH        | 8      |  23.24 m   |  27.63 m   | [SAFE]
11   | car        | LARGE       | RIGHT_ROADSIDE | 5      |  15.94 m   |  16.46 m   | [SAFE (RIGHT_ROADSIDE)]
12   | traffic light | SMALL       | IN_PATH        | 3      |   5.69 m   |   7.42 m   | [INFO (SIGNAL)]
================================================================================
```

### Key Observations:
1. **No False-Alarm Braking on Adjacent Traffic**:
   * Bus `ID=1` and Car `ID=2` driving alongside the ego vehicle at $X \approx +3.5\text{m}$ to $+4.5\text{m}$ are correctly classified as **`[SAFE (RIGHT_LANE)]`**.
2. **Accurate Lead Vehicle Identification**:
   * Car `ID=3` is detected at $X = -0.01\text{m}$ (directly in our lane) at $18\text{m}$ distance. At $28\text{km/h}$, it triggers **`[WARNING]`**, advising steady following distance without triggering emergency stops.
3. **Infrastructure Isolation**:
   * Overhead traffic light `ID=12` is isolated as **`[INFO (SIGNAL)]`**, preventing ghost-braking underneath gantry signals.

---

## 8. Integration with Reinforcement Learning & Control

In `UERLEnvironment.step()` or `MyAgent.act()`, the safe distance outputs can be directly integrated to compute reward penalties or enforce safety overrides:

```python
# In RL Agent or Safety Filter
detections = detector.detect_objects(obs.image, ego_speed_ms=ego_speed)

for obj in detections:
    if obj.get("in_ego_path") and obj.get("safety_status") == "CRITICAL":
        # Override RL action with emergency brake
        action.throttle = 0.0
        action.brake = True
        break
```
