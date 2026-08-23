# Capstone Autonomous Vehicle (AV) Python Server - Complete Walkthrough

Welcome to the comprehensive documentation for the **Capstone AV Python Server Infrastructure**.

This repository provides the core high-performance networking, binary serialization, protocol framing, state validation, and RL environment abstraction connecting **Unreal Engine 5 (UE5)** with **Python-side Perception & Reinforcement Learning models** (YOLOv8, Trajectron++, and RL driving policy agents).

---

## 1. What Does This Folder Do?

The goal of this repository is to solve the **networking and data synchronization problem** for autonomous vehicle simulation.

Instead of forcing RL and computer vision developers to write C++ socket code or parse raw binary byte streams, this codebase acts as a **transparent, production-grade bridge**:

```text
               Unreal Engine 5 (UE5 Simulator)
                             │
                             │ Binary Observation (Images, Sensors, Speed, State)
                             ▼
              ┌─────────────────────────────┐
              │      Python TCP Server      │
              │  (framing, header, codec)   │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │    Observation Decoder      │
              │ (Camera image, Sensor Data) │
              └──────────────┬──────────────┘
                             │
                             ▼
             ┌───────────────────────────────┐
             │ Perception & Inference Pipeline│
             │   - YOLOv8 (Object Detection) │
             │   - Trajectron++ (Trajectory) │
             │   - RL Agent (Control Policy) │
             └───────────────┬───────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │      Action Serializer      │
              │  (Throttle, Steering, Brake)│
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │      Python TCP Server      │
              └──────────────┬──────────────┘
                             │
                             │ Binary Action Payload
                             ▼
               Unreal Engine 5 (UE5 Vehicle)
```

---

## 2. What Have We Implemented? (Layer-by-Layer Breakdown)

### Layer 1: TCP Transport Layer (`python/transport/`)
* **`tcp_server.py`**: Socket creation, binding, non-blocking/blocking listener, connection acceptance, and stream cleanup.
* **Stream Fragmentation Fix (`receive_exact`)**: Solves the fundamental TCP byte-stream boundary problem where `recv(12)` may return partial data chunks. `receive_exact(size)` guarantees that exactly `size` bytes arrive before returning.

### Layer 2: Binary Protocol & Framing (`python/protocol/`)
* **Protocol Header (`framing.py`)**: 12-byte fixed header encoded as Little-Endian (`<IHHI`):
  * `uint32 magic`: Magic identifier (`0x524C5631`) to verify valid packets.
  * `uint16 version`: Protocol version (`1`).
  * `uint16 message_type`: Integer identifier (`1=OBSERVATION`, `2=ACTION`, `3=RESET`, `4=HELLO`, `5=HELLO_ACK`, `6=ERROR`).
  * `uint32 payload_size`: Byte size of the variable payload immediately following the header.
* **Handshake Protocol**: `HELLO` / `HELLO_ACK` exchange ensuring both UE5 and Python speak the same version before starting simulation.
* **Observation Codec (`observation_serializer.py`, `observation_deserializer.py`)**: Serializes/deserializes `EpisodeID`, `StepID`, `FrameID`, `SimulationTime`, `Reward`, `Termination`, optional JPEG/PNG/Raw `ImageData`, and variable-length array of `Sensor` structs.
* **Action Codec (`action_serializer.py`, `action_deserializer.py`)**: Serializes/deserializes fixed 25-byte vehicle control actions containing `EpisodeID`, `StepID`, `Throttle` ($[-1.0, 1.0]$), `Steering` ($[-1.0, 1.0]$), and `Brake` (`bool`).

### Layer 3: Validation Layer (`python/environment/validation.py`)
* **Sequence Integrity (`validate_observation`)**: Prevents out-of-order execution by verifying `episode_id` and `step_id` monotonically match expectation.
* **Action Safety Bounds (`validate_action`)**: Asserts that `throttle` and `steering` remain strictly in $[-1.0, 1.0]$ and `brake` is boolean. Out-of-bounds predictions raise explicit `InvalidActionError` before reaching Unreal Engine.

### Layer 4: RL Environment API & RESET Protocol (`python/environment/ue_environment.py`)
* **Gym / Gymnasium Compatible `UERLEnvironment`**:
  * `reset()`: Sends `MessageType.RESET` over TCP, increments internal `episode_id`, sets `step_id = 0`, and returns the initial `Observation`.
  * `step(action)`: Validates action, serializes & sends `MessageType.ACTION`, receives next `Observation`, validates sequence, detects episode termination, and returns `(obs, reward, terminated, info)`.

### Layer 5: Simulation & Verification Suite (`python/tests/`)
* **`fake_unreal_episodes.py`**: A fully functional mock Unreal Engine client that mimics multi-episode simulation, handshake, action verification, and reset transitions.
* **Integration Tests (`test_multi_episode_loop.py`, `test_validation.py`, `test_observation.py`, `test_action.py`)**: Automated test suite testing 100% of the communication pipeline.

---

## 3. How This Relates to the Real Project

When transitioning from this test harness to the full production pipeline, **the Python TCP protocol and `UERLEnvironment` remain 100% unchanged**.

Here is how each real component plugs directly into this infrastructure:

```text
               +----------------------------------------+
               |          REAL UNREAL ENGINE 5          |
               | - Vehicle Physics (Chaos Vehicles)     |
               | - Scene Cameras (Scene Capture 2D)     |
               | - Sensors (Speed, Collision, LiDAR)    |
               +----------------───┬────────────────----+
                                   │
                         TCP Protocol (Header + Msg)
                                   │
                                   ▼
               +----------------------------------------+
               |        UERLEnvironment.step()          |
               +----------------───┬────────────────----+
                                   │
                        decodes Observation
                                   │
             ┌─────────────────────┴─────────────────────┐
             ▼                                           ▼
   observation.image                           observation.sensors
             │                                           │
             ▼                                           ▼
     ┌───────────────┐                           ┌───────────────┐
     │    YOLOv8     │ (Bounding Boxes)          │ Trajectron++  │ (Predicted
     │ Object Detect │                           │ Trajectory    │  Histories)
     └───────┬───────┘                           └───────┬───────┘
             │                                           │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │  RL Policy Agent (PyTorch)   │
                   │  Fused Perception Input       │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                       Action(throttle, steering, brake)
                                   │
                                   ▼
               +----------------------------------------+
               |       UERLEnvironment.step(action)     |
               +----------------────────────────--------+
```

### Component Mapping Table:

| Real Project Component | Role in Real Project | How It Connects to Python Infrastructure | Current Mock Representation |
| :--- | :--- | :--- | :--- |
| **UE5 Client** | Runs 3D physics, renders frames, measures speed/collisions. | Connects via TCP to port `9000`, sends `HELLO`, receives `HELLO_ACK`, listens for `RESET`, sends `OBSERVATION`, receives `ACTION`. | `tests/fake_unreal_episodes.py` |
| **YOLOv8** | Detects pedestrians, vehicles, traffic lights from camera images. | Consumes `obs.image.data` (converted to NumPy array via OpenCV/PIL) and outputs bounding boxes. | `obs.image = None` or dummy bytes |
| **Trajectron++** | Predicts trajectories of surrounding dynamic obstacles. | Consumes `obs.sensors` (velocity, position histories) and outputs candidate future trajectories. | `Sensor(SensorType.SPEED, ...)` |
| **RL Model (PPO/SAC)** | Computes throttle, steering, brake values. | Consumes fused features (YOLO + Trajectron + Speed) inside `agent.act(obs)` and returns `Action`. | `agents/test_agent.py` (`MyAgent`) |

---

## 4. How to Make Sure This Mimics Our Real Project

To guarantee that this Python infrastructure behaves identically to the final UE5 production system:

1. **Protocol Rigidity**: Unreal Engine developers implement the exact same 12-byte header (`<IHHI`) and binary payload layouts as defined in `python/protocol/framing.py` and `python/protocol/models.py`.
2. **Step vs. Render Frequency**:
   * UE5 simulation rate: e.g. 60 FPS (16.6 ms/frame).
   * RL Decision rate: e.g. 5 Hz (200 ms/step).
   * UE5 holds the last received action for ~12 simulation frames before sending the next `OBSERVATION` at Step $t+1$.
3. **RESET Lifecycle**: When `env.reset()` is called, Unreal resets vehicle position to spawn point, resets collision flags, increments `episode_id`, sets `step_id = 0`, and sends initial `OBSERVATION`.

---

## 5. How to Check that This Is Running Perfectly (Commands & Expected Output)

### Step 1: Run the Automated Unit & Integration Tests

Open terminal run:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

#### Expected Output:
```text
.......
----------------------------------------------------------------------
Ran 7 tests in 0.658s

OK
```

---

### Step 2: Run Live Server and Multi-Episode Client Interactively

To watch the live TCP communication loop in action, open **two terminal windows**:

#### Terminal 1 (Python Server):
```bash
python main.py --port 9000 --episodes 3
```

#### Terminal 2 (Fake Unreal Multi-Episode Client):
```bash
python tests/fake_unreal_episodes.py --port 9000 --episodes 3 --steps-per-episode 5
```

---

### Step 3: Verify the Live Output Logs

#### Terminal 1 (Server Log Sample):
```text
Python RL Server listening on 127.0.0.1:9000...
Unreal client connected successfully.
Executing HELLO / HELLO_ACK protocol handshake...
Handshake OK!

=================== EPISODE 1 START ===================
[Server] Initial Obs received: ep=1, step=0, sim_time=0.00s

========== AGENT ACT ==========
Episode : 1
Step    : 0
Frame   : 1000
Reward  : 0.5
Sensors: 2
Action generated: throttle = 0.5, steering = 0.0, brake = False
===============================

[Server] Step 1 executed: reward=0.50, terminated=False, info={'reason': '', 'frame_id': 1001, 'simulation_time': 0.2}
...
[Server] Step 5 executed: reward=-10.00, terminated=True, info={'reason': 'Goal reached / Max steps', 'frame_id': 1004, 'simulation_time': 0.8}
=================== EPISODE 1 END ===================
Total Steps: 5 | Total Episode Reward: -8.00

=================== EPISODE 2 START ===================
...
All target episodes completed successfully!
Server closed.
```

#### Terminal 2 (Fake Unreal Client Log Sample):
```text
[FakeUE] Connecting to Python server at 127.0.0.1:9000...
[FakeUE] Connected.
[FakeUE] Sent HELLO. Waiting for HELLO_ACK...
[FakeUE] Handshake OK. Server reply: b'Hello from Python RL Environment!'

================ EPISODE 1 START ================
[FakeUE] Waiting for RESET message from Python environment...
[FakeUE] Received RESET! Initializing episode_id=1, step_id=0
[FakeUE] Sent Initial OBSERVATION (step=0, frame=1000)
[FakeUE] Received ACTION: ep=1, step=0, throttle=0.50, steering=0.00, brake=False
...
[FakeUE] Episode 1 terminated: Goal reached / Max steps
================ EPISODE 2 START ================
...
ALL EPISODES COMPLETED SUCCESSFULLY IN FAKE UNREAL!
[FakeUE] Disconnected.
```

---

## 6. Summary Checklist for Team Hand-off

| Requirement | Status | Verification Command |
| :--- | :---: | :--- |
| **TCP Socket & Packet Boundaries** | Pass | `python -m unittest tests/test_multi_episode_loop.py` |
| **Header & Framing Protocol** | Pass | `python -m unittest tests/test_observation.py` |
| **Observation & Action Serialization** | Pass | `python -m unittest tests/test_action.py` |
| **Sequence Integrity & Range Validation** | Pass | `python -m unittest tests/test_validation.py` |
| **Gym `reset()` & `step()` API** | Pass | `python main.py --port 9000` |
| **Multi-Episode RESET Handshake** | Pass | `python tests/fake_unreal_episodes.py --port 9000` |

---

*This WALKTHROUGH.md serves as the definitive reference document for the Python RL infrastructure.*
