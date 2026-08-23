# Capstone Autonomous Vehicle (AV) Python Server Infrastructure

Production-grade networking, binary serialization, state validation, and Gym-compatible environment bridge connecting **Unreal Engine 5 (UE5)** with **Python-side Perception & Reinforcement Learning models** (YOLOv8, Trajectron++, PyTorch RL agents).

---

## Project Architecture

```text
               Unreal Engine 5 (UE5 Simulator)
                             │
                             │ Binary Observation (Camera Images, Sensors, Speed)
                             ▼
              ┌─────────────────────────────┐
              │      Python TCP Server      │
              │  (Framing, Header, Codecs)  │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │    Observation Decoder      │
              │ (Camera Image, Sensor Data) │
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
                             │ Binary Action Payload
                             ▼
               Unreal Engine 5 (UE5 Vehicle)
```

---

## Repository Directory Structure

```text
Implementation/
├── README.md                  # Main setup and developer guide (this file)
├── WALKTHROUGH.md             # Deep-dive architecture and component integration guide
├── UNREAL_GUIDE.md            # Official Protocol v1 contract & C++ code snippets for UE5
└── python/
    ├── main.py                # Server entry point running multi-episode RL loops
    ├── transport/             # Low-level TCP socket listener and packet framing
    │   ├── tcp_server.py
    │   ├── tcp_client.py
    │   └── tcp_socket.py
    ├── protocol/              # Binary codecs, 12-byte protocol header, data models
    │   ├── constants.py       # Header Magic (0x524C5631), MessageType & Sensor enums
    │   ├── framing.py         # Binary header encoder/decoder (<IHHI)
    │   ├── models.py          # Data classes (Observation, Action, Sensor, Termination)
    │   ├── connection.py      # High-level ProtocolConnection abstraction
    │   ├── observation_serializer.py
    │   ├── observation_deserializer.py
    │   ├── action_serializer.py
    │   └── action_deserializer.py
    ├── environment/           # RL Environment abstraction & validation
    │   ├── validation.py      # Monotonic sequence & safety bounds validator
    │   ├── ue_environment.py  # Gym-compatible UERLEnvironment class (reset/step)
    │   ├── observation.py
    │   └── action.py
    ├── agents/                # RL Driving Policy Agent interface
    │   └── test_agent.py      # Baseline MyAgent implementation
    └── tests/                 # Automated test suite & mock Unreal Engine clients
        ├── test_multi_episode_loop.py
        ├── test_validation.py
        ├── test_observation.py
        ├── test_action.py
        ├── fake_unreal_episodes.py     # Multi-episode fake UE5 client
        ├── fake_unreal_observation.py  # Single-episode fake UE5 client
        └── fake_unreal_action_loop.py
```

---

## Quick Start & Verification

### Prerequisites
* Python **3.10** or higher installed.
* Standard Python libraries (no external dependencies required for core networking).

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/Shreyas-J-U/Capstone-AV-python_server.git
cd Capstone-AV-python_server/python
```

---

### Step 2: Run Automated Test Suite

Run the full unit and integration test suite to verify 100% build health:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

#### Expected Output:
```text
.......
----------------------------------------------------------------------
Ran 7 tests in 0.660s

OK
```

---

### Step 3: Run Interactive Multi-Episode Simulation

To run a live simulation loop, open **two terminal windows**:

#### **Terminal 1: Start the Python Server First**
```bash
cd python
python main.py --port 9000 --episodes 3
```

#### **Terminal 2: Start the Fake Unreal Client**
```bash
cd python
python -m tests.fake_unreal_episodes --host 127.0.0.1 --port 9000 --episodes 3
```

#### Expected Output:
Both terminals will exchange `HELLO` / `HELLO_ACK` handshakes, execute `RESET` signals across 3 episodes, generate vehicle action control commands, and report `ALL EPISODES COMPLETED SUCCESSFULLY!`.

---

## Key Documentation Links

* **[WALKTHROUGH.md](WALKTHROUGH.md)**: Explains what this codebase does, layer-by-layer architectural breakdown, and how YOLOv8, Trajectron++, and RL models plug into `UERLEnvironment`.
* **[UNREAL_GUIDE.md](UNREAL_GUIDE.md)**: Specification contract for Unreal Engine 5 developers. Contains C++ header structures (`#pragma pack(push, 1)`), socket receive helpers, and Chaos Vehicle control binding instructions.

---

## Frequently Asked Questions & Troubleshooting

#### Q: I see `ConnectionRefusedError: [WinError 10061]` when running the client.
**A:** In TCP client-server architecture, **the Python server (`main.py`) MUST be running first** before starting the client (`fake_unreal_episodes.py`). Ensure Terminal 1 is active.

#### Q: How do I change the target port or number of episodes?
**A:** Use the command line flags:
```bash
python main.py --host 127.0.0.1 --port 9876 --episodes 5
```

#### Q: Where do I add my custom RL model or PyTorch weights?
**A:** Implement your custom model inside `agents/test_agent.py` or extend the base `RLAgent` class in `rl/agent.py`. The `UERLEnvironment` passes standard `Observation` objects directly to `agent.act(obs)`.
