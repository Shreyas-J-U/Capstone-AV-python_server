# Capstone Autonomous Vehicle (AV) Python Server Infrastructure

Production-grade networking, binary serialization, state validation, and Gym-compatible environment bridge connecting **Unreal Engine 5 (UE5)** with **Python-side Perception & Reinforcement Learning models** (YOLOv8, Trajectron++, PyTorch RL agents).

---

## Quick Start Commands

### 1. Run Automated Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 2. Run Interactive Server & Client (2 Terminals)

#### Terminal 1 (Server):
```bash
python main.py --port 9000 --episodes 3
```

#### Terminal 2 (Fake Unreal Client):
```bash
python -m tests.fake_unreal_episodes --host 127.0.0.1 --port 9000 --episodes 3
```

---

## Key Documentation Links

* **[WALKTHROUGH.md](../WALKTHROUGH.md)**: Deep dive into architecture, layer-by-layer design, and perception model integration (YOLOv8, Trajectron++).
* **[UNREAL_GUIDE.md](../UNREAL_GUIDE.md)**: Official Protocol v1 contract, C++ header structs, and TCP receive helpers for the Unreal Engine developer.
