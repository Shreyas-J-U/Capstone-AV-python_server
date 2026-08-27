python/
│
├── main.py
│
├── transport/
│
├── protocol/
│
├── environment/
│
├── perception/              ← NEW
│   ├── __init__.py
│   ├── yolo_detector.py
│   └── tracker.py
│
├── prediction/              ← NEW
│   ├── __init__.py
│   ├── trajectron_model.py
│   └── trajectory_buffer.py
│
├── risk/                    ← NEW
│   ├── __init__.py
│   └── risk_engine.py
│
├── agents/
│   ├── test_agent.py
│   └── ppo_agent.py         ← NEW
│
├── control/                 ← NEW, if needed
│   ├── __init__.py
│   └── safety_controller.py
│
└── tests/
    ├── ...
    ├── test_yolo.py
    ├── test_tracking.py
    ├── test_trajectory.py
    ├── test_risk.py
    └── test_ppo.py