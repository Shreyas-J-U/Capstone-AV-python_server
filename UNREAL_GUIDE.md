# Unreal Engine 5 Developer Integration Guide & Protocol Contract (v1)

This document is the **official protocol specification and integration guide** for the Unreal Engine 5 (UE5) developer. 

By adhering to this guide, the UE5 C++/Blueprint simulation will connect seamlessly with the Python Reinforcement Learning (RL) Server without requiring any changes to the Python infrastructure.

---

## 1. High-Level Architecture & Responsibility

```text
              Unreal Engine 5 (Simulator)
                           │
                           │ 1. Connect TCP to Server Port (e.g., 9000)
                           │ 2. Send HELLO, receive HELLO_ACK
                           │
                 +─────────┴─────────+
                 │   EPISODE LOOP    │
                 +─────────┬─────────+
                           │
                           │ 3. Wait for RESET message from Python
                           │ 4. Reset vehicle position, physics & collision state
                           │ 5. Send Initial OBSERVATION (step=0)
                           │
                 +─────────┴─────────+
                 │     STEP LOOP     │
                 +─────────┬─────────+
                           │
                           │ 6. Wait for ACTION message from Python
                           │ 7. Apply throttle, steering & brake to Chaos Vehicle
                           │ 8. Simulate for 200 ms (e.g. 12 physics frames @ 60 FPS)
                           │ 9. Measure new speed, position, camera & collision
                           │ 10. Send next OBSERVATION (step=t+1)
                           │ 11. Repeat until episode termination
                           ▼
```

---

## 2. Official Protocol v1 Specification

All TCP communication uses **Little-Endian byte order** (`<` in Python / standard `x86_64` format).

### 2.1 Fixed Protocol Header (12 Bytes)

Every message sent across the socket **MUST** begin with this 12-byte header:

```text
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                          Magic (uint32)                       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        Version (uint16)       |      MessageType (uint16)     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       PayloadSize (uint32)                    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

| Field | C++ Type | Size | Description |
| :--- | :--- | :--- | :--- |
| `Magic` | `uint32` | 4 Bytes | Must be `0x524C5631` (ASCII `'RLV1'`). |
| `Version` | `uint16` | 2 Bytes | Protocol version. Must be `1`. |
| `MessageType` | `uint16` | 2 Bytes | Message type identifier (see table below). |
| `PayloadSize` | `uint32` | 4 Bytes | Size in bytes of the payload following this header. |

#### Message Types (`MessageType` Enum):
* `1` = `OBSERVATION` (UE5 $\rightarrow$ Python)
* `2` = `ACTION` (Python $\rightarrow$ UE5)
* `3` = `RESET` (Python $\rightarrow$ UE5)
* `4` = `HELLO` (UE5 $\rightarrow$ Python)
* `5` = `HELLO_ACK` (Python $\rightarrow$ UE5)
* `6` = `ERROR`

---

### 2.2 HELLO / HELLO_ACK Handshake

Upon opening the TCP socket, UE5 **MUST** send `HELLO` (`MessageType = 4`).

* **Payload**: Raw UTF-8 string bytes (e.g. `b"Hello from Unreal Engine 5!"`).
* **Python Reply**: `HELLO_ACK` (`MessageType = 5`) with payload `b"Hello from Python RL Environment!"`.

---

### 2.3 RESET Message (`MessageType = 3`)

Sent from Python to UE5 to initiate a new episode.

* **Payload**: Empty (`0` bytes) or optional configuration bytes.
* **UE5 Action Requirements**:
  1. Reset vehicle transform to start location.
  2. Reset vehicle linear/angular velocity to zero.
  3. Reset collision counters and objective timers.
  4. Increment local `episode_id`, set `step_id = 0`.
  5. Immediately capture and send initial `OBSERVATION` (`step_id = 0`).

---

### 2.4 OBSERVATION Payload Layout (`MessageType = 1`)

Sent from UE5 to Python containing simulator state, camera image, and sensors.

| Offset / Order | Field Name | C++ Type | Size | Notes |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `episode_id` | `uint64` | 8 Bytes | Current episode ID (starts at 1). |
| 2 | `step_id` | `uint64` | 8 Bytes | Step ID within episode (starts at 0). |
| 3 | `frame_id` | `uint64` | 8 Bytes | UE5 frame counter / tick number. |
| 4 | `simulation_time` | `double` | 8 Bytes | Total simulation time in seconds. |
| 5 | `reward` | `float` | 4 Bytes | Reward calculated for previous action. |
| 6 | `terminated` | `uint8` | 1 Byte | `0` = False, `1` = True (collision, goal reached, off road). |
| 7 | `reason_length` | `uint32` | 4 Bytes | Length of termination reason string ($N$). |
| 8 | `reason` | `char[N]` | $N$ Bytes | UTF-8 encoded string (e.g. `"Collision with curb"`). |
| 9 | `image_width` | `uint32` | 4 Bytes | Camera image width in pixels ($0$ if no image). |
| 10 | `image_height` | `uint32` | 4 Bytes | Camera image height in pixels ($0$ if no image). |
| 11 | `image_channels` | `uint16` | 2 Bytes | Number of channels (e.g. $3$ for RGB, $4$ for RGBA). |
| 12 | `image_data_length` | `uint32` | 4 Bytes | Byte length of image buffer ($M$). |
| 13 | `image_data` | `uint8[M]`| $M$ Bytes | Compressed JPEG/PNG or raw camera bytes. |
| 14 | `sensor_count` | `uint16` | 2 Bytes | Number of sensor entries following ($K$). |

#### Sensor Entry Binary Structure (Repeated $K$ times):
| Field | C++ Type | Size | Notes |
| :--- | :--- | :--- | :--- |
| `sensor_type` | `uint16` | 2 Bytes | `1`=SPEED, `2`=COLLISION, `3`=DISTANCE_TO_OBJECTIVE, `4`=ON_FOOTPATH, `5`=VELOCITY, `6`=CAMERA |
| `sensor_format` | `uint16` | 2 Bytes | `1`=FLOAT32, `2`=BOOL, `3`=VECTOR3, `4`=IMAGE |
| `sensor_data_length`| `uint32` | 4 Bytes | Byte size of raw value ($L$). |
| `sensor_value` | `uint8[L]`| $L$ Bytes | Binary value (e.g. 4 bytes for float speed). |

---

### 2.5 ACTION Payload Layout (`MessageType = 2`)

Sent from Python to UE5 containing control commands for the vehicle. **Fixed size: 25 Bytes**.

| Offset | Field Name | C++ Type | Size | Range / Constraints |
| :--- | :--- | :--- | :--- | :--- |
| 0 | `episode_id` | `uint64` | 8 Bytes | Must match active episode ID. |
| 8 | `step_id` | `uint64` | 8 Bytes | Must match active step ID. |
| 16 | `throttle` | `float` | 4 Bytes | Normalized throttle/reverse range $[-1.0, 1.0]$. |
| 20 | `steering` | `float` | 4 Bytes | Normalized steering angle $[-1.0, 1.0]$. |
| 24 | `brake` | `uint8` | 1 Byte | `0` = False, `1` = True. |

---

## 3. C++ Reference Implementation for Unreal Engine

### 3.1 Header Data Structure (Packing `1`)

Add this header structure to your UE5 C++ plugin / module:

```cpp
#pragma pack(push, 1)
struct FProtocolHeader
{
    uint32 Magic       = 0x524C5631; // 'RLV1'
    uint16 Version     = 1;
    uint16 MessageType = 0;
    uint32 PayloadSize = 0;
};
#pragma pack(pop)
```

---

### 3.2 Reliable Exact Socket Receive Helper

TCP is a byte stream and may fragment packets. Use this helper method in C++ to receive full messages:

```cpp
bool ReceiveExact(FSocket* Socket, uint8* Destination, int32 TotalBytes)
{
    int32 BytesReceivedTotal = 0;
    while (BytesReceivedTotal < TotalBytes)
    {
        int32 BytesRead = 0;
        bool bReadSuccess = Socket->Recv(
            Destination + BytesReceivedTotal,
            TotalBytes - BytesReceivedTotal,
            BytesRead
        );

        if (!bReadSuccess || BytesRead <= 0)
        {
            UE_LOG(LogTemp, Error, TEXT("TCP Socket disconnected or error while reading exact bytes."));
            return false;
        }

        BytesReceivedTotal += BytesRead;
    }
    return true;
}
```

---

### 3.3 Receiving and Applying Vehicle Actions in UE5

```cpp
#include "ChaosVehicleMovementComponent.h"

void AAVVehiclePawn::ApplyActionFromPython(const TArray<uint8>& ActionPayload, uint64 CurrentEp, uint64 CurrentStep)
{
    if (ActionPayload.Num() < 25)
    {
        UE_LOG(LogTemp, Error, TEXT("Invalid Action payload size: %d bytes (expected 25)"), ActionPayload.Num());
        return;
    }

    FMemoryReader Reader(ActionPayload, true);
    
    uint64 ReceivedEpisode = 0;
    uint64 ReceivedStep = 0;
    float Throttle = 0.0f;
    float Steering = 0.0f;
    uint8 bBrake = 0;

    Reader << ReceivedEpisode;
    Reader << ReceivedStep;
    Reader << Throttle;
    Reader << Steering;
    Reader << bBrake;

    // Validate sequence
    check(ReceivedEpisode == CurrentEp);
    check(ReceivedStep == CurrentStep);

    // Apply to Chaos Vehicle Movement Component
    UChaosVehicleMovementComponent* VehicleMovement = GetVehicleMovementComponent();
    if (VehicleMovement)
    {
        if (Throttle >= 0.0f)
        {
            VehicleMovement->SetThrottleInput(Throttle);
            VehicleMovement->SetBrakeInput(bBrake ? 1.0f : 0.0f);
        }
        else
        {
            // Reverse controls
            VehicleMovement->SetThrottleInput(0.0f);
            VehicleMovement->SetBrakeInput(FMath::Abs(Throttle));
        }

        VehicleMovement->SetSteeringInput(Steering);
    }
}
```

---

## 4. How to Test Your UE5 Client Against the Python Server

You do **not** need to modify any Python code. The Python server is ready and running.

### Step 1: Start the Python Server
In a terminal on your development machine:

```bash
python main.py --port 9000 --episodes 3
```

You will see:
```text
Python RL Server listening on 127.0.0.1:9000...
Waiting for Unreal client...
```

```bash
# Start Fake UE Client (separate terminal)
python -m tests.fake_unreal_episodes --port 9000 --episodes 3
```

### Step 2: Connect UE5
1. Open your UE5 project and trigger your C++ TCP client component targeting `127.0.0.1:9000`.
2. UE5 sends `HELLO`.
3. UE5 receives `HELLO_ACK`.
4. UE5 waits for `RESET`.
5. UE5 sends `OBSERVATION` (`step_id = 0`).
6. UE5 receives `ACTION`, applies throttle/steering/brake, advances simulation, and sends next `OBSERVATION`.

If all steps complete without errors, **your Unreal Engine integration is 100% complete!**
