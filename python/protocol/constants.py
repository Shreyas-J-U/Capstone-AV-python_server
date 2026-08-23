from enum import IntEnum


MAGIC = 0x524C5631
PROTOCOL_VERSION = 1


class MessageType(IntEnum):
    OBSERVATION = 1
    ACTION = 2
    RESET = 3
    HELLO = 4
    HELLO_ACK = 5
    ERROR = 6


class SensorType(IntEnum):
    SPEED = 1
    COLLISION = 2
    DISTANCE_TO_OBJECTIVE = 3
    ON_FOOTPATH = 4
    VELOCITY = 5
    CAMERA = 6


class SensorFormat(IntEnum):
    FLOAT32 = 1
    BOOL = 2
    VECTOR3 = 3
    IMAGE = 4