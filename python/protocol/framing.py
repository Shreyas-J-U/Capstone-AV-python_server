import struct

from .constants import MAGIC, PROTOCOL_VERSION


HEADER = struct.Struct("<IHHI")

HEADER_SIZE = HEADER.size


def encode_message(message_type, payload: bytes) -> bytes:

    return HEADER.pack(
        MAGIC,
        PROTOCOL_VERSION,
        int(message_type),
        len(payload),
    ) + payload


def decode_header(data: bytes):

    if len(data) != HEADER_SIZE:
        raise ValueError(
            f"Invalid header size: {len(data)}"
        )

    (
        magic,
        version,
        message_type,
        payload_size,
    ) = HEADER.unpack(data)

    if magic != MAGIC:
        raise ValueError(
            f"Invalid magic: {hex(magic)}"
        )

    if version != PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported protocol version: {version}"
        )

    return message_type, payload_size