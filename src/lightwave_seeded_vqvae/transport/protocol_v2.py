"""K128 software-loopback transport v2; the legacy six-bit v1 is unchanged.

Wire layout: 64-byte authenticated header | 12-byte nonce | ciphertext | tag16.
Header (>4sBBBBHHI32s16s): magic, version, bits, dataset, sender state, H, W,
tokens per map, SHA256 model/bank binding, random session identifier.
Dataset 0 is generic; 1/2 require MNIST 7x7 / STL-10 24x24. K_B=K_R=128 is
implicit in v2. Payload: row-major I_B followed immediately by row-major I_R,
MSB-first seven-bit integers, with zero low-bit padding only at the very end.

SessionSender owns a fresh OS-random AES-256 key for one in-memory session.
The 96-bit nonce is fixed32 || counter64be, never reset under that key. Keys
cannot be imported/exported through the sender API. Restart means a fresh key.
This loopback API is not a key-distribution service or physical optical link.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import operator
import os
import struct
import threading

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"LWVQ"
VERSION = 2
INDEX_BITS = 7
HEADER = struct.Struct(">4sBBBBHHI32s16s")
HEADER_SIZE = HEADER.size
NONCE_SIZE = 12
TAG_SIZE = 16
MAX_TOKENS = 1 << 20
MAX_PACKET_SIZE = HEADER_SIZE + NONCE_SIZE + (14 * MAX_TOKENS + 7) // 8 + TAG_SIZE


class ProtocolError(ValueError):
    """Reject malformed or incompatible data without returning indices."""


class AuthenticationError(ProtocolError):
    """Reject unauthenticated data without returning plaintext."""


def _integer(value, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ProtocolError(f"{name} must be an integer, not bool")
    try:
        value = operator.index(value)
    except TypeError:
        raise ProtocolError(f"{name} must be an integer") from None
    if not minimum <= value <= maximum:
        raise ProtocolError(f"{name} outside [{minimum}, {maximum}]")
    return value


def _bytes(value, name: str, size: int | None = None) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ProtocolError(f"{name} must be bytes-like")
    value = bytes(value)
    if size is not None and len(value) != size:
        raise ProtocolError(f"{name} must contain exactly {size} bytes")
    return value


def _shape(shape) -> tuple[int, int]:
    try:
        height, width = shape
    except (TypeError, ValueError):
        raise ProtocolError("shape must contain exactly H and W") from None
    height = _integer(height, "height", 1, 65535)
    width = _integer(width, "width", 1, 65535)
    if height * width > MAX_TOKENS:
        raise ProtocolError("latent exceeds the v2 token limit")
    return height, width


def pack_indices(base, residual) -> bytes:
    """Losslessly serialize two equal, nonempty integer HxW maps."""
    base, residual = np.asarray(base), np.asarray(residual)
    if base.ndim != 2 or base.shape != residual.shape:
        raise ProtocolError("both index maps must have identical HxW shape")
    _shape(base.shape)
    for name, array in (("base", base), ("residual", residual)):
        if array.dtype.kind not in "iu":
            raise ProtocolError(f"{name} indices must have integer dtype")
        if np.any(array < 0) or np.any(array > 127):
            raise ProtocolError(f"{name} indices outside [0, 127]")
    values = np.concatenate((base.ravel(order="C"), residual.ravel(order="C")))
    bits = (values.astype(np.uint8)[:, None] >> np.arange(6, -1, -1)) & 1
    return np.packbits(bits.reshape(-1), bitorder="big").tobytes()


def unpack_indices(payload, shape) -> tuple[np.ndarray, np.ndarray]:
    height, width = _shape(shape)
    payload = _bytes(payload, "payload")
    count = height * width
    valid_bits = 14 * count
    expected = (valid_bits + 7) // 8
    if len(payload) != expected:
        raise ProtocolError(f"payload length must be {expected}, got {len(payload)}")
    padding = expected * 8 - valid_bits
    if padding and payload[-1] & ((1 << padding) - 1):
        raise ProtocolError("nonzero trailing padding bits")
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")
    values = (bits[:valid_bits].reshape(-1, 7) * (1 << np.arange(6, -1, -1))).sum(1)
    return (values[:count].reshape(height, width).astype(np.int64),
            values[count:].reshape(height, width).astype(np.int64))


def model_binding(checkpoint_sha256: str, bank_sha256: str) -> bytes:
    digests = []
    for value in (checkpoint_sha256, bank_sha256):
        if not isinstance(value, str) or len(value) != 64:
            raise ProtocolError("model/bank fingerprint must be SHA-256 hex")
        try:
            digest = bytes.fromhex(value)
        except ValueError:
            raise ProtocolError("model/bank fingerprint must be SHA-256 hex") from None
        if len(digest) != 32:
            raise ProtocolError("model/bank fingerprint must be SHA-256 hex")
        digests.append(digest)
    return hashlib.sha256(b"LWVQ/transport/model-bank/v2\0" + b"".join(digests)).digest()


def budget(shape) -> dict[str, int]:
    height, width = _shape(shape)
    bits = 14 * height * width
    payload_bytes = (bits + 7) // 8
    return dict(height=height, width=width, indices_per_map=height * width,
                total_index_count=2 * height * width, valid_index_bits=bits,
                alignment_padding_bits=payload_bytes * 8 - bits,
                plaintext_bytes=payload_bytes, header_bytes=HEADER_SIZE,
                nonce_bytes=NONCE_SIZE, tag_bytes=TAG_SIZE,
                packet_bytes=payload_bytes + HEADER_SIZE + NONCE_SIZE + TAG_SIZE)


def _metadata(dataset_id, state_id, shape):
    dataset_id = _integer(dataset_id, "dataset_id", 0, 2)
    state_id = _integer(state_id, "sender_state", 0, 9)
    shape = _shape(shape)
    expected = {1: (7, 7), 2: (24, 24)}.get(dataset_id)
    if expected is not None and shape != expected:
        raise ProtocolError("latent shape is incompatible with dataset_id")
    return dataset_id, state_id, shape


@dataclass(frozen=True)
class DecodedFrame:
    base_indices: np.ndarray
    residual_indices: np.ndarray
    sender_state: int
    dataset_id: int
    nonce: bytes


class SessionReceiver:
    """Authenticate first, validate metadata/padding next, then release indices.

    This bounded validation receiver rejects replay within its lifetime. A new
    receiver is a new replay context; replay persistence is outside this study.
    """
    def __init__(self, *, key: bytes, model_id: bytes, session_id: bytes, dataset_id: int):
        self._aes = AESGCM(_bytes(key, "AES-256 key", 32))
        self._model = _bytes(model_id, "model_id", 32)
        self._session = _bytes(session_id, "session_id", 16)
        self._dataset = _integer(dataset_id, "dataset_id", 0, 2)
        self._seen: set[bytes] = set()
        self._lock = threading.Lock()

    def open(self, packet) -> DecodedFrame:
        packet = _bytes(packet, "packet")
        if not HEADER_SIZE + NONCE_SIZE + TAG_SIZE <= len(packet) <= MAX_PACKET_SIZE:
            raise ProtocolError("packet length outside v2 bounds")
        header = packet[:HEADER_SIZE]
        nonce = packet[HEADER_SIZE:HEADER_SIZE + NONCE_SIZE]
        # No field in the untrusted header controls parsing or allocation yet.
        try:
            plaintext = self._aes.decrypt(nonce, packet[HEADER_SIZE + NONCE_SIZE:], header)
        except InvalidTag:
            raise AuthenticationError("AES-GCM authentication failed") from None
        magic, version, bits, dataset, state, height, width, count, model, session = HEADER.unpack(header)
        if magic != MAGIC or version != VERSION:
            raise ProtocolError("unsupported protocol magic/version")
        if bits != INDEX_BITS:
            raise ProtocolError("unsupported index bit width")
        _metadata(dataset, state, (height, width))
        if count != height * width:
            raise ProtocolError("index count inconsistent with latent shape")
        if (model, session, dataset) != (self._model, self._session, self._dataset):
            raise ProtocolError("authenticated model/session/dataset mismatch")
        base, residual = unpack_indices(plaintext, (height, width))
        with self._lock:
            if nonce in self._seen:
                raise ProtocolError("replayed packet")
            self._seen.add(nonce)
        return DecodedFrame(base, residual, state, dataset, nonce)


class SessionSender:
    """Single-process, nonserializable sender with a fresh independent key."""
    def __init__(self, *, model_id: bytes, dataset_id: int):
        self._model = _bytes(model_id, "model_id", 32)
        self._dataset = _integer(dataset_id, "dataset_id", 0, 2)
        self._key = AESGCM.generate_key(bit_length=256)
        self._aes = AESGCM(self._key)
        self._session = os.urandom(16)
        self._prefix = os.urandom(4)
        self._counter = 0
        self._pid = os.getpid()
        self._lock = threading.Lock()

    def __reduce_ex__(self, protocol):
        raise TypeError("sender must not be copied or serialized; create a fresh session")

    def receiver(self) -> SessionReceiver:
        """In-process provisioning for software loopback; does not export a key."""
        return SessionReceiver(key=self._key, model_id=self._model,
                               session_id=self._session, dataset_id=self._dataset)

    def seal(self, base, residual, *, sender_state: int) -> bytes:
        payload = pack_indices(base, residual)
        dataset, state, (height, width) = _metadata(self._dataset, sender_state, np.shape(base))
        header = HEADER.pack(MAGIC, VERSION, INDEX_BITS, dataset, state, height, width,
                             height * width, self._model, self._session)
        with self._lock:
            if os.getpid() != self._pid:
                raise ProtocolError("forked sender must create a fresh key/session")
            if self._counter >= 1 << 64:
                raise ProtocolError("nonce counter exhausted; create a fresh key/session")
            nonce = self._prefix + self._counter.to_bytes(8, "big")
            self._counter += 1  # Consume before encryption; failures cannot reuse it.
            encrypted = self._aes.encrypt(nonce, payload, header)
        return header + nonce + encrypted
