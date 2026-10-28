"""Versioned AES-GCM framing for two streams of six-bit VQ indices.

The 64-byte packet header is the complete AES-GCM associated data (AAD).  The
encrypted payload is the independently byte-aligned, MSB-first packed base
indices followed by the packed residual indices; unused low bits in either
stream are zero.  Authentication always completes before indices are parsed.

``transport_master_key`` must be generated and provisioned independently from
all optical measurements, chip seeds, and codebook-generation inputs.  Those
values are deliberately absent from this module's API.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import operator
import struct
import threading
from collections.abc import Iterable

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


PROTOCOL_MAGIC = b"LWVQ"
PROTOCOL_VERSION = 1
INDEX_BITS = 6
GCM_TAG_SIZE = 16
SESSION_KEY_SIZE = 32

# Exactly 64 bytes, all in network byte order:
# magic, protocol version, flags, header size, model id, session id,
# generator version, state id, index bits, H, W, K_b, K_r, stream id, frame seq.
_HEADER = struct.Struct(">4sBBH16s16sHBBHHHHIQ")
HEADER_SIZE = _HEADER.size
_FLAGS = 0
_HKDF_INFO = b"LIGHTWAVE_SEEDED_VQVAE/transport/aes-gcm/session-key/v1"
_MODEL_ID_CONTEXT = b"LIGHTWAVE_SEEDED_VQVAE/transport/model-id/v1\x00"


class TransportError(Exception):
    """Base class for errors that reject a transport packet."""


class PacketFormatError(TransportError):
    """The packet or packed-index payload is not canonical."""


class CompatibilityError(TransportError):
    """An authenticated packet is incompatible with this receiver."""


class AuthenticationError(TransportError):
    """AES-GCM authentication failed; no payload was released."""


class ReplayError(TransportError):
    """A frame sequence was duplicated or fell outside the replay window."""


class NonceReuseError(TransportError):
    """A sender attempted to reuse or roll back a stream sequence."""


def _bytes(value: bytes | bytearray | memoryview, *, name: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be bytes-like")
    return bytes(value)


def _identifier(value: bytes | bytearray | memoryview, *, name: str) -> bytes:
    result = _bytes(value, name=name)
    if len(result) != 16:
        raise ValueError(f"{name} must contain exactly 16 bytes")
    if not any(result):
        raise ValueError(f"{name} must not be the all-zero identifier")
    return result


def _integer(value: int, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {result}")
    return result


def derive_session_key(
    *,
    transport_master_key: bytes | bytearray | memoryview,
    session_id: bytes | bytearray | memoryview,
) -> bytes:
    """Derive a 256-bit AES session key using HKDF-SHA256.

    The caller is responsible for provisioning ``transport_master_key`` as a
    cryptographic secret independent of the chip/codebook seed.  A distinct
    16-byte session identifier gives a distinct HKDF salt and therefore a
    distinct AES key.
    """

    master_key = _bytes(transport_master_key, name="transport_master_key")
    if len(master_key) < SESSION_KEY_SIZE:
        raise ValueError("transport_master_key must contain at least 32 bytes")
    normalized_session_id = _identifier(session_id, name="session_id")
    return HKDF(
        algorithm=hashes.SHA256(),
        length=SESSION_KEY_SIZE,
        salt=normalized_session_id,
        info=_HKDF_INFO,
    ).derive(master_key)


def derive_model_id(*, checkpoint_sha256: str, base_bank_sha256: str) -> bytes:
    """Bind the compact authenticated model id to full checkpoint and bank hashes."""

    def decode_digest(value: str, name: str) -> bytes:
        hexadecimal = "0123456789abcdefABCDEF"
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in hexadecimal for character in value)
        ):
            raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
        return bytes.fromhex(value)

    checkpoint = decode_digest(checkpoint_sha256, "checkpoint_sha256")
    bank = decode_digest(base_bank_sha256, "base_bank_sha256")
    return hashlib.sha256(_MODEL_ID_CONTEXT + checkpoint + bank).digest()[:16]


def build_nonce(*, stream_id: int, frame_seq: int) -> bytes:
    """Build the unique 96-bit GCM nonce ``stream_id || frame_seq``.

    Both integers use unsigned, big-endian network byte order.  A stream id is
    32 bits and a frame sequence is 64 bits.
    """

    normalized_stream = _integer(
        stream_id, name="stream_id", minimum=0, maximum=(1 << 32) - 1
    )
    normalized_sequence = _integer(
        frame_seq, name="frame_seq", minimum=0, maximum=(1 << 64) - 1
    )
    return struct.pack(">IQ", normalized_stream, normalized_sequence)


def pack_indices_6bit(indices: Iterable[int]) -> bytes:
    """Pack indices in input order as a canonical MSB-first six-bit stream."""

    output = bytearray()
    bit_buffer = 0
    buffered_bits = 0
    for position, value in enumerate(indices):
        normalized = _integer(
            value, name=f"indices[{position}]", minimum=0, maximum=(1 << INDEX_BITS) - 1
        )
        bit_buffer = (bit_buffer << INDEX_BITS) | normalized
        buffered_bits += INDEX_BITS
        while buffered_bits >= 8:
            buffered_bits -= 8
            output.append((bit_buffer >> buffered_bits) & 0xFF)
            bit_buffer &= (1 << buffered_bits) - 1 if buffered_bits else 0
    if buffered_bits:
        output.append((bit_buffer << (8 - buffered_bits)) & 0xFF)
    return bytes(output)


def unpack_indices_6bit(
    packed: bytes | bytearray | memoryview,
    count: int,
) -> tuple[int, ...]:
    """Unpack exactly ``count`` indices and reject extra/non-zero padding."""

    data = _bytes(packed, name="packed")
    normalized_count = _integer(
        count, name="count", minimum=0, maximum=(1 << 63) - 1
    )
    expected_bytes = (normalized_count * INDEX_BITS + 7) // 8
    if len(data) != expected_bytes:
        raise PacketFormatError(
            f"six-bit stream for {normalized_count} indices must contain "
            f"{expected_bytes} bytes, got {len(data)}"
        )

    output: list[int] = []
    bit_buffer = 0
    buffered_bits = 0
    for byte in data:
        bit_buffer = (bit_buffer << 8) | byte
        buffered_bits += 8
        while buffered_bits >= INDEX_BITS and len(output) < normalized_count:
            buffered_bits -= INDEX_BITS
            output.append((bit_buffer >> buffered_bits) & ((1 << INDEX_BITS) - 1))
            bit_buffer &= (1 << buffered_bits) - 1 if buffered_bits else 0

    if len(output) != normalized_count:
        raise PacketFormatError("truncated six-bit index stream")
    if bit_buffer != 0:
        raise PacketFormatError("six-bit index stream has non-zero padding bits")
    return tuple(output)


@dataclass(frozen=True, slots=True)
class PacketHeader:
    """Authenticated metadata for one two-stage index frame.

    ``model_id`` is an opaque 128-bit deployment-bundle fingerprint.  The caller
    should obtain it from a canonical digest that includes the exact checkpoint,
    base-bank artifact, and decode-relevant configuration.  The receiver compares
    it with a locally trusted value; a packet cannot select its own decoder bundle.
    """

    model_id: bytes
    session_id: bytes
    generator_version: int
    state_id: int
    latent_height: int
    latent_width: int
    base_codebook_size: int
    residual_codebook_size: int
    stream_id: int
    frame_seq: int

    @property
    def token_count(self) -> int:
        return self.latent_height * self.latent_width

    @property
    def nonce(self) -> bytes:
        return build_nonce(stream_id=self.stream_id, frame_seq=self.frame_seq)

    @property
    def packed_index_bytes(self) -> int:
        return (self.token_count * INDEX_BITS + 7) // 8

    def validate_authenticated(self, *, state_count: int | None = None) -> None:
        """Validate semantic fields after the header has authenticated."""

        _identifier(self.model_id, name="model_id")
        _identifier(self.session_id, name="session_id")
        _integer(
            self.generator_version,
            name="generator_version",
            minimum=1,
            maximum=(1 << 16) - 1,
        )
        _integer(self.state_id, name="state_id", minimum=0, maximum=(1 << 8) - 1)
        _integer(
            self.latent_height,
            name="latent_height",
            minimum=1,
            maximum=(1 << 16) - 1,
        )
        _integer(
            self.latent_width,
            name="latent_width",
            minimum=1,
            maximum=(1 << 16) - 1,
        )
        _integer(
            self.base_codebook_size,
            name="base_codebook_size",
            minimum=1,
            maximum=1 << INDEX_BITS,
        )
        _integer(
            self.residual_codebook_size,
            name="residual_codebook_size",
            minimum=1,
            maximum=1 << INDEX_BITS,
        )
        build_nonce(stream_id=self.stream_id, frame_seq=self.frame_seq)
        if state_count is not None:
            normalized_state_count = _integer(
                state_count, name="state_count", minimum=1, maximum=1 << 8
            )
            if self.state_id >= normalized_state_count:
                raise CompatibilityError(
                    f"state_id {self.state_id} is outside receiver state_count "
                    f"{normalized_state_count}"
                )

    def to_bytes(self) -> bytes:
        self.validate_authenticated()
        return _HEADER.pack(
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            _FLAGS,
            HEADER_SIZE,
            bytes(self.model_id),
            bytes(self.session_id),
            self.generator_version,
            self.state_id,
            INDEX_BITS,
            self.latent_height,
            self.latent_width,
            self.base_codebook_size,
            self.residual_codebook_size,
            self.stream_id,
            self.frame_seq,
        )

    @classmethod
    def from_bytes(cls, encoded: bytes | bytearray | memoryview) -> "PacketHeader":
        """Parse an untrusted header; do not use its metadata before authentication."""

        data = _bytes(encoded, name="encoded header")
        if len(data) != HEADER_SIZE:
            raise PacketFormatError(
                f"AAD header must contain exactly {HEADER_SIZE} bytes, got {len(data)}"
            )
        (
            magic,
            version,
            flags,
            header_size,
            model_id,
            session_id,
            generator_version,
            state_id,
            index_bits,
            latent_height,
            latent_width,
            base_codebook_size,
            residual_codebook_size,
            stream_id,
            frame_seq,
        ) = _HEADER.unpack(data)
        if magic != PROTOCOL_MAGIC:
            raise CompatibilityError("unrecognized transport magic")
        if version != PROTOCOL_VERSION:
            raise CompatibilityError(
                f"unsupported transport version {version}; expected {PROTOCOL_VERSION}"
            )
        if flags != _FLAGS:
            raise CompatibilityError(f"unsupported transport flags 0x{flags:02x}")
        if header_size != HEADER_SIZE:
            raise CompatibilityError(
                f"unsupported AAD header size {header_size}; expected {HEADER_SIZE}"
            )
        if index_bits != INDEX_BITS:
            raise CompatibilityError(
                f"unsupported index width {index_bits}; expected {INDEX_BITS}"
            )
        return cls(
            model_id=model_id,
            session_id=session_id,
            generator_version=generator_version,
            state_id=state_id,
            latent_height=latent_height,
            latent_width=latent_width,
            base_codebook_size=base_codebook_size,
            residual_codebook_size=residual_codebook_size,
            stream_id=stream_id,
            frame_seq=frame_seq,
        )


@dataclass(frozen=True, slots=True)
class DecodedIndexFrame:
    """Authenticated flat index streams plus their trusted frame header."""

    header: PacketHeader
    base_indices: tuple[int, ...]
    residual_indices: tuple[int, ...]


def _normalized_indices(
    indices: Iterable[int],
    *,
    expected_count: int,
    codebook_size: int,
    name: str,
) -> tuple[int, ...]:
    try:
        values = tuple(indices)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of integer indices") from error
    if len(values) != expected_count:
        raise ValueError(
            f"{name} must contain H*W={expected_count} indices, got {len(values)}"
        )
    normalized: list[int] = []
    for position, value in enumerate(values):
        normalized.append(
            _integer(
                value,
                name=f"{name}[{position}]",
                minimum=0,
                maximum=codebook_size - 1,
            )
        )
    return tuple(normalized)


class _ReplayWindow:
    def __init__(self, size: int) -> None:
        self.size = _integer(size, name="replay_window_size", minimum=1, maximum=4096)
        self._mask = (1 << self.size) - 1
        self._streams: dict[int, tuple[int, int]] = {}
        self._lock = threading.Lock()

    def check_and_mark(self, stream_id: int, frame_seq: int) -> None:
        with self._lock:
            prior = self._streams.get(stream_id)
            if prior is None:
                self._streams[stream_id] = (frame_seq, 1)
                return
            highest, bitmap = prior
            if frame_seq > highest:
                shift = frame_seq - highest
                bitmap = 1 if shift >= self.size else ((bitmap << shift) | 1) & self._mask
                self._streams[stream_id] = (frame_seq, bitmap)
                return

            age = highest - frame_seq
            if age >= self.size:
                raise ReplayError(
                    f"frame {frame_seq} is outside the {self.size}-frame replay window"
                )
            bit = 1 << age
            if bitmap & bit:
                raise ReplayError(f"frame {frame_seq} has already been accepted")
            self._streams[stream_id] = (highest, bitmap | bit)


class IndexPacketSender:
    """Encrypt index frames and enforce monotonic sequence numbers per stream.

    Sequence state is process-local.  A production caller must persist the last
    sequence or allocate a fresh stream id before restarting, so that a session
    key never sees the same 96-bit nonce twice.
    """

    def __init__(
        self,
        *,
        transport_master_key: bytes | bytearray | memoryview,
        model_id: bytes | bytearray | memoryview,
        session_id: bytes | bytearray | memoryview,
        generator_version: int,
        base_codebook_size: int = 64,
        residual_codebook_size: int = 64,
    ) -> None:
        self.model_id = _identifier(model_id, name="model_id")
        self.session_id = _identifier(session_id, name="session_id")
        self.generator_version = _integer(
            generator_version,
            name="generator_version",
            minimum=1,
            maximum=(1 << 16) - 1,
        )
        self.base_codebook_size = _integer(
            base_codebook_size,
            name="base_codebook_size",
            minimum=1,
            maximum=1 << INDEX_BITS,
        )
        self.residual_codebook_size = _integer(
            residual_codebook_size,
            name="residual_codebook_size",
            minimum=1,
            maximum=1 << INDEX_BITS,
        )
        session_key = derive_session_key(
            transport_master_key=transport_master_key,
            session_id=self.session_id,
        )
        self._aesgcm = AESGCM(session_key)
        self._last_sequence_by_stream: dict[int, int] = {}
        self._lock = threading.Lock()

    def seal(
        self,
        base_indices: Iterable[int],
        residual_indices: Iterable[int],
        *,
        state_id: int,
        latent_height: int,
        latent_width: int,
        stream_id: int,
        frame_seq: int,
    ) -> bytes:
        """Serialize and authenticate one frame, returning ``AAD || ciphertext || tag``."""

        header = PacketHeader(
            model_id=self.model_id,
            session_id=self.session_id,
            generator_version=self.generator_version,
            state_id=state_id,
            latent_height=latent_height,
            latent_width=latent_width,
            base_codebook_size=self.base_codebook_size,
            residual_codebook_size=self.residual_codebook_size,
            stream_id=stream_id,
            frame_seq=frame_seq,
        )
        header.validate_authenticated()
        normalized_base = _normalized_indices(
            base_indices,
            expected_count=header.token_count,
            codebook_size=header.base_codebook_size,
            name="base_indices",
        )
        normalized_residual = _normalized_indices(
            residual_indices,
            expected_count=header.token_count,
            codebook_size=header.residual_codebook_size,
            name="residual_indices",
        )
        plaintext = pack_indices_6bit(normalized_base) + pack_indices_6bit(
            normalized_residual
        )
        aad = header.to_bytes()

        with self._lock:
            previous = self._last_sequence_by_stream.get(header.stream_id)
            if previous is not None and header.frame_seq <= previous:
                raise NonceReuseError(
                    f"frame_seq must increase within stream {header.stream_id}; "
                    f"last={previous}, requested={header.frame_seq}"
                )
            ciphertext_and_tag = self._aesgcm.encrypt(header.nonce, plaintext, aad)
            self._last_sequence_by_stream[header.stream_id] = header.frame_seq
        return aad + ciphertext_and_tag


class IndexPacketReceiver:
    """Authenticate, compatibility-check, replay-check, then unpack index frames."""

    def __init__(
        self,
        *,
        transport_master_key: bytes | bytearray | memoryview,
        expected_model_id: bytes | bytearray | memoryview,
        expected_session_id: bytes | bytearray | memoryview,
        expected_generator_version: int,
        expected_base_codebook_size: int = 64,
        expected_residual_codebook_size: int = 64,
        state_count: int = 10,
        replay_window_size: int = 64,
    ) -> None:
        self._transport_master_key = _bytes(
            transport_master_key, name="transport_master_key"
        )
        if len(self._transport_master_key) < SESSION_KEY_SIZE:
            raise ValueError("transport_master_key must contain at least 32 bytes")
        self.expected_model_id = _identifier(expected_model_id, name="expected_model_id")
        self.expected_session_id = _identifier(
            expected_session_id, name="expected_session_id"
        )
        self.expected_generator_version = _integer(
            expected_generator_version,
            name="expected_generator_version",
            minimum=1,
            maximum=(1 << 16) - 1,
        )
        self.expected_base_codebook_size = _integer(
            expected_base_codebook_size,
            name="expected_base_codebook_size",
            minimum=1,
            maximum=1 << INDEX_BITS,
        )
        self.expected_residual_codebook_size = _integer(
            expected_residual_codebook_size,
            name="expected_residual_codebook_size",
            minimum=1,
            maximum=1 << INDEX_BITS,
        )
        self.state_count = _integer(
            state_count, name="state_count", minimum=1, maximum=1 << 8
        )
        self._replay = _ReplayWindow(replay_window_size)

    def open(self, packet: bytes | bytearray | memoryview) -> DecodedIndexFrame:
        """Return indices only after GCM authentication and all rejection checks."""

        encoded = _bytes(packet, name="packet")
        if len(encoded) < HEADER_SIZE + GCM_TAG_SIZE:
            raise PacketFormatError(
                f"packet must contain a {HEADER_SIZE}-byte header and a full GCM tag"
            )
        aad = encoded[:HEADER_SIZE]
        header = PacketHeader.from_bytes(aad)

        # The candidate session id is untrusted until this decryption succeeds.
        try:
            candidate_key = derive_session_key(
                transport_master_key=self._transport_master_key,
                session_id=header.session_id,
            )
        except ValueError:
            # The session id came from unauthenticated bytes.  Do not expose a
            # distinguishable parser/derivation oracle for an invalid candidate.
            raise AuthenticationError("AES-GCM authentication failed") from None
        try:
            plaintext = AESGCM(candidate_key).decrypt(
                header.nonce, encoded[HEADER_SIZE:], aad
            )
        except InvalidTag:
            raise AuthenticationError("AES-GCM authentication failed") from None

        # From here onward every header field and payload byte is authenticated.
        try:
            header.validate_authenticated(state_count=self.state_count)
        except CompatibilityError:
            raise
        except (TypeError, ValueError) as error:
            raise CompatibilityError(
                f"authenticated packet has invalid header metadata: {error}"
            ) from None
        mismatches: list[str] = []
        if header.model_id != self.expected_model_id:
            mismatches.append("model_id")
        if header.session_id != self.expected_session_id:
            mismatches.append("session_id")
        if header.generator_version != self.expected_generator_version:
            mismatches.append("generator_version")
        if header.base_codebook_size != self.expected_base_codebook_size:
            mismatches.append("base_codebook_size")
        if header.residual_codebook_size != self.expected_residual_codebook_size:
            mismatches.append("residual_codebook_size")
        if mismatches:
            raise CompatibilityError(
                "authenticated packet is incompatible in: " + ", ".join(mismatches)
            )

        expected_payload_size = 2 * header.packed_index_bytes
        if len(plaintext) != expected_payload_size:
            raise PacketFormatError(
                f"authenticated payload must contain {expected_payload_size} bytes, "
                f"got {len(plaintext)}"
            )

        # Authentication and compatibility checks precede replay mutation and parsing.
        self._replay.check_and_mark(header.stream_id, header.frame_seq)
        split = header.packed_index_bytes
        base_indices = unpack_indices_6bit(plaintext[:split], header.token_count)
        residual_indices = unpack_indices_6bit(plaintext[split:], header.token_count)
        if any(value >= header.base_codebook_size for value in base_indices):
            raise PacketFormatError("authenticated base index exceeds K_b")
        if any(value >= header.residual_codebook_size for value in residual_indices):
            raise PacketFormatError("authenticated residual index exceeds K_r")
        return DecodedIndexFrame(
            header=header,
            base_indices=base_indices,
            residual_indices=residual_indices,
        )
