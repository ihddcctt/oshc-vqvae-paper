"""Authenticated transport for serialized VQ index frames.

The transport master key is a separately provisioned cryptographic secret.  This
package intentionally has no dependency on the optical-seed or codebook modules.
"""

from .protocol import (
    AuthenticationError,
    CompatibilityError,
    DecodedIndexFrame,
    GCM_TAG_SIZE,
    HEADER_SIZE,
    INDEX_BITS,
    IndexPacketReceiver,
    IndexPacketSender,
    NonceReuseError,
    PacketFormatError,
    PacketHeader,
    PROTOCOL_VERSION,
    ReplayError,
    TransportError,
    build_nonce,
    derive_model_id,
    derive_session_key,
    pack_indices_6bit,
    unpack_indices_6bit,
)

__all__ = [
    "AuthenticationError",
    "CompatibilityError",
    "DecodedIndexFrame",
    "GCM_TAG_SIZE",
    "HEADER_SIZE",
    "INDEX_BITS",
    "IndexPacketReceiver",
    "IndexPacketSender",
    "NonceReuseError",
    "PacketFormatError",
    "PacketHeader",
    "PROTOCOL_VERSION",
    "ReplayError",
    "TransportError",
    "build_nonce",
    "derive_model_id",
    "derive_session_key",
    "pack_indices_6bit",
    "unpack_indices_6bit",
]
