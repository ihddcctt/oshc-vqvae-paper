from __future__ import annotations

import pytest

from lightwave_seeded_vqvae.transport import (
    PacketFormatError,
    build_nonce,
    pack_indices_6bit,
    unpack_indices_6bit,
)


def test_six_bit_packing_is_msb_first_and_canonical() -> None:
    indices = (0, 1, 2, 63)
    packed = pack_indices_6bit(indices)
    assert packed == bytes.fromhex("0010bf")
    assert unpack_indices_6bit(packed, len(indices)) == indices


@pytest.mark.parametrize(
    ("indices", "expected"),
    [
        ((), b""),
        ((0,), b"\x00"),
        ((63,), b"\xfc"),
        ((1, 2), b"\x04\x20"),
        ((63, 63, 63, 63), b"\xff\xff\xff"),
    ],
)
def test_six_bit_roundtrip_boundaries(indices: tuple[int, ...], expected: bytes) -> None:
    assert pack_indices_6bit(indices) == expected
    assert unpack_indices_6bit(expected, len(indices)) == indices


def test_six_bit_parser_rejects_values_lengths_and_nonzero_padding() -> None:
    with pytest.raises(ValueError):
        pack_indices_6bit([64])
    with pytest.raises(ValueError):
        pack_indices_6bit([-1])
    with pytest.raises(PacketFormatError, match="must contain"):
        unpack_indices_6bit(b"\x00\x00", 1)
    with pytest.raises(PacketFormatError, match="padding"):
        unpack_indices_6bit(b"\xfd", 1)


def test_nonce_is_stream_id_followed_by_frame_sequence_msb_first() -> None:
    nonce = build_nonce(stream_id=0x01020304, frame_seq=0x05060708090A0B0C)
    assert nonce == bytes.fromhex("0102030405060708090a0b0c")
    assert len(nonce) == 12
