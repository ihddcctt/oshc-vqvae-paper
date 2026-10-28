from __future__ import annotations

import struct

import pytest

from lightwave_seeded_vqvae.transport import (
    AuthenticationError,
    CompatibilityError,
    HEADER_SIZE,
    INDEX_BITS,
    IndexPacketReceiver,
    IndexPacketSender,
    NonceReuseError,
    PacketHeader,
    PROTOCOL_VERSION,
    ReplayError,
    derive_model_id,
    derive_session_key,
)


MASTER_KEY = bytes(range(32))
OTHER_MASTER_KEY = bytes(range(1, 33))
MODEL_ID = bytes.fromhex("00112233445566778899aabbccddeeff")
OTHER_MODEL_ID = bytes.fromhex("10112233445566778899aabbccddeeff")
SESSION_ID = bytes.fromhex("ffeeddccbbaa99887766554433221100")
OTHER_SESSION_ID = bytes.fromhex("efeeddccbbaa99887766554433221100")


def make_sender(**overrides) -> IndexPacketSender:
    arguments = {
        "transport_master_key": MASTER_KEY,
        "model_id": MODEL_ID,
        "session_id": SESSION_ID,
        "generator_version": 3,
        "base_codebook_size": 64,
        "residual_codebook_size": 64,
    }
    arguments.update(overrides)
    return IndexPacketSender(**arguments)


def make_receiver(**overrides) -> IndexPacketReceiver:
    arguments = {
        "transport_master_key": MASTER_KEY,
        "expected_model_id": MODEL_ID,
        "expected_session_id": SESSION_ID,
        "expected_generator_version": 3,
        "expected_base_codebook_size": 64,
        "expected_residual_codebook_size": 64,
        "state_count": 10,
    }
    arguments.update(overrides)
    return IndexPacketReceiver(**arguments)


def seal_frame(
    sender: IndexPacketSender,
    *,
    frame_seq: int = 7,
    stream_id: int = 5,
    state_id: int = 4,
) -> bytes:
    return sender.seal(
        [0, 1, 2, 63],
        [63, 2, 1, 0],
        state_id=state_id,
        latent_height=2,
        latent_width=2,
        stream_id=stream_id,
        frame_seq=frame_seq,
    )


def test_header_layout_is_fixed_versioned_and_exactly_64_bytes() -> None:
    packet = seal_frame(make_sender())
    aad = packet[:HEADER_SIZE]
    assert HEADER_SIZE == 64
    assert aad[:4] == b"LWVQ"
    assert aad[4] == PROTOCOL_VERSION
    assert aad[5] == 0
    assert struct.unpack(">H", aad[6:8])[0] == HEADER_SIZE
    assert aad[8:24] == MODEL_ID
    assert aad[24:40] == SESSION_ID
    assert struct.unpack(">H", aad[40:42])[0] == 3
    assert aad[42] == 4
    assert aad[43] == INDEX_BITS
    assert struct.unpack(">HHHHIQ", aad[44:64]) == (2, 2, 64, 64, 5, 7)
    assert PacketHeader.from_bytes(aad).nonce == bytes.fromhex(
        "000000050000000000000007"
    )


def test_hkdf_session_key_is_deterministic_and_session_separated() -> None:
    first = derive_session_key(
        transport_master_key=MASTER_KEY, session_id=SESSION_ID
    )
    again = derive_session_key(
        transport_master_key=MASTER_KEY, session_id=SESSION_ID
    )
    other_session = derive_session_key(
        transport_master_key=MASTER_KEY, session_id=OTHER_SESSION_ID
    )
    other_master = derive_session_key(
        transport_master_key=OTHER_MASTER_KEY, session_id=SESSION_ID
    )
    assert len(first) == 32
    assert first == again
    assert first != other_session
    assert first != other_master


def test_model_id_binds_checkpoint_and_base_bank_hashes() -> None:
    checkpoint = "01" * 32
    bank = "02" * 32
    model_id = derive_model_id(
        checkpoint_sha256=checkpoint, base_bank_sha256=bank
    )
    assert len(model_id) == 16
    assert model_id == derive_model_id(
        checkpoint_sha256=checkpoint, base_bank_sha256=bank
    )
    assert model_id != derive_model_id(
        checkpoint_sha256="03" * 32, base_bank_sha256=bank
    )
    assert model_id != derive_model_id(
        checkpoint_sha256=checkpoint, base_bank_sha256="04" * 32
    )
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        derive_model_id(checkpoint_sha256="not-a-digest", base_bank_sha256=bank)


def test_correct_aes_roundtrip_preserves_both_index_streams() -> None:
    packet = seal_frame(make_sender())
    decoded = make_receiver().open(packet)
    assert decoded.base_indices == (0, 1, 2, 63)
    assert decoded.residual_indices == (63, 2, 1, 0)
    assert decoded.header.state_id == 4
    assert (decoded.header.latent_height, decoded.header.latent_width) == (2, 2)


def test_wrong_key_rejects_without_consuming_replay_state() -> None:
    packet = seal_frame(make_sender())
    wrong_receiver = make_receiver(transport_master_key=OTHER_MASTER_KEY)
    with pytest.raises(AuthenticationError):
        wrong_receiver.open(packet)

    receiver = make_receiver()
    tampered = bytearray(packet)
    tampered[HEADER_SIZE] ^= 0x01
    with pytest.raises(AuthenticationError):
        receiver.open(tampered)
    assert receiver.open(packet).base_indices == (0, 1, 2, 63)


@pytest.mark.parametrize("mutation", ["truncate", "append"])
def test_truncated_or_extended_ciphertext_and_tag_fail_authentication(
    mutation: str,
) -> None:
    packet = seal_frame(make_sender())
    malformed = packet[:-1] if mutation == "truncate" else packet + b"\x00"
    with pytest.raises(AuthenticationError):
        make_receiver().open(malformed)


@pytest.mark.parametrize("location", [8, HEADER_SIZE, -1])
def test_one_bit_flip_in_aad_ciphertext_or_tag_fails_authentication(location: int) -> None:
    packet = bytearray(seal_frame(make_sender()))
    packet[location] ^= 0x01
    with pytest.raises(AuthenticationError):
        make_receiver().open(packet)


def test_invalid_unauthenticated_session_identifier_is_generic_auth_failure() -> None:
    packet = bytearray(seal_frame(make_sender()))
    packet[24:40] = bytes(16)
    with pytest.raises(AuthenticationError):
        make_receiver().open(packet)


def test_protocol_version_and_authenticated_metadata_mismatch_are_distinct() -> None:
    packet = bytearray(seal_frame(make_sender()))
    packet[4] = PROTOCOL_VERSION + 1
    with pytest.raises(CompatibilityError, match="version"):
        make_receiver().open(packet)

    valid_packet = seal_frame(make_sender(), frame_seq=8)
    with pytest.raises(CompatibilityError, match="model_id"):
        make_receiver(expected_model_id=OTHER_MODEL_ID).open(valid_packet)
    with pytest.raises(CompatibilityError, match="session_id"):
        make_receiver(expected_session_id=OTHER_SESSION_ID).open(valid_packet)
    with pytest.raises(CompatibilityError, match="generator_version"):
        make_receiver(expected_generator_version=4).open(valid_packet)


def test_sender_rejects_nonce_reuse_and_sequence_rollback() -> None:
    sender = make_sender()
    seal_frame(sender, frame_seq=10)
    with pytest.raises(NonceReuseError):
        seal_frame(sender, frame_seq=10)
    with pytest.raises(NonceReuseError):
        seal_frame(sender, frame_seq=9)


def test_receiver_replay_window_accepts_fresh_reordering_once() -> None:
    sender = make_sender()
    packets = [seal_frame(sender, frame_seq=sequence) for sequence in range(6)]
    receiver = make_receiver(replay_window_size=4)

    assert receiver.open(packets[5]).header.frame_seq == 5
    assert receiver.open(packets[3]).header.frame_seq == 3
    with pytest.raises(ReplayError, match="already"):
        receiver.open(packets[3])
    with pytest.raises(ReplayError, match="outside"):
        receiver.open(packets[0])


def test_authenticated_state_and_codebook_incompatibility_are_rejected() -> None:
    state_packet = seal_frame(make_sender(), state_id=9)
    with pytest.raises(CompatibilityError, match="state_id"):
        make_receiver(state_count=5).open(state_packet)

    small_sender = make_sender(base_codebook_size=32)
    small_packet = small_sender.seal(
        [0, 1, 2, 31],
        [3, 4, 5, 6],
        state_id=1,
        latent_height=2,
        latent_width=2,
        stream_id=9,
        frame_seq=1,
    )
    with pytest.raises(CompatibilityError, match="base_codebook_size"):
        make_receiver().open(small_packet)
