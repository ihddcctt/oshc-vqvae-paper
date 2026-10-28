"""Independent wire oracle, strict parser checks, and fail-closed AEAD tests."""
import copy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from lightwave_seeded_vqvae.transport import protocol_v2 as p


@pytest.mark.parametrize("shape", [(1, 1), (1, 3), (1, 4), (1, 128), (7, 7), (24, 24)])
@pytest.mark.parametrize("pattern", ["zero", "max", "mixed", "range", "random"])
def test_pack_roundtrip_and_independent_bit_oracle(shape, pattern):
    n = np.prod(shape)
    rng = np.random.default_rng(20261021)
    values = {"zero": np.zeros(n), "max": np.full(n, 127),
              "mixed": np.resize([0, 127, 1, 64, 126], n),
              "range": np.arange(n) % 128, "random": rng.integers(0, 128, n)}[pattern]
    base = values.astype(np.int64).reshape(shape)
    residual = (127 - base).copy()
    packed = p.pack_indices(base, residual)
    text = "".join(f"{int(v):07b}" for v in np.concatenate([base.ravel(), residual.ravel()]))
    text += "0" * (-len(text) % 8)
    oracle = bytes(int(text[i:i + 8], 2) for i in range(0, len(text), 8))
    assert packed == oracle
    b, r = p.unpack_indices(packed, shape)
    assert b.shape == r.shape == shape
    assert np.array_equal(b, base) and np.array_equal(r, residual)


def test_known_answer_and_budget():
    assert p.pack_indices([[0]], [[127]]) == bytes.fromhex("01fc")
    assert p.pack_indices([[127]], [[0]]) == bytes.fromhex("fe00")
    assert p.budget((7, 7)) == dict(height=7, width=7, indices_per_map=49,
        total_index_count=98, valid_index_bits=686, alignment_padding_bits=2,
        plaintext_bytes=86, header_bytes=64, nonce_bytes=12, tag_bytes=16, packet_bytes=178)
    assert p.budget((24, 24))["packet_bytes"] == 1100
    assert p.budget((24, 24))["alignment_padding_bits"] == 0


@pytest.mark.parametrize("value", [-1, 128, 256, 2**64-1, 0.5, 1.0, np.nan, True, "1"])
def test_invalid_index_never_truncated_or_wrapped(value):
    with pytest.raises(p.ProtocolError):
        p.pack_indices([[value]], [[0]])
    with pytest.raises(p.ProtocolError):
        p.pack_indices([[0]], [[value]])


@pytest.mark.parametrize("base,residual", [([], []), ([0], [0]), ([[0]], [[0, 1]]),
                                          (np.zeros((1, 1, 1), dtype=int), [[0]])])
def test_invalid_map_shape(base, residual):
    with pytest.raises(p.ProtocolError):
        p.pack_indices(base, residual)


@pytest.mark.parametrize("shape", [(0, 1), (-1, 1), (1,), (1, 1, 1), (1.0, 1),
                                 (True, 1), (65536, 1), (65535, 65535)])
def test_invalid_decode_shape(shape):
    with pytest.raises(p.ProtocolError):
        p.unpack_indices(b"", shape)


@pytest.mark.parametrize("shape", [(1, 1), (7, 7), (24, 24)])
def test_truncation_trailing_bytes_and_padding(shape):
    data = p.pack_indices(np.zeros(shape, dtype=int), np.zeros(shape, dtype=int))
    for size in range(len(data)):
        with pytest.raises(p.ProtocolError, match="length"):
            p.unpack_indices(data[:size], shape)
    with pytest.raises(p.ProtocolError, match="length"):
        p.unpack_indices(data + b"\0", shape)
    if p.budget(shape)["alignment_padding_bits"]:
        with pytest.raises(p.ProtocolError, match="padding"):
            p.unpack_indices(data[:-1] + bytes([data[-1] | 1]), shape)


def session(dataset=0):
    return p.SessionSender(model_id=p.model_binding("01" * 32, "02" * 32), dataset_id=dataset)


@pytest.mark.parametrize("dataset,shape", [(0, (1, 128)), (1, (7, 7)), (2, (24, 24))])
def test_aes_roundtrip_plaintext_and_header(dataset, shape):
    sender = session(dataset)
    base = np.random.default_rng(123).integers(0, 128, shape)
    residual = 127 - base
    packet = sender.seal(base, residual, sender_state=9)
    fields = p.HEADER.unpack(packet[:64])
    assert fields[:8] == (b"LWVQ", 2, 7, dataset, 9, *shape, np.prod(shape))
    plaintext = AESGCM(sender._key).decrypt(packet[64:76], packet[76:], packet[:64])
    assert plaintext == p.pack_indices(base, residual)
    decoded = sender.receiver().open(packet)
    assert np.array_equal(base, decoded.base_indices)
    assert np.array_equal(residual, decoded.residual_indices)
    assert decoded.sender_state == 9
    assert len(packet) == p.budget(shape)["packet_bytes"]


def test_wrong_key_rejected():
    sender = session()
    packet = sender.seal([[0]], [[127]], sender_state=0)
    wrong = p.SessionReceiver(key=AESGCM.generate_key(bit_length=256), model_id=sender._model,
                              session_id=sender._session, dataset_id=0)
    with pytest.raises(p.AuthenticationError):
        wrong.open(packet)


@pytest.mark.parametrize("byte_position", list(range(64)) + list(range(64, 76)) + [76, 77] + list(range(78, 94)))
def test_every_header_nonce_ciphertext_and_tag_byte_authenticated(byte_position, monkeypatch):
    sender = session()
    packet = bytearray(sender.seal([[0]], [[127]], sender_state=0))
    packet[byte_position] ^= 1
    def forbidden(*args):
        pytest.fail("indices must not be unpacked on authentication failure")
    monkeypatch.setattr(p, "unpack_indices", forbidden)
    with pytest.raises(p.AuthenticationError):
        sender.receiver().open(packet)


def test_malformed_packets_and_no_release():
    sender = session()
    packet = sender.seal([[0]], [[127]], sender_state=0)
    for size in range(len(packet)):
        with pytest.raises(p.ProtocolError):
            sender.receiver().open(packet[:size])
    with pytest.raises(p.ProtocolError):
        sender.receiver().open(packet + b"\0")
    with pytest.raises(p.ProtocolError):
        sender.receiver().open(b"\0" * (p.MAX_PACKET_SIZE + 1))


@pytest.mark.parametrize("field,value", [(0, b"NOPE"), (1, 1), (1, 3), (2, 6), (3, 3),
                                       (4, 10), (5, 0), (6, 0), (7, 3),
                                       (8, b"x" * 32), (9, b"x" * 16)])
def test_authenticated_invalid_metadata_rejected(field, value):
    sender = session()
    packet = sender.seal([[0]], [[127]], sender_state=0)
    fields = list(p.HEADER.unpack(packet[:64]))
    fields[field] = value
    header = p.HEADER.pack(*fields)
    # A separate fresh nonce for each deliberately malformed authenticated packet.
    nonce = sender._prefix + (100).to_bytes(8, "big")
    candidate = header + nonce + AESGCM(sender._key).encrypt(nonce, b"\x01\xfc", header)
    with pytest.raises(p.ProtocolError):
        sender.receiver().open(candidate)


@pytest.mark.parametrize("plaintext", [b"", b"\x01", b"\x01\xfc\0", b"\x01\xfd"])
def test_authenticated_wrong_length_and_padding(plaintext):
    sender = session()
    packet = sender.seal([[0]], [[127]], sender_state=0)
    nonce = sender._prefix + (100).to_bytes(8, "big")
    candidate = packet[:64] + nonce + AESGCM(sender._key).encrypt(nonce, plaintext, packet[:64])
    with pytest.raises(p.ProtocolError):
        sender.receiver().open(candidate)


def test_nonce_lifecycle_replay_and_thread_safety(monkeypatch):
    sender = session()
    with ThreadPoolExecutor(max_workers=8) as executor:
        packets = list(executor.map(lambda _: sender.seal([[0]], [[127]], sender_state=0), range(128)))
    assert len({x[64:76] for x in packets}) == 128
    assert sorted(int.from_bytes(x[68:76], "big") for x in packets) == list(range(128))
    receiver = sender.receiver()
    receiver.open(packets[0])
    with pytest.raises(p.ProtocolError, match="replayed"):
        receiver.open(packets[0])
    sender._counter = 1 << 64
    with pytest.raises(p.ProtocolError, match="exhausted"):
        sender.seal([[0]], [[127]], sender_state=0)
    with pytest.raises(TypeError):
        copy.copy(sender)
    sender._pid = -1
    with pytest.raises(p.ProtocolError, match="forked"):
        sender.seal([[0]], [[127]], sender_state=0)


def test_dataset_shape_and_new_session_key():
    first, second = session(1), session(1)
    assert first._key != second._key
    with pytest.raises(p.ProtocolError, match="shape"):
        first.seal([[0]], [[0]], sender_state=0)
    with pytest.raises(p.ProtocolError):
        first.seal(np.zeros((7, 7), dtype=int), np.zeros((7, 7), dtype=int), sender_state=10)
