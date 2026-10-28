from __future__ import annotations

import hashlib

import numpy as np
import pytest

from lightwave_seeded_vqvae.cli.evaluate_aes_gcm_ablation import (
    REGISTERED_PROBE_CASES,
    build_parser,
    evaluation_scope,
    frame_format_summary,
    run_authentication_probes,
    timing_summary,
    validate_artifact_compatibility,
)
from lightwave_seeded_vqvae.seed.codebook import GENERATOR_VERSION
from lightwave_seeded_vqvae.transport import IndexPacketReceiver, IndexPacketSender


MASTER_KEY = bytes(range(32))
MODEL_ID = bytes.fromhex("00112233445566778899aabbccddeeff")
SESSION_ID = bytes.fromhex("ffeeddccbbaa99887766554433221100")


def test_cli_marks_smoke_explicitly_and_defaults_to_formal() -> None:
    common = [
        "--checkpoint",
        "selected.pt",
        "--codebook",
        "bank.npz",
        "--manifest",
        "bank.manifest.json",
        "--output",
        "aes.json",
    ]
    formal = build_parser().parse_args(common)
    smoke = build_parser().parse_args(common + ["--smoke-max-batches", "1"])
    assert evaluation_scope(formal.smoke_max_batches) == "formal"
    assert evaluation_scope(smoke.smoke_max_batches) == "smoke"
    with pytest.raises(SystemExit):
        build_parser().parse_args(common + ["--smoke-max-batches", "0"])


def test_mnist_frame_format_reports_plaintext_and_authenticated_overhead() -> None:
    report = frame_format_summary(
        latent_height=7,
        latent_width=7,
        base_codebook_size=64,
        residual_codebook_size=64,
    )
    assert report["tokens_per_frame"] == 49
    assert report["nominal_index_bits_per_frame"] == 588
    assert report["base_packed_bytes"] == 37
    assert report["residual_packed_bytes"] == 37
    assert report["plaintext_bytes"] == 74
    assert report["aad_header_bytes"] == 64
    assert report["gcm_tag_bytes"] == 16
    assert report["authenticated_packet_bytes"] == 154
    assert report["authenticated_overhead_bytes"] == 80


def test_timing_summary_reports_total_and_per_frame_values() -> None:
    report = timing_summary(2_000_000, 4)
    assert report["total_seconds"] == pytest.approx(0.002)
    assert report["mean_microseconds_per_frame"] == pytest.approx(500.0)
    assert report["frames_per_second"] == pytest.approx(2000.0)


def _artifact_fixture() -> tuple[dict, np.ndarray, dict]:
    generator = np.random.default_rng(7)
    bank = generator.normal(size=(10, 64, 64)).astype(np.float32)
    bank /= np.linalg.norm(bank, axis=-1, keepdims=True)
    canonical = np.ascontiguousarray(bank, dtype="<f4")
    bank_hash = hashlib.sha256(canonical.tobytes()).hexdigest()
    checkpoint = {
        "config": {
            "dataset": {"name": "mnist", "root": "data", "download": False},
            "model": {
                "quantizer_type": "seeded_residual",
                "latent_dim": 64,
                "base_codebook_size": 64,
                "residual_codebook_size": 64,
            },
        },
        "model": {},
        "base_bank_array_sha256": bank_hash,
    }
    manifest = {
        "shape": [10, 64, 64],
        "generator_version": GENERATOR_VERSION,
        "bank_array_sha256": bank_hash,
        "states": [{"index": index} for index in range(10)],
    }
    return checkpoint, bank, manifest


def test_artifact_validation_binds_checkpoint_manifest_and_bank() -> None:
    checkpoint, bank, manifest = _artifact_fixture()
    validated = validate_artifact_compatibility(checkpoint, bank, manifest)
    assert validated["latent_dim"] == 64
    assert validated["bank_shape"] == [10, 64, 64]
    manifest["bank_array_sha256"] = "00" * 32
    with pytest.raises(ValueError, match="manifest"):
        validate_artifact_compatibility(checkpoint, bank, manifest)


def test_registered_probe_cases_all_fail_authentication_without_decoder() -> None:
    sender = IndexPacketSender(
        transport_master_key=MASTER_KEY,
        model_id=MODEL_ID,
        session_id=SESSION_ID,
        generator_version=1,
    )
    receiver = IndexPacketReceiver(
        transport_master_key=MASTER_KEY,
        expected_model_id=MODEL_ID,
        expected_session_id=SESSION_ID,
        expected_generator_version=1,
    )
    packet = sender.seal(
        [0, 1, 2, 63],
        [63, 2, 1, 0],
        state_id=3,
        latent_height=2,
        latent_width=2,
        stream_id=5,
        frame_seq=0,
    )
    assert receiver.open(packet).base_indices == (0, 1, 2, 63)

    report = run_authentication_probes(
        packet,
        master_key=MASTER_KEY,
        model_id=MODEL_ID,
        session_id=SESSION_ID,
        base_codebook_size=64,
        residual_codebook_size=64,
        state_count=10,
        probes_per_case=3,
    )
    assert report["registered_cases"] == list(REGISTERED_PROBE_CASES)
    assert report["total_attempted"] == 12
    assert report["total_authentication_rejections"] == 12
    assert report["total_unexpected_acceptances"] == 0
    assert report["total_decoder_invocations"] == 0
    for case in report["cases"].values():
        assert case["authentication_rejections"] == 3
        assert case["decoder_invocations"] == 0
