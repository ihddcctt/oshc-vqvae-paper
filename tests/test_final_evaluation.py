from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from lightwave_seeded_vqvae.cli.evaluate_final import (
    EVALUATION_SCHEMA,
    _base_codebook_state_distances,
    _compatibility_report,
    _finalize_quantization_geometry,
    _load_model,
    _lpips_enabled,
    _lpips_per_image,
    _model_parameter_summary,
    _new_quantization_geometry,
    _nominal_bits,
    _update_quantization_geometry,
    _write_base_distance_csv,
    build_parser,
    evaluation_scope,
)
from lightwave_seeded_vqvae.evaluation import (
    fixed_state_schedule,
    gaussian_ssim_per_image,
    metric_summary,
    paired_correct_wrong_summary,
    pseudorandom_balanced_schedule,
    schedule_sha256,
    summarize_cross_state,
    summarize_schedule,
    usage_summary,
)
from lightwave_seeded_vqvae.models.vqvae import SeededResidualVQVAE


def test_parser_and_scope_distinguish_formal_from_smoke() -> None:
    common = [
        "--seeded-checkpoint",
        "seeded.pt",
        "--software-checkpoint",
        "software.pt",
        "--codebook",
        "bank.npz",
        "--manifest",
        "bank.manifest.json",
        "--output-dir",
        "result",
    ]
    formal = build_parser().parse_args(common)
    smoke = build_parser().parse_args(common + ["--smoke-max-batches", "1"])
    assert evaluation_scope(formal.smoke_max_batches) == "final"
    assert EVALUATION_SCHEMA == "lightwave-seeded-vqvae-final-evaluation-v2"
    assert formal.lpips == "auto"
    assert _lpips_enabled(formal.lpips, "cifar10")
    assert not _lpips_enabled(formal.lpips, "mnist")
    assert evaluation_scope(smoke.smoke_max_batches) == "smoke"
    with pytest.raises(SystemExit):
        build_parser().parse_args(common + ["--smoke-max-batches", "0"])


def test_usage_summary_reports_coverage_entropy_and_normalized_perplexity() -> None:
    summary = usage_summary(np.asarray([2, 2, 0, 0]))
    assert summary["token_count"] == 4
    assert summary["used_codes"] == 2
    assert summary["coverage"] == pytest.approx(0.5)
    assert summary["entropy_bits"] == pytest.approx(1.0)
    assert summary["perplexity"] == pytest.approx(2.0)
    assert summary["normalized_perplexity"] == pytest.approx(0.5)


def test_pseudorandom_schedule_is_reproducible_and_block_balanced() -> None:
    first = pseudorandom_balanced_schedule(30, 10, 20260907)
    second = pseudorandom_balanced_schedule(30, 10, 20260907)
    different = pseudorandom_balanced_schedule(30, 10, 7)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    np.testing.assert_array_equal(np.bincount(first, minlength=10), np.full(10, 3))
    assert schedule_sha256(first) == schedule_sha256(second)


def test_ssim_is_one_for_identical_images_and_lower_after_change() -> None:
    target = torch.linspace(0.0, 1.0, 28 * 28).reshape(1, 1, 28, 28)
    identical = gaussian_ssim_per_image(target, target)
    changed = gaussian_ssim_per_image(torch.zeros_like(target), target)
    assert identical.item() == pytest.approx(1.0, abs=1e-6)
    assert changed.item() < identical.item()


def test_cross_state_and_time_shift_summaries_keep_wrong_frames_separate() -> None:
    sample_count = 20
    state_count = 10
    mse = np.full((sample_count, state_count, state_count), 0.25, dtype=np.float32)
    ssim = np.full_like(mse, 0.2)
    for state in range(state_count):
        mse[:, state, state] = 0.01
        ssim[:, state, state] = 0.9
    labels = [f"state_{index}" for index in range(state_count)]
    cross = summarize_cross_state(
        mse,
        ssim,
        state_labels=labels,
        trained_offset=1,
        pixels_per_image=28 * 28,
    )
    assert cross["correct_pooled"]["mse"] == pytest.approx(0.01)
    assert cross["wrong_pooled"]["mse"] == pytest.approx(0.25)
    assert len(cross["by_state_offset"]) == 9
    assert cross["selection_monitored_offset"]["state_offset"] == 1
    assert "trained_offset" not in cross
    assert "unseen_offsets" not in cross

    fixed = fixed_state_schedule(sample_count, state_count)
    rows = summarize_schedule(
        mse,
        ssim,
        schedule_name="fixed",
        send_schedule=fixed,
        pixels_per_image=28 * 28,
    )
    assert rows[0]["matched_frame_count"] == sample_count
    assert rows[0]["unmatched_only"] is None
    for row in rows[1:]:
        assert row["matched_frame_count"] == 0
        assert row["unmatched_only"]["mse"] == pytest.approx(0.25)


def test_metric_summary_reports_pooled_rmse_and_optional_lpips() -> None:
    summary = metric_summary(
        np.asarray([0.01, 0.09]),
        np.asarray([0.9, 0.7]),
        pixels_per_observation=12,
        per_observation_lpips=np.asarray([0.2, 0.4]),
    )
    assert summary["mse"] == pytest.approx(0.05)
    assert summary["rmse"] == pytest.approx(np.sqrt(0.05))
    assert summary["lpips_mean"] == pytest.approx(0.3)
    assert summary["squared_error_sum"] == pytest.approx(1.2)


def test_lpips_adapter_maps_unit_interval_to_signed_range() -> None:
    class RecordingMetric(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.received: tuple[torch.Tensor, torch.Tensor] | None = None

        def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
            self.received = (first.detach().clone(), second.detach().clone())
            return (first - second).square().mean(dim=(1, 2, 3), keepdim=True)

    metric = RecordingMetric()
    reconstruction = torch.zeros(2, 3, 32, 32)
    target = torch.ones_like(reconstruction)
    score = _lpips_per_image(metric, reconstruction, target)
    assert metric.received is not None
    first, second = metric.received
    assert first.min().item() == pytest.approx(-1.0)
    assert second.max().item() == pytest.approx(1.0)
    torch.testing.assert_close(score, torch.full((2,), 4.0))


def test_cross_state_lpips_and_wrong_correct_bootstrap_are_reproducible() -> None:
    sample_count = 8
    state_count = 3
    mse = np.full((sample_count, state_count, state_count), 0.16)
    ssim = np.full_like(mse, 0.2)
    lpips = np.full_like(mse, 0.8)
    states = np.arange(state_count)
    mse[:, states, states] = 0.04
    ssim[:, states, states] = 0.9
    lpips[:, states, states] = 0.1
    cross = summarize_cross_state(
        mse,
        ssim,
        state_labels=["a", "b", "c"],
        trained_offset=1,
        pixels_per_image=3 * 32 * 32,
        per_image_lpips=lpips,
    )
    assert cross["correct_pooled"]["lpips_mean"] == pytest.approx(0.1)
    assert cross["wrong_pooled"]["lpips_mean"] == pytest.approx(0.8)
    assert np.asarray(cross["lpips_mean_matrix"]).shape == (3, 3)

    first = paired_correct_wrong_summary(
        mse,
        per_image_lpips=lpips,
        per_image_ssim=ssim,
        seed=91,
        replicates=50,
    )
    second = paired_correct_wrong_summary(
        mse,
        per_image_lpips=lpips,
        per_image_ssim=ssim,
        seed=91,
        replicates=50,
    )
    assert first == second
    assert first["pooled"]["wrong_to_correct_mse_ratio"] == pytest.approx(4.0)
    assert first["pooled"]["wrong_to_correct_rmse_ratio"] == pytest.approx(2.0)
    assert first["pooled"]["wrong_minus_correct_lpips"] == pytest.approx(0.7)
    assert first["pooled"]["correct_minus_wrong_ssim"] == pytest.approx(0.7)
    assert first["bootstrap"]["wrong_to_correct_mse_ratio"][
        "ci95_percentile_lower"
    ] == pytest.approx(4.0)


def test_quantization_geometry_is_token_pooled_and_tracks_error_reduction() -> None:
    generator = torch.Generator().manual_seed(17)
    bank = torch.nn.functional.normalize(
        torch.randn(10, 2, 4, generator=generator), dim=-1
    )
    model = SeededResidualVQVAE(
        bank,
        in_channels=3,
        latent_dim=4,
        residual_codebook_size=2,
        pretrained_encoder=False,
    )
    with torch.no_grad():
        model.quantizer.residual_codebook.zero_()
        model.quantizer.residual_codebook[1, 0] = 0.1
    latent = torch.randn(2, 4, 2, 2, generator=generator)
    accumulator = _new_quantization_geometry(10, 4)
    for state in range(10):
        state_ids = torch.full((2,), state, dtype=torch.long)
        quantized = model.quantizer(latent, state_ids)
        _update_quantization_geometry(
            accumulator, model, latent, quantized, state
        )
    geometry = _finalize_quantization_geometry(
        model, accumulator, [f"state_{index}" for index in range(10)]
    )
    pooled = geometry["selected_on_held_out_test"]["pooled"]
    assert pooled["quantization_error"]["token_count"] == 80
    assert pooled["quantization_error"]["final_to_first_error_ratio"] <= 1.0
    assert geometry["codebook_unweighted_supplement"]["base_pooled"]["count"] == 20
    assert geometry["codebook_unweighted_supplement"][
        "base_plus_residual_all_state_code_pairs"
    ]["count"] == 40


def _matched_checkpoint_config() -> dict[str, object]:
    return {
        "experiment": {"seed": 20260907},
        "dataset": {
            "name": "cifar10",
            "root": "/tmp/cifar10",
            "validation_size": 5000,
        },
        "model": {
            "quantizer_type": "seeded_residual",
            "latent_dim": 4,
            "base_codebook_size": 2,
            "residual_codebook_size": 2,
            "pretrained_encoder": True,
            "imagenet_weights": "IMAGENET1K_V1",
            "normalize_latents": True,
            "loss_reference_dim": 64,
            "decoder_blocks_per_stage": 1,
        },
        "training": {},
    }


def test_checkpoint_compatibility_normalizes_new_fields_and_detects_mismatch() -> None:
    legacy = _matched_checkpoint_config()
    explicit = copy.deepcopy(legacy)
    explicit["model"]["encoder_backbone"] = "resnet18"
    explicit["model"]["decoder_blocks_by_stage"] = [1, 1, 1]
    explicit["training"]["reconstruction_loss"] = "mse"
    explicit["training"]["perceptual_weight"] = 0.0

    report = _compatibility_report(legacy, explicit)
    assert report["mismatches"] == {}
    assert "model.decoder_blocks_by_stage" in report["matched_fields"]
    assert "model.decoder_zero_init_extra_residual_blocks" in report["matched_fields"]
    assert "model.encoder_backbone" in report["matched_fields"]
    assert "training.reconstruction_loss" in report["matched_fields"]
    assert "training.perceptual_weight" in report["matched_fields"]

    mismatched = copy.deepcopy(explicit)
    mismatched["model"]["decoder_blocks_by_stage"] = [3, 2, 2]
    with pytest.raises(ValueError, match="not matched"):
        _compatibility_report(legacy, mismatched)

    mismatched = copy.deepcopy(explicit)
    mismatched["model"]["decoder_zero_init_extra_residual_blocks"] = True
    with pytest.raises(ValueError, match="not matched"):
        _compatibility_report(legacy, mismatched)

    mismatched = copy.deepcopy(explicit)
    mismatched["training"]["reconstruction_loss"] = "l1_lpips"
    mismatched["training"]["perceptual_weight"] = 0.05
    with pytest.raises(ValueError, match="not matched"):
        _compatibility_report(legacy, mismatched)

    mismatched = copy.deepcopy(explicit)
    mismatched["model"]["encoder_backbone"] = "resnet34"
    with pytest.raises(ValueError, match="not matched"):
        _compatibility_report(legacy, mismatched)


def test_checkpoint_loader_restores_per_stage_decoder_profile_strictly() -> None:
    generator = torch.Generator().manual_seed(41)
    bank = torch.nn.functional.normalize(
        torch.randn(10, 2, 4, generator=generator), dim=-1
    )
    config = _matched_checkpoint_config()
    config["model"]["decoder_blocks_by_stage"] = [3, 2, 2]
    config["model"]["decoder_zero_init_extra_residual_blocks"] = True
    source = SeededResidualVQVAE(
        bank,
        in_channels=3,
        latent_dim=4,
        residual_codebook_size=2,
        pretrained_encoder=False,
        decoder_blocks_by_stage=[3, 2, 2],
        decoder_zero_init_extra_residual_blocks=True,
    )
    checkpoint = {"config": config, "model": source.state_dict()}

    restored = _load_model(
        checkpoint,
        expected_quantizer_type="seeded_residual",
        bank=bank.numpy(),
    )
    assert restored.decoder.blocks_by_stage == (3, 2, 2)
    assert restored.decoder.zero_init_extra_residual_blocks is True
    assert tuple(restored.state_dict()) == tuple(source.state_dict())


def test_checkpoint_loader_keeps_legacy_decoder_checkpoint_compatible() -> None:
    generator = torch.Generator().manual_seed(43)
    bank = torch.nn.functional.normalize(
        torch.randn(10, 2, 4, generator=generator), dim=-1
    )
    config = _matched_checkpoint_config()
    source = SeededResidualVQVAE(
        bank,
        in_channels=3,
        latent_dim=4,
        residual_codebook_size=2,
        pretrained_encoder=False,
    )

    restored = _load_model(
        {"config": config, "model": source.state_dict()},
        expected_quantizer_type="seeded_residual",
        bank=bank.numpy(),
    )
    assert restored.decoder.blocks_by_stage == (1, 1, 1)
    assert tuple(restored.state_dict()) == tuple(source.state_dict())


def test_model_parameter_summary_separates_trainable_parameters() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    model[0].weight.requires_grad_(False)

    summary = _model_parameter_summary(model)
    assert summary["total_parameters"] == 26
    assert summary["trainable_parameters"] == 14
    assert summary["non_trainable_parameters"] == 12


def test_corresponding_base_state_distances_and_csv(tmp_path) -> None:
    bank = np.zeros((3, 2, 2), dtype=np.float32)
    bank[1, :, 0] = 1.0
    bank[2, :, 0] = 3.0
    summary = _base_codebook_state_distances(bank, ["a", "b", "c"])
    matrix = np.asarray(summary["mean_l2_matrix"])

    np.testing.assert_allclose(
        matrix,
        np.asarray([[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]]),
    )
    assert summary["different_state_pooled"]["count"] == 12
    assert len(summary["by_state_pair"]) == 9
    assert summary["by_state_pair"][1]["mean"] == pytest.approx(1.0)

    destination = tmp_path / "base_distances.csv"
    _write_base_distance_csv(destination, summary)
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10
    assert "first_state_index" in lines[0]


def test_nominal_rate_reports_k64_and_k128_cifar_payloads() -> None:
    k64 = _nominal_bits(
        64,
        64,
        latent_shape=[8, 8],
        image_shape=[3, 32, 32],
    )
    assert k64["total_index_bits_per_token"] == 12
    assert k64["nominal_index_bits_per_image"] == 768
    assert k64["raw_reference_bits_per_pixel"] == 24
    assert k64["raw_reference_bits_per_image"] == 24576
    assert k64["nominal_raw_to_index_compression_ratio"] == pytest.approx(32.0)

    k128 = _nominal_bits(
        128,
        128,
        latent_shape=[8, 8],
        image_shape=[3, 32, 32],
    )
    assert k128["total_index_bits_per_token"] == 14
    assert k128["nominal_index_bits_per_image"] == 896
    assert k128["nominal_raw_to_index_compression_ratio"] == pytest.approx(
        24576 / 896
    )
