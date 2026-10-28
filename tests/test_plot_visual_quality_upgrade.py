from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightwave_seeded_vqvae.cli.build_fixed_sample_registry import (
    EXPECTED_STATE_IDS,
    FIXED_STATE_IDS,
    select_quantile_samples,
)
from lightwave_seeded_vqvae.cli.plot_visual_quality_upgrade import (
    INFERENCE_MSE_ATOL,
    INFERENCE_MSE_RTOL,
    _validate_evaluation_provenance,
    _validate_inference_mse,
    plot_candidate_mismatch_heatmap,
    plot_paired_correct_distribution,
    plot_quantile_overview,
    resolve_registry_sample_ids,
    run_upgrade_inference,
)


def _registry(targets: np.ndarray, cross_mse: np.ndarray) -> dict[str, object]:
    scores = np.diagonal(cross_mse, axis1=1, axis2=2).mean(axis=1)
    return {
        "schema": "lightwave-seeded-vqvae-fixed-sample-registry-v1",
        "state_order": [
            {"index": index, "state_id": state_id}
            for index, state_id in enumerate(EXPECTED_STATE_IDS)
        ],
        "fixed_state_indices": dict(FIXED_STATE_IDS),
        "fixed_states": {
            role: {"index": index, "state_id": EXPECTED_STATE_IDS[index]}
            for role, index in FIXED_STATE_IDS.items()
        },
        "classes": select_quantile_samples(targets, scores),
    }


def test_registry_order_is_reproduced_without_candidate_selection() -> None:
    targets = np.repeat(np.arange(10), 4)
    cross = np.zeros((40, 10, 10), dtype=np.float64)
    for sample_id in range(40):
        for state in range(10):
            cross[sample_id, state, state] = sample_id % 4 + state * 0.01
    registry = _registry(targets, cross)

    selected = resolve_registry_sample_ids(registry, targets, cross)

    assert selected["q25"] == [4 * class_id + 1 for class_id in range(10)]
    assert selected["q50"] == [4 * class_id + 1 for class_id in range(10)]
    assert selected["q75"] == [4 * class_id + 2 for class_id in range(10)]
    registry["classes"][0]["quantiles"]["q50"]["sample_id"] = 3
    with pytest.raises(ValueError, match="does not reproduce"):
        resolve_registry_sample_ids(registry, targets, cross)


class _FakeQuantizer:
    def __init__(self, owner: "_FakeModel") -> None:
        self.owner = owner
        self.calls = 0

    def __call__(self, latent: torch.Tensor, state_ids: torch.Tensor) -> SimpleNamespace:
        self.calls += 1
        self.owner.latent = latent
        shape = (latent.shape[0], 2, 2)
        return SimpleNamespace(
            base_indices=torch.zeros(shape, dtype=torch.long, device=latent.device),
            residual_indices=torch.ones(shape, dtype=torch.long, device=latent.device),
        )


class _FakeModel:
    def __init__(self, offset: float) -> None:
        self.offset = offset
        self.latent = torch.empty(0)
        self.quantizer = _FakeQuantizer(self)
        self.receiver_states: list[int] = []

    def encoder(self, images: torch.Tensor) -> torch.Tensor:
        return images

    def decode_indices(
        self,
        base_indices: torch.Tensor,
        residual_indices: torch.Tensor,
        state_ids: torch.Tensor,
    ) -> torch.Tensor:
        del base_indices, residual_indices
        state = int(state_ids[0])
        self.receiver_states.append(state)
        return torch.full_like(self.latent, self.offset + state / 20.0)


def test_upgrade_inference_quantizes_each_model_once_and_fixes_states() -> None:
    baseline = _FakeModel(0.0)
    candidate = _FakeModel(0.1)
    images = torch.zeros((3, 3, 32, 32), dtype=torch.float32)

    arrays, provenance = run_upgrade_inference(
        baseline, candidate, images, device=torch.device("cpu")
    )

    assert baseline.quantizer.calls == candidate.quantizer.calls == 1
    assert baseline.receiver_states == candidate.receiver_states == [2, 7, 4, 1]
    assert list(arrays) == [
        "original",
        "baseline_correct",
        "candidate_correct",
        "candidate_cross_pump",
        "candidate_same_pump",
        "candidate_representative",
    ]
    np.testing.assert_allclose(arrays["baseline_correct"], 0.1)
    np.testing.assert_allclose(arrays["candidate_cross_pump"], 0.45)
    assert provenance["fixed_state_indices"] == FIXED_STATE_IDS


def test_inference_mse_validation_records_differences_and_enforces_tolerance() -> None:
    sample_ids = [0, 1]
    original = np.zeros((2, 4, 4, 3), dtype=np.float64)
    reconstruction = np.full_like(original, 0.1)
    arrays = {"original": original}
    arrays.update(
        {
            key: reconstruction.copy()
            for key in (
                "baseline_correct",
                "candidate_correct",
                "candidate_cross_pump",
                "candidate_same_pump",
                "candidate_representative",
            )
        }
    )
    baseline_mse = np.zeros((2, 10, 10), dtype=np.float64)
    candidate_mse = np.zeros((2, 10, 10), dtype=np.float64)
    near_boundary_delta = INFERENCE_MSE_ATOL + INFERENCE_MSE_RTOL * 0.01
    expected = 0.01 + near_boundary_delta
    baseline_mse[:, 2, 2] = expected
    for receiver in (2, 7, 4, 1):
        candidate_mse[:, 2, receiver] = expected

    diagnostics = _validate_inference_mse(
        arrays,
        {"mse": baseline_mse},
        {"mse": candidate_mse},
        sample_ids,
    )

    assert set(diagnostics) == {
        "baseline_correct",
        "candidate_correct",
        "candidate_cross_pump",
        "candidate_same_pump",
        "candidate_representative",
    }
    assert diagnostics["baseline_correct"][
        "maximum_absolute_mse_difference"
    ] == pytest.approx(near_boundary_delta)
    assert INFERENCE_MSE_RTOL == 2e-3
    assert INFERENCE_MSE_ATOL == 2e-6

    candidate_mse[:, 2, 7] = 0.01 + 1.1 * near_boundary_delta
    with pytest.raises(ValueError, match="candidate_cross_pump"):
        _validate_inference_mse(
            arrays,
            {"mse": baseline_mse},
            {"mse": candidate_mse},
            sample_ids,
        )


def test_formal_evaluator_manifest_schema_mismatch_is_rejected(tmp_path) -> None:
    placeholder = tmp_path / "placeholder"
    with pytest.raises(ValueError, match="evaluation manifest schema"):
        _validate_evaluation_provenance(
            label="candidate",
            checkpoint_path=placeholder,
            codebook_path=placeholder,
            codebook_manifest_path=placeholder,
            summary_path=placeholder,
            metrics_path=placeholder,
            evaluation_manifest_path=placeholder,
            evaluation_manifest={"schema": "unexpected-schema", "scope": "final"},
            codebook_manifest={},
            bank=np.zeros((10, 2, 3), dtype=np.float32),
        )


def test_small_fake_arrays_generate_all_required_png_and_pdf(tmp_path) -> None:
    rng = np.random.default_rng(7)
    original = rng.uniform(0.0, 1.0, size=(10, 32, 32, 3)).astype(np.float32)
    arrays = {"original": original}
    metric_records: dict[str, dict[str, np.ndarray]] = {}
    for index, key in enumerate(
        (
            "baseline_correct",
            "candidate_correct",
            "candidate_cross_pump",
            "candidate_same_pump",
            "candidate_representative",
        )
    ):
        arrays[key] = np.clip(original + 0.01 * (index + 1), 0.0, 1.0)
        mse = np.full(10, 0.001 * (index + 1), dtype=np.float64)
        metric_records[key] = {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "ssim": np.full(10, 0.9 - index * 0.05),
            "lpips": np.full(10, 0.1 + index * 0.05),
            "psnr_db": 10.0 * np.log10(1.0 / mse),
        }
    panel = plot_quantile_overview(
        arrays,
        metric_records,
        quantile_name="q50",
        sample_ids=list(range(10)),
        class_ids=list(range(10)),
        output_dir=tmp_path,
        dpi=20,
    )

    matrix = np.full((10, 10), 0.04, dtype=np.float64)
    np.fill_diagonal(matrix, 0.01)
    heatmap = plot_candidate_mismatch_heatmap(
        matrix,
        state_labels=EXPECTED_STATE_IDS,
        output_dir=tmp_path,
        dpi=20,
    )
    baseline = np.broadcast_to(matrix, (24, 10, 10)).copy()
    candidate = baseline * 0.8
    distribution = plot_paired_correct_distribution(
        baseline,
        candidate,
        output_dir=tmp_path,
        dpi=20,
    )

    for result in (panel, heatmap, distribution):
        for file_type in ("png", "pdf"):
            assert Path(result[file_type]["path"]).is_file()
            assert result[file_type]["sha256"]
    assert panel["sample_ids"] == list(range(10))
    assert distribution["paired_statistics"]["sample_count"] == 24
