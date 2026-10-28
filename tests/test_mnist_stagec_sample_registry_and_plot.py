from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lightwave_seeded_vqvae.cli.build_mnist_stagec_sample_registry import (
    EXPECTED_TEST_SAMPLES,
    MNIST_DIGIT_NAMES,
    REGISTRY_SCHEMA,
    build_mnist_stagec_sample_registry,
    select_mnist_quantile_samples,
)
from lightwave_seeded_vqvae.cli.plot_mnist_stagec import (
    BRANCHES,
    grayscale_absolute_error,
    nchw_to_grayscale,
    plot_quantile_gallery,
    run_mnist_formal_batch_context_inference,
    run_mnist_presentation_inference,
    selected_metric_records,
    validate_mnist_stagec_bank_shape,
    validate_mnist_stagec_checkpoint_config,
    validate_registry_dataset_contract,
    validate_rendered_mse,
    write_selected_metrics_csv,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _physical_states() -> list[dict[str, int | str]]:
    return [
        {
            "index": index,
            "state_id": f"{pump}nm_{power}mW",
            "pump_nm": pump,
            "power_mw": power,
        }
        for index, (pump, power) in enumerate(
            [
                (405, 0),
                (405, 5),
                (405, 10),
                (405, 15),
                (405, 20),
                (640, 0),
                (640, 5),
                (640, 10),
                (640, 15),
                (640, 20),
            ]
        )
    ]


def _codebook_manifest() -> dict[str, object]:
    return {"schema": "test-codebook", "states": _physical_states()}


class _FakeMNIST:
    def __init__(self, labels: np.ndarray) -> None:
        self.targets = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.targets.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = torch.full((1, 28, 28), index / EXPECTED_TEST_SAMPLES)
        return image, int(self.targets[index])


def test_per_digit_quantiles_reuse_linear_rule_and_lower_id_tie() -> None:
    labels = np.repeat(np.arange(10), 4)
    scores = np.tile(np.asarray([0.0, 1.0, 2.0, 3.0]), 10)
    selected = select_mnist_quantile_samples(labels, scores)
    assert len(selected) == 10
    assert selected[0]["quantiles"]["q25"]["sample_id"] == 1
    assert selected[0]["quantiles"]["q50"]["sample_id"] == 1
    assert selected[0]["quantiles"]["q75"]["sample_id"] == 2
    assert selected[9]["class_name"] == "9"


def test_registry_uses_complete_mnist_test_and_physical_state_roles(
    tmp_path: Path,
) -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), 1_000)
    cross = np.full((EXPECTED_TEST_SAMPLES, 10, 10), 0.2, dtype=np.float32)
    states = np.arange(10)
    diagonal = (
        np.arange(EXPECTED_TEST_SAMPLES, dtype=np.float32) % 1_000 + 1.0
    ) / 10_000.0
    cross[:, states, states] = diagonal[:, None]
    metrics = tmp_path / "per_image_metrics.npz"
    np.savez(metrics, seeded_cross_mse=cross)

    codebook_manifest_path = tmp_path / "codebook_manifest.json"
    codebook_manifest_path.write_text(
        json.dumps(_codebook_manifest()), encoding="utf-8"
    )
    label_hash = hashlib.sha256(
        np.ascontiguousarray(labels, dtype="<i8").tobytes()
    ).hexdigest()
    evaluation_manifest = {
        "scope": "final",
        "dataset": {
            "name": "MNIST",
            "expected_test_samples": EXPECTED_TEST_SAMPLES,
            "processed_test_samples": EXPECTED_TEST_SAMPLES,
            "test_truncated": False,
            "processed_labels_int64_sha256": label_hash,
        },
        "outputs": {"per_image_metrics.npz": _sha256(metrics)},
        "inputs": {
            "codebook": {"manifest_sha256": _sha256(codebook_manifest_path)}
        },
        "state_order": [
            {"index": row["index"], "state_id": row["state_id"]}
            for row in _physical_states()
        ],
    }
    (tmp_path / "evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest), encoding="utf-8"
    )
    calls: list[tuple[str, bool, bool]] = []

    def factory(*, root: str, train: bool, download: bool) -> _FakeMNIST:
        calls.append((root, train, download))
        return _FakeMNIST(labels)

    registry = build_mnist_stagec_sample_registry(
        metrics,
        tmp_path / "MNIST_ROOT",
        codebook_manifest_path,
        dataset_factory=factory,
    )
    assert registry["schema"] == REGISTRY_SCHEMA
    assert calls == [(str((tmp_path / "MNIST_ROOT").resolve()), False, False)]
    assert registry["dataset"]["image_shape_chw"] == [1, 28, 28]
    assert registry["dataset"]["class_names"] == list(MNIST_DIGIT_NAMES)
    assert registry["selection_algorithm"]["manual_visual_selection"] is False
    assert registry["fixed_state_indices"]["sender_state"] == 2
    assert registry["fixed_state_indices"]["wrong_cross_pump_same_power"] == 7
    assert registry["fixed_state_indices"]["wrong_same_pump_other_power"] == 4


def test_registry_dataset_contract_and_bank_shape_are_strict() -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), 1_000)
    dataset = _FakeMNIST(labels)
    stored = {
        "dataset": {
            "name": "mnist",
            "train": False,
            "split": "test",
            "sample_count": EXPECTED_TEST_SAMPLES,
            "image_shape_chw": [1, 28, 28],
            "class_names": list(MNIST_DIGIT_NAMES),
        }
    }
    contract = validate_registry_dataset_contract(stored, dataset)
    assert contract["verified_against_current_dataset"] is True
    assert contract["image_shape_chw"] == [1, 28, 28]

    wrong = {"dataset": dict(stored["dataset"])}
    wrong["dataset"]["image_shape_chw"] = [3, 28, 28]
    with pytest.raises(ValueError, match="image_shape_chw"):
        validate_registry_dataset_contract(wrong, dataset)

    validate_mnist_stagec_bank_shape(np.zeros((10, 128, 64), dtype=np.float32))
    with pytest.raises(ValueError, match="requires frozen B_s shape"):
        validate_mnist_stagec_bank_shape(
            np.zeros((10, 64, 64), dtype=np.float32)
        )


def test_checkpoint_config_rejects_legacy_decoder_and_wrong_quantizer() -> None:
    config = {
        "dataset": {"name": "MNIST", "native_resolution": [28, 28]},
        "model": {
            "quantizer_type": "seeded_residual",
            "latent_dim": 64,
            "base_codebook_size": 128,
            "residual_codebook_size": 128,
            "pretrained_encoder": True,
            "imagenet_weights": "IMAGENET1K_V1",
            "decoder_blocks_by_stage": [3, 2, 2],
        },
    }
    contract = validate_mnist_stagec_checkpoint_config(
        config, expected_quantizer_type="seeded_residual"
    )
    assert contract["encoder_backbone"] == "resnet18"
    assert contract["decoder_blocks_by_stage"] == [3, 2, 2]

    legacy = json.loads(json.dumps(config))
    legacy["model"].pop("decoder_blocks_by_stage")
    with pytest.raises(ValueError, match="decoder_blocks_by_stage"):
        validate_mnist_stagec_checkpoint_config(
            legacy, expected_quantizer_type="seeded_residual"
        )

    with pytest.raises(ValueError, match="quantizer_type"):
        validate_mnist_stagec_checkpoint_config(
            config, expected_quantizer_type="trainable_residual"
        )


class _FakeSeededQuantizer:
    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []

    def __call__(self, latent: torch.Tensor, state_ids: torch.Tensor) -> SimpleNamespace:
        self.calls += 1
        batch, _, height, width = latent.shape
        self.batch_sizes.append(batch)
        assert bool((state_ids == state_ids[0]).all())
        return SimpleNamespace(
            base_indices=torch.zeros(batch, height, width, dtype=torch.long),
            residual_indices=torch.ones(batch, height, width, dtype=torch.long),
        )


class _FakeSeededModel:
    def __init__(self) -> None:
        self.quantizer = _FakeSeededQuantizer()
        self.receiver_calls: list[int] = []

    def encoder(self, images: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(images, kernel_size=4, stride=4)

    def decode_indices(
        self,
        base_indices: torch.Tensor,
        residual_indices: torch.Tensor,
        state_ids: torch.Tensor,
    ) -> torch.Tensor:
        receiver = int(state_ids[0])
        self.receiver_calls.append(receiver)
        batch = int(base_indices.shape[0])
        return torch.full((batch, 1, 28, 28), receiver / 10.0)


class _FakeSoftwareModel:
    def __call__(self, images: torch.Tensor, state_ids: torch.Tensor) -> SimpleNamespace:
        assert bool((state_ids == 0).all())
        return SimpleNamespace(reconstruction=images * 0.5)


def test_grayscale_helpers_and_presentation_inference_preserve_native_shape() -> None:
    images = torch.rand(2, 1, 28, 28)
    assert nchw_to_grayscale(images).shape == (2, 28, 28)
    assert np.allclose(
        grayscale_absolute_error(np.zeros((1, 28, 28)), np.ones((1, 28, 28))),
        1.0,
    )

    seeded = _FakeSeededModel()
    arrays, provenance = run_mnist_presentation_inference(
        seeded,
        _FakeSoftwareModel(),
        images,
        sender_state=2,
        cross_pump_state=7,
        same_pump_state=4,
        representative_state=8,
        device=torch.device("cpu"),
    )
    assert set(arrays) == {"original", *BRANCHES}
    assert all(value.shape == (2, 28, 28) for value in arrays.values())
    assert seeded.quantizer.calls == 1
    assert seeded.receiver_calls == [2, 7, 4, 8]
    assert provenance["seeded"]["base_indices_shape"] == [2, 7, 7]


def test_selected_inference_reuses_formal_batches_and_requested_order() -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), 1_000)
    dataset = _FakeMNIST(labels)
    seeded = _FakeSeededModel()
    sample_ids = [129, 2, 9_999, 130, 2]
    arrays, provenance = run_mnist_formal_batch_context_inference(
        seeded,
        _FakeSoftwareModel(),
        dataset,
        sample_ids,
        formal_batch_size=128,
        sender_state=2,
        cross_pump_state=7,
        same_pump_state=4,
        representative_state=8,
        device=torch.device("cpu"),
    )
    assert provenance["mode"] == "formal-evaluation-batch-context-v1"
    assert provenance["unique_formal_batch_count"] == 3
    assert seeded.quantizer.batch_sizes == [128, 128, 16]
    assert seeded.receiver_calls == [2, 7, 4, 8] * 3
    assert np.allclose(
        arrays["original"][:, 0, 0],
        np.asarray(sample_ids, dtype=np.float32) / EXPECTED_TEST_SAMPLES,
    )


def _selected_gallery_metrics(
    arrays: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    original = arrays["original"]
    result: dict[str, dict[str, np.ndarray]] = {}
    for branch in BRANCHES:
        mse = np.mean((arrays[branch] - original) ** 2, axis=(1, 2))
        result[branch] = {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "ssim": np.full(10, 0.8),
            "lpips": np.full(10, 0.2),
        }
    return result


def test_gallery_outputs_png_pdf_and_checks_rendered_mse(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    arrays = {
        key: rng.random((10, 28, 28), dtype=np.float32)
        for key in ("original", *BRANCHES)
    }
    metrics = _selected_gallery_metrics(arrays)
    checks = validate_rendered_mse(arrays, metrics, list(range(10)))
    assert set(checks) == set(BRANCHES)

    gallery = plot_quantile_gallery(
        arrays,
        metrics,
        quantile_name="q25",
        sample_ids=list(range(10)),
        digit_names=MNIST_DIGIT_NAMES,
        output_dir=tmp_path,
        dpi=25,
        row_start=5,
        row_stop=10,
    )
    assert Path(gallery["png"]["path"]).name == "mnist_01_q25_digits_5_9.png"
    assert Path(gallery["png"]["path"]).is_file()
    assert Path(gallery["pdf"]["path"]).is_file()
    assert gallery["native_resolution"] == [28, 28]
    assert gallery["absolute_error_scale"]["display_range"] == [0.0, 1.0]
    assert gallery["absolute_error_scale"]["shared_across_all_error_maps"] is True

    mismatched = {
        key: {metric: value.copy() for metric, value in branch.items()}
        for key, branch in metrics.items()
    }
    mismatched["correct"]["mse"][0] += 0.1
    with pytest.raises(ValueError, match="MNIST ID 0"):
        validate_rendered_mse(arrays, mismatched, list(range(10)))


def test_selected_metrics_csv_uses_native_grayscale_resolution(tmp_path: Path) -> None:
    sample_count = 20
    cross_mse = np.full((sample_count, 10, 10), 0.09)
    cross_ssim = np.full_like(cross_mse, 0.35)
    cross_lpips = np.full_like(cross_mse, 0.65)
    states = np.arange(10)
    cross_mse[:, states, states] = 0.01
    cross_ssim[:, states, states] = 0.90
    cross_lpips[:, states, states] = 0.10
    formal = {
        "software_mse": np.full(sample_count, 0.008),
        "software_ssim": np.full(sample_count, 0.92),
        "software_lpips": np.full(sample_count, 0.08),
        "seeded_cross_mse": cross_mse,
        "seeded_cross_ssim": cross_ssim,
        "seeded_cross_lpips": cross_lpips,
    }
    fixed = {
        "sender_state": 2,
        "wrong_cross_pump_same_power": 7,
        "wrong_same_pump_other_power": 4,
        "representative_median_wrong": 8,
    }
    registry = {
        "fixed_state_indices": fixed,
        "classes": [
            {
                "class_id": digit,
                "class_name": str(digit),
                "quantiles": {
                    name: {"sample_id": digit} for name in ("q25", "q50", "q75")
                },
            }
            for digit in range(10)
        ],
    }
    csv_path = tmp_path / "selected.csv"
    count = write_selected_metrics_csv(
        csv_path,
        registry=registry,
        formal=formal,
        state_records=_physical_states(),
    )
    assert count == 3 * 10 * 5
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["metric_resolution"] for row in rows} == {
        "native 28x28 grayscale [0,1]"
    }
    selected = selected_metric_records(formal, [0], fixed_state_indices=fixed)
    assert selected["correct"]["rmse"][0] == pytest.approx(0.1)
