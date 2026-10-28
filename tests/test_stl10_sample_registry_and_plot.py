from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightwave_seeded_vqvae.cli.build_stl10_sample_registry import (
    EXPECTED_TEST_SAMPLES,
    REGISTRY_SCHEMA,
    STL10_CLASS_NAMES,
    build_stl10_sample_registry,
    resolve_controlled_states,
    select_quantile_samples,
)
from lightwave_seeded_vqvae.cli.plot_stl10_stagec import (
    BRANCHES,
    per_image_correct_wrong_distributions,
    plot_correct_wrong_distributions,
    plot_quantile_gallery,
    plot_state_metric_heatmaps,
    resolve_registry_ids,
    run_stl10_formal_batch_context_inference,
    run_stl10_presentation_inference,
    selected_metric_records,
    validate_registry_dataset_contract,
    validate_stagec_bank_shape,
    write_selected_metrics_csv,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _physical_states() -> list[dict[str, int | str]]:
    # Deliberately do not place the presentation roles at their historical CIFAR
    # numeric indices.  The STL tools must resolve roles from physical metadata.
    pairs = [
        (640, 20),
        (405, 0),
        (640, 10),
        (405, 20),
        (640, 0),
        (405, 10),
        (640, 5),
        (405, 5),
        (640, 15),
        (405, 15),
    ]
    return [
        {
            "index": index,
            "state_id": f"{pump}nm_{power}mW",
            "pump_nm": pump,
            "power_mw": power,
        }
        for index, (pump, power) in enumerate(pairs)
    ]


def _codebook_manifest() -> dict[str, object]:
    return {"schema": "test-codebook", "states": _physical_states()}


def _formal_arrays(sample_count: int) -> dict[str, np.ndarray]:
    cross_mse = np.full((sample_count, 10, 10), 0.09, dtype=np.float64)
    cross_ssim = np.full_like(cross_mse, 0.35)
    cross_lpips = np.full_like(cross_mse, 0.65)
    states = np.arange(10)
    cross_mse[:, states, states] = 0.01
    cross_ssim[:, states, states] = 0.90
    cross_lpips[:, states, states] = 0.10
    return {
        "software_mse": np.full(sample_count, 0.008),
        "software_ssim": np.full(sample_count, 0.92),
        "software_lpips": np.full(sample_count, 0.08),
        "seeded_cross_mse": cross_mse,
        "seeded_cross_ssim": cross_ssim,
        "seeded_cross_lpips": cross_lpips,
    }


def test_per_class_quantiles_use_ten_correct_states_and_lower_id_tie() -> None:
    labels = np.repeat(np.arange(10), 4)
    scores = np.tile(np.asarray([0.0, 1.0, 2.0, 3.0]), 10)
    selections = select_quantile_samples(labels, scores)
    assert len(selections) == 10
    assert selections[0]["quantiles"]["q25"]["sample_id"] == 1
    assert selections[0]["quantiles"]["q50"]["sample_id"] == 1
    assert selections[0]["quantiles"]["q75"]["sample_id"] == 2
    assert selections[1]["quantiles"]["q50"]["sample_id"] == 5


def test_controlled_states_are_resolved_from_manifest_physics() -> None:
    mse = np.full((10, 10), 0.2)
    np.fill_diagonal(mse, 0.01)
    mse[5] = np.asarray([0.10, 0.11, 0.12, 0.13, 0.14, 0.01, 0.16, 0.17, 0.18, 0.19])
    mapping = resolve_controlled_states(_codebook_manifest(), mse)
    assert mapping["fixed_state_indices"]["sender_state"] == 5
    assert mapping["fixed_state_indices"]["wrong_cross_pump_same_power"] == 2
    assert mapping["fixed_state_indices"]["wrong_same_pump_other_power"] == 3
    representative = mapping["fixed_state_indices"]["representative_median_wrong"]
    assert representative not in {2, 3, 5}


class _FakeSTL10:
    classes = STL10_CLASS_NAMES

    def __init__(self, labels: np.ndarray) -> None:
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(3, 96, 96), int(self.labels[index])


def test_registry_uses_full_official_test_and_formal_hashes(tmp_path: Path) -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), 800)
    cross = np.full((8_000, 10, 10), 0.2, dtype=np.float32)
    for state in range(10):
        cross[:, state, state] = (
            np.arange(8_000, dtype=np.float32) % 800 + state / 100.0
        ) / 1000.0
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
            "name": "STL-10",
            "expected_test_samples": 8_000,
            "processed_test_samples": 8_000,
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
    calls: list[tuple[str, str, bool]] = []

    def factory(*, root: str, split: str, download: bool) -> _FakeSTL10:
        calls.append((root, split, download))
        return _FakeSTL10(labels)

    registry = build_stl10_sample_registry(
        metrics,
        tmp_path / "STL10",
        codebook_manifest_path,
        dataset_factory=factory,
    )
    assert registry["schema"] == REGISTRY_SCHEMA
    assert calls == [(str((tmp_path / "STL10").resolve()), "test", False)]
    assert registry["dataset"]["sample_count"] == 8_000
    assert registry["selection_algorithm"]["manual_visual_selection"] is False
    assert registry["fixed_state_indices"]["sender_state"] == 5
    assert len(registry["classes"]) == 10


def test_registry_dataset_contract_and_stagec_bank_shape_are_strict() -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), 800)
    dataset = _FakeSTL10(labels)
    stored = {
        "dataset": {
            "name": "stl10",
            "split": "test",
            "sample_count": 8_000,
            "image_shape_chw": [3, 96, 96],
            "class_names": list(STL10_CLASS_NAMES),
        }
    }
    contract = validate_registry_dataset_contract(stored, dataset)
    assert contract["verified_against_current_dataset"] is True
    assert contract["image_shape_chw"] == [3, 96, 96]

    wrong_names = {"dataset": dict(stored["dataset"])}
    wrong_names["dataset"]["class_names"] = list(reversed(STL10_CLASS_NAMES))
    with pytest.raises(ValueError, match="class_names"):
        validate_registry_dataset_contract(wrong_names, dataset)

    validate_stagec_bank_shape(np.zeros((10, 128, 64), dtype=np.float32))
    with pytest.raises(ValueError, match="requires frozen B_s shape"):
        validate_stagec_bank_shape(np.zeros((10, 64, 64), dtype=np.float32))

    cross = np.full((8_000, 10, 10), 0.2, dtype=np.float32)
    states = np.arange(10)
    cross[:, states, states] = np.linspace(0.001, 0.05, 8_000)[:, None]
    classes = select_quantile_samples(
        labels, np.diagonal(cross, axis1=1, axis2=2).mean(axis=1)
    )
    registry = {"classes": classes}
    resolved = resolve_registry_ids(registry, labels, cross)
    assert all(len(ids) == 10 for ids in resolved.values())
    registry["classes"][0]["class_name"] = "not-airplane"
    with pytest.raises(ValueError, match="class name/order"):
        resolve_registry_ids(registry, labels, cross)


class _FakeSeededQuantizer:
    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []

    def __call__(self, latent: torch.Tensor, state_ids: torch.Tensor) -> SimpleNamespace:
        self.calls += 1
        batch, _, height, width = latent.shape
        self.batch_sizes.append(batch)
        return SimpleNamespace(
            base_indices=torch.zeros(batch, height, width, dtype=torch.long),
            residual_indices=torch.ones(batch, height, width, dtype=torch.long),
        )


class _FakeSeededModel:
    def __init__(self) -> None:
        self.quantizer = _FakeSeededQuantizer()
        self.receiver_calls: list[int] = []

    def encoder(self, images: torch.Tensor) -> torch.Tensor:
        return images

    def decode_indices(
        self,
        base_indices: torch.Tensor,
        residual_indices: torch.Tensor,
        state_ids: torch.Tensor,
    ) -> torch.Tensor:
        receiver = int(state_ids[0])
        self.receiver_calls.append(receiver)
        batch, height, width = base_indices.shape
        return torch.full((batch, 3, height, width), receiver / 10.0)


class _FakeSoftwareModel:
    def __call__(self, images: torch.Tensor, state_ids: torch.Tensor) -> SimpleNamespace:
        assert bool((state_ids == 0).all())
        return SimpleNamespace(reconstruction=images * 0.5)


def test_presentation_inference_preserves_96_and_reuses_sender_indices() -> None:
    seeded = _FakeSeededModel()
    images = torch.rand(2, 3, 96, 96)
    arrays, provenance = run_stl10_presentation_inference(
        seeded,
        _FakeSoftwareModel(),
        images,
        sender_state=5,
        cross_pump_state=2,
        same_pump_state=3,
        representative_state=8,
        device=torch.device("cpu"),
    )
    assert set(arrays) == {"original", *BRANCHES}
    assert all(value.shape == (2, 96, 96, 3) for value in arrays.values())
    assert seeded.quantizer.calls == 1
    assert seeded.receiver_calls == [5, 2, 3, 8]
    assert provenance["seeded"]["sender_quantizer_calls"] == 1


class _IndexedFakeSTL10(_FakeSTL10):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = torch.full((3, 96, 96), index / EXPECTED_TEST_SAMPLES)
        return image, int(self.labels[index])


def test_selected_inference_reuses_formal_batch_boundaries_and_order() -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), 800)
    dataset = _IndexedFakeSTL10(labels)
    seeded = _FakeSeededModel()
    sample_ids = [129, 2, 7_999, 130, 2]
    arrays, provenance = run_stl10_formal_batch_context_inference(
        seeded,
        _FakeSoftwareModel(),
        dataset,
        sample_ids,
        formal_batch_size=128,
        sender_state=5,
        cross_pump_state=2,
        same_pump_state=3,
        representative_state=8,
        device=torch.device("cpu"),
    )
    assert provenance["mode"] == "formal-evaluation-batch-context-v1"
    assert provenance["unique_formal_batch_count"] == 3
    assert provenance["seeded_sender_quantizer_calls"] == 3
    assert provenance["software_forward_calls"] == 3
    assert seeded.quantizer.calls == 3
    assert seeded.quantizer.batch_sizes == [128, 128, 64]
    assert seeded.receiver_calls == [5, 2, 3, 8] * 3
    assert [row["formal_batch_sample_count"] for row in provenance["context_batches"]] == [
        128,
        128,
        64,
    ]
    assert np.allclose(
        arrays["original"][:, 0, 0, 0],
        np.asarray(sample_ids, dtype=np.float32) / EXPECTED_TEST_SAMPLES,
    )
    assert arrays["original"].shape == (len(sample_ids), 96, 96, 3)


def test_plot_helpers_and_selected_csv_emit_required_artifacts(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    arrays = {
        key: rng.random((10, 96, 96, 3), dtype=np.float32)
        for key in ("original", *BRANCHES)
    }
    formal = _formal_arrays(20)
    fixed = {
        "sender_state": 5,
        "wrong_cross_pump_same_power": 2,
        "wrong_same_pump_other_power": 3,
        "representative_median_wrong": 8,
    }
    sample_ids = list(range(10))
    metrics = selected_metric_records(formal, sample_ids, fixed_state_indices=fixed)
    gallery = plot_quantile_gallery(
        arrays,
        metrics,
        quantile_name="q50",
        sample_ids=sample_ids,
        class_names=STL10_CLASS_NAMES,
        output_dir=tmp_path,
        dpi=25,
        row_start=0,
        row_stop=5,
    )
    display = gallery["display_scale_at_png_export"]
    assert display["source_pixels_hw"] == [96, 96]
    assert display["export_dpi"] == 25
    assert display["scale_x"] > 0.0
    assert display["scale_y"] > 0.0
    assert display["axis_pixels_hw_approx"][0] == pytest.approx(
        96 * display["scale_y"]
    )
    assert display["axis_pixels_hw_approx"][1] == pytest.approx(
        96 * display["scale_x"]
    )
    heatmap = plot_state_metric_heatmaps(
        formal,
        state_labels=[row["state_id"] for row in _physical_states()],
        output_dir=tmp_path,
        dpi=25,
    )
    distribution = plot_correct_wrong_distributions(formal, output_dir=tmp_path, dpi=25)
    for record in (gallery, heatmap, distribution):
        assert Path(record["png"]["path"]).is_file()
        assert Path(record["pdf"]["path"]).is_file()

    per_image = per_image_correct_wrong_distributions(formal)
    assert per_image["mse"]["correct"].shape == (20,)
    assert np.allclose(per_image["mse"]["correct"], 0.01)
    assert np.allclose(per_image["mse"]["wrong"], 0.09)

    registry = {
        "fixed_state_indices": fixed,
        "classes": [
            {
                "class_id": class_id,
                "class_name": STL10_CLASS_NAMES[class_id],
                "quantiles": {
                    name: {"sample_id": class_id} for name in ("q25", "q50", "q75")
                },
            }
            for class_id in range(10)
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
    assert len(rows) == 150
    assert {row["branch"] for row in rows} == set(BRANCHES)
    assert {row["metric_resolution"] for row in rows} == {
        "native 96x96 RGB [0,1]"
    }
