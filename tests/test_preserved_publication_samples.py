from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from lightwave_seeded_vqvae.cli import plot_mnist_stagec as mnist
from lightwave_seeded_vqvae.cli import plot_stl10_stagec as stl
from lightwave_seeded_vqvae.cli.build_stl10_sample_registry import (
    resolve_controlled_states, select_quantile_samples,
)


def registry_for(dataset, class_names):
    labels = np.repeat(np.arange(10), 4)
    scores = np.tile(np.arange(4), 10) * 0.01
    classes = select_quantile_samples(labels, scores, class_names=class_names)
    registry = {
        "dataset": {"name": dataset}, "classes": classes,
        "inputs": {f"{dataset}_test_labels": {"canonical_int64_sha256": hashlib.sha256(
            np.ascontiguousarray(labels, dtype="<i8").tobytes()
        ).hexdigest()}},
    }
    cross = np.full((len(labels), 10, 10), 0.1)
    cross[:, np.arange(10), np.arange(10)] = scores[:, None] + 1e-4
    return registry, labels, cross


@pytest.mark.parametrize("module,names,dataset", [
    (mnist, tuple(str(i) for i in range(10)), "mnist"),
    (stl, stl.STL10_CLASS_NAMES, "stl10"),
])
def test_preservation_keeps_ids_when_current_quantile_ranks_change(module, names, dataset):
    registry, labels, cross = registry_for(dataset, names)
    original = module.resolve_registry_ids(registry, labels, cross)
    changed = cross[::-1].copy()
    with pytest.raises(ValueError, match="sample IDs do not reproduce"):
        module.resolve_registry_ids(registry, labels, changed)
    before = copy.deepcopy(registry)
    preserved = module.resolve_registry_ids(
        registry, labels, changed, preserve_registered_samples=True
    )
    assert preserved == original
    assert registry == before


@pytest.mark.parametrize("mutation,match", [
    ("label", "wrong class label"),
    ("range", "test range"),
    ("class", "class name/order"),
    ("labels_hash", "labels hash"),
    ("quantile", "quantile label"),
    ("boolean_id", "test range"),
])
def test_preservation_does_not_relax_sample_or_label_validation(mutation, match):
    registry, labels, cross = registry_for("mnist", tuple(str(i) for i in range(10)))
    first = registry["classes"][0]
    if mutation == "label":
        first["quantiles"]["q25"]["sample_id"] = 5
    elif mutation == "range":
        first["quantiles"]["q25"]["sample_id"] = len(labels)
    elif mutation == "class":
        first["class_name"] = "wrong"
    elif mutation == "labels_hash":
        registry["inputs"]["mnist_test_labels"]["canonical_int64_sha256"] = "0" * 64
    elif mutation == "quantile":
        first["quantiles"]["q25"]["quantile"] = 0.9
    else:
        first["quantiles"]["q25"]["sample_id"] = True
    with pytest.raises(ValueError, match=match):
        mnist.resolve_registry_ids(registry, labels, cross, preserve_registered_samples=True)


def test_only_old_metric_and_evaluation_hashes_can_differ(tmp_path):
    paths = {}
    registry = {"inputs": {}}
    for key in ("per_image_metrics", "evaluation_manifest", "codebook_manifest"):
        path = tmp_path / key
        path.write_bytes(key.encode())
        paths[key] = path
        registry["inputs"][key] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    registry["inputs"]["per_image_metrics"]["sha256"] = "1" * 64
    registry["inputs"]["evaluation_manifest"]["sha256"] = "2" * 64
    with pytest.raises(ValueError, match="input hash mismatch"):
        stl.validate_registry_input_hashes(registry, paths)
    result = stl.validate_registry_input_hashes(registry, paths, preserve_registered_samples=True)
    assert result["per_image_metrics"]["registered_sha256"] == "1" * 64
    assert result["per_image_metrics"]["current_evaluation_sha256"] != "1" * 64
    assert result["codebook_manifest"]["identical"] is True
    registry["inputs"]["codebook_manifest"]["sha256"] = "3" * 64
    with pytest.raises(ValueError, match="codebook_manifest"):
        stl.validate_registry_input_hashes(registry, paths, preserve_registered_samples=True)


def physical_fixture():
    records = [
        {"index": i, "state_id": f"{pump}nm_{power}mW", "pump_nm": pump, "power_mw": power}
        for i, (pump, power) in enumerate((pump, power) for pump in (405, 640) for power in (0, 5, 10, 15, 20))
    ]
    manifest = {"states": records}
    old = np.full((10, 10), 0.2)
    old[2] = np.arange(10) * 0.01 + 0.02
    old[2, 2] = 0.001
    mapping = resolve_controlled_states(manifest, old)
    registry = {"state_order": records, **{key: mapping[key] for key in (
        "fixed_state_indices", "fixed_states", "representative_selection"
    )}}
    return manifest, registry, old


def test_preservation_keeps_original_median_receiver_without_reranking(monkeypatch):
    manifest, registry, old = physical_fixture()
    def reject(*args, **kwargs):
        raise AssertionError("preservation must not invoke current metric selection")
    monkeypatch.setattr(stl, "resolve_controlled_states", reject)
    before = copy.deepcopy(registry)
    result = stl.resolve_gallery_states(
        manifest, old * 2, registry, preserve_registered_samples=True
    )
    assert result["fixed_state_indices"] == registry["fixed_state_indices"]
    assert registry == before
    assert "not reranked" in result["representative_selection"]["selection_epoch"]


@pytest.mark.parametrize("mutation", ["order", "role", "representative"])
def test_preservation_rejects_changed_physical_roles(mutation):
    manifest, registry, old = physical_fixture()
    if mutation == "order":
        registry["state_order"] = list(reversed(registry["state_order"]))
    elif mutation == "role":
        registry["fixed_state_indices"]["wrong_cross_pump_same_power"] = 6
        registry["fixed_states"]["wrong_cross_pump_same_power"] = manifest["states"][6]
    else:
        registry["representative_selection"]["receiver_state"] = 2
    with pytest.raises(ValueError):
        stl.resolve_gallery_states(manifest, old, registry, preserve_registered_samples=True)


@pytest.mark.parametrize("module", [mnist, stl])
def test_preservation_is_an_explicit_cli_opt_in(module):
    required = ["--seeded-checkpoint", "a", "--software-checkpoint", "b", "--codebook", "c",
                "--codebook-manifest", "d", "--registry", "e", "--evaluation-dir", "f", "--output-dir", "g"]
    assert module.build_parser().parse_args(required).preserve_registered_samples is False
    assert module.build_parser().parse_args(required + ["--preserve-registered-samples"]).preserve_registered_samples is True


@pytest.mark.parametrize("module,rows", [(mnist, 5), (mnist, 10), (stl, 5), (stl, 10)])
def test_preserved_gallery_footer_clears_last_row_metric_labels(module, rows, tmp_path):
    shape = (10, 28, 28) if module is mnist else (10, 96, 96, 3)
    arrays = {key: np.zeros(shape, dtype=np.float32) for key in ("original", *stl.BRANCHES)}
    metrics = {
        key: {metric: np.full(10, 0.0123) for metric in ("mse", "rmse", "ssim", "lpips")}
        for key in stl.BRANCHES
    }
    names = {"digit_names": [str(i) for i in range(10)]} if module is mnist else {"class_names": stl.STL10_CLASS_NAMES}
    result = module.plot_quantile_gallery(
        arrays, metrics, quantile_name="q50", sample_ids=list(range(10)),
        output_dir=tmp_path, dpi=40, row_stop=rows,
        preserve_registered_samples=True, **names,
    )
    assert result["footer_clearance"]["verified_no_overlap"] is True
    assert result["footer_clearance"]["clearance_points"] >= 4.0
