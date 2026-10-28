"""Integrity checks operate on full-size synthetic arrays, never model weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import summarize_results as summaries
import verify_release as verifier


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_evaluation(path, dataset):
    path.mkdir(parents=True)
    n = summaries.EXPECTED_COUNTS[dataset]
    arrays = {}
    diagonal = np.arange(10)
    for metric, (software, correct, wrong) in {
        "mse": (0.002, 0.01, 0.09),
        "ssim": (0.98, 0.9, 0.2),
        "lpips": (0.02, 0.05, 0.4),
    }.items():
        arrays[f"software_{metric}"] = np.full(n, software, dtype=np.float32)
        values = np.full((n, 10, 10), wrong, dtype=np.float32)
        values[:, diagonal, diagonal] = correct
        arrays[f"seeded_cross_{metric}"] = values
    arrays["fixed_schedule"] = (np.arange(n) % 10).astype(np.int16)
    np.savez_compressed(path / "per_image_metrics.npz", **arrays)

    def branch(which):
        selected = {}
        for metric in ("mse", "ssim", "lpips"):
            if which == "software":
                values = arrays[f"software_{metric}"]
            elif which == "correct":
                values = arrays[f"seeded_cross_{metric}"][:, diagonal, diagonal]
            else:
                values = arrays[f"seeded_cross_{metric}"][:, ~np.eye(10, dtype=bool)]
            selected[metric] = float(values.astype(np.float64).mean())
        return {"observation_count": n * {"software": 1, "correct": 10, "wrong": 90}[which],
                "mse": selected["mse"], "rmse": selected["mse"] ** 0.5,
                "ssim_mean": selected["ssim"], "lpips_mean": selected["lpips"]}

    correct, wrong = branch("correct"), branch("wrong")
    summary = {"dataset": dataset, "scope": "final", "test_truncated": False,
               "expected_test_images": n, "processed_test_images": n,
               "software": {"overall": branch("software")},
               "seeded": {"cross_state": {"sample_count": n, "state_count": 10,
                                           "correct_pooled": correct, "wrong_pooled": wrong,
                                           "wrong_to_correct_mse_ratio": wrong["mse"] / correct["mse"]}}}
    _write_json(path / "reconstruction_summary.json", summary)
    manifest = {"scope": "final",
                "dataset": {"name": dataset, "expected_test_samples": n, "processed_test_samples": n,
                            "processed_sample_count": n, "test_truncated": False, "shuffle": False},
                "metrics": {"ssim": {"implementation": summaries.EXPECTED_SSIM}, "lpips": {"enabled": True}},
                "state_order": [{"index": i, "state_id": value}
                                for i, value in enumerate(summaries.EXPECTED_STATES)],
                "outputs": {name: _hash(path / name)
                            for name in ("per_image_metrics.npz", "reconstruction_summary.json")}}
    _write_json(path / "evaluation_manifest.json", manifest)


def _manifest_for(root):
    files, errors = verifier._payload_paths(root)
    assert not errors
    _write_json(root / "release_manifest.json", {
        "files": [{"path": relative, "bytes": path.stat().st_size, "sha256": _hash(path)}
                  for relative, path in sorted(files.items())]})


@pytest.fixture(scope="module")
def complete_release(tmp_path_factory):
    root = tmp_path_factory.mktemp("public-release-script-fixture")
    for dataset in ("mnist", "stl10"):
        _fixture_evaluation(root / "results/recomputed" / dataset, dataset)
    (root / "README.md").write_text("Synthetic release integrity fixture.\n", encoding="utf-8")
    _manifest_for(root)
    return root


@pytest.fixture
def release_copy(complete_release, tmp_path):
    target = tmp_path / "release"
    shutil.copytree(complete_release, target)
    return target


def test_array_pooling_and_complete_manifest(complete_release):
    report = verifier.verify_release(complete_release, require_manifest=True)
    assert report["ok"], report["errors"]
    assert report["release_manifest_verified"]
    result = report["evaluations"][0]
    assert result["images"] == 10000
    assert result["branches"]["wrong"]["observations"] == 900000
    assert result["wrong_to_correct_mse_ratio"] == pytest.approx(9, rel=1e-6)
    assert result["branches"]["correct"]["rmse"] == pytest.approx(0.1, rel=1e-6)


def test_legacy_ssim_requires_explicit_audit_mode(release_copy):
    path = release_copy / "results/recomputed/mnist/evaluation_manifest.json"
    manifest = summaries.load_json(path)
    manifest["metrics"]["ssim"]["implementation"] = "torch-gaussian-valid-v1"
    _write_json(path, manifest)
    with pytest.raises(ValueError, match="expected SSIM"):
        summaries.summarize_dataset(path.parent, "mnist")
    result = summaries.summarize_dataset(path.parent, "mnist", require_ssim_v2=False)
    assert not result["publication_metric_version"]


@pytest.mark.parametrize("mutation", ("smoke", "truncated", "count", "summary", "nan", "shape"))
def test_bad_evaluation_is_rejected_even_with_updated_hashes(release_copy, mutation):
    directory = release_copy / "results/recomputed/mnist"
    manifest_path = directory / "evaluation_manifest.json"
    summary_path = directory / "reconstruction_summary.json"
    manifest = summaries.load_json(manifest_path)
    summary = summaries.load_json(summary_path)
    if mutation == "smoke":
        manifest["scope"] = "smoke"
    elif mutation == "truncated":
        summary["test_truncated"] = True
    elif mutation == "count":
        summary["processed_test_images"] = 9999
    elif mutation == "summary":
        summary["seeded"]["cross_state"]["correct_pooled"]["mse"] += 0.01
    else:
        with np.load(directory / "per_image_metrics.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        if mutation == "nan":
            arrays["seeded_cross_lpips"][0, 1, 2] = np.nan
        else:
            arrays["seeded_cross_mse"] = arrays["seeded_cross_mse"][:, :9, :9]
        np.savez_compressed(directory / "per_image_metrics.npz", **arrays)
    _write_json(summary_path, summary)
    manifest["outputs"] = {name: _hash(directory / name) for name in manifest["outputs"]}
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError):
        summaries.summarize_dataset(directory, "mnist")


def test_manifest_missing_extra_and_modified_payload(release_copy):
    (release_copy / "release_manifest.json").unlink()
    report = verifier.verify_release(release_copy)
    assert report["ok"] and report["warnings"] and not report["release_manifest_verified"]
    assert not verifier.verify_release(release_copy, require_manifest=True)["ok"]
    _manifest_for(release_copy)
    (release_copy / "extra.txt").write_text("Unlisted.", encoding="utf-8")
    report = verifier.verify_release(release_copy, require_manifest=True)
    assert any("Unlisted release payload: extra.txt" in error for error in report["errors"])
    (release_copy / "README.md").write_text("Modified.", encoding="utf-8")
    report = verifier.verify_release(release_copy, require_manifest=True)
    assert any("mismatch: README.md" in error for error in report["errors"])


def test_editable_install_metadata_does_not_change_release_inventory(release_copy):
    generated = release_copy / "src/lightwave_seeded_vqvae.egg-info"
    generated.mkdir(parents=True)
    (generated / "PKG-INFO").write_text("Generated installation metadata", encoding="utf-8")
    report = verifier.verify_release(release_copy, require_manifest=True)
    assert report["ok"], report["errors"]


def test_codebook_npz_is_allowed_but_model_weights_are_not(release_copy):
    generated = release_copy / "artifacts/codebooks"
    generated.mkdir(parents=True)
    np.savez_compressed(generated / "bank.npz", codebooks=np.zeros((10, 128, 64), dtype=np.float32))
    report = verifier.verify_release(release_copy, require_manifest=True)
    assert report["ok"], report["errors"]
    (generated / "accidental.pt").write_bytes(b"not an allowed public artifact")
    (release_copy / "model.PTH").write_bytes(b"also forbidden")
    report = verifier.verify_release(release_copy, require_manifest=True)
    assert any("accidental.pt" in error for error in report["errors"])
    assert any("model.PTH" in error for error in report["errors"])


def test_manifest_path_traversal_and_size_limit(release_copy, monkeypatch):
    manifest_path = release_copy / "release_manifest.json"
    manifest = summaries.load_json(manifest_path)
    manifest["files"].append({"path": "../escape.txt", "bytes": 0, "sha256": "0" * 64})
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(verifier, "MAX_FILE_BYTES", 50)
    report = verifier.verify_release(release_copy, require_manifest=True)
    assert any("Unsafe" in error for error in report["errors"])
    assert any("exceeds 50 bytes" in error for error in report["errors"])


def test_optional_cifar_is_checked_when_present(release_copy):
    _fixture_evaluation(release_copy / "results/recomputed/cifar10", "cifar10")
    _manifest_for(release_copy)
    report = verifier.verify_release(release_copy, require_manifest=True)
    assert report["ok"], report["errors"]
    assert [value["dataset"] for value in report["evaluations"]] == ["mnist", "stl10", "cifar10"]
