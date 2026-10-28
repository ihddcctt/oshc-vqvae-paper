"""Recompute publication metrics from arrays, without model weights or PyTorch.

Examples:
    python scripts/summarize_results.py
    python scripts/summarize_results.py --dataset stl10 --json
    python scripts/summarize_results.py --output-json /new/path/pooled_metrics.json

Only full, corrected MNIST/STL-10 evaluations pass the default checks. The
--allow-legacy-ssim switch is an explicit audit mode, not a publication result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COUNTS = {"mnist": 10000, "stl10": 8000, "cifar10": 10000}
PRIMARY_DATASETS = ("mnist", "stl10")
EXPECTED_SSIM = "torch-gaussian-valid-v2"
EXPECTED_STATES = [f"{pump}nm_{power}mW" for pump in (405, 640) for power in (0, 5, 10, 15, 20)]
ATOL = 1e-12
RTOL = 1e-9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return value


def _same_number(actual: float, recorded: Any, location: str) -> None:
    if isinstance(recorded, bool) or not isinstance(recorded, (int, float)):
        raise ValueError(f"Missing/non-numeric summary field: {location}")
    if not math.isfinite(recorded) or not math.isclose(actual, recorded, rel_tol=RTOL, abs_tol=ATOL):
        raise ValueError(f"Summary mismatch at {location}: recomputed={actual:.17g}, recorded={recorded!r}")


def _pool(mse: np.ndarray, ssim: np.ndarray, lpips: np.ndarray) -> dict[str, float | int]:
    value = float(mse.mean(dtype=np.float64))
    return {
        "observations": int(mse.size),
        "mse": value,
        "rmse": math.sqrt(value),
        "ssim": float(ssim.mean(dtype=np.float64)),
        "lpips": float(lpips.mean(dtype=np.float64)),
    }


def summarize_dataset(directory: Path, dataset: str, *, require_ssim_v2: bool = True) -> dict[str, Any]:
    """Return verified scalar metrics, or raise on incomplete/inconsistent data."""
    if dataset not in EXPECTED_COUNTS:
        raise ValueError(f"Unsupported final dataset: {dataset}")
    directory = Path(directory).resolve()
    expected = EXPECTED_COUNTS[dataset]
    summary_path = directory / "reconstruction_summary.json"
    manifest_path = directory / "evaluation_manifest.json"
    arrays_path = directory / "per_image_metrics.npz"
    for path in (summary_path, manifest_path, arrays_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty required result: {path}")
    summary = load_json(summary_path)
    manifest = load_json(manifest_path)
    metadata = manifest.get("dataset", {})
    if summary.get("dataset") != dataset or metadata.get("name") != dataset:
        raise ValueError(f"Dataset identity mismatch for {dataset}")
    if summary.get("scope") != "final" or manifest.get("scope") != "final":
        raise ValueError(f"{dataset}: smoke/partial evaluation cannot be a final result")
    if summary.get("test_truncated") is not False or metadata.get("test_truncated") is not False:
        raise ValueError(f"{dataset}: test_truncated must be explicitly false")
    for key in ("expected_test_images", "processed_test_images"):
        if summary.get(key) != expected:
            raise ValueError(f"{dataset}: summary.{key} must be {expected}")
    for key in ("expected_test_samples", "processed_test_samples", "processed_sample_count"):
        if metadata.get(key) != expected:
            raise ValueError(f"{dataset}: manifest.dataset.{key} must be {expected}")
    if metadata.get("shuffle") is not False:
        raise ValueError(f"{dataset}: official test stream must be explicitly unshuffled")

    metric_spec = manifest.get("metrics", {})
    ssim_version = metric_spec.get("ssim", {}).get("implementation")
    if require_ssim_v2 and ssim_version != EXPECTED_SSIM:
        raise ValueError(f"{dataset}: expected SSIM {EXPECTED_SSIM}, found {ssim_version!r}")
    if metric_spec.get("lpips", {}).get("enabled") is not True:
        raise ValueError(f"{dataset}: LPIPS must be enabled for the final comparison")
    state_order = manifest.get("state_order", [])
    if [entry.get("state_id") for entry in state_order] != EXPECTED_STATES:
        raise ValueError(f"{dataset}: state labels/order differ from the published bank")
    if [entry.get("index") for entry in state_order] != list(range(10)):
        raise ValueError(f"{dataset}: state indices must be 0 through 9")

    inputs_hashed = {}
    for path in (arrays_path, summary_path):
        actual = sha256_file(path)
        expected_hash = manifest.get("outputs", {}).get(path.name)
        if actual != expected_hash:
            raise ValueError(f"{dataset}: evaluation-manifest hash mismatch for {path.name}")
        inputs_hashed[path.name] = actual
    inputs_hashed[manifest_path.name] = sha256_file(manifest_path)

    arrays = {}
    with np.load(arrays_path, allow_pickle=False) as saved:
        for name in saved.files:
            value = saved[name]
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise ValueError(f"{dataset}: non-finite or non-numeric NPZ array: {name}")
            if value.ndim < 1 or value.shape[0] != expected:
                raise ValueError(f"{dataset}: {name} has no complete {expected}-image axis")
            if name.endswith("_mse") and bool((value < 0).any()):
                raise ValueError(f"{dataset}: negative MSE in {name}")
            arrays[name] = value
    for metric in ("mse", "ssim", "lpips"):
        for name, shape in ((f"software_{metric}", (expected,)),
                            (f"seeded_cross_{metric}", (expected, 10, 10))):
            if name not in arrays or arrays[name].shape != shape:
                raise ValueError(f"{dataset}: required {name} must have shape {shape}")

    diagonal = np.arange(10)
    wrong_mask = ~np.eye(10, dtype=bool)
    correct = [arrays[f"seeded_cross_{metric}"][:, diagonal, diagonal]
               for metric in ("mse", "ssim", "lpips")]
    wrong = [arrays[f"seeded_cross_{metric}"][:, wrong_mask]
             for metric in ("mse", "ssim", "lpips")]
    software = [arrays[f"software_{metric}"] for metric in ("mse", "ssim", "lpips")]
    branches = {"software": _pool(*software), "correct": _pool(*correct), "wrong": _pool(*wrong)}
    cross = summary.get("seeded", {}).get("cross_state", {})
    if cross.get("sample_count") != expected or cross.get("state_count") != 10:
        raise ValueError(f"{dataset}: summary cross-state cardinality mismatch")
    recorded_branches = {
        "software": summary.get("software", {}).get("overall", {}),
        "correct": cross.get("correct_pooled", {}),
        "wrong": cross.get("wrong_pooled", {}),
    }
    summary_names = {"mse": "mse", "rmse": "rmse", "ssim": "ssim_mean", "lpips": "lpips_mean"}
    for name, calculated in branches.items():
        recorded = recorded_branches[name]
        if recorded.get("observation_count") != calculated["observations"]:
            raise ValueError(f"{dataset}: {name} observation_count mismatch")
        for metric, source_key in summary_names.items():
            _same_number(calculated[metric], recorded.get(source_key), f"{dataset}.{name}.{source_key}")

    if branches["correct"]["mse"] <= 0:
        raise ValueError(f"{dataset}: zero correct MSE prevents a finite wrong/correct ratio")
    ratio = branches["wrong"]["mse"] / branches["correct"]["mse"]
    _same_number(ratio, cross.get("wrong_to_correct_mse_ratio"), f"{dataset}.wrong_to_correct_mse_ratio")
    return {
        "dataset": dataset,
        "images": expected,
        "sender_states": 10,
        "ordered_wrong_pairs": 90,
        "ssim_implementation": ssim_version,
        "publication_metric_version": ssim_version == EXPECTED_SSIM,
        "summary_verified": True,
        "array_count": len(arrays),
        "branches": branches,
        "wrong_to_correct_mse_ratio": ratio,
        "wrong_minus_correct_lpips": branches["wrong"]["lpips"] - branches["correct"]["lpips"],
        "correct_minus_wrong_ssim": branches["correct"]["ssim"] - branches["wrong"]["ssim"],
        "source_sha256": inputs_hashed,
    }


def format_table(results: list[dict[str, Any]]) -> str:
    lines = ["dataset   branch        images          MSE         RMSE         SSIM        LPIPS",
             "--------- ------------ ------- ------------ ------------ ------------ ------------"]
    for result in results:
        for branch, values in result["branches"].items():
            lines.append(f"{result['dataset']:<9} {branch:<12} {result['images']:>7} "
                         + " ".join(f"{values[key]:12.8f}" for key in ("mse", "rmse", "ssim", "lpips")))
        lines.append(f"  wrong/correct MSE: {result['wrong_to_correct_mse_ratio']:.6f}; "
                     f"wrong-correct LPIPS: {result['wrong_minus_correct_lpips']:.8f}; "
                     f"{result['ssim_implementation']}; verified against arrays and summary")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/recomputed")
    parser.add_argument("--dataset", choices=("all", "mnist", "stl10", "cifar10"), default="all")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the human-readable table")
    parser.add_argument("--output-json", type=Path, help="Write JSON exclusively to a new file")
    parser.add_argument("--allow-legacy-ssim", action="store_true", help="Audit historical results; never label them corrected")
    args = parser.parse_args()
    datasets = ([*PRIMARY_DATASETS, *(["cifar10"] if (args.results_root / "cifar10").exists() else [])]
                if args.dataset == "all" else [args.dataset])
    results, errors = [], []
    for dataset in datasets:
        try:
            results.append(summarize_dataset(args.results_root / dataset, dataset,
                                             require_ssim_v2=not args.allow_legacy_ssim))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({"dataset": dataset, "error": str(error)})
    report = {"schema": "oshc-vqvae-array-summary-v1", "ok": not errors,
              "legacy_audit_mode": args.allow_legacy_ssim, "results": results, "errors": errors}
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output_json:
        try:
            with args.output_json.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
        except OSError as error:
            print(f"Cannot write new JSON output: {error}", file=sys.stderr)
            return 1
    if args.json:
        print(encoded, end="")
    else:
        print(format_table(results))
        if args.allow_legacy_ssim:
            print("AUDIT MODE: historical SSIM is not a corrected publication metric.")
        for error in errors:
            print(f"ERROR [{error['dataset']}]: {error['error']}", file=sys.stderr)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
