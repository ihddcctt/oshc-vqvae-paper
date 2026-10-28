from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


COMPARISON_SCHEMA = "lightwave-seeded-vqvae-visual-quality-comparison-v1"
STATE_COUNT = 10
FINGERPRINT_FIELDS = (
    "processed_images_float32_sha256",
    "processed_labels_int64_sha256",
    "processed_sample_order_sha256",
    "processed_sample_count",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1)")
    return parsed


def _ratio_at_least_one(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be finite and at least 1")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be finite and in [0, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two matched visual-quality stages with paired image bootstrap "
            "intervals and a conservative, machine-readable promotion audit."
        )
    )
    parser.add_argument("--baseline-vq-summary", required=True, type=Path)
    parser.add_argument("--baseline-vq-npz", required=True, type=Path)
    parser.add_argument("--candidate-vq-summary", required=True, type=Path)
    parser.add_argument("--candidate-vq-npz", required=True, type=Path)
    parser.add_argument("--baseline-continuous-summary", type=Path)
    parser.add_argument("--baseline-continuous-npz", type=Path)
    parser.add_argument("--candidate-continuous-summary", type=Path)
    parser.add_argument("--candidate-continuous-npz", type=Path)
    parser.add_argument("--baseline-runtime", required=True, type=Path)
    parser.add_argument("--candidate-runtime", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=20260910)
    parser.add_argument(
        "--bootstrap-replicates", type=_positive_int, default=2000
    )
    parser.add_argument(
        "--max-separation-degradation-fraction",
        type=_fraction,
        default=0.10,
        help=(
            "Largest tolerated fractional loss of the wrong/correct MSE ratio or "
            "wrong-minus-correct LPIPS gap. Default: 0.10."
        ),
    )
    parser.add_argument(
        "--min-r-only-to-full-mse-ratio",
        type=_ratio_at_least_one,
        default=1.5,
        help=(
            "Minimum candidate transmitted-R-only MSE divided by full correct "
            "B_s+R MSE. Default: 1.5."
        ),
    )
    parser.add_argument(
        "--max-r-only-ssim-retention",
        type=_unit_interval,
        default=0.95,
        help=(
            "Maximum candidate transmitted-R-only SSIM divided by full correct "
            "B_s+R SSIM. Default: 0.95."
        ),
    )
    parser.add_argument(
        "--min-r-only-to-full-lpips-ratio",
        type=_ratio_at_least_one,
        default=1.25,
        help=(
            "Minimum candidate transmitted-R-only LPIPS divided by full correct "
            "B_s+R LPIPS. Default: 1.25."
        ),
    )
    parser.add_argument(
        "--min-requantized-r-only-psnr-gap-db",
        type=_nonnegative_float,
        default=3.0,
        help=(
            "Minimum full-correct minus re-quantized-R-only pooled PSNR gap. "
            "The point estimate and paired-bootstrap CI lower bound must both "
            "meet it. Default: 3.0 dB, matching training selection."
        ),
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must contain an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"missing JSON field: {'.'.join(keys)}")
        current = current[key]
    return current


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number")
    return parsed


def _expected_output_hash(
    manifest: Mapping[str, Any], path: Path, logical_name: str
) -> str:
    outputs = _require_mapping(manifest.get("outputs"), "manifest.outputs")
    candidates = (path.name, logical_name)
    for candidate in candidates:
        value = outputs.get(candidate)
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping) and isinstance(value.get("sha256"), str):
            return str(value["sha256"])
    raise ValueError(
        f"evaluation manifest does not record a SHA-256 for {path.name}"
    )


def _verified_file(
    manifest: Mapping[str, Any], path: Path, logical_name: str
) -> dict[str, str]:
    expected = _expected_output_hash(manifest, path, logical_name)
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path}: manifest={expected}, actual={actual}"
        )
    return {"path": str(path), "sha256": actual}


def _dataset_identity(manifest: Mapping[str, Any], label: str) -> dict[str, Any]:
    dataset = _require_mapping(manifest.get("dataset"), f"{label}.dataset")
    identity: dict[str, Any] = {
        "name": str(dataset.get("name", "")).strip().lower().replace("-", ""),
    }
    if not identity["name"]:
        raise ValueError(f"{label}.dataset.name must be non-empty")
    for field in FINGERPRINT_FIELDS:
        if field not in dataset:
            raise ValueError(f"{label}.dataset is missing {field}")
        identity[field] = dataset[field]
    for field in FINGERPRINT_FIELDS[:3]:
        value = identity[field]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{label}.dataset.{field} must be a SHA-256 string")
    count = identity["processed_sample_count"]
    if isinstance(count, bool) or int(count) <= 0:
        raise ValueError(f"{label}.dataset.processed_sample_count must be positive")
    identity["processed_sample_count"] = int(count)
    return identity


def _canonical_state_order(manifest: Mapping[str, Any], label: str) -> list[str]:
    state_order = manifest.get("state_order")
    if not isinstance(state_order, list) or len(state_order) != STATE_COUNT:
        raise ValueError(f"{label}.state_order must contain exactly ten states")
    labels: list[str] = []
    for expected_index, row in enumerate(state_order):
        if not isinstance(row, Mapping) or int(row.get("index", -1)) != expected_index:
            raise ValueError(f"{label}.state_order is not in canonical index order")
        state_id = row.get("state_id")
        if not isinstance(state_id, str) or not state_id:
            raise ValueError(f"{label}.state_order has an invalid state_id")
        labels.append(state_id)
    if len(set(labels)) != STATE_COUNT:
        raise ValueError(f"{label}.state_order contains duplicate state IDs")
    return labels


@dataclass(frozen=True)
class EvaluationInput:
    label: str
    summary_path: Path
    metrics_path: Path
    manifest_path: Path
    summary: dict[str, Any]
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]
    dataset_identity: dict[str, Any]
    scope: str
    files: dict[str, dict[str, str]]
    state_order: list[str] | None


def _load_evaluation(
    label: str, summary_path: Path, metrics_path: Path, *, kind: str
) -> EvaluationInput:
    summary_path = summary_path.expanduser().resolve()
    metrics_path = metrics_path.expanduser().resolve()
    if summary_path.parent != metrics_path.parent:
        raise ValueError(f"{label} summary and NPZ must be sibling artifacts")
    manifest_path = summary_path.parent / "evaluation_manifest.json"
    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path)
    files = {
        "summary": _verified_file(
            manifest, summary_path, "reconstruction_summary"
        ),
        "metrics": _verified_file(manifest, metrics_path, "per_image_metrics"),
        "manifest": {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
        },
    }
    try:
        with np.load(metrics_path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError(f"unable to load safe NumPy metrics archive: {metrics_path}") from error

    summary_scope = str(summary.get("scope", "")).strip().lower()
    manifest_scope = str(manifest.get("scope", "")).strip().lower()
    if summary_scope not in {"final", "smoke"} or manifest_scope != summary_scope:
        raise ValueError(f"{label} summary/manifest scope is missing or inconsistent")
    dataset_identity = _dataset_identity(manifest, label)
    state_order = (
        _canonical_state_order(manifest, label) if kind == "vq" else None
    )
    return EvaluationInput(
        label=label,
        summary_path=summary_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        summary=summary,
        manifest=manifest,
        arrays=arrays,
        dataset_identity=dataset_identity,
        scope=summary_scope,
        files=files,
        state_order=state_order,
    )


def _finite_array(
    arrays: Mapping[str, np.ndarray], key: str, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    if key not in arrays:
        raise ValueError(f"metrics archive is missing required array: {key}")
    array = np.asarray(arrays[key], dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{key} must have shape {shape}, received {array.shape}")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{key} must contain finite values")
    return array


def _validate_nonnegative(array: np.ndarray, key: str) -> None:
    if bool((array < 0.0).any()):
        raise ValueError(f"{key} must be non-negative")


def _point_metrics(
    mse: np.ndarray, ssim: np.ndarray, lpips: np.ndarray
) -> dict[str, float]:
    pooled_mse = float(np.asarray(mse, dtype=np.float64).mean(dtype=np.float64))
    return {
        "mse": pooled_mse,
        "rmse": math.sqrt(pooled_mse),
        "ssim": float(np.asarray(ssim, dtype=np.float64).mean(dtype=np.float64)),
        "lpips": float(np.asarray(lpips, dtype=np.float64).mean(dtype=np.float64)),
    }


def _summary_metrics(container: Mapping[str, Any], label: str) -> dict[str, float]:
    return {
        "mse": _finite_float(container.get("mse"), f"{label}.mse"),
        "rmse": _finite_float(container.get("rmse"), f"{label}.rmse"),
        "ssim": _finite_float(container.get("ssim_mean"), f"{label}.ssim_mean"),
        "lpips": _finite_float(container.get("lpips_mean"), f"{label}.lpips_mean"),
    }


def _assert_metric_match(
    computed: Mapping[str, float], recorded: Mapping[str, float], label: str
) -> None:
    for metric in ("mse", "rmse", "ssim", "lpips"):
        if not math.isclose(
            computed[metric], recorded[metric], rel_tol=2e-6, abs_tol=1e-10
        ):
            raise ValueError(
                f"{label}.{metric} does not match its per-image metrics archive: "
                f"summary={recorded[metric]}, computed={computed[metric]}"
            )


@dataclass(frozen=True)
class VQMetrics:
    correct: dict[str, np.ndarray]
    wrong: dict[str, np.ndarray]
    residual_only: dict[str, np.ndarray]
    requantized_residual_only: dict[str, np.ndarray]
    software: dict[str, np.ndarray]
    sample_count: int


def _load_vq_metrics(evaluation: EvaluationInput) -> VQMetrics:
    arrays = evaluation.arrays
    cross_mse = _finite_array(arrays, "seeded_cross_mse")
    if cross_mse.ndim != 3 or cross_mse.shape[1:] != (STATE_COUNT, STATE_COUNT):
        raise ValueError(
            "seeded_cross_mse must have shape (N, 10, 10), including all 90 "
            "ordered wrong-state pairs"
        )
    sample_count = int(cross_mse.shape[0])
    shape = (sample_count, STATE_COUNT, STATE_COUNT)
    cross_ssim = _finite_array(arrays, "seeded_cross_ssim", shape)
    cross_lpips = _finite_array(arrays, "seeded_cross_lpips", shape)
    _validate_nonnegative(cross_mse, "seeded_cross_mse")
    _validate_nonnegative(cross_lpips, "seeded_cross_lpips")

    branch_shape = (sample_count, STATE_COUNT)
    residual_mse = _finite_array(
        arrays, "seeded_transmitted_residual_only_mse", branch_shape
    )
    residual_ssim = _finite_array(
        arrays, "seeded_transmitted_residual_only_ssim", branch_shape
    )
    residual_lpips = _finite_array(
        arrays, "seeded_transmitted_residual_only_lpips", branch_shape
    )
    _validate_nonnegative(residual_mse, "seeded_transmitted_residual_only_mse")
    _validate_nonnegative(residual_lpips, "seeded_transmitted_residual_only_lpips")
    requantized_residual_only = {
        "mse": _finite_array(
            arrays, "seeded_requantized_residual_only_mse", (sample_count,)
        ),
        "ssim": _finite_array(
            arrays, "seeded_requantized_residual_only_ssim", (sample_count,)
        ),
        "lpips": _finite_array(
            arrays, "seeded_requantized_residual_only_lpips", (sample_count,)
        ),
    }
    _validate_nonnegative(
        requantized_residual_only["mse"],
        "seeded_requantized_residual_only_mse",
    )
    _validate_nonnegative(
        requantized_residual_only["lpips"],
        "seeded_requantized_residual_only_lpips",
    )

    states = np.arange(STATE_COUNT)
    wrong_mask = ~np.eye(STATE_COUNT, dtype=bool)
    correct = {
        "mse": cross_mse[:, states, states].mean(axis=1, dtype=np.float64),
        "ssim": cross_ssim[:, states, states].mean(axis=1, dtype=np.float64),
        "lpips": cross_lpips[:, states, states].mean(axis=1, dtype=np.float64),
    }
    wrong = {
        "mse": cross_mse[:, wrong_mask].mean(axis=1, dtype=np.float64),
        "ssim": cross_ssim[:, wrong_mask].mean(axis=1, dtype=np.float64),
        "lpips": cross_lpips[:, wrong_mask].mean(axis=1, dtype=np.float64),
    }
    residual_only = {
        "mse": residual_mse.mean(axis=1, dtype=np.float64),
        "ssim": residual_ssim.mean(axis=1, dtype=np.float64),
        "lpips": residual_lpips.mean(axis=1, dtype=np.float64),
    }
    software = {
        "mse": _finite_array(arrays, "software_mse", (sample_count,)),
        "ssim": _finite_array(arrays, "software_ssim", (sample_count,)),
        "lpips": _finite_array(arrays, "software_lpips", (sample_count,)),
    }
    _validate_nonnegative(software["mse"], "software_mse")
    _validate_nonnegative(software["lpips"], "software_lpips")

    summary = evaluation.summary
    correct_summary = _summary_metrics(
        _require_mapping(
            _nested(summary, "seeded", "cross_state", "correct_pooled"),
            "seeded.cross_state.correct_pooled",
        ),
        "seeded.cross_state.correct_pooled",
    )
    wrong_summary = _summary_metrics(
        _require_mapping(
            _nested(summary, "seeded", "cross_state", "wrong_pooled"),
            "seeded.cross_state.wrong_pooled",
        ),
        "seeded.cross_state.wrong_pooled",
    )
    residual_summary = _summary_metrics(
        _require_mapping(
            _nested(
                summary,
                "seeded",
                "branches",
                "transmitted_residual_only",
                "pooled",
            ),
            "seeded.branches.transmitted_residual_only.pooled",
        ),
        "seeded.branches.transmitted_residual_only.pooled",
    )
    requantized_summary = _summary_metrics(
        _require_mapping(
            _nested(summary, "seeded", "branches", "requantized_residual_only"),
            "seeded.branches.requantized_residual_only",
        ),
        "seeded.branches.requantized_residual_only",
    )
    software_summary = _summary_metrics(
        _require_mapping(
            _nested(summary, "software", "overall"),
            "software.overall",
        ),
        "software.overall",
    )
    _assert_metric_match(
        _point_metrics(**correct), correct_summary, f"{evaluation.label}.correct"
    )
    _assert_metric_match(
        _point_metrics(**wrong), wrong_summary, f"{evaluation.label}.wrong"
    )
    _assert_metric_match(
        _point_metrics(**residual_only),
        residual_summary,
        f"{evaluation.label}.transmitted_residual_only",
    )
    _assert_metric_match(
        _point_metrics(**requantized_residual_only),
        requantized_summary,
        f"{evaluation.label}.requantized_residual_only",
    )
    _assert_metric_match(
        _point_metrics(**software), software_summary, f"{evaluation.label}.software"
    )
    if evaluation.dataset_identity["processed_sample_count"] != sample_count:
        raise ValueError(
            f"{evaluation.label} manifest sample count does not match its NPZ"
        )
    return VQMetrics(
        correct=correct,
        wrong=wrong,
        residual_only=residual_only,
        requantized_residual_only=requantized_residual_only,
        software=software,
        sample_count=sample_count,
    )


@dataclass(frozen=True)
class ContinuousMetrics:
    values: dict[str, np.ndarray]
    sample_indices: np.ndarray
    labels: np.ndarray
    sample_count: int


def _load_continuous_metrics(evaluation: EvaluationInput) -> ContinuousMetrics:
    arrays = evaluation.arrays
    mse = _finite_array(arrays, "mse")
    if mse.ndim != 1:
        raise ValueError("continuous mse must be one-dimensional")
    sample_count = int(mse.size)
    shape = (sample_count,)
    rmse = _finite_array(arrays, "rmse", shape)
    ssim = _finite_array(arrays, "ssim", shape)
    lpips = _finite_array(arrays, "lpips", shape)
    _validate_nonnegative(mse, "continuous mse")
    _validate_nonnegative(rmse, "continuous rmse")
    _validate_nonnegative(lpips, "continuous lpips")
    if not np.allclose(np.square(rmse), mse, rtol=2e-6, atol=1e-10):
        raise ValueError("continuous per-image RMSE is inconsistent with MSE")
    sample_indices = np.asarray(arrays.get("sample_indices"))
    labels = np.asarray(arrays.get("labels"))
    if sample_indices.shape != shape or labels.shape != shape:
        raise ValueError("continuous sample_indices and labels must have shape (N,)")
    if sample_indices.dtype.kind not in "iu" or labels.dtype.kind not in "iu":
        raise ValueError("continuous sample_indices and labels must be integers")
    if np.unique(sample_indices).size != sample_count:
        raise ValueError("continuous sample_indices must be unique")
    recorded = _summary_metrics(
        _require_mapping(
            _nested(evaluation.summary, "continuous", "overall"),
            "continuous.overall",
        ),
        "continuous.overall",
    )
    values = {"mse": mse, "ssim": ssim, "lpips": lpips}
    _assert_metric_match(
        _point_metrics(**values), recorded, f"{evaluation.label}.continuous"
    )
    if evaluation.dataset_identity["processed_sample_count"] != sample_count:
        raise ValueError(
            f"{evaluation.label} manifest sample count does not match its NPZ"
        )
    return ContinuousMetrics(
        values=values,
        sample_indices=sample_indices.astype(np.int64, copy=False),
        labels=labels.astype(np.int64, copy=False),
        sample_count=sample_count,
    )


def _assert_same_identity(inputs: Sequence[EvaluationInput]) -> dict[str, Any]:
    if not inputs:
        raise ValueError("at least one evaluation is required")
    reference = inputs[0]
    for evaluation in inputs[1:]:
        if evaluation.dataset_identity != reference.dataset_identity:
            raise ValueError(
                "dataset/sample-order fingerprint mismatch between "
                f"{reference.label} and {evaluation.label}"
            )
        if evaluation.scope != reference.scope:
            raise ValueError(
                f"evaluation scope mismatch: {reference.label}={reference.scope}, "
                f"{evaluation.label}={evaluation.scope}"
            )
    return {
        "verified": True,
        "method": (
            "exact equality of image SHA-256, label SHA-256, combined sample-order "
            "SHA-256, sample count, and normalized dataset name"
        ),
        "identity": reference.dataset_identity,
        "scope": reference.scope,
        "evaluations": [evaluation.label for evaluation in inputs],
    }


def _percentile_interval(values: np.ndarray) -> dict[str, float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {"lower": float(lower), "upper": float(upper)}


def _relative_change(candidate: float, baseline: float) -> float | None:
    if abs(baseline) <= np.finfo(np.float64).tiny:
        return None
    return float(candidate / baseline - 1.0)


def _bootstrap_metric_bundle(
    baseline: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    seed: int,
    replicates: int,
) -> dict[str, dict[str, Any]]:
    required = ("mse", "ssim", "lpips")
    sample_count = int(np.asarray(baseline["mse"]).size)
    for name in required:
        first = np.asarray(baseline[name], dtype=np.float64)
        second = np.asarray(candidate[name], dtype=np.float64)
        if first.shape != (sample_count,) or second.shape != first.shape:
            raise ValueError(f"paired {name} arrays must share shape (N,)")
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            raise ValueError(f"paired {name} arrays must be finite")

    baseline_point = _point_metrics(
        baseline["mse"], baseline["ssim"], baseline["lpips"]
    )
    candidate_point = _point_metrics(
        candidate["mse"], candidate["ssim"], candidate["lpips"]
    )
    absolute_samples = {
        name: np.empty(replicates, dtype=np.float64)
        for name in ("mse", "rmse", "ssim", "lpips")
    }
    relative_samples = {
        name: np.empty(replicates, dtype=np.float64)
        for name in ("mse", "rmse", "ssim", "lpips")
    }
    generator = np.random.Generator(np.random.PCG64(seed))
    chunk_size = min(128, replicates)
    tiny = np.finfo(np.float64).tiny
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = generator.integers(
            0, sample_count, size=(stop - start, sample_count), dtype=np.int32
        )
        baseline_mse = np.asarray(baseline["mse"])[indices].mean(axis=1)
        candidate_mse = np.asarray(candidate["mse"])[indices].mean(axis=1)
        aggregate = {
            "mse": (baseline_mse, candidate_mse),
            "rmse": (np.sqrt(baseline_mse), np.sqrt(candidate_mse)),
            "ssim": (
                np.asarray(baseline["ssim"])[indices].mean(axis=1),
                np.asarray(candidate["ssim"])[indices].mean(axis=1),
            ),
            "lpips": (
                np.asarray(baseline["lpips"])[indices].mean(axis=1),
                np.asarray(candidate["lpips"])[indices].mean(axis=1),
            ),
        }
        for name, (first, second) in aggregate.items():
            absolute_samples[name][start:stop] = second - first
            denominator = np.where(np.abs(first) > tiny, first, np.nan)
            relative_samples[name][start:stop] = second / denominator - 1.0

    results: dict[str, dict[str, Any]] = {}
    for name in ("mse", "rmse", "ssim", "lpips"):
        preference = "increase" if name == "ssim" else "decrease"
        absolute_change = candidate_point[name] - baseline_point[name]
        relative_change = _relative_change(candidate_point[name], baseline_point[name])
        relative_values = relative_samples[name]
        relative_ci = (
            _percentile_interval(relative_values)
            if np.isfinite(relative_values).all()
            else None
        )
        absolute_ci = _percentile_interval(absolute_samples[name])
        favorable_point = (
            absolute_change > 0.0 if preference == "increase" else absolute_change < 0.0
        )
        favorable_ci = (
            absolute_ci["lower"] > 0.0
            if preference == "increase"
            else absolute_ci["upper"] < 0.0
        )
        results[name] = {
            "baseline": baseline_point[name],
            "candidate": candidate_point[name],
            "absolute_change_candidate_minus_baseline": float(absolute_change),
            "relative_change_candidate_over_baseline_minus_one": relative_change,
            "paired_bootstrap_ci95_absolute_change": absolute_ci,
            "paired_bootstrap_ci95_relative_change": relative_ci,
            "preferred_direction": preference,
            "point_estimate_improved": bool(favorable_point),
            "ci95_entirely_favorable": bool(favorable_ci),
        }
    return results


def _candidate_chip_vs_software_gap(
    candidate: VQMetrics, *, seed: int, replicates: int
) -> dict[str, Any]:
    """Compare matched candidate-stage software and chip reconstructions."""

    raw = _bootstrap_metric_bundle(
        candidate.software,
        candidate.correct,
        seed=seed,
        replicates=replicates,
    )
    metrics: dict[str, Any] = {}
    for name, row in raw.items():
        metrics[name] = {
            "software": row["baseline"],
            "chip_correct": row["candidate"],
            "absolute_gap_chip_minus_software": row[
                "absolute_change_candidate_minus_baseline"
            ],
            "relative_gap_chip_over_software_minus_one": row[
                "relative_change_candidate_over_baseline_minus_one"
            ],
            "paired_bootstrap_ci95_absolute_gap": row[
                "paired_bootstrap_ci95_absolute_change"
            ],
            "paired_bootstrap_ci95_relative_gap": row[
                "paired_bootstrap_ci95_relative_change"
            ],
            "preferred_direction_for_chip": row["preferred_direction"],
        }
    return {
        "definition": (
            "candidate-stage chip B_s+R correct-state reconstruction minus the "
            "matched candidate-stage fully trainable Software VQ reconstruction"
        ),
        "metrics": metrics,
    }


def _bootstrap_separation(
    baseline: VQMetrics,
    candidate: VQMetrics,
    *,
    seed: int,
    replicates: int,
) -> dict[str, dict[str, Any]]:
    sample_count = baseline.sample_count
    generator = np.random.Generator(np.random.PCG64(seed))
    values = {
        "wrong_to_correct_mse_ratio": {
            "baseline": float(
                baseline.wrong["mse"].mean()
                / max(baseline.correct["mse"].mean(), np.finfo(np.float64).tiny)
            ),
            "candidate": float(
                candidate.wrong["mse"].mean()
                / max(candidate.correct["mse"].mean(), np.finfo(np.float64).tiny)
            ),
        },
        "wrong_minus_correct_lpips_gap": {
            "baseline": float(
                baseline.wrong["lpips"].mean()
                - baseline.correct["lpips"].mean()
            ),
            "candidate": float(
                candidate.wrong["lpips"].mean()
                - candidate.correct["lpips"].mean()
            ),
        },
        "correct_minus_wrong_ssim_gap": {
            "baseline": float(
                baseline.correct["ssim"].mean()
                - baseline.wrong["ssim"].mean()
            ),
            "candidate": float(
                candidate.correct["ssim"].mean()
                - candidate.wrong["ssim"].mean()
            ),
        },
    }
    samples = {
        name: {
            "baseline": np.empty(replicates),
            "candidate": np.empty(replicates),
            "retention": np.empty(replicates),
            "absolute_change": np.empty(replicates),
        }
        for name in values
    }
    chunk_size = min(128, replicates)
    tiny = np.finfo(np.float64).tiny
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = generator.integers(
            0, sample_count, size=(stop - start, sample_count), dtype=np.int32
        )
        baseline_correct_mse = baseline.correct["mse"][indices].mean(axis=1)
        baseline_wrong_mse = baseline.wrong["mse"][indices].mean(axis=1)
        candidate_correct_mse = candidate.correct["mse"][indices].mean(axis=1)
        candidate_wrong_mse = candidate.wrong["mse"][indices].mean(axis=1)
        aggregates = {
            "wrong_to_correct_mse_ratio": (
                baseline_wrong_mse / np.maximum(baseline_correct_mse, tiny),
                candidate_wrong_mse / np.maximum(candidate_correct_mse, tiny),
            ),
            "wrong_minus_correct_lpips_gap": (
                baseline.wrong["lpips"][indices].mean(axis=1)
                - baseline.correct["lpips"][indices].mean(axis=1),
                candidate.wrong["lpips"][indices].mean(axis=1)
                - candidate.correct["lpips"][indices].mean(axis=1),
            ),
            "correct_minus_wrong_ssim_gap": (
                baseline.correct["ssim"][indices].mean(axis=1)
                - baseline.wrong["ssim"][indices].mean(axis=1),
                candidate.correct["ssim"][indices].mean(axis=1)
                - candidate.wrong["ssim"][indices].mean(axis=1),
            ),
        }
        for name, (first, second) in aggregates.items():
            samples[name]["baseline"][start:stop] = first
            samples[name]["candidate"][start:stop] = second
            samples[name]["absolute_change"][start:stop] = second - first
            samples[name]["retention"][start:stop] = np.where(
                first > tiny, second / first, np.nan
            )

    result: dict[str, dict[str, Any]] = {}
    for name, point in values.items():
        baseline_value = point["baseline"]
        candidate_value = point["candidate"]
        retention = (
            candidate_value / baseline_value if baseline_value > tiny else None
        )
        retention_samples = samples[name]["retention"]
        result[name] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "absolute_change_candidate_minus_baseline": float(
                candidate_value - baseline_value
            ),
            "candidate_over_baseline_retention": retention,
            "bootstrap_ci95_baseline": _percentile_interval(
                samples[name]["baseline"]
            ),
            "bootstrap_ci95_candidate": _percentile_interval(
                samples[name]["candidate"]
            ),
            "paired_bootstrap_ci95_absolute_change": _percentile_interval(
                samples[name]["absolute_change"]
            ),
            "paired_bootstrap_ci95_retention": (
                _percentile_interval(retention_samples)
                if np.isfinite(retention_samples).all()
                else None
            ),
        }
    return result


def _bootstrap_r_only(
    baseline: VQMetrics,
    candidate: VQMetrics,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    tiny = np.finfo(np.float64).tiny

    def point(metrics: VQMetrics) -> dict[str, float | None]:
        full_mse = float(metrics.correct["mse"].mean())
        residual_mse = float(metrics.residual_only["mse"].mean())
        full_ssim = float(metrics.correct["ssim"].mean())
        residual_ssim = float(metrics.residual_only["ssim"].mean())
        full_lpips = float(metrics.correct["lpips"].mean())
        residual_lpips = float(metrics.residual_only["lpips"].mean())
        return {
            "r_only_to_full_mse_ratio": residual_mse / max(full_mse, tiny),
            "r_only_ssim_performance_retention": (
                residual_ssim / full_ssim if full_ssim > tiny else None
            ),
            "r_only_to_full_lpips_ratio": residual_lpips / max(full_lpips, tiny),
            "r_only_minus_full_lpips": residual_lpips - full_lpips,
        }

    baseline_point = point(baseline)
    candidate_point = point(candidate)
    generator = np.random.Generator(np.random.PCG64(seed))
    mse_ratio_samples = np.empty(replicates, dtype=np.float64)
    ssim_retention_samples = np.empty(replicates, dtype=np.float64)
    lpips_ratio_samples = np.empty(replicates, dtype=np.float64)
    chunk_size = min(128, replicates)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = generator.integers(
            0,
            candidate.sample_count,
            size=(stop - start, candidate.sample_count),
            dtype=np.int32,
        )
        full_mse = candidate.correct["mse"][indices].mean(axis=1)
        residual_mse = candidate.residual_only["mse"][indices].mean(axis=1)
        full_ssim = candidate.correct["ssim"][indices].mean(axis=1)
        residual_ssim = candidate.residual_only["ssim"][indices].mean(axis=1)
        full_lpips = candidate.correct["lpips"][indices].mean(axis=1)
        residual_lpips = candidate.residual_only["lpips"][indices].mean(axis=1)
        mse_ratio_samples[start:stop] = residual_mse / np.maximum(full_mse, tiny)
        ssim_retention_samples[start:stop] = np.where(
            full_ssim > tiny, residual_ssim / full_ssim, np.nan
        )
        lpips_ratio_samples[start:stop] = residual_lpips / np.maximum(
            full_lpips, tiny
        )

    def finite_interval(values: np.ndarray) -> dict[str, float] | None:
        return _percentile_interval(values) if np.isfinite(values).all() else None

    return {
        "definition": (
            "transmitted residual indices decoded with B disabled, compared with "
            "correct-state full B_s+R reconstruction"
        ),
        "baseline": baseline_point,
        "candidate": candidate_point,
        "candidate_r_only_to_full_mse_ratio_bootstrap_ci95": (
            finite_interval(mse_ratio_samples)
        ),
        "candidate_r_only_ssim_performance_retention_bootstrap_ci95": (
            finite_interval(ssim_retention_samples)
        ),
        "candidate_r_only_to_full_lpips_ratio_bootstrap_ci95": (
            finite_interval(lpips_ratio_samples)
        ),
    }


def _bootstrap_requantized_r_only(
    baseline: VQMetrics,
    candidate: VQMetrics,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Audit the strong R-only branch that reselects R without using B_s."""

    tiny = np.finfo(np.float64).tiny

    def point(metrics: VQMetrics) -> dict[str, Any]:
        full = _point_metrics(**metrics.correct)
        residual = _point_metrics(**metrics.requantized_residual_only)
        mse_ratio = residual["mse"] / max(full["mse"], tiny)
        return {
            "full_correct": full,
            "requantized_r_only": residual,
            "requantized_r_only_to_full_mse_ratio": float(mse_ratio),
            "full_minus_requantized_r_only_psnr_db": float(
                10.0 * math.log10(max(mse_ratio, tiny))
            ),
            "requantized_r_only_ssim_performance_retention": (
                float(residual["ssim"] / full["ssim"])
                if full["ssim"] > tiny
                else None
            ),
            "requantized_r_only_to_full_lpips_ratio": float(
                residual["lpips"] / max(full["lpips"], tiny)
            ),
            "requantized_r_only_minus_full_lpips": float(
                residual["lpips"] - full["lpips"]
            ),
        }

    baseline_point = point(baseline)
    candidate_point = point(candidate)
    generator = np.random.Generator(np.random.PCG64(seed))
    mse_ratio_samples = np.empty(replicates, dtype=np.float64)
    psnr_gap_samples = np.empty(replicates, dtype=np.float64)
    ssim_retention_samples = np.empty(replicates, dtype=np.float64)
    lpips_ratio_samples = np.empty(replicates, dtype=np.float64)
    chunk_size = min(128, replicates)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = generator.integers(
            0,
            candidate.sample_count,
            size=(stop - start, candidate.sample_count),
            dtype=np.int32,
        )
        full_mse = candidate.correct["mse"][indices].mean(axis=1)
        residual_mse = candidate.requantized_residual_only["mse"][indices].mean(
            axis=1
        )
        full_ssim = candidate.correct["ssim"][indices].mean(axis=1)
        residual_ssim = candidate.requantized_residual_only["ssim"][indices].mean(
            axis=1
        )
        full_lpips = candidate.correct["lpips"][indices].mean(axis=1)
        residual_lpips = candidate.requantized_residual_only["lpips"][indices].mean(
            axis=1
        )
        mse_ratio = residual_mse / np.maximum(full_mse, tiny)
        mse_ratio_samples[start:stop] = mse_ratio
        psnr_gap_samples[start:stop] = 10.0 * np.log10(
            np.maximum(mse_ratio, tiny)
        )
        ssim_retention_samples[start:stop] = np.where(
            full_ssim > tiny, residual_ssim / full_ssim, np.nan
        )
        lpips_ratio_samples[start:stop] = residual_lpips / np.maximum(
            full_lpips, tiny
        )

    def finite_interval(values: np.ndarray) -> dict[str, float] | None:
        return _percentile_interval(values) if np.isfinite(values).all() else None

    return {
        "definition": (
            "encoder tokens are re-quantized directly against shared R with B_s "
            "absent; this is the strong R-only control used by training selection"
        ),
        "baseline": baseline_point,
        "candidate": candidate_point,
        "candidate_requantized_r_only_to_full_mse_ratio_bootstrap_ci95": (
            finite_interval(mse_ratio_samples)
        ),
        "candidate_full_minus_requantized_r_only_psnr_db_bootstrap_ci95": (
            finite_interval(psnr_gap_samples)
        ),
        "candidate_requantized_r_only_ssim_performance_retention_bootstrap_ci95": (
            finite_interval(ssim_retention_samples)
        ),
        "candidate_requantized_r_only_to_full_lpips_ratio_bootstrap_ci95": (
            finite_interval(lpips_ratio_samples)
        ),
        "hard_gate_metric": "full_minus_requantized_r_only_psnr_db",
        "ssim_and_lpips_policy": "reported_without_additional_hard_thresholds",
    }


def _runtime_summary(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    payload = _read_json(path)
    elapsed = _finite_float(payload.get("elapsed_seconds"), "runtime.elapsed_seconds")
    if elapsed <= 0.0:
        raise ValueError("runtime.elapsed_seconds must be positive")
    completed_epochs = payload.get("completed_epochs")
    if isinstance(completed_epochs, bool) or int(completed_epochs) <= 0:
        raise ValueError("runtime.completed_epochs must be positive")
    completed_epochs = int(completed_epochs)
    raw_counts = _require_mapping(
        payload.get("parameter_counts"), "runtime.parameter_counts"
    )
    counts: dict[str, int] = {}
    for key, value in raw_counts.items():
        if isinstance(value, bool):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            counts[str(key)] = parsed
    for required in ("total", "trainable"):
        if required not in counts:
            raise ValueError(f"runtime.parameter_counts is missing {required}")
    return {
        "file": {"path": str(path), "sha256": _sha256_file(path)},
        "elapsed_seconds": elapsed,
        "completed_epochs": completed_epochs,
        "seconds_per_completed_epoch": elapsed / completed_epochs,
        "parameter_counts": counts,
        "peak_gpu_memory_bytes": payload.get("peak_gpu_memory_bytes"),
    }


def _validate_runtime_against_summary(
    runtime: Mapping[str, Any], evaluation: EvaluationInput
) -> None:
    recorded = _nested(evaluation.summary, "seeded", "model_parameters")
    recorded = _require_mapping(recorded, "seeded.model_parameters")
    summary_total = int(recorded.get("total_parameters", -1))
    runtime_total = int(_nested(runtime, "parameter_counts", "total"))
    if summary_total != runtime_total:
        raise ValueError(
            f"{evaluation.label} runtime total parameter count ({runtime_total}) "
            f"does not match evaluation summary ({summary_total})"
        )


def _numeric_change(candidate: float, baseline: float) -> dict[str, float | None]:
    return {
        "baseline": baseline,
        "candidate": candidate,
        "absolute_change_candidate_minus_baseline": candidate - baseline,
        "relative_change_candidate_over_baseline_minus_one": _relative_change(
            candidate, baseline
        ),
    }


def _runtime_comparison(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    parameter_rows: dict[str, Any] = {}
    baseline_counts = _require_mapping(
        baseline["parameter_counts"], "baseline.parameter_counts"
    )
    candidate_counts = _require_mapping(
        candidate["parameter_counts"], "candidate.parameter_counts"
    )
    for key in sorted(set(baseline_counts) & set(candidate_counts)):
        parameter_rows[key] = _numeric_change(
            float(candidate_counts[key]), float(baseline_counts[key])
        )
    return {
        "parameters": parameter_rows,
        "training_elapsed_seconds": _numeric_change(
            float(candidate["elapsed_seconds"]), float(baseline["elapsed_seconds"])
        ),
        "seconds_per_completed_epoch": _numeric_change(
            float(candidate["seconds_per_completed_epoch"]),
            float(baseline["seconds_per_completed_epoch"]),
        ),
        "completed_epochs": {
            "baseline": int(baseline["completed_epochs"]),
            "candidate": int(candidate["completed_epochs"]),
        },
        "peak_gpu_memory_bytes": {
            "baseline": baseline.get("peak_gpu_memory_bytes"),
            "candidate": candidate.get("peak_gpu_memory_bytes"),
        },
        "interpretation": (
            "wall-clock totals are descriptive; seconds per completed epoch is more "
            "comparable when epoch counts differ"
        ),
    }


def _gate_check(name: str, passed: bool, evidence: Any, rule: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "rule": rule, "evidence": evidence}


def _build_gates(
    correct: Mapping[str, Mapping[str, Any]],
    separation: Mapping[str, Mapping[str, Any]],
    r_only: Mapping[str, Any],
    requantized_r_only: Mapping[str, Any],
    *,
    scope: str,
    maximum_separation_degradation: float,
    minimum_r_only_mse_ratio: float,
    maximum_r_only_ssim_retention: float,
    minimum_r_only_lpips_ratio: float,
    minimum_requantized_r_only_psnr_gap_db: float,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for metric in ("mse", "ssim", "lpips"):
        row = correct[metric]
        checks.append(
            _gate_check(
                f"correct_{metric}_confident_improvement",
                bool(row["point_estimate_improved"])
                and bool(row["ci95_entirely_favorable"]),
                {
                    "absolute_change": row[
                        "absolute_change_candidate_minus_baseline"
                    ],
                    "ci95": row["paired_bootstrap_ci95_absolute_change"],
                },
                (
                    "candidate minus baseline and its paired-image 95% CI must be "
                    + ("above zero" if metric == "ssim" else "below zero")
                ),
            )
        )

    minimum_retention = 1.0 - maximum_separation_degradation
    mse_separation = separation["wrong_to_correct_mse_ratio"]
    lpips_separation = separation["wrong_minus_correct_lpips_gap"]
    ssim_separation = separation["correct_minus_wrong_ssim_gap"]
    mse_candidate_ci = mse_separation["bootstrap_ci95_candidate"]
    mse_retention_ci = mse_separation["paired_bootstrap_ci95_retention"]
    lpips_candidate_ci = lpips_separation["bootstrap_ci95_candidate"]
    lpips_retention_ci = lpips_separation["paired_bootstrap_ci95_retention"]
    ssim_candidate_ci = ssim_separation["bootstrap_ci95_candidate"]
    ssim_retention_ci = ssim_separation["paired_bootstrap_ci95_retention"]
    checks.extend(
        [
            _gate_check(
                "wrong_correct_mse_separation_positive",
                mse_separation["candidate"] > 1.0
                and mse_candidate_ci["lower"] > 1.0,
                {
                    "candidate": mse_separation["candidate"],
                    "candidate_ci95": mse_candidate_ci,
                },
                "candidate wrong/correct MSE ratio and CI lower bound must exceed 1",
            ),
            _gate_check(
                "wrong_correct_mse_separation_not_degraded",
                mse_separation["candidate_over_baseline_retention"] is not None
                and mse_separation["candidate_over_baseline_retention"]
                >= minimum_retention
                and mse_retention_ci is not None
                and mse_retention_ci["lower"] >= minimum_retention,
                {
                    "retention": mse_separation[
                        "candidate_over_baseline_retention"
                    ],
                    "retention_ci95": mse_retention_ci,
                    "minimum": minimum_retention,
                },
                "ratio retention and paired-bootstrap CI lower bound must meet tolerance",
            ),
            _gate_check(
                "correct_wrong_lpips_gap_positive",
                lpips_separation["candidate"] > 0.0
                and lpips_candidate_ci["lower"] > 0.0,
                {
                    "candidate": lpips_separation["candidate"],
                    "candidate_ci95": lpips_candidate_ci,
                },
                "candidate wrong-minus-correct LPIPS gap and CI must be positive",
            ),
            _gate_check(
                "correct_wrong_lpips_gap_not_degraded",
                lpips_separation["candidate_over_baseline_retention"] is not None
                and lpips_separation["candidate_over_baseline_retention"]
                >= minimum_retention
                and lpips_retention_ci is not None
                and lpips_retention_ci["lower"] >= minimum_retention,
                {
                    "retention": lpips_separation[
                        "candidate_over_baseline_retention"
                    ],
                    "retention_ci95": lpips_retention_ci,
                    "minimum": minimum_retention,
                },
                "LPIPS-gap retention and CI lower bound must meet tolerance",
            ),
            _gate_check(
                "correct_wrong_ssim_gap_positive",
                ssim_separation["candidate"] > 0.0
                and ssim_candidate_ci["lower"] > 0.0,
                {
                    "candidate": ssim_separation["candidate"],
                    "candidate_ci95": ssim_candidate_ci,
                },
                "candidate correct-minus-wrong SSIM gap and CI must be positive",
            ),
            _gate_check(
                "correct_wrong_ssim_gap_not_degraded",
                ssim_separation["candidate_over_baseline_retention"] is not None
                and ssim_separation["candidate_over_baseline_retention"]
                >= minimum_retention
                and ssim_retention_ci is not None
                and ssim_retention_ci["lower"] >= minimum_retention,
                {
                    "retention": ssim_separation[
                        "candidate_over_baseline_retention"
                    ],
                    "retention_ci95": ssim_retention_ci,
                    "minimum": minimum_retention,
                },
                "SSIM-gap retention and CI lower bound must meet tolerance",
            ),
        ]
    )
    candidate_mse_ratio = r_only["candidate"]["r_only_to_full_mse_ratio"]
    candidate_mse_ratio_ci = r_only[
        "candidate_r_only_to_full_mse_ratio_bootstrap_ci95"
    ]
    candidate_ssim_retention = r_only["candidate"][
        "r_only_ssim_performance_retention"
    ]
    candidate_ssim_retention_ci = r_only[
        "candidate_r_only_ssim_performance_retention_bootstrap_ci95"
    ]
    candidate_lpips_ratio = r_only["candidate"]["r_only_to_full_lpips_ratio"]
    candidate_lpips_ratio_ci = r_only[
        "candidate_r_only_to_full_lpips_ratio_bootstrap_ci95"
    ]
    checks.extend(
        [
            _gate_check(
                "r_only_mse_not_close_to_full",
                candidate_mse_ratio_ci is not None
                and candidate_mse_ratio >= minimum_r_only_mse_ratio
                and candidate_mse_ratio_ci["lower"] >= minimum_r_only_mse_ratio,
                {
                    "candidate_ratio": candidate_mse_ratio,
                    "candidate_ratio_ci95": candidate_mse_ratio_ci,
                    "minimum": minimum_r_only_mse_ratio,
                },
                "candidate R-only/full MSE ratio and CI lower bound must meet threshold",
            ),
            _gate_check(
                "r_only_ssim_not_close_to_full",
                candidate_ssim_retention is not None
                and candidate_ssim_retention_ci is not None
                and candidate_ssim_retention <= maximum_r_only_ssim_retention
                and candidate_ssim_retention_ci["upper"]
                <= maximum_r_only_ssim_retention,
                {
                    "candidate_retention": candidate_ssim_retention,
                    "candidate_retention_ci95": candidate_ssim_retention_ci,
                    "maximum": maximum_r_only_ssim_retention,
                },
                (
                    "candidate R-only/full SSIM retention and CI upper bound must "
                    "stay below threshold"
                ),
            ),
            _gate_check(
                "r_only_lpips_not_close_to_full",
                candidate_lpips_ratio_ci is not None
                and candidate_lpips_ratio >= minimum_r_only_lpips_ratio
                and candidate_lpips_ratio_ci["lower"]
                >= minimum_r_only_lpips_ratio,
                {
                    "candidate_ratio": candidate_lpips_ratio,
                    "candidate_ratio_ci95": candidate_lpips_ratio_ci,
                    "minimum": minimum_r_only_lpips_ratio,
                },
                "candidate R-only/full LPIPS ratio and CI lower bound must meet threshold",
            ),
        ]
    )
    candidate_requantized_gap = requantized_r_only["candidate"][
        "full_minus_requantized_r_only_psnr_db"
    ]
    candidate_requantized_gap_ci = requantized_r_only[
        "candidate_full_minus_requantized_r_only_psnr_db_bootstrap_ci95"
    ]
    checks.append(
        _gate_check(
            "requantized_r_only_psnr_gap_meets_training_selector",
            candidate_requantized_gap_ci is not None
            and candidate_requantized_gap
            >= minimum_requantized_r_only_psnr_gap_db
            and candidate_requantized_gap_ci["lower"]
            >= minimum_requantized_r_only_psnr_gap_db,
            {
                "candidate_full_minus_requantized_r_only_psnr_db": (
                    candidate_requantized_gap
                ),
                "candidate_paired_bootstrap_ci95": candidate_requantized_gap_ci,
                "minimum_psnr_gap_db": minimum_requantized_r_only_psnr_gap_db,
                "equivalent_minimum_requantized_to_full_mse_ratio": float(
                    10.0 ** (minimum_requantized_r_only_psnr_gap_db / 10.0)
                ),
            },
            (
                "candidate pooled full-minus-re-quantized-R-only PSNR gap and "
                "paired-image bootstrap CI lower bound must meet the training "
                "selector threshold"
            ),
        )
    )
    numerical_pass = all(check["passed"] for check in checks)
    if scope != "final":
        numeric_status = "not_eligible_smoke"
        promotion_status = "not_eligible_smoke"
    elif numerical_pass:
        numeric_status = "pass"
        promotion_status = "pending_manual_review"
    else:
        numeric_status = "fail"
        promotion_status = "not_promoted_numeric_gate_failed"
    return {
        "numeric_gate": {
            "status": numeric_status,
            "all_checks_passed": numerical_pass,
            "checks": checks,
            "rmse_note": (
                "RMSE is reported but not duplicated as a gate because it is the "
                "monotonic square root of pooled MSE."
            ),
        },
        "visual_artifact_gate": {
            "status": "requires_manual_review",
            "auto_evaluable": False,
            "required_checks": [
                "fixed-sample correct reconstructions are visibly clearer",
                "no checkerboard artifacts",
                "no systematic color shift",
                "no perceptual-loss hallucinated or false texture",
                "median, 25th-percentile, and 75th-percentile panels support the claim",
            ],
        },
        "promotion_decision": {
            "status": promotion_status,
            "auto_promoted": False,
            "reason": (
                "Numerical evidence passed; visual artifact review is still mandatory."
                if promotion_status == "pending_manual_review"
                else (
                    "Smoke comparisons are diagnostic and cannot promote a stage."
                    if promotion_status == "not_eligible_smoke"
                    else "At least one numerical promotion rule failed."
                )
            ),
        },
    }


def _comparison_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for population in ("correct", "wrong_all_90_ordered_pairs"):
        for metric, values in payload["vq_metrics"][population].items():
            absolute_ci = values["paired_bootstrap_ci95_absolute_change"]
            relative_ci = values["paired_bootstrap_ci95_relative_change"] or {}
            rows.append(
                {
                    "category": "vq_reconstruction",
                    "population": population,
                    "metric": metric,
                    "baseline": values["baseline"],
                    "candidate": values["candidate"],
                    "absolute_change": values[
                        "absolute_change_candidate_minus_baseline"
                    ],
                    "relative_change": values[
                        "relative_change_candidate_over_baseline_minus_one"
                    ],
                    "ci95_lower": absolute_ci["lower"],
                    "ci95_upper": absolute_ci["upper"],
                    "relative_ci95_lower": relative_ci.get("lower"),
                    "relative_ci95_upper": relative_ci.get("upper"),
                    "status": (
                        "confident_improvement"
                        if values["ci95_entirely_favorable"]
                        else "not_confidently_improved"
                    ),
                }
            )
    for metric, values in payload["software_metrics"].items():
        absolute_ci = values["paired_bootstrap_ci95_absolute_change"]
        relative_ci = values["paired_bootstrap_ci95_relative_change"] or {}
        rows.append(
            {
                "category": "software_vq_reconstruction",
                "population": "held_out_test_images",
                "metric": metric,
                "baseline": values["baseline"],
                "candidate": values["candidate"],
                "absolute_change": values[
                    "absolute_change_candidate_minus_baseline"
                ],
                "relative_change": values[
                    "relative_change_candidate_over_baseline_minus_one"
                ],
                "ci95_lower": absolute_ci["lower"],
                "ci95_upper": absolute_ci["upper"],
                "relative_ci95_lower": relative_ci.get("lower"),
                "relative_ci95_upper": relative_ci.get("upper"),
                "status": (
                    "confident_improvement"
                    if values["ci95_entirely_favorable"]
                    else "not_confidently_improved"
                ),
            }
        )
    for metric, values in payload["requantized_r_only_metrics"].items():
        absolute_ci = values["paired_bootstrap_ci95_absolute_change"]
        relative_ci = values["paired_bootstrap_ci95_relative_change"] or {}
        rows.append(
            {
                "category": "requantized_r_only_reconstruction",
                "population": "held_out_test_images",
                "metric": metric,
                "baseline": values["baseline"],
                "candidate": values["candidate"],
                "absolute_change": values[
                    "absolute_change_candidate_minus_baseline"
                ],
                "relative_change": values[
                    "relative_change_candidate_over_baseline_minus_one"
                ],
                "ci95_lower": absolute_ci["lower"],
                "ci95_upper": absolute_ci["upper"],
                "relative_ci95_lower": relative_ci.get("lower"),
                "relative_ci95_upper": relative_ci.get("upper"),
                "status": "diagnostic_not_a_stage_promotion_metric",
            }
        )
    for metric, values in payload["candidate_chip_vs_software_gap"][
        "metrics"
    ].items():
        absolute_ci = values["paired_bootstrap_ci95_absolute_gap"]
        relative_ci = values["paired_bootstrap_ci95_relative_gap"] or {}
        rows.append(
            {
                "category": "candidate_chip_vs_software",
                "population": "held_out_test_images",
                "metric": metric,
                "baseline": values["software"],
                "candidate": values["chip_correct"],
                "absolute_change": values["absolute_gap_chip_minus_software"],
                "relative_change": values[
                    "relative_gap_chip_over_software_minus_one"
                ],
                "ci95_lower": absolute_ci["lower"],
                "ci95_upper": absolute_ci["upper"],
                "relative_ci95_lower": relative_ci.get("lower"),
                "relative_ci95_upper": relative_ci.get("upper"),
                "status": "diagnostic_not_a_promotion_gate",
            }
        )
    if payload.get("continuous_metrics") is not None:
        for metric, values in payload["continuous_metrics"].items():
            absolute_ci = values["paired_bootstrap_ci95_absolute_change"]
            relative_ci = values["paired_bootstrap_ci95_relative_change"] or {}
            rows.append(
                {
                    "category": "continuous_upper_bound",
                    "population": "held_out_test_images",
                    "metric": metric,
                    "baseline": values["baseline"],
                    "candidate": values["candidate"],
                    "absolute_change": values[
                        "absolute_change_candidate_minus_baseline"
                    ],
                    "relative_change": values[
                        "relative_change_candidate_over_baseline_minus_one"
                    ],
                    "ci95_lower": absolute_ci["lower"],
                    "ci95_upper": absolute_ci["upper"],
                    "relative_ci95_lower": relative_ci.get("lower"),
                    "relative_ci95_upper": relative_ci.get("upper"),
                    "status": (
                        "confident_improvement"
                        if values["ci95_entirely_favorable"]
                        else "not_confidently_improved"
                    ),
                }
            )
    for metric, values in payload["separation"].items():
        ci = values["paired_bootstrap_ci95_absolute_change"]
        rows.append(
            {
                "category": "state_separation",
                "population": "correct_vs_wrong_all_90",
                "metric": metric,
                "baseline": values["baseline"],
                "candidate": values["candidate"],
                "absolute_change": values[
                    "absolute_change_candidate_minus_baseline"
                ],
                "relative_change": (
                    None
                    if values["candidate_over_baseline_retention"] is None
                    else values["candidate_over_baseline_retention"] - 1.0
                ),
                "ci95_lower": ci["lower"],
                "ci95_upper": ci["upper"],
                "relative_ci95_lower": None,
                "relative_ci95_upper": None,
                "status": "diagnostic",
            }
        )
    r_only = payload["r_only_closeness"]
    r_only_fields = {
        "r_only_to_full_mse_ratio": (
            "candidate_r_only_to_full_mse_ratio_bootstrap_ci95"
        ),
        "r_only_ssim_performance_retention": (
            "candidate_r_only_ssim_performance_retention_bootstrap_ci95"
        ),
        "r_only_to_full_lpips_ratio": (
            "candidate_r_only_to_full_lpips_ratio_bootstrap_ci95"
        ),
    }
    for metric, interval_key in r_only_fields.items():
        interval = r_only[interval_key] or {}
        baseline_value = r_only["baseline"][metric]
        candidate_value = r_only["candidate"][metric]
        rows.append(
            {
                "category": "r_only_closeness",
                "population": "candidate_correct_state",
                "metric": metric,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "absolute_change": (
                    candidate_value - baseline_value
                    if baseline_value is not None and candidate_value is not None
                    else None
                ),
                "relative_change": None,
                "ci95_lower": interval.get("lower"),
                "ci95_upper": interval.get("upper"),
                "relative_ci95_lower": None,
                "relative_ci95_upper": None,
                "status": "candidate_interval",
            }
        )
    requantized_r_only = payload["requantized_r_only_closeness"]
    requantized_fields = {
        "requantized_r_only_to_full_mse_ratio": (
            "candidate_requantized_r_only_to_full_mse_ratio_bootstrap_ci95"
        ),
        "full_minus_requantized_r_only_psnr_db": (
            "candidate_full_minus_requantized_r_only_psnr_db_bootstrap_ci95"
        ),
        "requantized_r_only_ssim_performance_retention": (
            "candidate_requantized_r_only_ssim_performance_retention_bootstrap_ci95"
        ),
        "requantized_r_only_to_full_lpips_ratio": (
            "candidate_requantized_r_only_to_full_lpips_ratio_bootstrap_ci95"
        ),
    }
    for metric, interval_key in requantized_fields.items():
        interval = requantized_r_only[interval_key] or {}
        baseline_value = requantized_r_only["baseline"][metric]
        candidate_value = requantized_r_only["candidate"][metric]
        rows.append(
            {
                "category": "requantized_r_only_closeness",
                "population": "candidate_correct_state",
                "metric": metric,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "absolute_change": (
                    candidate_value - baseline_value
                    if baseline_value is not None and candidate_value is not None
                    else None
                ),
                "relative_change": None,
                "ci95_lower": interval.get("lower"),
                "ci95_upper": interval.get("upper"),
                "relative_ci95_lower": None,
                "relative_ci95_upper": None,
                "status": (
                    "hard_gate_input"
                    if metric == "full_minus_requantized_r_only_psnr_db"
                    else "diagnostic_no_additional_hard_threshold"
                ),
            }
        )
    resources = payload["resources"]["comparison"]
    for metric, values in resources["parameters"].items():
        rows.append(
            {
                "category": "resources",
                "population": "training_run",
                "metric": f"parameters.{metric}",
                "baseline": values["baseline"],
                "candidate": values["candidate"],
                "absolute_change": values[
                    "absolute_change_candidate_minus_baseline"
                ],
                "relative_change": values[
                    "relative_change_candidate_over_baseline_minus_one"
                ],
                "ci95_lower": None,
                "ci95_upper": None,
                "relative_ci95_lower": None,
                "relative_ci95_upper": None,
                "status": "descriptive",
            }
        )
    for metric in ("training_elapsed_seconds", "seconds_per_completed_epoch"):
        values = resources[metric]
        rows.append(
            {
                "category": "resources",
                "population": "training_run",
                "metric": metric,
                "baseline": values["baseline"],
                "candidate": values["candidate"],
                "absolute_change": values[
                    "absolute_change_candidate_minus_baseline"
                ],
                "relative_change": values[
                    "relative_change_candidate_over_baseline_minus_one"
                ],
                "ci95_lower": None,
                "ci95_upper": None,
                "relative_ci95_lower": None,
                "relative_ci95_upper": None,
                "status": "descriptive",
            }
        )
    for check in payload["gates"]["numeric_gate"]["checks"]:
        rows.append(
            {
                "category": "gate",
                "population": "numeric",
                "metric": check["name"],
                "baseline": None,
                "candidate": None,
                "absolute_change": None,
                "relative_change": None,
                "ci95_lower": None,
                "ci95_upper": None,
                "relative_ci95_lower": None,
                "relative_ci95_upper": None,
                "status": "pass" if check["passed"] else "fail",
            }
        )
    rows.append(
        {
            "category": "gate",
            "population": "visual",
            "metric": "artifact_review",
            "baseline": None,
            "candidate": None,
            "absolute_change": None,
            "relative_change": None,
            "ci95_lower": None,
            "ci95_upper": None,
            "relative_ci95_lower": None,
            "relative_ci95_upper": None,
            "status": "requires_manual_review",
        }
    )
    return rows


def compare(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"comparison output directory already exists: {output_dir}")

    baseline_vq = _load_evaluation(
        "baseline_vq", args.baseline_vq_summary, args.baseline_vq_npz, kind="vq"
    )
    candidate_vq = _load_evaluation(
        "candidate_vq", args.candidate_vq_summary, args.candidate_vq_npz, kind="vq"
    )
    if baseline_vq.state_order != candidate_vq.state_order:
        raise ValueError("baseline and candidate VQ state order differs")

    optional_paths = (
        args.baseline_continuous_summary,
        args.baseline_continuous_npz,
        args.candidate_continuous_summary,
        args.candidate_continuous_npz,
    )
    if any(path is not None for path in optional_paths) and not all(
        path is not None for path in optional_paths
    ):
        raise ValueError(
            "continuous comparison requires all four baseline/candidate summary/NPZ paths"
        )
    continuous_evaluations: tuple[EvaluationInput, EvaluationInput] | None = None
    if all(path is not None for path in optional_paths):
        continuous_evaluations = (
            _load_evaluation(
                "baseline_continuous",
                args.baseline_continuous_summary,
                args.baseline_continuous_npz,
                kind="continuous",
            ),
            _load_evaluation(
                "candidate_continuous",
                args.candidate_continuous_summary,
                args.candidate_continuous_npz,
                kind="continuous",
            ),
        )

    identity_inputs = [baseline_vq, candidate_vq]
    if continuous_evaluations is not None:
        identity_inputs.extend(continuous_evaluations)
    identity = _assert_same_identity(identity_inputs)

    baseline_metrics = _load_vq_metrics(baseline_vq)
    candidate_metrics = _load_vq_metrics(candidate_vq)
    if baseline_metrics.sample_count != candidate_metrics.sample_count:
        raise ValueError("baseline and candidate VQ sample counts differ")

    continuous_result = None
    continuous_input_files = None
    if continuous_evaluations is not None:
        baseline_continuous = _load_continuous_metrics(continuous_evaluations[0])
        candidate_continuous = _load_continuous_metrics(continuous_evaluations[1])
        if not np.array_equal(
            baseline_continuous.sample_indices, candidate_continuous.sample_indices
        ) or not np.array_equal(baseline_continuous.labels, candidate_continuous.labels):
            raise ValueError(
                "baseline and candidate continuous sample order/labels differ"
            )
        if baseline_continuous.sample_count != baseline_metrics.sample_count:
            raise ValueError("continuous and VQ sample counts differ")
        continuous_result = _bootstrap_metric_bundle(
            baseline_continuous.values,
            candidate_continuous.values,
            seed=args.bootstrap_seed + 3,
            replicates=args.bootstrap_replicates,
        )
        continuous_input_files = {
            evaluation.label: evaluation.files
            for evaluation in continuous_evaluations
        }

    baseline_runtime = _runtime_summary(args.baseline_runtime)
    candidate_runtime = _runtime_summary(args.candidate_runtime)
    _validate_runtime_against_summary(baseline_runtime, baseline_vq)
    _validate_runtime_against_summary(candidate_runtime, candidate_vq)

    correct = _bootstrap_metric_bundle(
        baseline_metrics.correct,
        candidate_metrics.correct,
        seed=args.bootstrap_seed,
        replicates=args.bootstrap_replicates,
    )
    wrong = _bootstrap_metric_bundle(
        baseline_metrics.wrong,
        candidate_metrics.wrong,
        seed=args.bootstrap_seed + 1,
        replicates=args.bootstrap_replicates,
    )
    separation = _bootstrap_separation(
        baseline_metrics,
        candidate_metrics,
        seed=args.bootstrap_seed + 2,
        replicates=args.bootstrap_replicates,
    )
    r_only = _bootstrap_r_only(
        baseline_metrics,
        candidate_metrics,
        seed=args.bootstrap_seed + 4,
        replicates=args.bootstrap_replicates,
    )
    software = _bootstrap_metric_bundle(
        baseline_metrics.software,
        candidate_metrics.software,
        seed=args.bootstrap_seed + 5,
        replicates=args.bootstrap_replicates,
    )
    candidate_chip_vs_software = _candidate_chip_vs_software_gap(
        candidate_metrics,
        seed=args.bootstrap_seed + 6,
        replicates=args.bootstrap_replicates,
    )
    requantized_r_only_metrics = _bootstrap_metric_bundle(
        baseline_metrics.requantized_residual_only,
        candidate_metrics.requantized_residual_only,
        seed=args.bootstrap_seed + 7,
        replicates=args.bootstrap_replicates,
    )
    requantized_r_only = _bootstrap_requantized_r_only(
        baseline_metrics,
        candidate_metrics,
        seed=args.bootstrap_seed + 8,
        replicates=args.bootstrap_replicates,
    )
    gates = _build_gates(
        correct,
        separation,
        r_only,
        requantized_r_only,
        scope=identity["scope"],
        maximum_separation_degradation=args.max_separation_degradation_fraction,
        minimum_r_only_mse_ratio=args.min_r_only_to_full_mse_ratio,
        maximum_r_only_ssim_retention=args.max_r_only_ssim_retention,
        minimum_r_only_lpips_ratio=args.min_r_only_to_full_lpips_ratio,
        minimum_requantized_r_only_psnr_gap_db=(
            args.min_requantized_r_only_psnr_gap_db
        ),
    )

    payload: dict[str, Any] = {
        "schema": COMPARISON_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": identity["scope"],
        "formal": identity["scope"] == "final",
        "sample_identity": identity,
        "state_order": baseline_vq.state_order,
        "bootstrap": {
            "seed": int(args.bootstrap_seed),
            "replicates": int(args.bootstrap_replicates),
            "sampling_unit": "held-out test image",
            "pairing": (
                "correct states or all 90 ordered wrong-state pairs are averaged "
                "within each image before paired image resampling"
            ),
            "uncertainty_boundary": (
                "test-image sampling uncertainty only; this is not uncertainty "
                "across independent training runs"
            ),
        },
        "thresholds": {
            "max_separation_degradation_fraction": float(
                args.max_separation_degradation_fraction
            ),
            "min_r_only_to_full_mse_ratio": float(
                args.min_r_only_to_full_mse_ratio
            ),
            "max_r_only_ssim_retention": float(args.max_r_only_ssim_retention),
            "min_r_only_to_full_lpips_ratio": float(
                args.min_r_only_to_full_lpips_ratio
            ),
            "min_requantized_r_only_psnr_gap_db": float(
                args.min_requantized_r_only_psnr_gap_db
            ),
            "equivalent_min_requantized_r_only_to_full_mse_ratio": float(
                10.0 ** (args.min_requantized_r_only_psnr_gap_db / 10.0)
            ),
            "correct_metric_rule": (
                "point estimate and paired-bootstrap 95% CI must be favorable"
            ),
        },
        "inputs": {
            "baseline_vq": baseline_vq.files,
            "candidate_vq": candidate_vq.files,
            "continuous": continuous_input_files,
            "baseline_runtime": baseline_runtime["file"],
            "candidate_runtime": candidate_runtime["file"],
        },
        "vq_metrics": {
            "correct": correct,
            "wrong_all_90_ordered_pairs": wrong,
        },
        "software_metrics": software,
        "candidate_chip_vs_software_gap": candidate_chip_vs_software,
        "requantized_r_only_metrics": requantized_r_only_metrics,
        "continuous_metrics": continuous_result,
        "separation": separation,
        "r_only_closeness": r_only,
        "requantized_r_only_closeness": requantized_r_only,
        "resources": {
            "baseline": {
                key: value for key, value in baseline_runtime.items() if key != "file"
            },
            "candidate": {
                key: value for key, value in candidate_runtime.items() if key != "file"
            },
            "comparison": _runtime_comparison(
                baseline_runtime, candidate_runtime
            ),
        },
        "gates": gates,
    }
    return payload


def _write_outputs(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "stage_comparison.json"
    csv_path = output_dir / "stage_comparison.csv"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = _comparison_rows(payload)
    fieldnames = [
        "category",
        "population",
        "metric",
        "baseline",
        "candidate",
        "absolute_change",
        "relative_change",
        "ci95_lower",
        "ci95_upper",
        "relative_ci95_lower",
        "relative_ci95_upper",
        "status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = compare(args)
    output_dir = args.output_dir.expanduser().resolve()
    _write_outputs(output_dir, payload)
    print(json.dumps(payload["gates"], ensure_ascii=False, indent=2))
    print(f"comparison directory: {output_dir}")


if __name__ == "__main__":
    main()
