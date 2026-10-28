from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data import build_dataset
from ..seed.codebook import load_codebook_bank
from .build_fixed_sample_registry import (
    EXPECTED_STATE_IDS,
    FIXED_STATE_IDS,
    QUANTILES,
    REGISTRY_SCHEMA,
    select_quantile_samples,
    validate_state_order,
)
from .evaluate_final import (
    EVALUATION_SCHEMA,
    _array_sha256,
    _load_model,
    _read_json,
    _sha256_file,
    _state_labels,
)
from .plot_cifar_natural_examples import (
    CIFAR10_CLASS_NAMES_EN,
    RC_PARAMS,
    _dataset_fingerprint,
    _save_figure,
    _targets_from_dataset,
    nchw_to_rgb,
    rgb_absolute_error,
    run_controlled_inference,
    state_records_from_manifest,
    validate_controlled_state_mapping,
)
from .plot_final_evaluation import compact_state_label


UPGRADE_VISUAL_SCHEMA = "lightwave-seeded-vqvae-visual-quality-upgrade-v1"
INFERENCE_MSE_RTOL = 2e-3
INFERENCE_MSE_ATOL = 2e-6
DISPLAY_KEYS = (
    "baseline_correct",
    "candidate_correct",
    "candidate_cross_pump",
    "candidate_same_pump",
    "candidate_representative",
)
DISPLAY_TITLES = {
    "baseline_correct": "Pre-upgrade correct\n2 → 2",
    "candidate_correct": "Candidate correct\n2 → 2",
    "candidate_cross_pump": "Candidate cross-pump\n2 → 7",
    "candidate_same_pump": "Candidate wrong power\n2 → 4",
    "candidate_representative": "Candidate representative wrong\n2 → 1",
}
PAIR_BY_DISPLAY_KEY = {
    "baseline_correct": (2, 2),
    "candidate_correct": (2, 2),
    "candidate_cross_pump": (2, 7),
    "candidate_same_pump": (2, 4),
    "candidate_representative": (2, 1),
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the frozen-sample pre-upgrade versus candidate CIFAR-10 "
            "PPT visual package."
        )
    )
    parser.add_argument("--baseline-seeded-checkpoint", required=True, type=Path)
    parser.add_argument("--candidate-seeded-checkpoint", required=True, type=Path)
    parser.add_argument("--baseline-codebook", required=True, type=Path)
    parser.add_argument("--baseline-codebook-manifest", required=True, type=Path)
    parser.add_argument("--candidate-codebook", required=True, type=Path)
    parser.add_argument("--candidate-codebook-manifest", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--baseline-metrics", required=True, type=Path)
    parser.add_argument("--candidate-summary", required=True, type=Path)
    parser.add_argument("--candidate-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dpi", type=_positive_int, default=300)
    return parser


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _int64_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _require_file(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _evaluation_manifest_for(summary: Path, metrics: Path) -> Path:
    if summary.parent != metrics.parent:
        raise ValueError("each evaluation summary and NPZ must share one directory")
    manifest = summary.parent / "evaluation_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"formal evaluation_manifest.json is required beside summary/NPZ: {manifest}"
        )
    return manifest


def _validate_registry_states(registry: Mapping[str, Any]) -> None:
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("unsupported fixed sample registry schema")
    validate_state_order(registry.get("state_order"))
    fixed = registry.get("fixed_state_indices")
    if fixed != FIXED_STATE_IDS:
        raise ValueError(
            "registry fixed states must be sender=2, cross-pump=7, "
            "same-pump wrong-power=4, representative=1"
        )
    verbose = registry.get("fixed_states")
    if not isinstance(verbose, dict):
        raise ValueError("registry is missing verbose fixed-state records")
    for role, state_index in FIXED_STATE_IDS.items():
        record = verbose.get(role)
        if not isinstance(record, dict) or record.get("index") != state_index:
            raise ValueError(f"registry fixed-state record is invalid: {role}")
        if record.get("state_id") != EXPECTED_STATE_IDS[state_index]:
            raise ValueError(f"registry fixed-state label is invalid: {role}")


def resolve_registry_sample_ids(
    registry: Mapping[str, Any],
    targets: Sequence[int] | np.ndarray,
    baseline_cross_mse: np.ndarray,
) -> dict[str, list[int]]:
    """Verify and return the immutable q25/q50/q75 sample order.

    Registry IDs are checked against a fresh deterministic recomputation from
    the designated baseline, but never selected from the candidate metrics.
    """

    _validate_registry_states(registry)
    labels = np.asarray(targets, dtype=np.int64)
    cross = np.asarray(baseline_cross_mse, dtype=np.float64)
    if cross.shape != (labels.size, 10, 10):
        raise ValueError("baseline seeded_cross_mse and test labels have incompatible shapes")
    scores = np.diagonal(cross, axis1=1, axis2=2).mean(axis=1, dtype=np.float64)
    expected = select_quantile_samples(labels, scores)
    classes = registry.get("classes")
    if not isinstance(classes, list) or len(classes) != 10:
        raise ValueError("registry must contain exactly ten class records")

    ordered: dict[str, list[int]] = {name: [] for name, _ in QUANTILES}
    for class_id, (stored_class, expected_class) in enumerate(zip(classes, expected)):
        if not isinstance(stored_class, dict) or stored_class.get("class_id") != class_id:
            raise ValueError("registry classes must appear exactly in CIFAR class order 0..9")
        records = stored_class.get("quantiles")
        if not isinstance(records, dict):
            raise ValueError(f"registry class {class_id} is missing quantile records")
        for quantile_name, _ in QUANTILES:
            stored = records.get(quantile_name)
            expected_record = expected_class["quantiles"][quantile_name]
            if not isinstance(stored, dict):
                raise ValueError(
                    f"registry class {class_id} is missing {quantile_name}"
                )
            sample_id = int(stored.get("sample_id", -1))
            if sample_id != int(expected_record["sample_id"]):
                raise ValueError(
                    "registry sample does not reproduce the designated baseline "
                    f"selection: class {class_id} {quantile_name}"
                )
            for field in (
                "baseline_empirical_quantile_mse",
                "selected_sample_baseline_diagonal_mean_mse",
            ):
                if not np.isclose(
                    float(stored.get(field, np.nan)),
                    float(expected_record[field]),
                    rtol=1e-10,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"registry baseline MSE is inconsistent: class {class_id} "
                        f"{quantile_name} {field}"
                    )
            if labels[sample_id] != class_id:
                raise ValueError("registry sample ID does not belong to its recorded class")
            ordered[quantile_name].append(sample_id)
    return ordered


def _load_cross_arrays(path: Path, *, sample_count: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "seeded_cross_mse",
            "seeded_cross_ssim",
            "seeded_cross_lpips",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"per-image archive is missing required keys: {missing}")
        arrays = {
            "mse": np.asarray(archive["seeded_cross_mse"], dtype=np.float64),
            "ssim": np.asarray(archive["seeded_cross_ssim"], dtype=np.float64),
            "lpips": np.asarray(archive["seeded_cross_lpips"], dtype=np.float64),
        }
    expected_shape = (sample_count, 10, 10)
    for name, array in arrays.items():
        if array.shape != expected_shape or not np.isfinite(array).all():
            raise ValueError(f"seeded_cross_{name} must be finite shape {expected_shape}")
    if np.any(arrays["mse"] < 0.0) or np.any(arrays["lpips"] < 0.0):
        raise ValueError("MSE and LPIPS arrays must be non-negative")
    return arrays


def _validate_summary(
    summary: Mapping[str, Any], arrays: Mapping[str, np.ndarray], *, sample_count: int
) -> None:
    dataset_name = str(summary.get("dataset", "")).lower().replace("-", "")
    if summary.get("scope") != "final" or dataset_name != "cifar10":
        raise ValueError("upgrade visuals require a formal full-test CIFAR-10 summary")
    if bool(summary.get("test_truncated")):
        raise ValueError("upgrade visuals reject truncated evaluations")
    if int(summary.get("processed_test_images", -1)) != sample_count:
        raise ValueError("summary sample count does not match the CIFAR-10 test set")
    cross = summary.get("seeded", {}).get("cross_state", {})
    summary_fields = {
        "mse": "mse_matrix",
        "ssim": "ssim_mean_matrix",
        "lpips": "lpips_mean_matrix",
    }
    for metric, field in summary_fields.items():
        matrix = np.asarray(cross.get(field), dtype=np.float64)
        if matrix.shape != (10, 10) or not np.allclose(
            matrix, arrays[metric].mean(axis=0), rtol=1e-6, atol=1e-9
        ):
            raise ValueError(f"summary {field} is inconsistent with its per-image NPZ")


def _validate_dataset_fingerprint_value(
    actual: Mapping[str, Any], evaluation_manifest: Mapping[str, Any], label: str
) -> None:
    expected = evaluation_manifest.get("dataset")
    if not isinstance(expected, dict):
        raise ValueError(f"{label} evaluation manifest has no dataset fingerprint")
    fields = (
        "processed_images_float32_sha256",
        "processed_labels_int64_sha256",
        "processed_sample_order_sha256",
        "processed_sample_count",
    )
    mismatches = {
        field: {"expected": expected.get(field), "actual": actual.get(field)}
        for field in fields
        if expected.get(field) != actual.get(field)
    }
    if mismatches:
        raise ValueError(
            f"{label} evaluation sample fingerprint mismatch: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def _validate_evaluation_provenance(
    *,
    label: str,
    checkpoint_path: Path,
    codebook_path: Path,
    codebook_manifest_path: Path,
    summary_path: Path,
    metrics_path: Path,
    evaluation_manifest_path: Path,
    evaluation_manifest: Mapping[str, Any],
    codebook_manifest: Mapping[str, Any],
    bank: np.ndarray,
) -> tuple[list[str], list[dict[str, Any]], str]:
    if evaluation_manifest.get("schema") != EVALUATION_SCHEMA:
        raise ValueError(
            f"{label} evaluation manifest schema must be {EVALUATION_SCHEMA!r}"
        )
    if evaluation_manifest.get("scope") != "final":
        raise ValueError(f"{label} evaluation manifest is not formal/full-test")
    outputs = evaluation_manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"{label} evaluation manifest is missing output hashes")
    for filename, path in (
        ("reconstruction_summary.json", summary_path),
        ("per_image_metrics.npz", metrics_path),
    ):
        if outputs.get(filename) != _sha256_file(path):
            raise ValueError(f"{label} evaluation artifact hash mismatch: {filename}")
    inputs = evaluation_manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"{label} evaluation manifest is missing input hashes")
    if inputs.get("seeded_checkpoint", {}).get("sha256") != _sha256_file(
        checkpoint_path
    ):
        raise ValueError(f"{label} checkpoint hash differs from formal evaluation")

    labels = _state_labels(dict(codebook_manifest))
    records = state_records_from_manifest(dict(codebook_manifest))
    if labels != EXPECTED_STATE_IDS:
        raise ValueError(f"{label} codebook uses an unexpected ten-state order")
    validate_state_order(evaluation_manifest.get("state_order"))
    bank_hash = _array_sha256(bank)
    if codebook_manifest.get("bank_array_sha256") != bank_hash:
        raise ValueError(f"{label} codebook manifest array hash mismatch")
    codebook_input = inputs.get("codebook", {})
    if codebook_input.get("array_sha256") != bank_hash:
        raise ValueError(f"{label} evaluation codebook array hash mismatch")
    if codebook_input.get("file_sha256") != _sha256_file(codebook_path):
        raise ValueError(f"{label} evaluation codebook file hash mismatch")
    if codebook_input.get("manifest_sha256") != _sha256_file(codebook_manifest_path):
        raise ValueError(f"{label} evaluation codebook manifest hash mismatch")
    if not evaluation_manifest_path.is_file():
        raise FileNotFoundError(evaluation_manifest_path)
    return labels, records, bank_hash


@torch.inference_mode()
def run_upgrade_inference(
    baseline_model: Any,
    candidate_model: Any,
    images: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run both models once per sender and reuse each model's transmitted indices."""

    kwargs = {
        "send_state": FIXED_STATE_IDS["sender_state"],
        "cross_pump_receiver_state": FIXED_STATE_IDS[
            "wrong_cross_pump_same_power"
        ],
        "same_pump_receiver_state": FIXED_STATE_IDS[
            "wrong_same_pump_other_power"
        ],
        "representative_receiver_state": FIXED_STATE_IDS[
            "representative_wrong_state"
        ],
        "device": device,
    }
    baseline, baseline_provenance = run_controlled_inference(
        baseline_model, images, **kwargs
    )
    candidate, candidate_provenance = run_controlled_inference(
        candidate_model, images, **kwargs
    )
    if not np.array_equal(baseline["original"], candidate["original"]):
        raise RuntimeError("baseline and candidate inference did not use identical images")
    arrays = {
        "original": baseline["original"],
        "baseline_correct": baseline["correct"],
        "candidate_correct": candidate["correct"],
        "candidate_cross_pump": candidate["cross_pump"],
        "candidate_same_pump": candidate["same_pump"],
        "candidate_representative": candidate["representative"],
    }
    provenance = {
        "baseline": baseline_provenance,
        "candidate": candidate_provenance,
        "fixed_state_indices": dict(FIXED_STATE_IDS),
    }
    return arrays, provenance


def _metric_records(
    baseline: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    sample_ids: Sequence[int],
) -> dict[str, dict[str, np.ndarray]]:
    records: dict[str, dict[str, np.ndarray]] = {}
    for display_key in DISPLAY_KEYS:
        source = baseline if display_key == "baseline_correct" else candidate
        sender, receiver = PAIR_BY_DISPLAY_KEY[display_key]
        mse = source["mse"][sample_ids, sender, receiver]
        records[display_key] = {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "ssim": source["ssim"][sample_ids, sender, receiver],
            "lpips": source["lpips"][sample_ids, sender, receiver],
            "psnr_db": np.where(mse > 0.0, 10.0 * np.log10(1.0 / mse), np.inf),
        }
    return records


def _metric_label(values: Mapping[str, np.ndarray], index: int) -> str:
    psnr = float(values["psnr_db"][index])
    psnr_text = f"{psnr:.1f}" if np.isfinite(psnr) else "∞"
    return (
        f"MSE {values['mse'][index]:.4f} · RMSE {values['rmse'][index]:.3f}\n"
        f"SSIM {values['ssim'][index]:.3f} · LPIPS {values['lpips'][index]:.3f}\n"
        f"PSNR {psnr_text} dB (supp.)"
    )


def plot_quantile_overview(
    arrays: Mapping[str, np.ndarray],
    metrics: Mapping[str, Mapping[str, np.ndarray]],
    *,
    quantile_name: str,
    sample_ids: Sequence[int],
    class_ids: Sequence[int],
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    if quantile_name not in {name for name, _ in QUANTILES}:
        raise ValueError("quantile_name must be q25, q50, or q75")
    if list(class_ids) != list(range(10)) or len(sample_ids) != 10:
        raise ValueError("quantile overview requires CIFAR classes in exact order 0..9")
    expected_shape = (10, 32, 32, 3)
    for key in ("original", *DISPLAY_KEYS):
        if np.asarray(arrays[key]).shape != expected_shape:
            raise ValueError(f"{key} must have display shape {expected_shape}")

    figure, axes = plt.subplots(10, 11, figsize=(25.5, 22.0), squeeze=False)
    column_titles = ["Original"]
    for display_key in DISPLAY_KEYS:
        column_titles.extend([DISPLAY_TITLES[display_key], "Absolute error"])
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=9.2, fontweight="bold")

    original = np.asarray(arrays["original"], dtype=np.float32)
    for row, (sample_id, class_id) in enumerate(zip(sample_ids, class_ids)):
        axis = axes[row, 0]
        axis.imshow(original[row], vmin=0.0, vmax=1.0, interpolation="nearest")
        axis.set_ylabel(
            f"{class_id}: {CIFAR10_CLASS_NAMES_EN[class_id]}\nID {sample_id}",
            fontsize=8.0,
            fontweight="bold",
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for display_index, display_key in enumerate(DISPLAY_KEYS):
            reconstruction = np.asarray(arrays[display_key], dtype=np.float32)
            image_axis = axes[row, 1 + 2 * display_index]
            error_axis = axes[row, 2 + 2 * display_index]
            image_axis.imshow(
                np.clip(reconstruction[row], 0.0, 1.0),
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            error_axis.imshow(
                rgb_absolute_error(original[row : row + 1], reconstruction[row : row + 1])[0],
                cmap="inferno",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            image_axis.set_xlabel(_metric_label(metrics[display_key], row), fontsize=6.2)
            for current in (image_axis, error_axis):
                current.set_xticks([])
                current.set_yticks([])
        for current in axes[row]:
            for spine in current.spines.values():
                spine.set_visible(False)

    title = (
        f"Frozen CIFAR-10 {quantile_name.upper()} samples: visual-quality upgrade "
        "and controlled state mismatch"
    )
    figure.suptitle(title, fontsize=16, fontweight="bold", y=0.998)
    figure.text(
        0.5,
        0.002,
        "All IDs and ordering come from the pre-upgrade registry. Error maps share "
        "the fixed [0,1] inferno scale; no image-wise rescaling.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.065, right=0.995, bottom=0.035, top=0.972, wspace=0.12, hspace=0.54
    )
    result = _save_figure(
        figure,
        output_dir=output_dir,
        stem=f"upgrade_01_{quantile_name}_frozen_samples",
        title=title,
        dpi=dpi,
    )
    result["sample_ids"] = [int(value) for value in sample_ids]
    result["class_ids"] = [int(value) for value in class_ids]
    result["metric_annotation"] = "MSE, RMSE, SSIM, LPIPS; PSNR supplementary"
    return result


def plot_candidate_mismatch_heatmap(
    mse_matrix: np.ndarray,
    *,
    state_labels: Sequence[str],
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    mse = np.asarray(mse_matrix, dtype=np.float64)
    if mse.shape != (10, 10) or not np.isfinite(mse).all() or np.any(mse < 0.0):
        raise ValueError("candidate MSE matrix must be finite non-negative shape (10,10)")
    if list(state_labels) != EXPECTED_STATE_IDS:
        raise ValueError("candidate heatmap requires the registered ten-state order")
    labels = [compact_state_label(label) for label in state_labels]
    rmse = np.sqrt(mse)
    figure, axes = plt.subplots(1, 2, figsize=(14.2, 6.3))
    for axis, values, title, colorbar_label, cmap in (
        (axes[0], mse, "Candidate mismatch MSE", "MSE in [0,1] RGB", "magma"),
        (axes[1], rmse, "Candidate mismatch RMSE", "RMSE in [0,1] RGB", "viridis"),
    ):
        image = axis.imshow(values, cmap=cmap, aspect="equal")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04).set_label(
            colorbar_label
        )
        axis.set_xticks(range(10), labels=labels, rotation=45, ha="right")
        axis.set_yticks(range(10), labels=labels)
        axis.set_xlabel("Receiver state")
        axis.set_ylabel("Sender state")
        axis.set_title(title, fontweight="bold")
        threshold = float(np.median(values))
        for sender in range(10):
            for receiver in range(10):
                value = float(values[sender, receiver])
                axis.text(
                    receiver,
                    sender,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if value <= threshold else "black",
                    fontweight="bold" if sender == receiver else "normal",
                )
    title = "Candidate 10×10 receiver-state mismatch: MSE and RMSE"
    figure.suptitle(title, fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return _save_figure(
        figure,
        output_dir=output_dir,
        stem="upgrade_02_candidate_mismatch_mse_rmse",
        title=title,
        dpi=dpi,
    )


def plot_paired_correct_distribution(
    baseline_cross_mse: np.ndarray,
    candidate_cross_mse: np.ndarray,
    *,
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    baseline = np.diagonal(
        np.asarray(baseline_cross_mse, dtype=np.float64), axis1=1, axis2=2
    ).mean(axis=1)
    candidate = np.diagonal(
        np.asarray(candidate_cross_mse, dtype=np.float64), axis1=1, axis2=2
    ).mean(axis=1)
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("paired correct-state arrays must have equal [N,10,10] shape")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("paired correct-state MSE values must be finite")
    delta = candidate - baseline
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.7))
    axes[0].scatter(baseline, candidate, s=7, alpha=0.22, color="#0072B2")
    limit = float(max(baseline.max(), candidate.max()))
    axes[0].plot([0.0, limit], [0.0, limit], "--", color="#666666", linewidth=1)
    axes[0].set_xlabel("Pre-upgrade diagonal-mean MSE")
    axes[0].set_ylabel("Candidate diagonal-mean MSE")
    axes[0].set_title("Paired test images", fontweight="bold")

    for values, label, color in (
        (baseline, "Pre-upgrade", "#D55E00"),
        (candidate, "Candidate", "#0072B2"),
    ):
        ordered = np.sort(values)
        probability = np.arange(1, len(ordered) + 1) / len(ordered)
        axes[1].plot(ordered, probability, label=label, color=color, linewidth=1.7)
    axes[1].set_xlabel("Diagonal-mean MSE")
    axes[1].set_ylabel("Empirical cumulative probability")
    axes[1].set_title("Correct-state distribution", fontweight="bold")
    axes[1].legend(frameon=False)

    axes[2].hist(delta, bins=60, color="#009E73", alpha=0.82)
    axes[2].axvline(0.0, linestyle="--", color="#666666", linewidth=1)
    axes[2].axvline(float(np.median(delta)), color="#C83E4D", linewidth=1.5)
    axes[2].set_xlabel("Candidate − pre-upgrade MSE")
    axes[2].set_ylabel("Test-image count")
    axes[2].set_title("Paired change", fontweight="bold")

    title = "Pre-upgrade versus candidate correct-state reconstruction distribution"
    figure.suptitle(title, fontsize=14.5, fontweight="bold")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    result = _save_figure(
        figure,
        output_dir=output_dir,
        stem="upgrade_03_paired_correct_distribution",
        title=title,
        dpi=dpi,
    )
    result["paired_statistics"] = {
        "sample_count": int(baseline.size),
        "baseline_mean": float(baseline.mean()),
        "candidate_mean": float(candidate.mean()),
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "candidate_lower_fraction": float(np.mean(candidate < baseline)),
    }
    return result


def _validate_inference_mse(
    arrays: Mapping[str, np.ndarray],
    baseline_metrics: Mapping[str, np.ndarray],
    candidate_metrics: Mapping[str, np.ndarray],
    sample_ids: Sequence[int],
) -> dict[str, dict[str, float | int]]:
    original = np.asarray(arrays["original"], dtype=np.float64)
    diagnostics: dict[str, dict[str, float | int]] = {}
    for display_key in DISPLAY_KEYS:
        reconstruction = np.asarray(arrays[display_key], dtype=np.float64)
        actual = np.mean((reconstruction - original) ** 2, axis=(1, 2, 3))
        source = baseline_metrics if display_key == "baseline_correct" else candidate_metrics
        sender, receiver = PAIR_BY_DISPLAY_KEY[display_key]
        expected = source["mse"][sample_ids, sender, receiver]
        absolute_delta = np.abs(actual - expected)
        diagnostics[display_key] = {
            "sample_count": int(actual.size),
            "maximum_absolute_mse_difference": float(absolute_delta.max()),
            "mean_absolute_mse_difference": float(absolute_delta.mean()),
        }
        # The formal archive uses full evaluation batches, whereas the PPT
        # registry uses a 30-image batch. cuDNN can select different convolution
        # algorithms across those shapes; this tolerance remains orders of
        # magnitude below a meaningful candidate-model change.
        if not np.allclose(
            actual,
            expected,
            rtol=INFERENCE_MSE_RTOL,
            atol=INFERENCE_MSE_ATOL,
        ):
            worst_position = int(np.argmax(absolute_delta))
            raise ValueError(
                f"{display_key} checkpoint inference does not match per-image metrics: "
                f"sample_id={int(sample_ids[worst_position])}, "
                f"actual={float(actual[worst_position]):.10g}, "
                f"expected={float(expected[worst_position]):.10g}, "
                f"max_abs_delta={float(absolute_delta[worst_position]):.10g}"
            )
    return diagnostics


def generate_visual_quality_upgrade(
    *,
    baseline_seeded_checkpoint: str | Path,
    candidate_seeded_checkpoint: str | Path,
    baseline_codebook: str | Path,
    baseline_codebook_manifest: str | Path,
    candidate_codebook: str | Path,
    candidate_codebook_manifest: str | Path,
    registry_path: str | Path,
    baseline_summary: str | Path,
    baseline_metrics: str | Path,
    candidate_summary: str | Path,
    candidate_metrics: str | Path,
    output_dir: str | Path,
    device_name: str = "cuda:0",
    dpi: int = 300,
) -> dict[str, Any]:
    paths = {
        "baseline_checkpoint": _require_file(baseline_seeded_checkpoint),
        "candidate_checkpoint": _require_file(candidate_seeded_checkpoint),
        "baseline_codebook": _require_file(baseline_codebook),
        "baseline_codebook_manifest": _require_file(baseline_codebook_manifest),
        "candidate_codebook": _require_file(candidate_codebook),
        "candidate_codebook_manifest": _require_file(candidate_codebook_manifest),
        "registry": _require_file(registry_path),
        "baseline_summary": _require_file(baseline_summary),
        "baseline_metrics": _require_file(baseline_metrics),
        "candidate_summary": _require_file(candidate_summary),
        "candidate_metrics": _require_file(candidate_metrics),
    }
    paths["baseline_evaluation_manifest"] = _evaluation_manifest_for(
        paths["baseline_summary"], paths["baseline_metrics"]
    )
    paths["candidate_evaluation_manifest"] = _evaluation_manifest_for(
        paths["candidate_summary"], paths["candidate_metrics"]
    )
    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"output directory already exists: {output_path}")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    registry = _read_json(paths["registry"])
    _validate_registry_states(registry)
    baseline_eval_manifest = _read_json(paths["baseline_evaluation_manifest"])
    candidate_eval_manifest = _read_json(paths["candidate_evaluation_manifest"])
    recorded_baseline_metrics = registry.get("inputs", {}).get(
        "per_image_metrics", {}
    ).get("sha256")
    if recorded_baseline_metrics != _sha256_file(paths["baseline_metrics"]):
        raise ValueError("registry was not built from the supplied baseline metrics NPZ")
    recorded_baseline_manifest = registry.get("inputs", {}).get(
        "evaluation_manifest", {}
    ).get("sha256")
    if recorded_baseline_manifest != _sha256_file(
        paths["baseline_evaluation_manifest"]
    ):
        raise ValueError("registry baseline evaluation-manifest hash mismatch")

    baseline_manifest = _read_json(paths["baseline_codebook_manifest"])
    candidate_manifest = _read_json(paths["candidate_codebook_manifest"])
    baseline_bank = load_codebook_bank(paths["baseline_codebook"])
    candidate_bank = load_codebook_bank(paths["candidate_codebook"])
    baseline_checkpoint = torch.load(paths["baseline_checkpoint"], map_location="cpu")
    candidate_checkpoint = torch.load(paths["candidate_checkpoint"], map_location="cpu")
    baseline_labels, baseline_records, baseline_bank_hash = (
        _validate_evaluation_provenance(
            label="baseline",
            checkpoint_path=paths["baseline_checkpoint"],
            codebook_path=paths["baseline_codebook"],
            codebook_manifest_path=paths["baseline_codebook_manifest"],
            summary_path=paths["baseline_summary"],
            metrics_path=paths["baseline_metrics"],
            evaluation_manifest_path=paths["baseline_evaluation_manifest"],
            evaluation_manifest=baseline_eval_manifest,
            codebook_manifest=baseline_manifest,
            bank=baseline_bank,
        )
    )
    candidate_labels, candidate_records, candidate_bank_hash = (
        _validate_evaluation_provenance(
            label="candidate",
            checkpoint_path=paths["candidate_checkpoint"],
            codebook_path=paths["candidate_codebook"],
            codebook_manifest_path=paths["candidate_codebook_manifest"],
            summary_path=paths["candidate_summary"],
            metrics_path=paths["candidate_metrics"],
            evaluation_manifest_path=paths["candidate_evaluation_manifest"],
            evaluation_manifest=candidate_eval_manifest,
            codebook_manifest=candidate_manifest,
            bank=candidate_bank,
        )
    )
    if baseline_labels != candidate_labels or baseline_records != candidate_records:
        raise ValueError("baseline and candidate physical state mappings differ")
    validate_controlled_state_mapping(
        baseline_records,
        send_state=2,
        cross_pump_receiver_state=7,
        same_pump_receiver_state=4,
    )
    for label, checkpoint, bank_hash in (
        ("baseline", baseline_checkpoint, baseline_bank_hash),
        ("candidate", candidate_checkpoint, candidate_bank_hash),
    ):
        if checkpoint.get("base_bank_array_sha256") != bank_hash:
            raise ValueError(f"{label} checkpoint and codebook array hashes differ")
        config = checkpoint.get("config", {})
        if str(config.get("model", {}).get("quantizer_type")) != "seeded_residual":
            raise ValueError(f"{label} checkpoint is not a seeded residual model")
        name = str(config.get("dataset", {}).get("name", "")).lower().replace("-", "")
        if name != "cifar10":
            raise ValueError(f"{label} checkpoint is not configured for CIFAR-10")

    dataset_config = baseline_checkpoint["config"]["dataset"]
    dataset = build_dataset(
        str(dataset_config["name"]),
        str(dataset_config["root"]),
        train=False,
        download=False,
    )
    if len(dataset) != 10_000:
        raise ValueError("formal CIFAR-10 upgrade visuals require all 10,000 test images")
    dataset_fingerprint = _dataset_fingerprint(dataset)
    _validate_dataset_fingerprint_value(
        dataset_fingerprint, baseline_eval_manifest, "baseline"
    )
    _validate_dataset_fingerprint_value(
        dataset_fingerprint, candidate_eval_manifest, "candidate"
    )
    registry_labels_hash = registry.get("inputs", {}).get(
        "cifar10_test_labels", {}
    ).get("canonical_int64_sha256")
    targets = _targets_from_dataset(dataset)
    if registry_labels_hash != _int64_sha256(targets):
        raise ValueError("registry CIFAR-10 label/sample-order hash mismatch")

    baseline_arrays = _load_cross_arrays(paths["baseline_metrics"], sample_count=len(dataset))
    candidate_arrays = _load_cross_arrays(paths["candidate_metrics"], sample_count=len(dataset))
    baseline_summary_payload = _read_json(paths["baseline_summary"])
    candidate_summary_payload = _read_json(paths["candidate_summary"])
    _validate_summary(
        baseline_summary_payload, baseline_arrays, sample_count=len(dataset)
    )
    _validate_summary(
        candidate_summary_payload, candidate_arrays, sample_count=len(dataset)
    )
    sample_ids_by_quantile = resolve_registry_sample_ids(
        registry, targets, baseline_arrays["mse"]
    )
    all_sample_ids = [
        sample_id
        for quantile_name, _ in QUANTILES
        for sample_id in sample_ids_by_quantile[quantile_name]
    ]
    samples = [dataset[sample_id] for sample_id in all_sample_ids]
    images = torch.stack([sample[0] for sample in samples])
    class_ids = [int(sample[1]) for sample in samples]
    if class_ids != list(range(10)) * len(QUANTILES):
        raise ValueError("registry sample ordering is not q25/q50/q75 × class 0..9")

    baseline_model = _load_model(
        baseline_checkpoint, expected_quantizer_type="seeded_residual", bank=baseline_bank
    ).to(device).eval()
    candidate_model = _load_model(
        candidate_checkpoint, expected_quantizer_type="seeded_residual", bank=candidate_bank
    ).to(device).eval()
    inference_arrays, inference_provenance = run_upgrade_inference(
        baseline_model, candidate_model, images, device=device
    )
    inference_mse_diagnostics = _validate_inference_mse(
        inference_arrays,
        baseline_arrays,
        candidate_arrays,
        all_sample_ids,
    )

    output_path.mkdir(parents=True, exist_ok=False)
    figures: dict[str, Any] = {}
    with plt.rc_context(RC_PARAMS):
        for quantile_index, (quantile_name, _) in enumerate(QUANTILES):
            start = quantile_index * 10
            stop = start + 10
            panel_arrays = {
                key: value[start:stop] for key, value in inference_arrays.items()
            }
            ids = all_sample_ids[start:stop]
            panel_metrics = _metric_records(baseline_arrays, candidate_arrays, ids)
            figures[f"{quantile_name}_overview"] = plot_quantile_overview(
                panel_arrays,
                panel_metrics,
                quantile_name=quantile_name,
                sample_ids=ids,
                class_ids=list(range(10)),
                output_dir=output_path,
                dpi=dpi,
            )
        candidate_matrix = candidate_arrays["mse"].mean(axis=0)
        figures["candidate_mismatch_heatmap"] = plot_candidate_mismatch_heatmap(
            candidate_matrix,
            state_labels=candidate_labels,
            output_dir=output_path,
            dpi=dpi,
        )
        figures["paired_correct_distribution"] = plot_paired_correct_distribution(
            baseline_arrays["mse"],
            candidate_arrays["mse"],
            output_dir=output_path,
            dpi=dpi,
        )

    output_manifest: dict[str, Any] = {
        "schema": UPGRADE_VISUAL_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in paths.items()
        },
        "registry": {
            "schema": registry["schema"],
            "sample_ids_by_quantile": sample_ids_by_quantile,
            "flattened_inference_order": all_sample_ids,
            "manual_or_candidate_based_reselection": False,
        },
        "dataset_fingerprint": dataset_fingerprint,
        "state_order": [
            {"index": index, "state_id": label}
            for index, label in enumerate(candidate_labels)
        ],
        "fixed_state_indices": dict(FIXED_STATE_IDS),
        "codebooks": {
            "baseline": {
                "shape": list(baseline_bank.shape),
                "array_sha256": baseline_bank_hash,
            },
            "candidate": {
                "shape": list(candidate_bank.shape),
                "array_sha256": candidate_bank_hash,
            },
        },
        "checkpoint_metric_binding": {
            "method": (
                "selected images were inferred from each supplied checkpoint/codebook; "
                "raw [0,1] RGB MSE was matched to the corresponding NPZ entries"
            ),
            "validated_display_pairs": {
                key: list(pair) for key, pair in PAIR_BY_DISPLAY_KEY.items()
            },
            "inference_validation": {
                "device": str(device),
                "rtol": INFERENCE_MSE_RTOL,
                "atol": INFERENCE_MSE_ATOL,
                "comparison": "numpy.allclose(actual_inference_mse, recorded_npz_mse)",
                "absolute_mse_difference_by_display_condition": (
                    inference_mse_diagnostics
                ),
            },
            "transport_index_reuse": inference_provenance,
        },
        "rendering": {
            "formats": ["png", "pdf"],
            "dpi": dpi,
            "rgb_range": [0.0, 1.0],
            "absolute_error_range": [0.0, 1.0],
            "absolute_error_colormap": "inferno",
            "metrics": ["MSE", "RMSE", "SSIM", "LPIPS", "PSNR (supplementary)"],
        },
        "figures": figures,
    }
    output_manifest["manifest_payload_sha256"] = _json_sha256(output_manifest)
    manifest_path = output_path / "visual_quality_upgrade_manifest.json"
    manifest_path.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_manifest


def main() -> None:
    args = build_parser().parse_args()
    result = generate_visual_quality_upgrade(
        baseline_seeded_checkpoint=args.baseline_seeded_checkpoint,
        candidate_seeded_checkpoint=args.candidate_seeded_checkpoint,
        baseline_codebook=args.baseline_codebook,
        baseline_codebook_manifest=args.baseline_codebook_manifest,
        candidate_codebook=args.candidate_codebook,
        candidate_codebook_manifest=args.candidate_codebook_manifest,
        registry_path=args.registry,
        baseline_summary=args.baseline_summary,
        baseline_metrics=args.baseline_metrics,
        candidate_summary=args.candidate_summary,
        candidate_metrics=args.candidate_metrics,
        output_dir=args.output_dir,
        device_name=args.device,
        dpi=args.dpi,
    )
    print(json.dumps({"figures": sorted(result["figures"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
