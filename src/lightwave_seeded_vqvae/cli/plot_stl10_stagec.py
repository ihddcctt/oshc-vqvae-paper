from __future__ import annotations

import argparse
import csv
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
from torchvision import transforms
from torchvision.datasets import STL10

from ..data import resolve_dataset_root
from ..seed.codebook import load_codebook_bank
from .build_stl10_sample_registry import (
    DEFAULT_PHYSICAL_ROLES,
    EXPECTED_CLASSES,
    EXPECTED_TEST_SAMPLES,
    QUANTILES,
    REGISTRY_SCHEMA,
    STL10_CLASS_NAMES,
    dataset_labels,
    resolve_controlled_states,
    select_quantile_samples,
    state_records_from_manifest,
)
from .evaluate_final import (
    EVALUATION_SCHEMA,
    _array_sha256,
    _compatibility_report,
    _load_model,
)
from .plot_cifar_natural_examples import (
    _dataset_fingerprint,
    _git_metadata,
    nchw_to_rgb,
    rgb_absolute_error,
    run_controlled_inference,
)
from .plot_final_evaluation import compact_state_label


VISUAL_SCHEMA = "lightwave-seeded-vqvae-stl10-stagec-visuals-v1"
BRANCHES = (
    "software",
    "correct",
    "cross_pump_same_power",
    "same_pump_wrong_power",
    "representative_median_wrong",
)
SEEDED_BRANCHES = BRANCHES[1:]
BRANCH_TITLES = {
    "software": "Software VQ\ncorrect reconstruction",
    "correct": "B_s + R correct",
    "cross_pump_same_power": "Wrong wavelength\nsame energy",
    "same_pump_wrong_power": "Same wavelength\nwrong energy",
    "representative_median_wrong": "Representative\nmedian wrong state",
}
INFERENCE_MSE_RTOL = 3e-3
INFERENCE_MSE_ATOL = 3e-6
EXPECTED_STAGEC_BANK_SHAPE = (10, 128, 64)
EXPECTED_IMAGE_SHAPE_CHW = (3, 96, 96)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _normalized_dataset_name(value: object) -> str:
    return str(value).strip().lower().replace("-", "").replace("_", "")


def _read_json(path: Path) -> dict[str, Any]:
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


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_stagec_bank_shape(bank: np.ndarray) -> None:
    """Reject a self-consistent but method-incompatible frozen base bank."""

    shape = tuple(int(value) for value in np.asarray(bank).shape)
    if shape != EXPECTED_STAGEC_BANK_SHAPE:
        raise ValueError(
            "STL-10 Stage C requires frozen B_s shape "
            f"{EXPECTED_STAGEC_BANK_SHAPE}, received {shape}"
        )


def validate_registry_dataset_contract(
    registry: Mapping[str, Any], dataset: Any
) -> dict[str, Any]:
    """Cross-check every stored dataset identity field against official STL-10."""

    stored = registry.get("dataset")
    if not isinstance(stored, dict):
        raise ValueError("registry must contain dataset metadata")
    if _normalized_dataset_name(stored.get("name")) != "stl10":
        raise ValueError("registry dataset name must be STL-10")
    if str(stored.get("split", "")).strip().lower() != "test":
        raise ValueError("registry dataset split must be the official test split")

    sample_count = len(dataset)
    if sample_count != EXPECTED_TEST_SAMPLES or int(
        stored.get("sample_count", -1)
    ) != sample_count:
        raise ValueError("registry and current STL-10 dataset must contain 8,000 samples")

    raw_classes = getattr(dataset, "classes", None)
    if raw_classes is None:
        raise ValueError("current STL-10 dataset does not expose official class names")
    class_names = [str(value) for value in raw_classes]
    if class_names != list(STL10_CLASS_NAMES):
        raise ValueError("current STL-10 class names/order differ from the official contract")
    if stored.get("class_names") != class_names:
        raise ValueError("registry class_names differ from the current official dataset")

    first_sample = dataset[0]
    if not isinstance(first_sample, (tuple, list)) or not first_sample:
        raise ValueError("current STL-10 dataset must return an image/label sample")
    first_image = first_sample[0]
    image_shape = tuple(int(value) for value in np.asarray(first_image).shape)
    if image_shape != EXPECTED_IMAGE_SHAPE_CHW:
        raise ValueError(
            "current STL-10 test images must have CHW shape "
            f"{EXPECTED_IMAGE_SHAPE_CHW}, received {image_shape}"
        )
    if stored.get("image_shape_chw") != list(image_shape):
        raise ValueError("registry image_shape_chw differs from the current dataset")

    return {
        "name": "stl10",
        "split": "test",
        "sample_count": sample_count,
        "image_shape_chw": list(image_shape),
        "class_names": class_names,
        "verified_against_current_dataset": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the formal, fixed-sample STL-10 Stage C PNG/PDF package "
            "without post-processing or test-set visual selection."
        )
    )
    parser.add_argument("--seeded-checkpoint", required=True, type=Path)
    parser.add_argument("--software-checkpoint", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--codebook-manifest", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--dataset-root", type=Path,
        help="Read a relocated test split here; preserve the registry and its fingerprint checks.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dpi", type=_positive_int, default=300)
    parser.add_argument(
        "--preserve-registered-samples", action="store_true",
        help="Keep original-run quantile sample IDs and receiver states while annotating current formal metrics.",
    )
    return parser


def _resolve_inputs(
    *,
    seeded_checkpoint: str | Path,
    software_checkpoint: str | Path,
    codebook: str | Path,
    codebook_manifest: str | Path,
    registry: str | Path,
    evaluation_dir: str | Path,
) -> dict[str, Path]:
    evaluation = Path(evaluation_dir).expanduser().resolve()
    paths = {
        "seeded_checkpoint": Path(seeded_checkpoint).expanduser().resolve(),
        "software_checkpoint": Path(software_checkpoint).expanduser().resolve(),
        "codebook": Path(codebook).expanduser().resolve(),
        "codebook_manifest": Path(codebook_manifest).expanduser().resolve(),
        "registry": Path(registry).expanduser().resolve(),
        "evaluation_manifest": evaluation / "evaluation_manifest.json",
        "reconstruction_summary": evaluation / "reconstruction_summary.json",
        "per_image_metrics": evaluation / "per_image_metrics.npz",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def _load_metric_arrays(path: Path) -> dict[str, np.ndarray]:
    keys = {
        "software_mse": (EXPECTED_TEST_SAMPLES,),
        "software_ssim": (EXPECTED_TEST_SAMPLES,),
        "software_lpips": (EXPECTED_TEST_SAMPLES,),
        "seeded_cross_mse": (EXPECTED_TEST_SAMPLES, 10, 10),
        "seeded_cross_ssim": (EXPECTED_TEST_SAMPLES, 10, 10),
        "seeded_cross_lpips": (EXPECTED_TEST_SAMPLES, 10, 10),
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(keys).difference(archive.files))
        if missing:
            raise ValueError(f"formal per-image metrics are missing keys: {missing}")
        arrays = {
            key: np.asarray(archive[key], dtype=np.float64) for key in keys
        }
    for key, expected_shape in keys.items():
        value = arrays[key]
        if value.shape != expected_shape or not np.isfinite(value).all():
            raise ValueError(f"{key} must be finite with shape {expected_shape}")
    if np.any(arrays["software_mse"] < 0.0) or np.any(
        arrays["seeded_cross_mse"] < 0.0
    ):
        raise ValueError("MSE arrays must be non-negative")
    if np.any(arrays["software_lpips"] < 0.0) or np.any(
        arrays["seeded_cross_lpips"] < 0.0
    ):
        raise ValueError("LPIPS arrays must be non-negative")
    return arrays


def _validate_formal_artifacts(
    *,
    paths: Mapping[str, Path],
    evaluation_manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    codebook_manifest: Mapping[str, Any],
    registry: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    bank: np.ndarray,
    preserve_registered_samples: bool = False,
) -> list[dict[str, Any]]:
    validate_stagec_bank_shape(bank)
    if evaluation_manifest.get("schema") != EVALUATION_SCHEMA:
        raise ValueError("unsupported formal evaluation schema")
    if evaluation_manifest.get("scope") != "final":
        raise ValueError("STL-10 figures require a formal full-test evaluation")
    dataset = evaluation_manifest.get("dataset")
    if not isinstance(dataset, dict) or _normalized_dataset_name(
        dataset.get("name")
    ) != "stl10":
        raise ValueError("formal evaluation dataset must be STL-10")
    if bool(dataset.get("test_truncated", True)):
        raise ValueError("truncated evaluations cannot generate formal figures")
    if int(dataset.get("processed_test_samples", -1)) != EXPECTED_TEST_SAMPLES:
        raise ValueError("formal evaluation must contain all 8,000 STL-10 test images")

    if _normalized_dataset_name(summary.get("dataset")) != "stl10":
        raise ValueError("reconstruction summary dataset must be STL-10")
    if summary.get("scope") != "final" or bool(summary.get("test_truncated", True)):
        raise ValueError("reconstruction summary must cover the full test set")
    outputs = evaluation_manifest.get("outputs", {})
    for filename, key in (
        ("reconstruction_summary.json", "reconstruction_summary"),
        ("per_image_metrics.npz", "per_image_metrics"),
    ):
        if outputs.get(filename) != _sha256_file(paths[key]):
            raise ValueError(f"formal output hash mismatch: {filename}")

    inputs = evaluation_manifest.get("inputs", {})
    for key in ("seeded_checkpoint", "software_checkpoint"):
        if inputs.get(key, {}).get("sha256") != _sha256_file(paths[key]):
            raise ValueError(f"formal evaluation checkpoint hash mismatch: {key}")
    codebook_input = inputs.get("codebook", {})
    bank_hash = _array_sha256(bank)
    if codebook_manifest.get("bank_array_sha256") != bank_hash:
        raise ValueError("codebook manifest array hash mismatch")
    if codebook_input.get("array_sha256") != bank_hash:
        raise ValueError("formal evaluation codebook array hash mismatch")
    if codebook_input.get("file_sha256") != _sha256_file(paths["codebook"]):
        raise ValueError("formal evaluation codebook file hash mismatch")
    if codebook_input.get("manifest_sha256") != _sha256_file(
        paths["codebook_manifest"]
    ):
        raise ValueError("formal evaluation codebook manifest hash mismatch")

    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("unsupported STL-10 fixed-sample registry schema")
    validate_registry_input_hashes(
        registry, paths, preserve_registered_samples=preserve_registered_samples
    )

    cross = summary.get("seeded", {}).get("cross_state", {})
    for array_key, summary_key in (
        ("seeded_cross_mse", "mse_matrix"),
        ("seeded_cross_ssim", "ssim_mean_matrix"),
        ("seeded_cross_lpips", "lpips_mean_matrix"),
    ):
        matrix = np.asarray(cross.get(summary_key), dtype=np.float64)
        if matrix.shape != (10, 10) or not np.allclose(
            matrix, arrays[array_key].mean(axis=0), rtol=1e-6, atol=1e-9
        ):
            raise ValueError(f"summary {summary_key} differs from per-image metrics")
    software = summary.get("software", {}).get("overall", {})
    if not np.isclose(
        float(software.get("mse", np.nan)),
        float(arrays["software_mse"].mean()),
        rtol=1e-6,
        atol=1e-10,
    ):
        raise ValueError("summary software MSE differs from per-image metrics")

    records = state_records_from_manifest(dict(codebook_manifest))
    expected_order = [
        {"index": item["index"], "state_id": item["state_id"]} for item in records
    ]
    if evaluation_manifest.get("state_order") != expected_order:
        raise ValueError("formal evaluation state order differs from codebook manifest")
    return records


def validate_registry_input_hashes(
    registry: Mapping[str, Any], paths: Mapping[str, Path], *,
    preserve_registered_samples: bool = False,
) -> dict[str, Any]:
    """Only the explicit preservation mode may pair old selection with new metrics."""
    provenance = {}
    for key in ("per_image_metrics", "evaluation_manifest", "codebook_manifest"):
        original = registry.get("inputs", {}).get(key, {}).get("sha256")
        if not isinstance(original, str) or len(original) != 64 or any(
            character not in "0123456789abcdef" for character in original
        ):
            raise ValueError(f"registry input hash is invalid: {key}")
        current = _sha256_file(paths[key])
        changed = original != current
        if changed and not (
            preserve_registered_samples and key in {"per_image_metrics", "evaluation_manifest"}
        ):
            raise ValueError(f"registry input hash mismatch: {key}")
        provenance[key] = {
            "registered_sha256": original,
            "current_evaluation_sha256": current,
            "identical": not changed,
        }
    return provenance


def resolve_preserved_registry_ids(
    registry: Mapping[str, Any], labels: np.ndarray, class_names: Sequence[str],
) -> dict[str, list[int]]:
    """Validate archived IDs against actual labels without recomputing ranks."""
    classes = registry.get("classes")
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(class_names) != 10 or not isinstance(classes, list) or len(classes) != 10:
        raise ValueError("preserved registry must contain ten ordered classes and 1D labels")
    dataset_name = _normalized_dataset_name(registry.get("dataset", {}).get("name"))
    recorded_labels_hash = registry.get("inputs", {}).get(
        f"{dataset_name}_test_labels", {}
    ).get("canonical_int64_sha256")
    actual_labels_hash = hashlib.sha256(
        np.ascontiguousarray(labels, dtype="<i8").tobytes()
    ).hexdigest()
    if recorded_labels_hash != actual_labels_hash:
        raise ValueError("preserved registry labels hash differs from the current dataset")
    result = {name: [] for name, _ in QUANTILES}
    for class_id, stored in enumerate(classes):
        if stored.get("class_id") != class_id or stored.get("class_name") != str(class_names[class_id]):
            raise ValueError("preserved registry class name/order differs from the dataset")
        for name, quantile in QUANTILES:
            record = stored.get("quantiles", {}).get(name, {})
            sample_id = record.get("sample_id")
            if isinstance(sample_id, bool) or not isinstance(sample_id, int) or not 0 <= sample_id < len(labels):
                raise ValueError("preserved registry sample ID is outside the official test range")
            if int(labels[sample_id]) != class_id:
                raise ValueError("preserved registry sample ID has the wrong class label")
            if record.get("quantile") != quantile:
                raise ValueError("preserved registry quantile label is invalid")
            result[name].append(sample_id)
    return result


def resolve_gallery_states(
    codebook_manifest: Mapping[str, Any], mse_matrix: np.ndarray,
    registry: Mapping[str, Any], *, preserve_registered_samples: bool = False,
) -> dict[str, Any]:
    if not preserve_registered_samples:
        mapping = resolve_controlled_states(codebook_manifest, mse_matrix)
        if registry.get("fixed_state_indices") != mapping["fixed_state_indices"]:
            raise ValueError("registry controlled states do not reproduce from pump/power metadata")
        return mapping
    records = state_records_from_manifest(dict(codebook_manifest))
    if registry.get("state_order") != records:
        raise ValueError("preserved registry state order differs from codebook manifest")
    fixed = registry.get("fixed_state_indices", {})
    expected_roles = {*DEFAULT_PHYSICAL_ROLES, "representative_median_wrong"}
    if set(fixed) != expected_roles:
        raise ValueError("preserved registry must specify all four controlled state roles")
    states = {}
    for role, index in fixed.items():
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 10:
            raise ValueError("preserved registry state index is invalid")
        state = records[index]
        if registry.get("fixed_states", {}).get(role) != state:
            raise ValueError("preserved registry physical state metadata differs from codebook")
        if role in DEFAULT_PHYSICAL_ROLES and any(
            state[key] != value for key, value in DEFAULT_PHYSICAL_ROLES[role].items()
        ):
            raise ValueError("preserved registry wavelength/power role is incorrect")
        states[role] = dict(state)
    if len(set(fixed.values())) != 4:
        raise ValueError("preserved representative wrong state must differ from the three fixed roles")
    selection = dict(registry.get("representative_selection", {}))
    if selection.get("send_state") != fixed["sender_state"] or selection.get("receiver_state") != fixed["representative_median_wrong"]:
        raise ValueError("preserved representative selection does not match the registered states")
    selection["selection_epoch"] = "original registry evaluation; not reranked on current metrics"
    selection["current_evaluation_pooled_mse"] = float(
        mse_matrix[fixed["sender_state"], fixed["representative_median_wrong"]]
    )
    return {"records": records, "fixed_states": states,
            "fixed_state_indices": dict(fixed), "representative_selection": selection}


def resolve_registry_ids(
    registry: Mapping[str, Any],
    labels: np.ndarray,
    cross_mse: np.ndarray,
    *,
    class_names: Sequence[str] = STL10_CLASS_NAMES,
    preserve_registered_samples: bool = False,
) -> dict[str, list[int]]:
    if preserve_registered_samples:
        if list(class_names) != list(STL10_CLASS_NAMES):
            raise ValueError("current STL-10 class names/order differ from the official contract")
        return resolve_preserved_registry_ids(registry, labels, class_names)
    stored_classes = registry.get("classes")
    if not isinstance(stored_classes, list) or len(stored_classes) != EXPECTED_CLASSES:
        raise ValueError("registry must contain ten ordered STL-10 classes")
    official_class_names = [str(value) for value in class_names]
    if official_class_names != list(STL10_CLASS_NAMES):
        raise ValueError("current STL-10 class names/order differ from the official contract")
    expected = select_quantile_samples(
        labels,
        np.diagonal(cross_mse, axis1=1, axis2=2).mean(axis=1),
        class_names=official_class_names,
    )
    result = {name: [] for name, _ in QUANTILES}
    for class_id, (stored, recomputed) in enumerate(zip(stored_classes, expected)):
        if int(stored.get("class_id", -1)) != class_id:
            raise ValueError("registry classes must be stored in class-ID order")
        if str(stored.get("class_name", "")) != official_class_names[class_id]:
            raise ValueError("registry class name/order differs from official STL-10")
        for quantile_name, _ in QUANTILES:
            stored_record = stored.get("quantiles", {}).get(quantile_name, {})
            expected_record = recomputed["quantiles"][quantile_name]
            if int(stored_record.get("sample_id", -1)) != int(
                expected_record["sample_id"]
            ):
                raise ValueError("registry sample IDs do not reproduce from formal metrics")
            result[quantile_name].append(int(expected_record["sample_id"]))
    return result


@torch.inference_mode()
def run_stl10_presentation_inference(
    seeded_model: Any,
    software_model: Any,
    images: torch.Tensor,
    *,
    sender_state: int,
    cross_pump_state: int,
    same_pump_state: int,
    representative_state: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    seeded, seeded_provenance = run_controlled_inference(
        seeded_model,
        images,
        send_state=sender_state,
        cross_pump_receiver_state=cross_pump_state,
        same_pump_receiver_state=same_pump_state,
        representative_receiver_state=representative_state,
        device=device,
    )
    images_device = images.to(device)
    software_state = torch.zeros(
        images_device.shape[0], dtype=torch.long, device=device
    )
    software_output = software_model(images_device, software_state)
    arrays = {
        "original": seeded["original"],
        "software": nchw_to_rgb(software_output.reconstruction),
        "correct": seeded["correct"],
        "cross_pump_same_power": seeded["cross_pump"],
        "same_pump_wrong_power": seeded["same_pump"],
        "representative_median_wrong": seeded["representative"],
    }
    expected_shape = (images.shape[0], 96, 96, 3)
    for key, value in arrays.items():
        if value.shape != expected_shape:
            raise ValueError(f"{key} must retain native STL-10 shape {expected_shape}")
    return arrays, {
        "seeded": seeded_provenance,
        "software_forward_calls": 1,
        "software_state_ids": "all zero; software VQ is state-independent",
    }


@torch.inference_mode()
def run_stl10_formal_batch_context_inference(
    seeded_model: Any,
    software_model: Any,
    dataset: Any,
    sample_ids: Sequence[int],
    *,
    formal_batch_size: int,
    sender_state: int,
    cross_pump_state: int,
    same_pump_state: int,
    representative_state: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reproduce each selected image inside its original formal-evaluation batch.

    GPU convolution reductions can vary slightly with batch shape.  Near a VQ
    Voronoi boundary that small float32 change can flip an index, so running the
    selected images as one new 30-image batch is not guaranteed to reproduce the
    formal 8,000-image evaluation.  This function reconstructs only the formal
    batches containing selected IDs, then restores the requested sample order.
    """

    if formal_batch_size <= 0:
        raise ValueError("formal_batch_size must be positive")
    dataset_size = len(dataset)
    if dataset_size != EXPECTED_TEST_SAMPLES:
        raise ValueError("formal batch-context inference requires 8,000 test images")
    requested_ids = [int(value) for value in sample_ids]
    if not requested_ids:
        raise ValueError("sample_ids cannot be empty")
    if any(value < 0 or value >= dataset_size for value in requested_ids):
        raise ValueError("sample_ids contain an out-of-range official test index")

    positions_by_batch: dict[int, list[tuple[int, int]]] = {}
    for output_position, sample_id in enumerate(requested_ids):
        batch_start = (sample_id // formal_batch_size) * formal_batch_size
        positions_by_batch.setdefault(batch_start, []).append(
            (output_position, sample_id)
        )

    collected: dict[str, list[np.ndarray | None]] = {
        key: [None] * len(requested_ids) for key in ("original", *BRANCHES)
    }
    context_records: list[dict[str, Any]] = []
    for batch_start in sorted(positions_by_batch):
        batch_stop = min(batch_start + formal_batch_size, dataset_size)
        context_ids = range(batch_start, batch_stop)
        images = torch.stack([dataset[index][0] for index in context_ids])
        context_arrays, context_provenance = run_stl10_presentation_inference(
            seeded_model,
            software_model,
            images,
            sender_state=sender_state,
            cross_pump_state=cross_pump_state,
            same_pump_state=same_pump_state,
            representative_state=representative_state,
            device=device,
        )

        selected_context_positions: list[int] = []
        selected_ids: list[int] = []
        for output_position, sample_id in positions_by_batch[batch_start]:
            context_position = sample_id - batch_start
            selected_context_positions.append(context_position)
            selected_ids.append(sample_id)
            for key in collected:
                collected[key][output_position] = np.asarray(
                    context_arrays[key][context_position], dtype=np.float32
                ).copy()
        context_records.append(
            {
                "formal_batch_index": batch_start // formal_batch_size,
                "formal_batch_start_inclusive": batch_start,
                "formal_batch_stop_exclusive": batch_stop,
                "formal_batch_sample_count": batch_stop - batch_start,
                "selected_sample_ids": selected_ids,
                "selected_context_positions": selected_context_positions,
                "inference": context_provenance,
            }
        )

    missing = {
        key: [index for index, value in enumerate(values) if value is None]
        for key, values in collected.items()
        if any(value is None for value in values)
    }
    if missing:
        raise RuntimeError(f"selected inference outputs are incomplete: {missing}")
    arrays = {
        key: np.stack([value for value in values if value is not None])
        for key, values in collected.items()
    }
    expected_shape = (len(requested_ids), 96, 96, 3)
    for key, value in arrays.items():
        if value.shape != expected_shape:
            raise RuntimeError(f"{key} selected output shape differs from {expected_shape}")

    context_count = len(context_records)
    return arrays, {
        "mode": "formal-evaluation-batch-context-v1",
        "formal_evaluation_batch_size": formal_batch_size,
        "selected_sample_order": requested_ids,
        "unique_formal_batch_count": context_count,
        "seeded_sender_quantizer_calls": context_count,
        "software_forward_calls": context_count,
        "context_batches": context_records,
        "reason": (
            "preserve the formal evaluator's batch shape so float32 convolution "
            "roundoff cannot spuriously move a near-boundary latent token to a "
            "different VQ index"
        ),
        "transport_invariant": (
            "within every formal batch, each selected reconstruction reuses the "
            "single sender-produced base/residual index tensors; only receiver B_s changes"
        ),
    }


def selected_metric_records(
    formal: Mapping[str, np.ndarray],
    sample_ids: Sequence[int],
    *,
    fixed_state_indices: Mapping[str, int],
) -> dict[str, dict[str, np.ndarray]]:
    sender = int(fixed_state_indices["sender_state"])
    branch_pairs = {
        "correct": (sender, sender),
        "cross_pump_same_power": (
            sender,
            int(fixed_state_indices["wrong_cross_pump_same_power"]),
        ),
        "same_pump_wrong_power": (
            sender,
            int(fixed_state_indices["wrong_same_pump_other_power"]),
        ),
        "representative_median_wrong": (
            sender,
            int(fixed_state_indices["representative_median_wrong"]),
        ),
    }
    ids = np.asarray(sample_ids, dtype=np.int64)
    records = {
        "software": {
            metric: formal[f"software_{metric}"][ids]
            for metric in ("mse", "ssim", "lpips")
        }
    }
    for branch, (send, receive) in branch_pairs.items():
        records[branch] = {
            metric: formal[f"seeded_cross_{metric}"][ids, send, receive]
            for metric in ("mse", "ssim", "lpips")
        }
    for values in records.values():
        values["rmse"] = np.sqrt(values["mse"])
    return records


def validate_rendered_mse(
    arrays: Mapping[str, np.ndarray],
    metrics: Mapping[str, Mapping[str, np.ndarray]],
    sample_ids: Sequence[int],
) -> dict[str, Any]:
    original = np.asarray(arrays["original"], dtype=np.float64)
    result: dict[str, Any] = {}
    for branch in BRANCHES:
        rendered = np.asarray(arrays[branch], dtype=np.float64)
        actual = np.mean((rendered - original) ** 2, axis=(1, 2, 3))
        expected = np.asarray(metrics[branch]["mse"], dtype=np.float64)
        difference = np.abs(actual - expected)
        if not np.allclose(
            actual, expected, rtol=INFERENCE_MSE_RTOL, atol=INFERENCE_MSE_ATOL
        ):
            worst = int(np.argmax(difference))
            raise ValueError(
                f"rendered {branch} MSE differs from formal metrics at STL-10 ID "
                f"{int(sample_ids[worst])}: actual={actual[worst]:.10g}, "
                f"expected={expected[worst]:.10g}"
            )
        result[branch] = {
            "maximum_absolute_difference": float(difference.max(initial=0.0)),
            "rtol": INFERENCE_MSE_RTOL,
            "atol": INFERENCE_MSE_ATOL,
        }
    return result


def _save_figure(
    figure: plt.Figure, *, output_dir: Path, stem: str, title: str, dpi: int
) -> dict[str, Any]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    canvas = [float(value) for value in figure.get_size_inches()]
    figure.savefig(png, dpi=dpi, bbox_inches="tight")
    figure.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"Title": title, "Creator": "lightwave-stl10-stagec-visuals"},
    )
    plt.close(figure)
    return {
        "title": title,
        "png": {"path": str(png), "sha256": _sha256_file(png)},
        "pdf": {"path": str(pdf), "sha256": _sha256_file(pdf)},
        "canvas_inches": canvas,
    }


def _metric_caption(values: Mapping[str, np.ndarray], row: int) -> str:
    return (
        f"MSE {float(values['mse'][row]):.4f} | "
        f"RMSE {float(values['rmse'][row]):.3f}\n"
        f"SSIM {float(values['ssim'][row]):.3f} | "
        f"LPIPS {float(values['lpips'][row]):.3f}"
    )


def exported_axis_geometry(
    figure: plt.Figure,
    axis: plt.Axes,
    *,
    source_height: int,
    source_width: int,
    export_dpi: int,
) -> dict[str, Any]:
    """Measure one image axis after layout and project it to PNG export pixels."""

    if source_height <= 0 or source_width <= 0 or export_dpi <= 0:
        raise ValueError("source dimensions and export_dpi must be positive")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bounds = axis.get_window_extent(renderer=renderer)
    dpi_ratio = float(export_dpi) / float(figure.dpi)
    axis_width = float(bounds.width) * dpi_ratio
    axis_height = float(bounds.height) * dpi_ratio
    if not np.isfinite([axis_width, axis_height]).all() or min(
        axis_width, axis_height
    ) <= 0.0:
        raise ValueError("rendered image axis has invalid export dimensions")
    return {
        "source_pixels_hw": [source_height, source_width],
        "axis_pixels_hw_approx": [axis_height, axis_width],
        "scale_y": axis_height / source_height,
        "scale_x": axis_width / source_width,
        "export_dpi": int(export_dpi),
        "measurement": (
            "axis bbox measured after final subplot layout/draw at figure DPI, "
            "then linearly projected to PNG export DPI; bbox_inches='tight' crops "
            "the canvas without rescaling the axes"
        ),
    }


def plot_quantile_gallery(
    arrays: Mapping[str, np.ndarray],
    metrics: Mapping[str, Mapping[str, np.ndarray]],
    *,
    quantile_name: str,
    sample_ids: Sequence[int],
    class_names: Sequence[str],
    output_dir: Path,
    dpi: int,
    row_start: int = 0,
    row_stop: int = 10,
    preserve_registered_samples: bool = False,
) -> dict[str, Any]:
    if quantile_name not in {name for name, _ in QUANTILES}:
        raise ValueError("quantile_name must be q25, q50, or q75")
    if len(sample_ids) != EXPECTED_CLASSES or len(class_names) != EXPECTED_CLASSES:
        raise ValueError("gallery requires one sample from each of ten classes")
    if not (0 <= row_start < row_stop <= EXPECTED_CLASSES):
        raise ValueError("invalid gallery row range")
    for key in ("original", *BRANCHES):
        if np.asarray(arrays[key]).shape != (10, 96, 96, 3):
            raise ValueError(f"{key} must have native display shape (10, 96, 96, 3)")

    selected_rows = list(range(row_start, row_stop))
    paged = len(selected_rows) < EXPECTED_CLASSES
    figsize = (13.333, 7.5) if paged else (23.5, 23.5)
    figure, axes = plt.subplots(len(selected_rows), 10, figsize=figsize, squeeze=False)
    titles = ["Original", BRANCH_TITLES["software"]]
    for branch in SEEDED_BRANCHES:
        branch_title = (
            "Original-run median\nwrong state"
            if preserve_registered_samples and branch == "representative_median_wrong"
            else BRANCH_TITLES[branch]
        )
        titles.extend([branch_title, "Absolute error"])
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=7.8 if paged else 9.3, fontweight="bold")

    original = np.asarray(arrays["original"], dtype=np.float32)
    for output_row, source_row in enumerate(selected_rows):
        axes[output_row, 0].imshow(
            original[source_row], vmin=0.0, vmax=1.0, interpolation="nearest"
        )
        axes[output_row, 0].set_ylabel(
            f"{class_names[source_row]}\nID {int(sample_ids[source_row])}",
            fontsize=7.2 if paged else 8.5,
            fontweight="bold",
        )
        software = np.asarray(arrays["software"], dtype=np.float32)
        axes[output_row, 1].imshow(
            software[source_row], vmin=0.0, vmax=1.0, interpolation="nearest"
        )
        axes[output_row, 1].set_xlabel(
            _metric_caption(metrics["software"], source_row),
            fontsize=4.7 if paged else 6.2,
        )
        column = 2
        for branch in SEEDED_BRANCHES:
            reconstruction = np.asarray(arrays[branch], dtype=np.float32)
            axes[output_row, column].imshow(
                reconstruction[source_row],
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            axes[output_row, column].set_xlabel(
                _metric_caption(metrics[branch], source_row),
                fontsize=4.7 if paged else 6.2,
            )
            error = rgb_absolute_error(
                original[source_row : source_row + 1],
                reconstruction[source_row : source_row + 1],
            )[0]
            axes[output_row, column + 1].imshow(
                error,
                cmap="inferno",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            column += 2
        for axis in axes[output_row]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

    page_suffix = (
        f"_classes_{row_start}_{row_stop - 1}" if paged else "_all_classes"
    )
    title = (
        f"STL-10 Stage C {'original-run ' if preserve_registered_samples else ''}{quantile_name.upper()}: native 96×96 reconstructions "
        "and controlled receiver-state mismatch"
    )
    figure.suptitle(title, fontsize=13.2 if paged else 16.0, fontweight="bold", y=0.995)
    figure.subplots_adjust(
        left=0.055,
        right=0.995,
        bottom=0.11 if paged else 0.055,
        top=0.91 if paged else 0.965,
        wspace=0.10,
        hspace=0.50 if paged else 0.54,
    )
    display_geometry = exported_axis_geometry(
        figure,
        axes[0, 0],
        source_height=96,
        source_width=96,
        export_dpi=dpi,
    )
    axis_height, axis_width = display_geometry["axis_pixels_hw_approx"]
    footer = figure.text(
        0.5,
        0.004,
        "Native 96×96 arrays; canvas enlargement only. No sharpening, denoising, "
        "super-resolution, or per-image scaling. Error maps use one fixed [0,1] scale.\n"
        + ("Original-run samples and receiver states; current evaluation metrics; no reselection.\n" if preserve_registered_samples else "")
        +
        f"Approximate PNG display scale @{dpi} dpi: "
        f"{display_geometry['scale_x']:.2f}× width × "
        f"{display_geometry['scale_y']:.2f}× height "
        f"(image axis ≈{axis_width:.0f}×{axis_height:.0f} px from native 96×96).",
        ha="center",
        fontsize=7.2 if paged else 9.0,
    )
    footer_clearance = gallery_footer_clearance(figure, axes, footer)
    result = _save_figure(
        figure,
        output_dir=output_dir,
        stem=f"stl10_01_{quantile_name}{page_suffix}",
        title=title,
        dpi=dpi,
    )
    result.update(
        {
            "quantile": quantile_name,
            "quantile_reference": "original registry evaluation" if preserve_registered_samples else "current formal evaluation",
            "class_ids": selected_rows,
            "sample_ids": [int(sample_ids[index]) for index in selected_rows],
            "native_resolution": [96, 96],
            "post_processing": "none",
            "display_scale_at_png_export": display_geometry,
            "footer_clearance": footer_clearance,
        }
    )
    return result


def gallery_footer_clearance(figure: plt.Figure, axes: np.ndarray, footer: Any) -> dict[str, Any]:
    """Measure the footer against last-row labels, not just the image axes."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    footer_top = float(footer.get_window_extent(renderer).y1)
    last_row_bottom = min(float(axis.get_tightbbox(renderer).y0) for axis in axes[-1])
    clearance = last_row_bottom - footer_top
    if clearance < 0:
        raise RuntimeError("gallery footer overlaps the last row's metric labels")
    return {
        "verified_no_overlap": True,
        "clearance_pixels_at_canvas_dpi": clearance,
        "canvas_dpi": float(figure.dpi),
        "clearance_points": clearance * 72.0 / float(figure.dpi),
    }


def _configure_matrix_axis(axis: plt.Axes, labels: Sequence[str]) -> None:
    axis.set_xticks(range(10), labels=labels, rotation=45, ha="right", fontsize=7)
    axis.set_yticks(range(10), labels=labels, fontsize=7)
    axis.set_xlabel("Receiver state (nm / mJ, 10 s)")
    axis.set_ylabel("Sender state (nm / mJ, 10 s)")
    for state in range(10):
        axis.add_patch(
            plt.Rectangle(
                (state - 0.5, state - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#00D084",
                linewidth=1.2,
            )
        )


def plot_state_metric_heatmaps(
    formal: Mapping[str, np.ndarray],
    *,
    state_labels: Sequence[str],
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    labels = [compact_state_label(value) for value in state_labels]
    matrices = {
        "MSE": formal["seeded_cross_mse"].mean(axis=0),
        "RMSE": np.sqrt(formal["seeded_cross_mse"].mean(axis=0)),
        "SSIM": formal["seeded_cross_ssim"].mean(axis=0),
        "LPIPS": formal["seeded_cross_lpips"].mean(axis=0),
    }
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 12.3))
    specifications = (
        ("MSE", "magma", None, None, ".4f", "lower is better"),
        ("RMSE", "magma", None, None, ".3f", "lower is better"),
        ("SSIM", "viridis", 0.0, 1.0, ".2f", "higher is better"),
        ("LPIPS", "magma", None, None, ".3f", "lower is better"),
    )
    for axis, (name, cmap, vmin, vmax, fmt, direction) in zip(
        axes.flat, specifications
    ):
        values = matrices[name]
        image = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04).set_label(
            f"{name} ({direction})"
        )
        axis.set_title(f"{name} in native [0,1] RGB evaluation", fontweight="bold")
        _configure_matrix_axis(axis, labels)
        threshold = float(np.median(values))
        for sender in range(10):
            for receiver in range(10):
                value = float(values[sender, receiver])
                axis.text(
                    receiver,
                    sender,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    fontsize=5.0,
                    color="white" if value <= threshold else "black",
                    fontweight="bold" if sender == receiver else "normal",
                )
    title = "STL-10 Stage C: all 10×10 ordered sender/receiver state pairs"
    figure.suptitle(title, fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        0.006,
        "Sender indices are held fixed across each row; only receiver B_s changes. "
        "Metrics use all 8,000 native 96×96 test images.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.025, 1.0, 0.965))
    result = _save_figure(
        figure,
        output_dir=output_dir,
        stem="stl10_02_state_mismatch_10x10_metrics",
        title=title,
        dpi=dpi,
    )
    result["matrices"] = {key: value.tolist() for key, value in matrices.items()}
    return result


def per_image_correct_wrong_distributions(
    formal: Mapping[str, np.ndarray]
) -> dict[str, dict[str, np.ndarray]]:
    mask = ~np.eye(10, dtype=bool)
    result: dict[str, dict[str, np.ndarray]] = {}
    for metric in ("mse", "ssim", "lpips"):
        cross = np.asarray(formal[f"seeded_cross_{metric}"], dtype=np.float64)
        correct = np.diagonal(cross, axis1=1, axis2=2).mean(axis=1)
        wrong = cross[:, mask].mean(axis=1)
        result[metric] = {"correct": correct, "wrong": wrong}
    result["rmse"] = {
        role: np.sqrt(result["mse"][role]) for role in ("correct", "wrong")
    }
    return result


def plot_correct_wrong_distributions(
    formal: Mapping[str, np.ndarray], *, output_dir: Path, dpi: int
) -> dict[str, Any]:
    distributions = per_image_correct_wrong_distributions(formal)
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 9.2))
    for axis, metric in zip(axes.flat, ("mse", "rmse", "ssim", "lpips")):
        correct = distributions[metric]["correct"]
        wrong = distributions[metric]["wrong"]
        violin = axis.violinplot(
            [correct, wrong], positions=[1, 2], showmedians=True, showextrema=False
        )
        for body, color in zip(violin["bodies"], ("#0072B2", "#D55E00")):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.72)
        violin["cmedians"].set_color("black")
        axis.boxplot(
            [correct, wrong],
            positions=[1, 2],
            widths=0.16,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white", "alpha": 0.75},
            medianprops={"color": "black"},
        )
        axis.set_xticks([1, 2], ["Correct state", "90 wrong pairs"])
        axis.set_ylabel(metric.upper())
        axis.set_title(f"Per-image {metric.upper()} distribution", fontweight="bold")
        axis.grid(axis="y", alpha=0.2)
    title = "Correct versus wrong receiver-state distributions (test image is the unit)"
    figure.suptitle(title, fontsize=15, fontweight="bold")
    figure.text(
        0.5,
        0.006,
        "Each point is reduced within one test image: mean of 10 diagonal states "
        "versus mean of all 90 ordered mismatches; no pair is treated as an independent image.",
        ha="center",
        fontsize=8.7,
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.96))
    result = _save_figure(
        figure,
        output_dir=output_dir,
        stem="stl10_03_correct_wrong_metric_distributions",
        title=title,
        dpi=dpi,
    )
    result["reduction"] = (
        "per test image: mean over 10 correct states and mean over 90 wrong pairs"
    )
    return result


def write_selected_metrics_csv(
    path: Path,
    *,
    registry: Mapping[str, Any],
    formal: Mapping[str, np.ndarray],
    state_records: Sequence[Mapping[str, Any]],
) -> int:
    fixed = {key: int(value) for key, value in registry["fixed_state_indices"].items()}
    sender = fixed["sender_state"]
    receiver_by_branch: dict[str, int | None] = {
        "software": None,
        "correct": sender,
        "cross_pump_same_power": fixed["wrong_cross_pump_same_power"],
        "same_pump_wrong_power": fixed["wrong_same_pump_other_power"],
        "representative_median_wrong": fixed["representative_median_wrong"],
    }
    fields = [
        "quantile",
        "class_id",
        "class_name",
        "official_test_id",
        "branch",
        "sender_state",
        "sender_state_id",
        "receiver_state",
        "receiver_state_id",
        "receiver_pump_nm",
        "receiver_power_mw",
        "mse",
        "rmse",
        "ssim",
        "lpips",
        "metric_resolution",
    ]
    rows: list[dict[str, Any]] = []
    for quantile_name, _ in QUANTILES:
        for class_record in registry["classes"]:
            sample_id = int(class_record["quantiles"][quantile_name]["sample_id"])
            selected = selected_metric_records(
                formal, [sample_id], fixed_state_indices=fixed
            )
            for branch in BRANCHES:
                receiver = receiver_by_branch[branch]
                receiver_record = state_records[receiver] if receiver is not None else None
                rows.append(
                    {
                        "quantile": quantile_name,
                        "class_id": int(class_record["class_id"]),
                        "class_name": str(class_record["class_name"]),
                        "official_test_id": sample_id,
                        "branch": branch,
                        "sender_state": "" if branch == "software" else sender,
                        "sender_state_id": (
                            "" if branch == "software" else state_records[sender]["state_id"]
                        ),
                        "receiver_state": "" if receiver is None else receiver,
                        "receiver_state_id": (
                            "" if receiver_record is None else receiver_record["state_id"]
                        ),
                        "receiver_pump_nm": (
                            "" if receiver_record is None else receiver_record["pump_nm"]
                        ),
                        "receiver_power_mw": (
                            "" if receiver_record is None else receiver_record["power_mw"]
                        ),
                        "mse": float(selected[branch]["mse"][0]),
                        "rmse": float(selected[branch]["rmse"][0]),
                        "ssim": float(selected[branch]["ssim"][0]),
                        "lpips": float(selected[branch]["lpips"][0]),
                        "metric_resolution": "native 96x96 RGB [0,1]",
                    }
                )
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def generate_stl10_stagec_visuals(
    *,
    seeded_checkpoint: str | Path,
    software_checkpoint: str | Path,
    codebook: str | Path,
    codebook_manifest: str | Path,
    registry: str | Path,
    evaluation_dir: str | Path,
    output_dir: str | Path,
    device_name: str = "cuda:0",
    dpi: int = 300,
    dataset_root: str | Path | None = None,
    preserve_registered_samples: bool = False,
) -> dict[str, Any]:
    paths = _resolve_inputs(
        seeded_checkpoint=seeded_checkpoint,
        software_checkpoint=software_checkpoint,
        codebook=codebook,
        codebook_manifest=codebook_manifest,
        registry=registry,
        evaluation_dir=evaluation_dir,
    )
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    evaluation_manifest = _read_json(paths["evaluation_manifest"])
    summary = _read_json(paths["reconstruction_summary"])
    codebook_manifest_payload = _read_json(paths["codebook_manifest"])
    registry_payload = _read_json(paths["registry"])
    formal = _load_metric_arrays(paths["per_image_metrics"])
    bank = load_codebook_bank(paths["codebook"])
    state_records = _validate_formal_artifacts(
        paths=paths,
        evaluation_manifest=evaluation_manifest,
        summary=summary,
        codebook_manifest=codebook_manifest_payload,
        registry=registry_payload,
        arrays=formal,
        bank=bank,
        preserve_registered_samples=preserve_registered_samples,
    )

    physical_mapping = resolve_gallery_states(
        codebook_manifest_payload, formal["seeded_cross_mse"].mean(axis=0),
        registry_payload, preserve_registered_samples=preserve_registered_samples,
    )

    dataset_root, dataset_path_provenance = resolve_dataset_root(
        registry_payload["dataset"]["root"], dataset_root
    )
    dataset = STL10(
        root=str(dataset_root),
        split="test",
        transform=transforms.ToTensor(),
        download=False,
    )
    if len(dataset) != EXPECTED_TEST_SAMPLES:
        raise ValueError("official STL-10 test split must contain 8,000 images")
    registry_dataset_contract = validate_registry_dataset_contract(
        registry_payload, dataset
    )
    actual_fingerprint = _dataset_fingerprint(dataset)
    expected_fingerprint = evaluation_manifest["dataset"]
    fingerprint_fields = (
        "processed_images_float32_sha256",
        "processed_labels_int64_sha256",
        "processed_sample_order_sha256",
        "processed_sample_count",
    )
    mismatches = {
        field: {"expected": expected_fingerprint.get(field), "actual": actual_fingerprint[field]}
        for field in fingerprint_fields
        if expected_fingerprint.get(field) != actual_fingerprint[field]
    }
    if mismatches:
        raise ValueError(
            "local STL-10 test split differs from formal evaluation: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    labels = dataset_labels(dataset)
    ids_by_quantile = resolve_registry_ids(
        registry_payload,
        labels,
        formal["seeded_cross_mse"],
        class_names=registry_dataset_contract["class_names"],
        preserve_registered_samples=preserve_registered_samples,
    )

    seeded_payload = torch.load(paths["seeded_checkpoint"], map_location="cpu")
    software_payload = torch.load(paths["software_checkpoint"], map_location="cpu")
    bank_hash = _array_sha256(bank)
    if seeded_payload.get("base_bank_array_sha256") != bank_hash:
        raise ValueError("seeded checkpoint and frozen B_s bank hashes differ")
    if _normalized_dataset_name(
        seeded_payload["config"]["dataset"]["name"]
    ) != "stl10":
        raise ValueError("seeded checkpoint is not an STL-10 model")
    compatibility = _compatibility_report(
        seeded_payload["config"], software_payload["config"]
    )
    seeded_model = _load_model(
        seeded_payload, expected_quantizer_type="seeded_residual", bank=bank
    ).to(device).eval()
    software_model = _load_model(
        software_payload, expected_quantizer_type="trainable_residual", bank=None
    ).to(device).eval()

    flattened_ids = [
        sample_id
        for quantile_name, _ in QUANTILES
        for sample_id in ids_by_quantile[quantile_name]
    ]
    runtime = evaluation_manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("formal evaluation manifest is missing runtime metadata")
    try:
        formal_batch_size = int(runtime["batch_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "formal evaluation manifest must record a positive batch_size"
        ) from error
    if formal_batch_size <= 0:
        raise ValueError("formal evaluation manifest batch_size must be positive")
    fixed = physical_mapping["fixed_state_indices"]
    all_arrays, inference_provenance = run_stl10_formal_batch_context_inference(
        seeded_model,
        software_model,
        dataset,
        flattened_ids,
        formal_batch_size=formal_batch_size,
        sender_state=fixed["sender_state"],
        cross_pump_state=fixed["wrong_cross_pump_same_power"],
        same_pump_state=fixed["wrong_same_pump_other_power"],
        representative_state=fixed["representative_median_wrong"],
        device=device,
    )

    output.mkdir(parents=True, exist_ok=False)
    class_names = registry_dataset_contract["class_names"]
    figures: dict[str, Any] = {}
    inference_checks: dict[str, Any] = {}
    for quantile_index, (quantile_name, _) in enumerate(QUANTILES):
        start = quantile_index * EXPECTED_CLASSES
        stop = start + EXPECTED_CLASSES
        quantile_arrays = {key: value[start:stop] for key, value in all_arrays.items()}
        sample_ids = ids_by_quantile[quantile_name]
        quantile_metrics = selected_metric_records(
            formal, sample_ids, fixed_state_indices=fixed
        )
        inference_checks[quantile_name] = validate_rendered_mse(
            quantile_arrays, quantile_metrics, sample_ids
        )
        figures[f"{quantile_name}_all_classes"] = plot_quantile_gallery(
            quantile_arrays,
            quantile_metrics,
            quantile_name=quantile_name,
            preserve_registered_samples=preserve_registered_samples,
            sample_ids=sample_ids,
            class_names=class_names,
            output_dir=output,
            dpi=dpi,
        )
        for row_start, row_stop in ((0, 5), (5, 10)):
            figures[f"{quantile_name}_classes_{row_start}_{row_stop - 1}"] = (
                plot_quantile_gallery(
                    quantile_arrays,
                    quantile_metrics,
                    quantile_name=quantile_name,
                    preserve_registered_samples=preserve_registered_samples,
                    sample_ids=sample_ids,
                    class_names=class_names,
                    output_dir=output,
                    dpi=dpi,
                    row_start=row_start,
                    row_stop=row_stop,
                )
            )
    figures["state_mismatch_10x10"] = plot_state_metric_heatmaps(
        formal,
        state_labels=[record["state_id"] for record in state_records],
        output_dir=output,
        dpi=dpi,
    )
    figures["correct_wrong_distributions"] = plot_correct_wrong_distributions(
        formal, output_dir=output, dpi=dpi
    )

    selected_csv = output / "stl10_selected_sample_metrics.csv"
    selected_row_count = write_selected_metrics_csv(
        selected_csv,
        registry=registry_payload,
        formal=formal,
        state_records=state_records,
    )
    project_root = Path(__file__).resolve().parents[3]
    manifest: dict[str, Any] = {
        "schema": VISUAL_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(project_root),
        "inputs": {
            key: {"path": str(path), "sha256": _sha256_file(path)}
            for key, path in paths.items()
        },
        "formal_evaluation": {
            "scope": "final",
            "dataset": "stl10",
            "split": "test",
            "sample_count": EXPECTED_TEST_SAMPLES,
            "fingerprint": actual_fingerprint,
            "registry_dataset_contract": registry_dataset_contract,
            "dataset_path_provenance": dataset_path_provenance,
            "metric_definitions": evaluation_manifest.get("metrics"),
        },
        "checkpoint_compatibility": compatibility,
        "frozen_base_bank": {
            "shape": list(bank.shape),
            "array_sha256": bank_hash,
        },
        "selection": {
            "manual_visual_selection": False,
            "preserve_registered_samples": preserve_registered_samples,
            "quantile_reference": "original registry evaluation; not exact current-evaluation quantiles" if preserve_registered_samples else "current formal evaluation",
            "input_hash_provenance": validate_registry_input_hashes(
                registry_payload, paths, preserve_registered_samples=preserve_registered_samples
            ),
            "ids_by_quantile_and_class": ids_by_quantile,
            "registry_schema": registry_payload["schema"],
            "registry_sha256": _sha256_file(paths["registry"]),
            **({"preserved_registry_content": registry_payload} if preserve_registered_samples else {}),
        },
        "controlled_state_mapping": {
            "resolved_from": "codebook manifest pump_nm and power_mw fields",
            "fixed_state_indices": fixed,
            "fixed_states": physical_mapping["fixed_states"],
            "representative_selection": physical_mapping[
                "representative_selection"
            ],
            "transport_index_reuse": inference_provenance,
        },
        "rendering": {
            "source_and_metric_resolution": [96, 96],
            "rgb_range": [0.0, 1.0],
            "display_interpolation": "nearest",
            "display_enlargement": (
                "canvas only; source arrays remain native 96x96; each gallery "
                "figure records its measured approximate PNG axis scale"
            ),
            "post_processing": "none",
            "forbidden_operations_absent": [
                "sharpening",
                "denoising",
                "super-resolution",
                "per-image normalization",
            ],
            "absolute_error": (
                "mean absolute RGB error per native pixel; common fixed [0,1] display scale"
            ),
            "metric_annotation": "MSE, RMSE, SSIM, LPIPS; no dB primary labels",
            "formats": ["png", "pdf"],
            "dpi": dpi,
        },
        "inference_mse_checks": inference_checks,
        "selected_sample_metrics_csv": {
            "path": str(selected_csv),
            "sha256": _sha256_file(selected_csv),
            "row_count": selected_row_count,
        },
        "figures": figures,
        "interpretation_boundary": (
            "The gallery visualizes native STL-10 reconstruction and receiver-state "
            "mismatch. It does not establish cryptographic security or physical PUF claims."
        ),
    }
    manifest["manifest_payload_sha256"] = _json_sha256(manifest)
    manifest_path = output / "stl10_stagec_visual_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    manifest = generate_stl10_stagec_visuals(
        seeded_checkpoint=args.seeded_checkpoint,
        software_checkpoint=args.software_checkpoint,
        codebook=args.codebook,
        codebook_manifest=args.codebook_manifest,
        registry=args.registry,
        evaluation_dir=args.evaluation_dir,
        output_dir=args.output_dir,
        device_name=args.device,
        dpi=args.dpi,
        dataset_root=args.dataset_root,
        preserve_registered_samples=args.preserve_registered_samples,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output_dir).expanduser().resolve()),
                "selected_ids": manifest["selection"]["ids_by_quantile_and_class"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
