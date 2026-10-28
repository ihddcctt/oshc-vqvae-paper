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
from torchvision.datasets import MNIST

from ..data import resolve_dataset_root
from ..seed.codebook import load_codebook_bank
from .build_mnist_stagec_sample_registry import (
    EXPECTED_CLASSES,
    EXPECTED_TEST_SAMPLES,
    MNIST_DIGIT_NAMES,
    REGISTRY_SCHEMA,
    dataset_labels,
    select_mnist_quantile_samples,
)
from .build_stl10_sample_registry import (
    QUANTILES,
    state_records_from_manifest,
)
from .evaluate_final import (
    EVALUATION_SCHEMA,
    _array_sha256,
    _compatibility_report,
    _json_sha256,
    _load_model,
)
from .plot_cifar_natural_examples import _dataset_fingerprint, _git_metadata
from .plot_stl10_stagec import (
    BRANCHES,
    BRANCH_TITLES,
    SEEDED_BRANCHES,
    _metric_caption,
    exported_axis_geometry,
    selected_metric_records,
    validate_registry_input_hashes,
    resolve_preserved_registry_ids,
    resolve_gallery_states,
    gallery_footer_clearance,
)


VISUAL_SCHEMA = "lightwave-seeded-vqvae-mnist-stagec-visuals-v1"
EXPECTED_STAGEC_BANK_SHAPE = (10, 128, 64)
EXPECTED_IMAGE_SHAPE_CHW = (1, 28, 28)
EXPECTED_LATENT_SHAPE_HW = (7, 7)
EXPECTED_DECODER_BLOCKS_BY_STAGE = (3, 2, 2)
EXPECTED_ENCODER_BACKBONE = "resnet18"
INFERENCE_MSE_RTOL = 3e-3
INFERENCE_MSE_ATOL = 3e-6


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


def _int64_array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the formal, fixed-sample MNIST Stage-C PNG/PDF package "
            "with the same ten-column contract as the STL-10 gallery."
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


def validate_mnist_stagec_bank_shape(bank: np.ndarray) -> None:
    shape = tuple(int(value) for value in np.asarray(bank).shape)
    if shape != EXPECTED_STAGEC_BANK_SHAPE:
        raise ValueError(
            "MNIST Stage C requires frozen B_s shape "
            f"{EXPECTED_STAGEC_BANK_SHAPE}, received {shape}"
        )


def validate_mnist_stagec_checkpoint_config(
    config: Mapping[str, Any], *, expected_quantizer_type: str
) -> dict[str, Any]:
    """Reject a legacy or cross-dataset checkpoint before rendering figures."""
    dataset = config.get("dataset")
    model = config.get("model")
    if not isinstance(dataset, Mapping) or not isinstance(model, Mapping):
        raise ValueError("checkpoint config must contain dataset and model mappings")

    problems: list[str] = []
    if _normalized_dataset_name(dataset.get("name")) != "mnist":
        problems.append("dataset.name must be MNIST")
    try:
        native_resolution = tuple(int(value) for value in dataset["native_resolution"])
    except (KeyError, TypeError, ValueError):
        native_resolution = ()
    if native_resolution != EXPECTED_IMAGE_SHAPE_CHW[1:]:
        problems.append("dataset.native_resolution must be [28, 28]")

    if str(model.get("quantizer_type", "")) != expected_quantizer_type:
        problems.append(f"model.quantizer_type must be {expected_quantizer_type!r}")
    for key, expected in (
        ("latent_dim", EXPECTED_STAGEC_BANK_SHAPE[2]),
        ("base_codebook_size", EXPECTED_STAGEC_BANK_SHAPE[1]),
        ("residual_codebook_size", EXPECTED_STAGEC_BANK_SHAPE[1]),
    ):
        try:
            actual = int(model[key])
        except (KeyError, TypeError, ValueError):
            actual = None
        if actual != expected:
            problems.append(f"model.{key} must be {expected}")

    backbone = str(
        model.get("encoder_backbone", EXPECTED_ENCODER_BACKBONE)
    ).strip().lower()
    if backbone != EXPECTED_ENCODER_BACKBONE:
        problems.append(
            f"model.encoder_backbone must be {EXPECTED_ENCODER_BACKBONE!r}"
        )
    if model.get("pretrained_encoder") is not True:
        problems.append("model.pretrained_encoder must be true")
    try:
        decoder_blocks = tuple(
            int(value) for value in model["decoder_blocks_by_stage"]
        )
    except (KeyError, TypeError, ValueError):
        decoder_blocks = ()
    if decoder_blocks != EXPECTED_DECODER_BLOCKS_BY_STAGE:
        problems.append("model.decoder_blocks_by_stage must be [3, 2, 2]")

    if problems:
        raise ValueError(
            "checkpoint does not satisfy the MNIST Stage-C architecture contract: "
            + "; ".join(problems)
        )
    return {
        "dataset": "mnist",
        "native_resolution": list(native_resolution),
        "encoder_backbone": backbone,
        "pretrained_encoder": True,
        "latent_dim": EXPECTED_STAGEC_BANK_SHAPE[2],
        "base_codebook_size": EXPECTED_STAGEC_BANK_SHAPE[1],
        "residual_codebook_size": EXPECTED_STAGEC_BANK_SHAPE[1],
        "decoder_blocks_by_stage": list(decoder_blocks),
        "quantizer_type": expected_quantizer_type,
    }


def validate_registry_dataset_contract(
    registry: Mapping[str, Any], dataset: Any
) -> dict[str, Any]:
    stored = registry.get("dataset")
    if not isinstance(stored, dict):
        raise ValueError("registry must contain dataset metadata")
    if _normalized_dataset_name(stored.get("name")) != "mnist":
        raise ValueError("registry dataset name must be MNIST")
    if str(stored.get("split", "")).strip().lower() != "test" or bool(
        stored.get("train", True)
    ):
        raise ValueError("registry dataset must be the official MNIST test split")

    sample_count = len(dataset)
    if sample_count != EXPECTED_TEST_SAMPLES or int(
        stored.get("sample_count", -1)
    ) != sample_count:
        raise ValueError(
            "registry and current MNIST dataset must contain 10,000 samples"
        )
    if stored.get("class_names") != list(MNIST_DIGIT_NAMES):
        raise ValueError("registry class_names must be the ordered digits 0 through 9")

    first_sample = dataset[0]
    if not isinstance(first_sample, (tuple, list)) or not first_sample:
        raise ValueError("current MNIST dataset must return an image/label sample")
    first_image = first_sample[0]
    image_shape = tuple(int(value) for value in first_image.shape)
    if image_shape != EXPECTED_IMAGE_SHAPE_CHW:
        raise ValueError(
            "current MNIST test images must have CHW shape "
            f"{EXPECTED_IMAGE_SHAPE_CHW}, received {image_shape}"
        )
    if stored.get("image_shape_chw") != list(image_shape):
        raise ValueError("registry image_shape_chw differs from the current dataset")
    return {
        "name": "mnist",
        "split": "test",
        "train": False,
        "sample_count": sample_count,
        "image_shape_chw": list(image_shape),
        "class_names": list(MNIST_DIGIT_NAMES),
        "verified_against_current_dataset": True,
    }


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
            raise ValueError(
                "formal MNIST per-image metrics are missing keys: "
                f"{missing}; run formal evaluation with --lpips on"
            )
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
    validate_mnist_stagec_bank_shape(bank)
    if evaluation_manifest.get("schema") != EVALUATION_SCHEMA:
        raise ValueError("unsupported formal evaluation schema")
    if evaluation_manifest.get("scope") != "final":
        raise ValueError("MNIST figures require a formal full-test evaluation")
    dataset = evaluation_manifest.get("dataset")
    if not isinstance(dataset, dict) or _normalized_dataset_name(
        dataset.get("name")
    ) != "mnist":
        raise ValueError("formal evaluation dataset must be MNIST")
    if bool(dataset.get("test_truncated", True)):
        raise ValueError("truncated evaluations cannot generate formal figures")
    processed = dataset.get(
        "processed_test_samples", dataset.get("processed_sample_count", -1)
    )
    if int(processed) != EXPECTED_TEST_SAMPLES:
        raise ValueError("formal evaluation must contain all 10,000 MNIST test images")

    if _normalized_dataset_name(summary.get("dataset")) != "mnist":
        raise ValueError("reconstruction summary dataset must be MNIST")
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
        raise ValueError("unsupported MNIST Stage-C fixed-sample registry schema")
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


def resolve_registry_ids(
    registry: Mapping[str, Any],
    labels: np.ndarray,
    cross_mse: np.ndarray,
    *, preserve_registered_samples: bool = False,
) -> dict[str, list[int]]:
    if preserve_registered_samples:
        return resolve_preserved_registry_ids(registry, labels, MNIST_DIGIT_NAMES)
    stored_classes = registry.get("classes")
    if not isinstance(stored_classes, list) or len(stored_classes) != EXPECTED_CLASSES:
        raise ValueError("registry must contain ten ordered MNIST digit classes")
    expected = select_mnist_quantile_samples(
        labels,
        np.diagonal(cross_mse, axis1=1, axis2=2).mean(axis=1),
    )
    result = {name: [] for name, _ in QUANTILES}
    for class_id, (stored, recomputed) in enumerate(zip(stored_classes, expected)):
        if int(stored.get("class_id", -1)) != class_id:
            raise ValueError("registry digit classes must be stored in digit order")
        if str(stored.get("class_name", "")) != MNIST_DIGIT_NAMES[class_id]:
            raise ValueError("registry digit name/order differs from MNIST")
        for quantile_name, _ in QUANTILES:
            stored_record = stored.get("quantiles", {}).get(quantile_name, {})
            expected_record = recomputed["quantiles"][quantile_name]
            if int(stored_record.get("sample_id", -1)) != int(
                expected_record["sample_id"]
            ):
                raise ValueError("registry sample IDs do not reproduce from formal metrics")
            result[quantile_name].append(int(expected_record["sample_id"]))
    return result


def nchw_to_grayscale(images: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(images):
        array = images.detach().float().cpu().numpy()
    else:
        array = np.asarray(images, dtype=np.float32)
    if array.ndim != 4 or array.shape[1] != 1:
        raise ValueError("MNIST display tensors must have shape (N, 1, H, W)")
    return np.clip(array[:, 0], 0.0, 1.0)


def grayscale_absolute_error(
    original: np.ndarray, reconstruction: np.ndarray
) -> np.ndarray:
    first = np.asarray(original, dtype=np.float32)
    second = np.asarray(reconstruction, dtype=np.float32)
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("original and reconstruction must be equal-shaped NHW arrays")
    return np.clip(np.abs(first - second), 0.0, 1.0)


@torch.inference_mode()
def run_mnist_presentation_inference(
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
    if tuple(int(value) for value in images.shape[1:]) != EXPECTED_IMAGE_SHAPE_CHW:
        raise ValueError(
            "MNIST presentation inputs must retain native shape "
            f"{EXPECTED_IMAGE_SHAPE_CHW}"
        )
    images_device = images.to(device)
    count = int(images_device.shape[0])
    sender_ids = torch.full(
        (count,), sender_state, dtype=torch.long, device=device
    )
    latent = seeded_model.encoder(images_device)
    transmitted = seeded_model.quantizer(latent, sender_ids)
    expected_indices_shape = (count, *EXPECTED_LATENT_SHAPE_HW)
    if tuple(transmitted.base_indices.shape) != expected_indices_shape or tuple(
        transmitted.residual_indices.shape
    ) != expected_indices_shape:
        raise ValueError(
            "MNIST Stage-C transmitted index tensors must have shape "
            f"{expected_indices_shape}"
        )

    receiver_states = {
        "correct": sender_state,
        "cross_pump": cross_pump_state,
        "same_pump": same_pump_state,
        "representative": representative_state,
    }
    reconstructed: dict[str, torch.Tensor] = {}
    for name, receiver_state in receiver_states.items():
        receiver_ids = torch.full(
            (count,), receiver_state, dtype=torch.long, device=device
        )
        reconstructed[name] = seeded_model.decode_indices(
            transmitted.base_indices,
            transmitted.residual_indices,
            receiver_ids,
        )
    software_ids = torch.zeros(count, dtype=torch.long, device=device)
    software_output = software_model(images_device, software_ids)

    arrays = {
        "original": nchw_to_grayscale(images_device),
        "software": nchw_to_grayscale(software_output.reconstruction),
        "correct": nchw_to_grayscale(reconstructed["correct"]),
        "cross_pump_same_power": nchw_to_grayscale(reconstructed["cross_pump"]),
        "same_pump_wrong_power": nchw_to_grayscale(reconstructed["same_pump"]),
        "representative_median_wrong": nchw_to_grayscale(
            reconstructed["representative"]
        ),
    }
    expected_display_shape = (count, 28, 28)
    for key, value in arrays.items():
        if value.shape != expected_display_shape:
            raise ValueError(
                f"{key} must retain native MNIST shape {expected_display_shape}"
            )

    base_indices = transmitted.base_indices.detach().cpu().numpy()
    residual_indices = transmitted.residual_indices.detach().cpu().numpy()
    return arrays, {
        "seeded": {
            "sender_quantizer_calls": 1,
            "fixed_send_state": sender_state,
            "receiver_states": receiver_states,
            "base_indices_shape": list(base_indices.shape),
            "residual_indices_shape": list(residual_indices.shape),
            "base_indices_int64le_sha256": _int64_array_sha256(base_indices),
            "residual_indices_int64le_sha256": _int64_array_sha256(
                residual_indices
            ),
            "invariant": (
                "all four reconstructions reuse the exact base/residual index "
                "tensors; only receiver state_ids change"
            ),
        },
        "software_forward_calls": 1,
        "software_state_ids": "all zero; software VQ is state-independent",
    }


@torch.inference_mode()
def run_mnist_formal_batch_context_inference(
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
    """Reproduce selected examples inside their original formal-evaluation batches."""

    if formal_batch_size <= 0:
        raise ValueError("formal_batch_size must be positive")
    dataset_size = len(dataset)
    if dataset_size != EXPECTED_TEST_SAMPLES:
        raise ValueError(
            "formal batch-context inference requires 10,000 MNIST test images"
        )
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
        images = torch.stack(
            [dataset[index][0] for index in range(batch_start, batch_stop)]
        )
        context_arrays, context_provenance = run_mnist_presentation_inference(
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
    expected_shape = (len(requested_ids), 28, 28)
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
            "single sender-produced base/residual index tensors; only receiver "
            "B_s changes"
        ),
    }


def validate_rendered_mse(
    arrays: Mapping[str, np.ndarray],
    metrics: Mapping[str, Mapping[str, np.ndarray]],
    sample_ids: Sequence[int],
) -> dict[str, Any]:
    original = np.asarray(arrays["original"], dtype=np.float64)
    result: dict[str, Any] = {}
    for branch in BRANCHES:
        rendered = np.asarray(arrays[branch], dtype=np.float64)
        actual = np.mean((rendered - original) ** 2, axis=(1, 2))
        expected = np.asarray(metrics[branch]["mse"], dtype=np.float64)
        difference = np.abs(actual - expected)
        if not np.allclose(
            actual, expected, rtol=INFERENCE_MSE_RTOL, atol=INFERENCE_MSE_ATOL
        ):
            worst = int(np.argmax(difference))
            raise ValueError(
                f"rendered {branch} MSE differs from formal metrics at MNIST ID "
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
        metadata={"Title": title, "Creator": "lightwave-mnist-stagec-visuals"},
    )
    plt.close(figure)
    return {
        "title": title,
        "png": {"path": str(png), "sha256": _sha256_file(png)},
        "pdf": {"path": str(pdf), "sha256": _sha256_file(pdf)},
        "canvas_inches": canvas,
    }


def plot_quantile_gallery(
    arrays: Mapping[str, np.ndarray],
    metrics: Mapping[str, Mapping[str, np.ndarray]],
    *,
    quantile_name: str,
    sample_ids: Sequence[int],
    digit_names: Sequence[str],
    output_dir: Path,
    dpi: int,
    row_start: int = 0,
    row_stop: int = 10,
    preserve_registered_samples: bool = False,
) -> dict[str, Any]:
    if quantile_name not in {name for name, _ in QUANTILES}:
        raise ValueError("quantile_name must be q25, q50, or q75")
    if len(sample_ids) != EXPECTED_CLASSES or len(digit_names) != EXPECTED_CLASSES:
        raise ValueError("gallery requires one sample from each of ten digit classes")
    if not (0 <= row_start < row_stop <= EXPECTED_CLASSES):
        raise ValueError("invalid gallery row range")
    for key in ("original", *BRANCHES):
        if np.asarray(arrays[key]).shape != (10, 28, 28):
            raise ValueError(f"{key} must have native display shape (10, 28, 28)")

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
        axes[0, column].set_title(
            title, fontsize=7.8 if paged else 9.3, fontweight="bold"
        )

    original = np.asarray(arrays["original"], dtype=np.float32)
    for output_row, source_row in enumerate(selected_rows):
        axes[output_row, 0].imshow(
            original[source_row],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axes[output_row, 0].set_ylabel(
            f"digit {digit_names[source_row]}\nID {int(sample_ids[source_row])}",
            fontsize=7.2 if paged else 8.5,
            fontweight="bold",
        )
        software = np.asarray(arrays["software"], dtype=np.float32)
        axes[output_row, 1].imshow(
            software[source_row],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
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
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            axes[output_row, column].set_xlabel(
                _metric_caption(metrics[branch], source_row),
                fontsize=4.7 if paged else 6.2,
            )
            error = grayscale_absolute_error(
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
        f"_digits_{row_start}_{row_stop - 1}" if paged else "_all_digits"
    )
    title = (
        f"MNIST Stage C {'original-run ' if preserve_registered_samples else ''}{quantile_name.upper()}: native 28×28 reconstructions "
        "and controlled receiver-state mismatch"
    )
    figure.suptitle(
        title, fontsize=13.2 if paged else 16.0, fontweight="bold", y=0.995
    )
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
        source_height=28,
        source_width=28,
        export_dpi=dpi,
    )
    axis_height, axis_width = display_geometry["axis_pixels_hw_approx"]
    footer = figure.text(
        0.5,
        0.004,
        "Native 28×28 grayscale arrays; canvas enlargement only. No sharpening, "
        "denoising, super-resolution, or per-image scaling. Error maps use one "
        "fixed [0,1] scale.\n"
        + ("Original-run samples and receiver states; current evaluation metrics; no reselection.\n" if preserve_registered_samples else "")
        +
        f"Approximate PNG display scale @{dpi} dpi: "
        f"{display_geometry['scale_x']:.2f}× width × "
        f"{display_geometry['scale_y']:.2f}× height "
        f"(image axis ≈{axis_width:.0f}×{axis_height:.0f} px from native 28×28).",
        ha="center",
        fontsize=7.2 if paged else 9.0,
    )
    footer_clearance = gallery_footer_clearance(figure, axes, footer)
    result = _save_figure(
        figure,
        output_dir=output_dir,
        stem=f"mnist_01_{quantile_name}{page_suffix}",
        title=title,
        dpi=dpi,
    )
    result.update(
        {
            "quantile": quantile_name,
            "quantile_reference": "original registry evaluation" if preserve_registered_samples else "current formal evaluation",
            "digit_ids": selected_rows,
            "sample_ids": [int(sample_ids[index]) for index in selected_rows],
            "native_resolution": [28, 28],
            "post_processing": "none",
            "absolute_error_scale": {
                "definition": "absolute grayscale error per native pixel",
                "display_range": [0.0, 1.0],
                "shared_across_all_error_maps": True,
            },
            "display_scale_at_png_export": display_geometry,
            "footer_clearance": footer_clearance,
        }
    )
    return result


def write_selected_metrics_csv(
    path: Path,
    *,
    registry: Mapping[str, Any],
    formal: Mapping[str, np.ndarray],
    state_records: Sequence[Mapping[str, Any]],
) -> int:
    fixed = {
        key: int(value) for key, value in registry["fixed_state_indices"].items()
    }
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
        "digit",
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
        for digit_record in registry["classes"]:
            sample_id = int(
                digit_record["quantiles"][quantile_name]["sample_id"]
            )
            selected = selected_metric_records(
                formal, [sample_id], fixed_state_indices=fixed
            )
            for branch in BRANCHES:
                receiver = receiver_by_branch[branch]
                receiver_record = (
                    state_records[receiver] if receiver is not None else None
                )
                rows.append(
                    {
                        "quantile": quantile_name,
                        "digit": int(digit_record["class_id"]),
                        "official_test_id": sample_id,
                        "branch": branch,
                        "sender_state": "" if branch == "software" else sender,
                        "sender_state_id": (
                            ""
                            if branch == "software"
                            else state_records[sender]["state_id"]
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
                        "metric_resolution": "native 28x28 grayscale [0,1]",
                    }
                )
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def generate_mnist_stagec_visuals(
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
    dataset = MNIST(
        root=str(dataset_root),
        train=False,
        transform=transforms.ToTensor(),
        download=False,
    )
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
        field: {
            "expected": expected_fingerprint.get(field),
            "actual": actual_fingerprint[field],
        }
        for field in fingerprint_fields
        if expected_fingerprint.get(field) != actual_fingerprint[field]
    }
    if mismatches:
        raise ValueError(
            "local MNIST test split differs from formal evaluation: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    labels = dataset_labels(dataset)
    ids_by_quantile = resolve_registry_ids(
        registry_payload, labels, formal["seeded_cross_mse"],
        preserve_registered_samples=preserve_registered_samples,
    )

    seeded_payload = torch.load(paths["seeded_checkpoint"], map_location="cpu")
    software_payload = torch.load(paths["software_checkpoint"], map_location="cpu")
    bank_hash = _array_sha256(bank)
    if seeded_payload.get("base_bank_array_sha256") != bank_hash:
        raise ValueError("seeded checkpoint and frozen B_s bank hashes differ")
    seeded_architecture = validate_mnist_stagec_checkpoint_config(
        seeded_payload["config"], expected_quantizer_type="seeded_residual"
    )
    software_architecture = validate_mnist_stagec_checkpoint_config(
        software_payload["config"], expected_quantizer_type="trainable_residual"
    )
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
        raise ValueError("formal evaluation batch_size must be positive")
    fixed = physical_mapping["fixed_state_indices"]
    all_arrays, inference_provenance = run_mnist_formal_batch_context_inference(
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
    figures: dict[str, Any] = {}
    inference_checks: dict[str, Any] = {}
    for quantile_index, (quantile_name, _) in enumerate(QUANTILES):
        start = quantile_index * EXPECTED_CLASSES
        stop = start + EXPECTED_CLASSES
        quantile_arrays = {
            key: value[start:stop] for key, value in all_arrays.items()
        }
        sample_ids = ids_by_quantile[quantile_name]
        quantile_metrics = selected_metric_records(
            formal, sample_ids, fixed_state_indices=fixed
        )
        inference_checks[quantile_name] = validate_rendered_mse(
            quantile_arrays, quantile_metrics, sample_ids
        )
        figures[f"{quantile_name}_all_digits"] = plot_quantile_gallery(
            quantile_arrays,
            quantile_metrics,
            quantile_name=quantile_name,
            preserve_registered_samples=preserve_registered_samples,
            sample_ids=sample_ids,
            digit_names=MNIST_DIGIT_NAMES,
            output_dir=output,
            dpi=dpi,
        )
        for row_start, row_stop in ((0, 5), (5, 10)):
            key = f"{quantile_name}_digits_{row_start}_{row_stop - 1}"
            figures[key] = plot_quantile_gallery(
                quantile_arrays,
                quantile_metrics,
                quantile_name=quantile_name,
                preserve_registered_samples=preserve_registered_samples,
                sample_ids=sample_ids,
                digit_names=MNIST_DIGIT_NAMES,
                output_dir=output,
                dpi=dpi,
                row_start=row_start,
                row_stop=row_stop,
            )

    selected_csv = output / "mnist_selected_sample_metrics.csv"
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
            "dataset": "mnist",
            "split": "test",
            "sample_count": EXPECTED_TEST_SAMPLES,
            "fingerprint": actual_fingerprint,
            "registry_dataset_contract": registry_dataset_contract,
            "dataset_path_provenance": dataset_path_provenance,
            "metric_definitions": evaluation_manifest.get("metrics"),
        },
        "checkpoint_compatibility": compatibility,
        "architecture_contract": {
            "input_shape_chw": list(EXPECTED_IMAGE_SHAPE_CHW),
            "latent_shape_hwd": [7, 7, 64],
            "base_codebook_size": 128,
            "residual_codebook_size": 128,
            "decoder_blocks_by_stage": list(
                EXPECTED_DECODER_BLOCKS_BY_STAGE
            ),
            "seeded_checkpoint": seeded_architecture,
            "software_checkpoint": software_architecture,
        },
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
            "ids_by_quantile_and_digit": ids_by_quantile,
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
            "source_and_metric_resolution": [28, 28],
            "channels": 1,
            "grayscale_range": [0.0, 1.0],
            "display_interpolation": "nearest",
            "display_enlargement": (
                "canvas only; source arrays remain native 28x28; each gallery "
                "records its measured approximate PNG axis scale"
            ),
            "post_processing": "none",
            "forbidden_operations_absent": [
                "sharpening",
                "denoising",
                "super-resolution",
                "per-image normalization",
            ],
            "absolute_error": (
                "absolute grayscale error per native pixel; common fixed [0,1] "
                "display scale"
            ),
            "metric_annotation": "MSE, RMSE, SSIM, LPIPS; no dB primary labels",
            "lpips_evaluation_requirement": (
                "formal evaluator --lpips on; grayscale is repeated to three "
                "channels and resized to 32x32 only inside LPIPS evaluation"
            ),
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
            "The gallery visualizes native MNIST reconstruction and receiver-state "
            "mismatch under the Stage-C architecture. It does not establish "
            "cryptographic security or physical PUF claims."
        ),
    }
    manifest["manifest_payload_sha256"] = _json_sha256(manifest)
    manifest_path = output / "mnist_stagec_visual_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    manifest = generate_mnist_stagec_visuals(
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
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
                "selected_ids": manifest["selection"][
                    "ids_by_quantile_and_digit"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
