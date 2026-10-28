from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .build_fixed_sample_registry import write_registry_exclusive
from .build_stl10_sample_registry import (
    QUANTILES,
    resolve_controlled_states,
    select_quantile_samples,
    state_records_from_manifest,
)


REGISTRY_SCHEMA = "lightwave-seeded-vqvae-mnist-stagec-sample-registry-v1"
EXPECTED_TEST_SAMPLES = 10_000
EXPECTED_CLASSES = 10
MNIST_DIGIT_NAMES = tuple(str(value) for value in range(EXPECTED_CLASSES))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray, dtype: str) -> str:
    canonical = np.ascontiguousarray(array, dtype=dtype)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must contain an object: {path}")
    return payload


def _normalized_dataset_name(value: object) -> str:
    return str(value).strip().lower().replace("-", "").replace("_", "")


def dataset_labels(dataset: Any) -> np.ndarray:
    """Return zero-based MNIST labels without decoding image data unnecessarily."""

    raw_labels = getattr(dataset, "targets", None)
    if raw_labels is None:
        raw_labels = getattr(dataset, "labels", None)
    if raw_labels is None:
        raw_labels = [dataset[index][1] for index in range(len(dataset))]
    if hasattr(raw_labels, "detach"):
        raw_labels = raw_labels.detach().cpu().numpy()
    labels = np.asarray(raw_labels, dtype=np.int64)
    if labels.ndim != 1 or labels.shape[0] != len(dataset):
        raise ValueError("MNIST test labels must be a one-dimensional dataset-length array")
    if not np.all((labels >= 0) & (labels < EXPECTED_CLASSES)):
        raise ValueError("MNIST test labels must be integers in [0, 9]")
    return labels


def select_mnist_quantile_samples(
    labels: Sequence[int] | np.ndarray,
    correct_mean_mse: Sequence[float] | np.ndarray,
) -> list[dict[str, Any]]:
    """Reuse the audited STL rule for the ten MNIST digit classes."""

    return select_quantile_samples(
        labels,
        correct_mean_mse,
        class_names=MNIST_DIGIT_NAMES,
    )


def _validate_formal_manifest(
    manifest: Mapping[str, Any],
    *,
    sample_count: int,
    labels_hash: str,
    metrics_path: Path,
    codebook_manifest_path: Path,
    codebook_manifest: Mapping[str, Any],
) -> None:
    if manifest.get("scope") != "final":
        raise ValueError("MNIST Stage-C registry requires a formal full-test evaluation")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or _normalized_dataset_name(
        dataset.get("name")
    ) != "mnist":
        raise ValueError("evaluation manifest dataset must be MNIST")
    if bool(dataset.get("test_truncated", True)):
        raise ValueError("evaluation manifest must cover the complete MNIST test set")
    processed = dataset.get(
        "processed_test_samples", dataset.get("processed_sample_count")
    )
    expected = dataset.get("expected_test_samples", EXPECTED_TEST_SAMPLES)
    if int(processed if processed is not None else -1) != sample_count:
        raise ValueError("evaluation manifest sample count differs from per-image metrics")
    if int(expected) != EXPECTED_TEST_SAMPLES or sample_count != EXPECTED_TEST_SAMPLES:
        raise ValueError("formal MNIST evaluation must contain all 10,000 test images")
    if dataset.get("processed_labels_int64_sha256") != labels_hash:
        raise ValueError("local MNIST labels differ from the formal evaluation")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or outputs.get(
        "per_image_metrics.npz"
    ) != _sha256_file(metrics_path):
        raise ValueError("formal per-image metrics hash mismatch")
    codebook_input = manifest.get("inputs", {}).get("codebook", {})
    if codebook_input.get("manifest_sha256") != _sha256_file(
        codebook_manifest_path
    ):
        raise ValueError("formal evaluation and codebook manifest hashes differ")
    expected_state_order = [
        {"index": int(record["index"]), "state_id": str(record["state_id"])}
        for record in state_records_from_manifest(dict(codebook_manifest))
    ]
    if manifest.get("state_order") != expected_state_order:
        raise ValueError("formal evaluation state order differs from the codebook manifest")


def _default_dataset_factory(*, root: str, train: bool, download: bool) -> Any:
    from torchvision.datasets import MNIST

    return MNIST(root=root, train=train, download=download)


def build_mnist_stagec_sample_registry(
    per_image_metrics: str | Path,
    dataset_root: str | Path,
    codebook_manifest_path: str | Path,
    *,
    dataset_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Freeze per-digit q25/q50/q75 examples from a formal Stage-C evaluation."""

    metrics_path = Path(per_image_metrics).expanduser().resolve()
    root_path = Path(dataset_root).expanduser().resolve()
    codebook_path = Path(codebook_manifest_path).expanduser().resolve()
    evaluation_manifest_path = metrics_path.with_name("evaluation_manifest.json")
    for path in (metrics_path, codebook_path, evaluation_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(metrics_path, allow_pickle=False) as archive:
        if "seeded_cross_mse" not in archive.files:
            raise ValueError("per-image metrics are missing seeded_cross_mse")
        cross_mse_stored = np.asarray(archive["seeded_cross_mse"])
    expected_shape = (EXPECTED_TEST_SAMPLES, 10, 10)
    if cross_mse_stored.shape != expected_shape:
        raise ValueError(f"seeded_cross_mse must have shape {expected_shape}")
    if not np.issubdtype(cross_mse_stored.dtype, np.floating):
        raise ValueError("seeded_cross_mse must use a floating-point dtype")
    cross_mse = np.asarray(cross_mse_stored, dtype=np.float64)
    if not np.isfinite(cross_mse).all() or np.any(cross_mse < 0.0):
        raise ValueError("seeded_cross_mse must contain finite non-negative values")

    factory = dataset_factory or _default_dataset_factory
    dataset = factory(root=str(root_path), train=False, download=False)
    if len(dataset) != EXPECTED_TEST_SAMPLES:
        raise ValueError("official MNIST test split must contain 10,000 images")
    labels = dataset_labels(dataset)
    labels_hash = _array_sha256(labels, "<i8")

    evaluation_manifest = _read_json(evaluation_manifest_path)
    codebook_manifest = _read_json(codebook_path)
    _validate_formal_manifest(
        evaluation_manifest,
        sample_count=len(dataset),
        labels_hash=labels_hash,
        metrics_path=metrics_path,
        codebook_manifest_path=codebook_path,
        codebook_manifest=codebook_manifest,
    )
    diagonal_mean_mse = np.diagonal(cross_mse, axis1=1, axis2=2).mean(axis=1)
    classes = select_mnist_quantile_samples(labels, diagonal_mean_mse)
    state_mapping = resolve_controlled_states(codebook_manifest, cross_mse.mean(axis=0))

    return {
        "schema": REGISTRY_SCHEMA,
        "dataset": {
            "name": "mnist",
            "train": False,
            "split": "test",
            "root": str(root_path),
            "sample_count": len(dataset),
            "image_shape_chw": [1, 28, 28],
            "official_test_index_definition": (
                "zero-based index in torchvision.datasets.MNIST(train=False)"
            ),
            "class_names": list(MNIST_DIGIT_NAMES),
        },
        "selection_algorithm": {
            "name": "mnist-stagec-correct-state-mean-per-digit-quantiles-v1",
            "score": (
                "per-image arithmetic mean of the ten correct-state entries "
                "seeded_cross_mse[:, s, s]"
            ),
            "quantiles": [quantile for _, quantile in QUANTILES],
            "empirical_quantile_method": "numpy.quantile(method='linear')",
            "selection_rule": (
                "within each digit class choose the closest score to the empirical "
                "quantile; exact ties choose the lower official test ID"
            ),
            "manual_visual_selection": False,
        },
        "inputs": {
            "per_image_metrics": {
                "path": str(metrics_path),
                "sha256": _sha256_file(metrics_path),
                "seeded_cross_mse_shape": list(cross_mse_stored.shape),
                "seeded_cross_mse_dtype": str(cross_mse_stored.dtype),
                "seeded_cross_mse_float32le_sha256": _array_sha256(
                    cross_mse_stored, "<f4"
                ),
            },
            "evaluation_manifest": {
                "path": str(evaluation_manifest_path),
                "sha256": _sha256_file(evaluation_manifest_path),
                "schema": evaluation_manifest.get("schema"),
            },
            "codebook_manifest": {
                "path": str(codebook_path),
                "sha256": _sha256_file(codebook_path),
                "schema": codebook_manifest.get("schema"),
            },
            "mnist_test_labels": {"canonical_int64_sha256": labels_hash},
        },
        "state_order": state_mapping["records"],
        "fixed_state_indices": state_mapping["fixed_state_indices"],
        "fixed_states": state_mapping["fixed_states"],
        "representative_selection": state_mapping["representative_selection"],
        "classes": classes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze MNIST q25/q50/q75 examples from a formal D64/K128 Stage-C "
            "correct-state reconstruction distribution."
        )
    )
    parser.add_argument("--per-image-metrics", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--codebook-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.expanduser().exists():
        raise FileExistsError(args.output.expanduser())
    registry = build_mnist_stagec_sample_registry(
        args.per_image_metrics,
        args.dataset_root,
        args.codebook_manifest,
    )
    output = write_registry_exclusive(registry, args.output)
    print(json.dumps({"output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
