from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


REGISTRY_SCHEMA = "lightwave-seeded-vqvae-fixed-sample-registry-v1"
EXPECTED_STATE_IDS = [
    "405nm_0mW",
    "405nm_5mW",
    "405nm_10mW",
    "405nm_15mW",
    "405nm_20mW",
    "640nm_0mW",
    "640nm_5mW",
    "640nm_10mW",
    "640nm_15mW",
    "640nm_20mW",
]
FIXED_STATE_IDS = {
    "sender_state": 2,
    "wrong_cross_pump_same_power": 7,
    "wrong_same_pump_other_power": 4,
    "representative_wrong_state": 1,
}
CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
QUANTILES = (("q25", 0.25), ("q50", 0.50), ("q75", 0.75))


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


def validate_state_order(state_order: object) -> list[dict[str, Any]]:
    """Validate the physical ten-state order used by the controlled figures."""

    if not isinstance(state_order, list) or len(state_order) != len(
        EXPECTED_STATE_IDS
    ):
        raise ValueError("evaluation manifest must contain exactly ten ordered states")
    validated: list[dict[str, Any]] = []
    for expected_index, expected_id in enumerate(EXPECTED_STATE_IDS):
        record = state_order[expected_index]
        if not isinstance(record, dict):
            raise ValueError(f"state_order[{expected_index}] must be an object")
        try:
            index = int(record.get("index", -1))
        except (TypeError, ValueError) as error:
            raise ValueError(f"state_order[{expected_index}] has an invalid index") from error
        state_id = str(record.get("state_id", ""))
        if index != expected_index or state_id != expected_id:
            raise ValueError(
                "ten-state mapping mismatch at index "
                f"{expected_index}: expected {expected_id!r}, got {state_id!r}"
            )
        validated.append({"index": expected_index, "state_id": expected_id})

    fixed_labels = {
        role: EXPECTED_STATE_IDS[state_index]
        for role, state_index in FIXED_STATE_IDS.items()
    }
    if fixed_labels != {
        "sender_state": "405nm_10mW",
        "wrong_cross_pump_same_power": "640nm_10mW",
        "wrong_same_pump_other_power": "405nm_20mW",
        "representative_wrong_state": "405nm_5mW",
    }:
        raise RuntimeError("fixed controlled-state constants are internally inconsistent")
    return validated


def _dataset_targets(dataset: Any) -> np.ndarray:
    raw_targets = getattr(dataset, "targets", None)
    if raw_targets is None:
        raw_targets = [dataset[index][1] for index in range(len(dataset))]
    targets = np.asarray(raw_targets, dtype=np.int64)
    if targets.ndim != 1 or targets.shape[0] != len(dataset):
        raise ValueError("CIFAR-10 test targets must be a one-dimensional dataset-length array")
    return targets


def select_quantile_samples(
    targets: Sequence[int] | np.ndarray,
    diagonal_mean_mse: Sequence[float] | np.ndarray,
    *,
    class_count: int = 10,
) -> list[dict[str, Any]]:
    """Select fixed per-class samples using baseline scores only.

    The empirical target uses NumPy's linear quantile. Candidate rows are
    stably ordered by MSE and then official test-set ID. The closest MSE wins;
    equal distances are resolved by the lower official ID.
    """

    target_array = np.asarray(targets, dtype=np.int64)
    score_array = np.asarray(diagonal_mean_mse, dtype=np.float64)
    if target_array.ndim != 1 or score_array.ndim != 1:
        raise ValueError("targets and diagonal_mean_mse must be one-dimensional")
    if target_array.shape[0] != score_array.shape[0]:
        raise ValueError("targets and diagonal_mean_mse must have equal length")
    if not np.all(np.isfinite(score_array)) or np.any(score_array < 0.0):
        raise ValueError("diagonal_mean_mse must contain finite non-negative values")

    selections: list[dict[str, Any]] = []
    for class_id in range(class_count):
        candidate_ids = np.flatnonzero(target_array == class_id).astype(np.int64)
        if candidate_ids.size == 0:
            raise ValueError(f"CIFAR-10 test labels contain no sample for class {class_id}")
        candidate_scores = score_array[candidate_ids]
        stable_order = np.lexsort((candidate_ids, candidate_scores))
        ordered_ids = candidate_ids[stable_order]
        ordered_scores = candidate_scores[stable_order]

        quantile_records: dict[str, Any] = {}
        for quantile_name, quantile in QUANTILES:
            quantile_mse = float(np.quantile(ordered_scores, quantile, method="linear"))
            distances = np.abs(ordered_scores - quantile_mse)
            minimum_distance = float(distances.min())
            # Use an exact computational tie here.  The pre-upgrade gallery used
            # argmin(abs(score - median)); treating merely near-equal float64
            # distances as ties can silently change its already-published sample
            # IDs.  Exact equal minima still resolve to the lower official ID.
            tied = np.flatnonzero(distances == minimum_distance)
            selected_position = int(tied[np.argmin(ordered_ids[tied])])
            sample_id = int(ordered_ids[selected_position])
            quantile_records[quantile_name] = {
                "quantile": quantile,
                "sample_id": sample_id,
                "baseline_empirical_quantile_mse": quantile_mse,
                "selected_sample_baseline_diagonal_mean_mse": float(
                    score_array[sample_id]
                ),
            }
        selections.append(
            {
                "class_id": class_id,
                "class_name": CIFAR10_CLASS_NAMES[class_id],
                "candidate_count": int(candidate_ids.size),
                "quantiles": quantile_records,
            }
        )
    return selections


def _validate_baseline_manifest(
    manifest: dict[str, Any], *, sample_count: int, labels_hash: str
) -> list[dict[str, Any]]:
    if manifest.get("scope") != "final":
        raise ValueError("baseline evaluation manifest must have scope='final'")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("baseline evaluation manifest is missing dataset metadata")
    if str(dataset.get("name", "")).strip().lower().replace("-", "") != "cifar10":
        raise ValueError("baseline evaluation dataset must be CIFAR-10")
    if bool(dataset.get("test_truncated", True)):
        raise ValueError("baseline evaluation must cover the full CIFAR-10 test set")
    processed = dataset.get("processed_test_samples", dataset.get("processed_sample_count"))
    if processed is None or int(processed) != sample_count:
        raise ValueError("baseline manifest sample count does not match seeded_cross_mse")
    manifest_labels_hash = dataset.get("processed_labels_int64_sha256")
    if manifest_labels_hash is not None and manifest_labels_hash != labels_hash:
        raise ValueError("CIFAR-10 test-label hash does not match the baseline evaluation")
    return validate_state_order(manifest.get("state_order"))


def build_fixed_sample_registry(
    per_image_metrics: str | Path,
    dataset_root: str | Path,
    *,
    dataset_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    metrics_path = Path(per_image_metrics).expanduser().resolve()
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    manifest_path = metrics_path.with_name("evaluation_manifest.json")
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"sibling evaluation_manifest.json is required: {manifest_path}"
        )

    with np.load(metrics_path, allow_pickle=False) as payload:
        if "seeded_cross_mse" not in payload.files:
            raise ValueError("per_image_metrics.npz is missing seeded_cross_mse")
        cross_mse_stored = np.asarray(payload["seeded_cross_mse"])
    if cross_mse_stored.ndim != 3 or cross_mse_stored.shape[1:] != (10, 10):
        raise ValueError("seeded_cross_mse must have shape [N, 10, 10]")
    if not np.issubdtype(cross_mse_stored.dtype, np.floating):
        raise ValueError("seeded_cross_mse must use a floating-point dtype")
    cross_mse = np.asarray(cross_mse_stored, dtype=np.float64)
    if not np.all(np.isfinite(cross_mse)) or np.any(cross_mse < 0.0):
        raise ValueError("seeded_cross_mse must contain finite non-negative values")
    sample_count = int(cross_mse.shape[0])

    if dataset_factory is None:
        from ..data import build_dataset

        dataset_factory = build_dataset
    dataset = dataset_factory(
        "cifar10", dataset_root_path, train=False, download=False
    )
    if len(dataset) != sample_count:
        raise ValueError(
            "CIFAR-10 test-set length does not match seeded_cross_mse: "
            f"{len(dataset)} != {sample_count}"
        )
    targets = _dataset_targets(dataset)
    if not np.all((targets >= 0) & (targets < 10)):
        raise ValueError("CIFAR-10 test labels must be integers in [0, 9]")
    labels_hash = _array_sha256(targets, "<i8")

    manifest = _read_json(manifest_path)
    state_order = _validate_baseline_manifest(
        manifest, sample_count=sample_count, labels_hash=labels_hash
    )
    diagonal_mse = np.diagonal(cross_mse, axis1=1, axis2=2)
    diagonal_mean_mse = diagonal_mse.mean(axis=1, dtype=np.float64)
    classes = select_quantile_samples(targets, diagonal_mean_mse)

    fixed_states = {
        role: {
            "index": state_index,
            "state_id": state_order[state_index]["state_id"],
        }
        for role, state_index in FIXED_STATE_IDS.items()
    }
    return {
        "schema": REGISTRY_SCHEMA,
        "dataset": {
            "name": "cifar10",
            "split": "test",
            "root": str(dataset_root_path),
            "sample_count": sample_count,
            "official_test_index_definition": (
                "zero-based index in torchvision.datasets.CIFAR10(train=False)"
            ),
        },
        "selection_algorithm": {
            "name": "baseline-diagonal-mean-per-class-quantiles-v1",
            "candidate_model_metrics_used": False,
            "score": (
                "per-image arithmetic mean of the ten diagonal entries "
                "seeded_cross_mse[:, s, s] from the designated baseline"
            ),
            "quantiles": [quantile for _, quantile in QUANTILES],
            "empirical_quantile_method": "numpy.quantile(method='linear')",
            "deterministic_order": "ascending (baseline MSE, official test-set ID)",
            "selection_rule": (
                "within each class choose the sample whose baseline diagonal-mean "
                "MSE is closest to the empirical quantile; equal distances and "
                "equal MSE values choose the lower official test-set ID"
            ),
        },
        "inputs": {
            "per_image_metrics": {
                "path": str(metrics_path),
                "sha256": _sha256_file(metrics_path),
                "seeded_cross_mse_key": {
                    "shape": list(cross_mse_stored.shape),
                    "stored_dtype": str(cross_mse_stored.dtype),
                    "canonical_float32_sha256": _array_sha256(
                        cross_mse_stored, "<f4"
                    ),
                },
            },
            "evaluation_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256_file(manifest_path),
                "schema": manifest.get("schema"),
            },
            "cifar10_test_labels": {
                "canonical_int64_sha256": labels_hash,
            },
        },
        "state_order": state_order,
        "fixed_state_indices": dict(FIXED_STATE_IDS),
        "fixed_states": fixed_states,
        "classes": classes,
    }


def write_registry_exclusive(registry: dict[str, Any], output: str | Path) -> Path:
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(registry, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze CIFAR-10 presentation samples from one designated baseline "
            "per_image_metrics.npz without consulting candidate models."
        )
    )
    parser.add_argument("--per-image-metrics", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.expanduser().exists():
        raise FileExistsError(args.output.expanduser())
    registry = build_fixed_sample_registry(
        args.per_image_metrics,
        args.dataset_root,
    )
    output_path = write_registry_exclusive(registry, args.output)
    print(json.dumps({"output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
