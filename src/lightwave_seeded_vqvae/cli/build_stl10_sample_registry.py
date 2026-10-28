from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .build_fixed_sample_registry import write_registry_exclusive
REGISTRY_SCHEMA = "lightwave-seeded-vqvae-stl10-sample-registry-v1"
QUANTILES = (("q25", 0.25), ("q50", 0.50), ("q75", 0.75))
EXPECTED_TEST_SAMPLES = 8_000
EXPECTED_CLASSES = 10
DEFAULT_PHYSICAL_ROLES = {
    "sender_state": {"pump_nm": 405, "power_mw": 10},
    "wrong_cross_pump_same_power": {"pump_nm": 640, "power_mw": 10},
    "wrong_same_pump_other_power": {"pump_nm": 405, "power_mw": 20},
}
STL10_CLASS_NAMES = (
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
)


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


def state_records_from_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read ten unique physical states without assuming a numeric ordering."""

    states = manifest.get("states")
    if not isinstance(states, list) or len(states) != 10:
        raise ValueError("codebook manifest must contain exactly ten states")
    records: list[dict[str, Any]] = []
    for expected_index, state in enumerate(states):
        if not isinstance(state, dict):
            raise ValueError(f"codebook state {expected_index} must be an object")
        try:
            index = int(state["index"])
            pump_nm = int(state["pump_nm"])
            power_mw = int(state["power_mw"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "every codebook state must contain integer index, pump_nm, and power_mw"
            ) from error
        state_id = str(state.get("state_id", ""))
        if index != expected_index or not state_id:
            raise ValueError("codebook states must use canonical indices and non-empty IDs")
        records.append(
            {
                "index": index,
                "state_id": state_id,
                "pump_nm": pump_nm,
                "power_mw": power_mw,
            }
        )
    if len({record["state_id"] for record in records}) != 10:
        raise ValueError("codebook state IDs must be unique")
    if len({(record["pump_nm"], record["power_mw"]) for record in records}) != 10:
        raise ValueError("codebook pump/power pairs must be unique")
    return records


def select_representative_wrong_pair(
    mse_matrix: np.ndarray,
    *,
    send_state: int,
    excluded_pairs: Sequence[tuple[int, int]] = (),
) -> dict[str, float | int]:
    """Select the eligible receiver closest to all nine wrong-pair median MSE."""

    mse = np.asarray(mse_matrix, dtype=np.float64)
    if mse.shape != (10, 10) or not np.isfinite(mse).all() or np.any(mse <= 0.0):
        raise ValueError("mse_matrix must be finite positive shape (10, 10)")
    if not 0 <= send_state < 10:
        raise ValueError("send_state must lie in [0, 9]")
    excluded = {(int(send), int(receive)) for send, receive in excluded_pairs}
    wrong_receivers = [state for state in range(10) if state != send_state]
    target = float(np.median(mse[send_state, wrong_receivers]))
    candidates = [
        receive
        for receive in wrong_receivers
        if (send_state, receive) not in excluded
    ]
    if not candidates:
        raise ValueError("no eligible representative wrong receiver remains")
    receiver = min(
        candidates,
        key=lambda value: (abs(float(mse[send_state, value]) - target), value),
    )
    return {
        "send_state": send_state,
        "receiver_state": receiver,
        "pooled_mse": float(mse[send_state, receiver]),
        "row_relative_mse": float(
            mse[send_state, receiver] / mse[send_state, send_state]
        ),
        "row_wrong_median_pooled_mse": target,
        "row_wrong_pair_count": len(wrong_receivers),
        "eligible_pair_count": len(candidates),
    }


def dataset_labels(dataset: Any) -> np.ndarray:
    """Return zero-based STL-10 labels without decoding image data unnecessarily."""

    raw_labels = getattr(dataset, "labels", None)
    if raw_labels is None:
        raw_labels = getattr(dataset, "targets", None)
    if raw_labels is None:
        raw_labels = [dataset[index][1] for index in range(len(dataset))]
    labels = np.asarray(raw_labels, dtype=np.int64)
    if labels.ndim != 1 or labels.shape[0] != len(dataset):
        raise ValueError("STL-10 test labels must be a one-dimensional dataset-length array")
    if not np.all((labels >= 0) & (labels < EXPECTED_CLASSES)):
        raise ValueError("STL-10 test labels must be zero-based integers in [0, 9]")
    return labels


def select_quantile_samples(
    labels: Sequence[int] | np.ndarray,
    correct_mean_mse: Sequence[float] | np.ndarray,
    *,
    class_names: Sequence[str] = STL10_CLASS_NAMES,
) -> list[dict[str, Any]]:
    """Select q25/q50/q75 independently within every official STL-10 class.

    The score for one image is the arithmetic mean of its ten correct-state MSE
    values.  The sample nearest NumPy's linear empirical quantile is selected;
    an exact distance tie is resolved by the lower official test index.
    """

    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(correct_mean_mse, dtype=np.float64)
    if label_array.ndim != 1 or score_array.ndim != 1:
        raise ValueError("labels and correct_mean_mse must be one-dimensional")
    if label_array.shape != score_array.shape:
        raise ValueError("labels and correct_mean_mse must have equal length")
    if len(class_names) != EXPECTED_CLASSES:
        raise ValueError("STL-10 must provide exactly ten class names")
    if not np.isfinite(score_array).all() or np.any(score_array < 0.0):
        raise ValueError("correct_mean_mse must contain finite non-negative values")

    selections: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(class_names):
        candidate_ids = np.flatnonzero(label_array == class_id).astype(np.int64)
        if candidate_ids.size == 0:
            raise ValueError(f"STL-10 test labels contain no sample for class {class_id}")
        candidate_scores = score_array[candidate_ids]
        quantile_records: dict[str, Any] = {}
        for quantile_name, quantile in QUANTILES:
            target = float(np.quantile(candidate_scores, quantile, method="linear"))
            distances = np.abs(candidate_scores - target)
            minimum = float(distances.min())
            tied_ids = candidate_ids[distances == minimum]
            sample_id = int(tied_ids.min())
            quantile_records[quantile_name] = {
                "quantile": quantile,
                "sample_id": sample_id,
                "empirical_quantile_correct_mean_mse": target,
                "selected_sample_correct_mean_mse": float(score_array[sample_id]),
            }
        selections.append(
            {
                "class_id": class_id,
                "class_name": str(class_name),
                "candidate_count": int(candidate_ids.size),
                "quantiles": quantile_records,
            }
        )
    return selections


def _unique_state_for_physics(
    records: Sequence[Mapping[str, Any]], *, pump_nm: int, power_mw: int
) -> dict[str, Any]:
    matches = [
        dict(record)
        for record in records
        if int(record["pump_nm"]) == pump_nm and int(record["power_mw"]) == power_mw
    ]
    if len(matches) != 1:
        raise ValueError(
            "codebook manifest must contain exactly one state for "
            f"{pump_nm} nm / {power_mw} mW; found {len(matches)}"
        )
    return matches[0]


def resolve_controlled_states(
    codebook_manifest: Mapping[str, Any], mse_matrix: np.ndarray
) -> dict[str, Any]:
    """Resolve presentation state indices from physical manifest fields."""

    records = state_records_from_manifest(dict(codebook_manifest))
    fixed: dict[str, dict[str, Any]] = {}
    for role, physical in DEFAULT_PHYSICAL_ROLES.items():
        fixed[role] = _unique_state_for_physics(records, **physical)

    sender = int(fixed["sender_state"]["index"])
    cross = int(fixed["wrong_cross_pump_same_power"]["index"])
    same = int(fixed["wrong_same_pump_other_power"]["index"])
    representative = select_representative_wrong_pair(
        np.asarray(mse_matrix, dtype=np.float64),
        send_state=sender,
        excluded_pairs={(sender, cross), (sender, same)},
    )
    representative_index = int(representative["receiver_state"])
    fixed["representative_median_wrong"] = dict(records[representative_index])

    return {
        "records": records,
        "fixed_states": fixed,
        "fixed_state_indices": {
            role: int(record["index"]) for role, record in fixed.items()
        },
        "representative_selection": representative,
    }


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
        raise ValueError("STL-10 registry requires a formal full-test evaluation")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or _normalized_dataset_name(
        dataset.get("name")
    ) != "stl10":
        raise ValueError("evaluation manifest dataset must be STL-10")
    if bool(dataset.get("test_truncated", True)):
        raise ValueError("evaluation manifest must cover the full STL-10 test set")
    processed = dataset.get("processed_test_samples", dataset.get("processed_sample_count"))
    expected = dataset.get("expected_test_samples", EXPECTED_TEST_SAMPLES)
    if int(processed if processed is not None else -1) != sample_count:
        raise ValueError("evaluation manifest sample count differs from per-image metrics")
    if int(expected) != EXPECTED_TEST_SAMPLES or sample_count != EXPECTED_TEST_SAMPLES:
        raise ValueError("formal STL-10 evaluation must contain all 8,000 test images")
    recorded_labels_hash = dataset.get("processed_labels_int64_sha256")
    if recorded_labels_hash != labels_hash:
        raise ValueError("local STL-10 labels differ from the formal evaluation")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or outputs.get("per_image_metrics.npz") != _sha256_file(
        metrics_path
    ):
        raise ValueError("formal per-image metrics hash mismatch")
    codebook_input = manifest.get("inputs", {}).get("codebook", {})
    if codebook_input.get("manifest_sha256") != _sha256_file(codebook_manifest_path):
        raise ValueError("formal evaluation and codebook manifest hashes differ")
    expected_state_order = [
        {"index": int(record["index"]), "state_id": str(record["state_id"])}
        for record in state_records_from_manifest(dict(codebook_manifest))
    ]
    if manifest.get("state_order") != expected_state_order:
        raise ValueError("formal evaluation and codebook state order differ")


def _default_dataset_factory(*, root: str, split: str, download: bool) -> Any:
    from torchvision.datasets import STL10

    return STL10(root=root, split=split, download=download)


def build_stl10_sample_registry(
    per_image_metrics: str | Path,
    dataset_root: str | Path,
    codebook_manifest_path: str | Path,
    *,
    dataset_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
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
    if cross_mse_stored.shape != (EXPECTED_TEST_SAMPLES, 10, 10):
        raise ValueError("seeded_cross_mse must have shape (8000, 10, 10)")
    if not np.issubdtype(cross_mse_stored.dtype, np.floating):
        raise ValueError("seeded_cross_mse must use a floating-point dtype")
    cross_mse = np.asarray(cross_mse_stored, dtype=np.float64)
    if not np.isfinite(cross_mse).all() or np.any(cross_mse < 0.0):
        raise ValueError("seeded_cross_mse must contain finite non-negative values")

    factory = dataset_factory or _default_dataset_factory
    dataset = factory(root=str(root_path), split="test", download=False)
    if len(dataset) != EXPECTED_TEST_SAMPLES:
        raise ValueError("official STL-10 test split must contain 8,000 images")
    labels = dataset_labels(dataset)
    labels_hash = _array_sha256(labels, "<i8")
    raw_class_names = getattr(dataset, "classes", None) or STL10_CLASS_NAMES
    class_names = tuple(str(value) for value in raw_class_names)
    if len(class_names) != EXPECTED_CLASSES:
        raise ValueError("official STL-10 test split must contain ten classes")
    if class_names != STL10_CLASS_NAMES:
        raise ValueError(
            "STL-10 class names/order differ from the official torchvision contract"
        )

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
    classes = select_quantile_samples(
        labels, diagonal_mean_mse, class_names=class_names
    )
    state_mapping = resolve_controlled_states(codebook_manifest, cross_mse.mean(axis=0))

    return {
        "schema": REGISTRY_SCHEMA,
        "dataset": {
            "name": "stl10",
            "split": "test",
            "root": str(root_path),
            "sample_count": len(dataset),
            "image_shape_chw": [3, 96, 96],
            "official_test_index_definition": (
                "zero-based index in torchvision.datasets.STL10(split='test')"
            ),
            "class_names": list(class_names),
        },
        "selection_algorithm": {
            "name": "stl10-correct-state-mean-per-class-quantiles-v1",
            "score": (
                "per-image arithmetic mean of the ten correct-state entries "
                "seeded_cross_mse[:, s, s]"
            ),
            "quantiles": [quantile for _, quantile in QUANTILES],
            "empirical_quantile_method": "numpy.quantile(method='linear')",
            "selection_rule": (
                "within each official class choose the closest score to the "
                "empirical quantile; exact ties choose the lower official test ID"
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
            "stl10_test_labels": {"canonical_int64_sha256": labels_hash},
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
            "Freeze STL-10 q25/q50/q75 PPT samples from the formal Stage C "
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
    registry = build_stl10_sample_registry(
        args.per_image_metrics,
        args.dataset_root,
        args.codebook_manifest,
    )
    output = write_registry_exclusive(registry, args.output)
    print(json.dumps({"output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
