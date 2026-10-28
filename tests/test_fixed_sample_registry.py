from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lightwave_seeded_vqvae.cli.build_fixed_sample_registry import (
    EXPECTED_STATE_IDS,
    build_fixed_sample_registry,
    select_quantile_samples,
    validate_state_order,
    write_registry_exclusive,
)


def _state_order() -> list[dict[str, int | str]]:
    return [
        {"index": index, "state_id": state_id}
        for index, state_id in enumerate(EXPECTED_STATE_IDS)
    ]


class _FakeCIFAR10:
    def __init__(self, targets: np.ndarray) -> None:
        self.targets = targets.tolist()

    def __len__(self) -> int:
        return len(self.targets)


def test_quantile_selection_is_deterministic_and_breaks_ties_by_lower_id() -> None:
    targets = np.zeros(4, dtype=np.int64)
    scores = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

    selected = select_quantile_samples(targets, scores, class_count=1)[0]["quantiles"]

    assert selected["q25"]["sample_id"] == 1
    assert selected["q50"]["sample_id"] == 1
    assert selected["q75"]["sample_id"] == 2
    assert selected["q50"]["baseline_empirical_quantile_mse"] == 1.5
    assert selected["q50"]["selected_sample_baseline_diagonal_mean_mse"] == 1.0


def test_build_registry_uses_only_baseline_diagonal_scores_and_validates_inputs(
    tmp_path: Path,
) -> None:
    targets = np.repeat(np.arange(10, dtype=np.int64), 4)
    sample_count = int(targets.size)
    cross_mse = np.full((sample_count, 10, 10), 99.0, dtype=np.float32)
    for sample_id in range(sample_count):
        within_class = sample_id % 4
        for state in range(10):
            cross_mse[sample_id, state, state] = float(within_class + state * 0.01)

    metrics_path = tmp_path / "per_image_metrics.npz"
    np.savez(metrics_path, seeded_cross_mse=cross_mse, unrelated_candidate_mse=-cross_mse)
    labels_hash = hashlib.sha256(
        np.ascontiguousarray(targets, dtype="<i8").tobytes()
    ).hexdigest()
    manifest = {
        "schema": "lightwave-seeded-vqvae-final-evaluation-v2",
        "scope": "final",
        "dataset": {
            "name": "cifar10",
            "processed_test_samples": sample_count,
            "processed_sample_count": sample_count,
            "processed_labels_int64_sha256": labels_hash,
            "test_truncated": False,
        },
        "state_order": _state_order(),
    }
    manifest_path = tmp_path / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    calls: list[tuple[str, Path, bool, bool]] = []

    def fake_dataset_factory(
        name: str, root: Path, *, train: bool, download: bool
    ) -> _FakeCIFAR10:
        calls.append((name, root, train, download))
        return _FakeCIFAR10(targets)

    registry = build_fixed_sample_registry(
        metrics_path,
        tmp_path / "cifar-root",
        dataset_factory=fake_dataset_factory,
    )

    assert calls == [("cifar10", (tmp_path / "cifar-root").resolve(), False, False)]
    assert registry["selection_algorithm"]["candidate_model_metrics_used"] is False
    assert registry["fixed_state_indices"] == {
        "sender_state": 2,
        "wrong_cross_pump_same_power": 7,
        "wrong_same_pump_other_power": 4,
        "representative_wrong_state": 1,
    }
    assert registry["fixed_states"] == {
        "sender_state": {"index": 2, "state_id": "405nm_10mW"},
        "wrong_cross_pump_same_power": {"index": 7, "state_id": "640nm_10mW"},
        "wrong_same_pump_other_power": {"index": 4, "state_id": "405nm_20mW"},
        "representative_wrong_state": {"index": 1, "state_id": "405nm_5mW"},
    }
    assert len(registry["classes"]) == 10
    first_class = registry["classes"][0]
    assert first_class["quantiles"]["q25"]["sample_id"] == 1
    assert first_class["quantiles"]["q50"]["sample_id"] == 1
    assert first_class["quantiles"]["q75"]["sample_id"] == 2
    assert registry["inputs"]["per_image_metrics"]["sha256"] == hashlib.sha256(
        metrics_path.read_bytes()
    ).hexdigest()
    assert registry["inputs"]["evaluation_manifest"]["sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()


def test_state_order_rejects_swapped_physical_mapping() -> None:
    records = _state_order()
    records[2], records[7] = records[7], records[2]
    with pytest.raises(ValueError, match="ten-state mapping mismatch"):
        validate_state_order(records)


def test_registry_writer_never_overwrites_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "registry.json"
    write_registry_exclusive({"first": True}, output)
    first_bytes = output.read_bytes()

    with pytest.raises(FileExistsError):
        write_registry_exclusive({"second": True}, output)

    assert output.read_bytes() == first_bytes
