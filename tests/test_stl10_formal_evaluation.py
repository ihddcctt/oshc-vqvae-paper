from __future__ import annotations

import copy
import csv
import math
from pathlib import Path

import numpy as np
import pytest

from lightwave_seeded_vqvae.cli.evaluate_final import (
    _compatibility_report,
    _lpips_enabled,
    _nominal_bits,
    _write_metric_confidence_intervals_csv,
    _write_physical_mismatch_pairs_csv,
    _write_physical_mismatch_summary_csv,
    cross_state_metric_confidence_intervals,
    physical_mismatch_groups_from_manifest,
    summarize_physical_mismatch_groups,
)
from lightwave_seeded_vqvae.evaluation import (
    image_metric_bootstrap,
    paired_image_branch_bootstrap,
    summarize_cross_state,
)


def _physical_manifest() -> dict[str, object]:
    states = []
    for pump_nm in (405, 640):
        for power_mw in (0, 5, 10, 15, 20):
            states.append(
                {
                    "index": len(states),
                    "state_id": f"{pump_nm}nm_{power_mw}mW",
                    "pump_nm": pump_nm,
                    "power_mw": power_mw,
                }
            )
    return {"states": states}


def _matched_stl10_config() -> dict[str, object]:
    return {
        "experiment": {"seed": 20260907},
        "dataset": {
            "name": "stl10",
            "root": "/datasets/STL10",
            "validation_size": 5000,
            "train_split": "unlabeled",
            "validation_split": "train",
            "test_split": "test",
            "train_horizontal_flip_probability": 0.5,
        },
        "model": {
            "encoder_backbone": "resnet18",
            "latent_dim": 64,
            "base_codebook_size": 128,
            "residual_codebook_size": 128,
            "pretrained_encoder": True,
            "imagenet_weights": "IMAGENET1K_V1",
            "normalize_latents": True,
            "loss_reference_dim": 64,
            "decoder_blocks_per_stage": 1,
            "decoder_blocks_by_stage": [3, 2, 2],
        },
        "training": {"reconstruction_loss": "mse", "perceptual_weight": 0.0},
    }


@pytest.mark.parametrize(
    ("field", "mismatched_value"),
    (
        ("train_split", "train+unlabeled"),
        ("validation_split", "unlabeled"),
        ("test_split", "train"),
        ("train_horizontal_flip_probability", 0.0),
    ),
)
def test_stl10_compatibility_rejects_split_or_augmentation_mismatch(
    field: str, mismatched_value: object
) -> None:
    seeded = _matched_stl10_config()
    software = copy.deepcopy(seeded)
    software["dataset"][field] = mismatched_value
    with pytest.raises(ValueError, match=f"dataset.{field}"):
        _compatibility_report(seeded, software)


def test_stl10_compatibility_records_split_and_augmentation_fields() -> None:
    config = _matched_stl10_config()
    report = _compatibility_report(config, copy.deepcopy(config))
    for field in (
        "dataset.train_split",
        "dataset.validation_split",
        "dataset.test_split",
        "dataset.train_horizontal_flip_probability",
    ):
        assert field in report["matched_fields"]


def test_lpips_auto_includes_both_natural_image_datasets() -> None:
    assert _lpips_enabled("auto", "cifar10")
    assert _lpips_enabled("auto", "CIFAR-10")
    assert _lpips_enabled("auto", "stl10")
    assert _lpips_enabled("auto", "STL-10")
    assert not _lpips_enabled("auto", "mnist")
    assert _lpips_enabled("on", "mnist")
    assert not _lpips_enabled("off", "stl10")


def test_physical_groups_cover_every_ordered_wrong_pair_exactly_once() -> None:
    grouping = physical_mismatch_groups_from_manifest(_physical_manifest())
    exclusive = grouping["exclusive_groups"]
    assert grouping["ordered_wrong_pair_count"] == 90
    assert {name: row["pair_count"] for name, row in exclusive.items()} == {
        "cross_pump_same_power": 10,
        "same_pump_wrong_power": 40,
        "cross_pump_wrong_power": 40,
    }
    assert {
        name: row["pair_count"]
        for name, row in grouping["marginal_groups"].items()
    } == {"all_cross_pump": 50, "all_wrong_power": 80}

    pairs = [
        (pair["send_state_index"], pair["receive_state_index"])
        for group in exclusive.values()
        for pair in group["pairs"]
    ]
    assert len(pairs) == len(set(pairs)) == 90
    assert set(pairs) == {
        (send, receive)
        for send in range(10)
        for receive in range(10)
        if send != receive
    }


def test_physical_group_summary_uses_image_unit_and_sender_correct_pairing() -> None:
    sample_count = 12
    mse = np.full((sample_count, 10, 10), np.nan, dtype=np.float64)
    ssim = np.full_like(mse, np.nan)
    lpips = np.full_like(mse, np.nan)
    for state in range(10):
        mse[:, state, state] = 0.01
        ssim[:, state, state] = 0.90
        lpips[:, state, state] = 0.10

    grouping = physical_mismatch_groups_from_manifest(_physical_manifest())
    group_values = {
        "cross_pump_same_power": (0.04, 0.60, 0.30),
        "same_pump_wrong_power": (0.09, 0.40, 0.50),
        "cross_pump_wrong_power": (0.16, 0.20, 0.70),
    }
    for group_name, (group_mse, group_ssim, group_lpips) in group_values.items():
        for pair in grouping["exclusive_groups"][group_name]["pairs"]:
            send = pair["send_state_index"]
            receive = pair["receive_state_index"]
            mse[:, send, receive] = group_mse
            ssim[:, send, receive] = group_ssim
            lpips[:, send, receive] = group_lpips

    summary = summarize_physical_mismatch_groups(
        mse,
        ssim,
        grouping=grouping,
        pixels_per_image=3 * 96 * 96,
        per_image_lpips=lpips,
        bootstrap_seed=17,
        bootstrap_replicates=40,
    )
    cross_pump = summary["exclusive_groups"]["cross_pump_same_power"]
    assert cross_pump["wrong_metrics"]["mse"] == pytest.approx(0.04)
    assert cross_pump["wrong_metrics"]["rmse"] == pytest.approx(0.20)
    comparison = cross_pump["paired_against_sender_correct"]["comparisons"]
    assert comparison["wrong_to_correct_mse_ratio"]["estimate"] == pytest.approx(4.0)
    assert comparison["correct_minus_wrong_ssim"]["estimate"] == pytest.approx(0.30)
    assert comparison["wrong_minus_correct_lpips"]["estimate"] == pytest.approx(0.20)
    assert comparison["wrong_to_correct_mse_ratio"][
        "ci95_percentile_lower"
    ] == pytest.approx(4.0)

    all_cross_pump = summary["marginal_groups"]["all_cross_pump"]
    expected_mse = (10 * 0.04 + 40 * 0.16) / 50
    assert all_cross_pump["wrong_metrics"]["mse"] == pytest.approx(expected_mse)
    assert all_cross_pump["paired_against_sender_correct"]["comparisons"][
        "wrong_to_correct_mse_ratio"
    ]["estimate"] == pytest.approx(expected_mse / 0.01)


def test_metric_bootstrap_and_cross_state_intervals_report_all_metrics() -> None:
    mse = np.asarray([[0.01, 0.03], [0.05, 0.07]], dtype=np.float64)
    ssim = np.asarray([[0.9, 0.8], [0.7, 0.6]], dtype=np.float64)
    lpips = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    summary = image_metric_bootstrap(
        mse,
        ssim,
        per_image_lpips=lpips,
        seed=23,
        replicates=50,
    )
    assert summary["sampling_unit"] == "test_image"
    assert set(summary["metrics"]) == {"mse", "rmse", "ssim_mean", "lpips_mean"}
    assert summary["metrics"]["mse"]["estimate"] == pytest.approx(0.04)
    assert summary["metrics"]["rmse"]["estimate"] == pytest.approx(0.2)
    assert summary["metrics"]["ssim_mean"]["estimate"] == pytest.approx(0.75)
    assert summary["metrics"]["lpips_mean"]["estimate"] == pytest.approx(0.25)

    cross_mse = np.full((4, 10, 10), 0.09)
    cross_ssim = np.full_like(cross_mse, 0.3)
    cross_lpips = np.full_like(cross_mse, 0.7)
    states = np.arange(10)
    cross_mse[:, states, states] = 0.01
    cross_ssim[:, states, states] = 0.9
    cross_lpips[:, states, states] = 0.1
    intervals = cross_state_metric_confidence_intervals(
        cross_mse,
        cross_ssim,
        state_labels=[f"state_{index}" for index in range(10)],
        per_image_lpips=cross_lpips,
        bootstrap_seed=29,
        bootstrap_replicates=20,
    )
    assert len(intervals["correct_by_state"]) == 10
    assert intervals["correct_pooled"]["metrics"]["mse"]["estimate"] == pytest.approx(
        0.01
    )
    assert intervals["wrong_pooled"]["metrics"]["mse"]["estimate"] == pytest.approx(
        0.09
    )


def test_paired_branch_bootstrap_keeps_image_pairing() -> None:
    correct_mse = np.asarray([0.01, 0.04])
    wrong_mse = np.asarray([0.04, 0.16])
    correct_ssim = np.asarray([0.9, 0.8])
    wrong_ssim = np.asarray([0.5, 0.4])
    correct_lpips = np.asarray([0.1, 0.2])
    wrong_lpips = np.asarray([0.4, 0.5])
    summary = paired_image_branch_bootstrap(
        correct_mse,
        wrong_mse,
        correct_ssim=correct_ssim,
        wrong_ssim=wrong_ssim,
        correct_lpips=correct_lpips,
        wrong_lpips=wrong_lpips,
        seed=31,
        replicates=50,
    )
    comparisons = summary["comparisons"]
    assert comparisons["wrong_to_correct_mse_ratio"]["estimate"] == pytest.approx(4.0)
    assert comparisons["wrong_to_correct_rmse_ratio"]["estimate"] == pytest.approx(2.0)
    assert comparisons["correct_minus_wrong_ssim"]["estimate"] == pytest.approx(0.4)
    assert comparisons["wrong_minus_correct_lpips"]["estimate"] == pytest.approx(0.3)


def test_stl10_k128_nominal_rate_is_8064_bits_and_27_point_43_to_one() -> None:
    rate = _nominal_bits(
        128,
        128,
        latent_shape=[24, 24],
        image_shape=[3, 96, 96],
    )
    assert rate["base_index_bits_per_token"] == 7
    assert rate["residual_index_bits_per_token"] == 7
    assert rate["total_index_bits_per_token"] == 14
    assert rate["latent_positions_per_image"] == 576
    assert rate["nominal_index_bits_per_image"] == 8064
    assert rate["raw_reference_bits_per_image"] == 221184
    assert rate["nominal_raw_to_index_compression_ratio"] == pytest.approx(
        27.428571428571427
    )
    assert math.isclose(
        rate["nominal_raw_to_index_compression_ratio"], 27.4286, rel_tol=2e-6
    )


def test_supplementary_csv_tables_are_complete(tmp_path: Path) -> None:
    sample_count = 3
    mse = np.full((sample_count, 10, 10), 0.09, dtype=np.float64)
    ssim = np.full_like(mse, 0.3)
    lpips = np.full_like(mse, 0.7)
    states = np.arange(10)
    mse[:, states, states] = 0.01
    ssim[:, states, states] = 0.9
    lpips[:, states, states] = 0.1
    labels = [f"{pump}nm_{power}mW" for pump in (405, 640) for power in (0, 5, 10, 15, 20)]
    grouping = physical_mismatch_groups_from_manifest(_physical_manifest())
    cross = summarize_cross_state(
        mse,
        ssim,
        state_labels=labels,
        selection_monitored_offset=1,
        pixels_per_image=3 * 96 * 96,
        per_image_lpips=lpips,
    )
    physical = summarize_physical_mismatch_groups(
        mse,
        ssim,
        grouping=grouping,
        pixels_per_image=3 * 96 * 96,
        per_image_lpips=lpips,
        bootstrap_seed=37,
        bootstrap_replicates=10,
    )
    cross_ci = cross_state_metric_confidence_intervals(
        mse,
        ssim,
        state_labels=labels,
        per_image_lpips=lpips,
        bootstrap_seed=37,
        bootstrap_replicates=10,
    )
    software_ci = image_metric_bootstrap(
        np.full(sample_count, 0.02),
        np.full(sample_count, 0.8),
        per_image_lpips=np.full(sample_count, 0.2),
        seed=37,
        replicates=10,
    )

    pairs_path = tmp_path / "physical_mismatch_pairs.csv"
    physical_path = tmp_path / "physical_mismatch_summary.csv"
    confidence_path = tmp_path / "metric_confidence_intervals.csv"
    _write_physical_mismatch_pairs_csv(pairs_path, grouping, cross)
    _write_physical_mismatch_summary_csv(physical_path, physical)
    _write_metric_confidence_intervals_csv(
        confidence_path, software=software_ci, cross_state=cross_ci
    )

    with pairs_path.open(newline="", encoding="utf-8") as handle:
        pair_rows = list(csv.DictReader(handle))
    assert len(pair_rows) == 90
    assert len(
        {(row["send_state_index"], row["receive_state_index"]) for row in pair_rows}
    ) == 90
    assert {row["exclusive_group"] for row in pair_rows} == {
        "cross_pump_same_power",
        "same_pump_wrong_power",
        "cross_pump_wrong_power",
    }

    with physical_path.open(newline="", encoding="utf-8") as handle:
        physical_rows = list(csv.DictReader(handle))
    assert {row["group"] for row in physical_rows} == {
        "cross_pump_same_power",
        "same_pump_wrong_power",
        "cross_pump_wrong_power",
        "all_cross_pump",
        "all_wrong_power",
    }
    assert {row["bootstrap_seed"] for row in physical_rows} == {"37"}
    assert {row["bootstrap_replicates"] for row in physical_rows} == {"10"}
    with confidence_path.open(newline="", encoding="utf-8") as handle:
        confidence_rows = list(csv.DictReader(handle))
    assert len(
        [row for row in confidence_rows if row["population"] == "seeded_correct_state"]
    ) == 40
    assert len(
        [row for row in confidence_rows if row["population"] == "seeded_wrong_vs_correct"]
    ) == 6
