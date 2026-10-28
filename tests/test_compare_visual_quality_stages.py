from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lightwave_seeded_vqvae.cli.compare_visual_quality_stages import (
    COMPARISON_SCHEMA,
    build_parser,
    main,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pooled(mse: np.ndarray, ssim: np.ndarray, lpips: np.ndarray) -> dict:
    mean_mse = float(np.asarray(mse, dtype=np.float64).mean())
    return {
        "mse": mean_mse,
        "rmse": mean_mse**0.5,
        "ssim_mean": float(np.asarray(ssim, dtype=np.float64).mean()),
        "lpips_mean": float(np.asarray(lpips, dtype=np.float64).mean()),
    }


def _identity(order_hash: str = "c" * 64) -> dict:
    return {
        "name": "cifar10",
        "processed_images_float32_sha256": "a" * 64,
        "processed_labels_int64_sha256": "b" * 64,
        "processed_sample_order_sha256": order_hash,
        "processed_sample_count": 12,
    }


def _state_order() -> list[dict]:
    return [
        {"index": index, "state_id": f"state_{index}"}
        for index in range(10)
    ]


def _write_vq_evaluation(
    directory: Path,
    *,
    correct_mse: float,
    wrong_mse: float,
    correct_ssim: float,
    wrong_ssim: float,
    correct_lpips: float,
    wrong_lpips: float,
    residual_mse: float,
    requantized_mse: float,
    software_mse: float,
    software_ssim: float,
    software_lpips: float,
    total_parameters: int,
    residual_ssim: float = 0.45,
    residual_lpips: float = 0.48,
    requantized_ssim: float = 0.50,
    requantized_lpips: float = 0.40,
    order_hash: str = "c" * 64,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True)
    variation = np.linspace(0.9, 1.1, 12, dtype=np.float64)
    shape = (12, 10, 10)
    cross_mse = np.empty(shape, dtype=np.float64)
    cross_ssim = np.empty(shape, dtype=np.float64)
    cross_lpips = np.empty(shape, dtype=np.float64)
    cross_mse[:] = wrong_mse * variation[:, None, None]
    cross_ssim[:] = wrong_ssim - (variation[:, None, None] - 1.0) * 0.02
    cross_lpips[:] = wrong_lpips * variation[:, None, None]
    states = np.arange(10)
    cross_mse[:, states, states] = correct_mse * variation[:, None]
    cross_ssim[:, states, states] = (
        correct_ssim - (variation[:, None] - 1.0) * 0.02
    )
    cross_lpips[:, states, states] = correct_lpips * variation[:, None]

    residual_only_mse = np.repeat(
        (residual_mse * variation)[:, None], 10, axis=1
    )
    residual_only_ssim = np.repeat(
        (np.full(12, residual_ssim) - (variation - 1.0) * 0.01)[:, None],
        10,
        axis=1,
    )
    residual_only_lpips = np.repeat(
        (np.full(12, residual_lpips) * variation)[:, None], 10, axis=1
    )
    requantized_mse_values = requantized_mse * variation
    requantized_ssim_values = requantized_ssim - (variation - 1.0) * 0.01
    requantized_lpips_values = requantized_lpips * variation
    software_mse_values = software_mse * variation
    software_ssim_values = software_ssim - (variation - 1.0) * 0.02
    software_lpips_values = software_lpips * variation
    summary = {
        "schema": "synthetic-vq-evaluation",
        "scope": "final",
        "seeded": {
            "cross_state": {
                "correct_pooled": _pooled(
                    cross_mse[:, states, states],
                    cross_ssim[:, states, states],
                    cross_lpips[:, states, states],
                ),
                "wrong_pooled": _pooled(
                    cross_mse[:, ~np.eye(10, dtype=bool)],
                    cross_ssim[:, ~np.eye(10, dtype=bool)],
                    cross_lpips[:, ~np.eye(10, dtype=bool)],
                ),
            },
            "branches": {
                "transmitted_residual_only": {
                    "pooled": _pooled(
                        residual_only_mse,
                        residual_only_ssim,
                        residual_only_lpips,
                    )
                },
                "requantized_residual_only": _pooled(
                    requantized_mse_values,
                    requantized_ssim_values,
                    requantized_lpips_values,
                ),
            },
            "model_parameters": {
                "total_parameters": total_parameters,
                "trainable_parameters": total_parameters - 10,
            },
        },
        "software": {
            "overall": _pooled(
                software_mse_values,
                software_ssim_values,
                software_lpips_values,
            )
        },
    }
    summary_path = directory / "reconstruction_summary.json"
    metrics_path = directory / "per_image_metrics.npz"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    np.savez_compressed(
        metrics_path,
        seeded_cross_mse=cross_mse,
        seeded_cross_ssim=cross_ssim,
        seeded_cross_lpips=cross_lpips,
        seeded_transmitted_residual_only_mse=residual_only_mse,
        seeded_transmitted_residual_only_ssim=residual_only_ssim,
        seeded_transmitted_residual_only_lpips=residual_only_lpips,
        seeded_requantized_residual_only_mse=requantized_mse_values,
        seeded_requantized_residual_only_ssim=requantized_ssim_values,
        seeded_requantized_residual_only_lpips=requantized_lpips_values,
        software_mse=software_mse_values,
        software_ssim=software_ssim_values,
        software_lpips=software_lpips_values,
    )
    manifest = {
        "schema": "synthetic-vq-evaluation",
        "scope": "final",
        "dataset": _identity(order_hash),
        "state_order": _state_order(),
        "outputs": {
            summary_path.name: _sha256(summary_path),
            metrics_path.name: _sha256(metrics_path),
        },
    }
    (directory / "evaluation_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return summary_path, metrics_path


def _write_continuous_evaluation(
    directory: Path,
    *,
    mse_level: float,
    ssim_level: float,
    lpips_level: float,
    order_hash: str = "c" * 64,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True)
    variation = np.linspace(0.9, 1.1, 12, dtype=np.float64)
    mse = mse_level * variation
    rmse = np.sqrt(mse)
    ssim = ssim_level - (variation - 1.0) * 0.02
    lpips = lpips_level * variation
    summary = {
        "schema": "synthetic-continuous-evaluation",
        "scope": "final",
        "formal": True,
        "continuous": {"overall": _pooled(mse, ssim, lpips)},
    }
    summary_path = directory / "reconstruction_summary.json"
    metrics_path = directory / "per_image_metrics.npz"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    np.savez_compressed(
        metrics_path,
        sample_indices=np.arange(12, dtype=np.int64),
        labels=np.arange(12, dtype=np.int64) % 10,
        mse=mse,
        rmse=rmse,
        ssim=ssim,
        lpips=lpips,
    )
    manifest = {
        "schema": "synthetic-continuous-evaluation",
        "scope": "final",
        "dataset": _identity(order_hash),
        "outputs": {
            "reconstruction_summary": {
                "path": str(summary_path),
                "sha256": _sha256(summary_path),
            },
            "per_image_metrics": {
                "path": str(metrics_path),
                "sha256": _sha256(metrics_path),
            },
        },
    }
    (directory / "evaluation_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return summary_path, metrics_path


def _write_runtime(
    path: Path, *, total_parameters: int, elapsed_seconds: float
) -> Path:
    payload = {
        "elapsed_seconds": elapsed_seconds,
        "completed_epochs": 10,
        "seconds_by_epoch": [elapsed_seconds / 10.0] * 10,
        "parameter_counts": {
            "total": total_parameters,
            "trainable": total_parameters - 10,
            "encoder_total": 60,
            "decoder_total": total_parameters - 80,
            "quantizer_trainable": 10,
        },
        "peak_gpu_memory_bytes": 1024,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _inputs(
    tmp_path: Path,
    *,
    candidate_wrong_mse: float = 0.16,
    candidate_wrong_ssim: float = 0.25,
    candidate_wrong_lpips: float = 0.70,
    candidate_residual_mse: float = 0.08,
    candidate_residual_ssim: float = 0.45,
    candidate_residual_lpips: float = 0.48,
    candidate_requantized_mse: float = 0.06,
    candidate_requantized_ssim: float = 0.55,
    candidate_requantized_lpips: float = 0.35,
    candidate_order_hash: str = "c" * 64,
    with_continuous: bool = True,
) -> list[str]:
    baseline_summary, baseline_npz = _write_vq_evaluation(
        tmp_path / "baseline_vq",
        correct_mse=0.04,
        wrong_mse=0.16,
        correct_ssim=0.75,
        wrong_ssim=0.25,
        correct_lpips=0.20,
        wrong_lpips=0.70,
        residual_mse=0.12,
        requantized_mse=0.10,
        requantized_ssim=0.50,
        requantized_lpips=0.40,
        software_mse=0.03,
        software_ssim=0.80,
        software_lpips=0.15,
        total_parameters=100,
    )
    candidate_summary, candidate_npz = _write_vq_evaluation(
        tmp_path / "candidate_vq",
        correct_mse=0.025,
        wrong_mse=candidate_wrong_mse,
        correct_ssim=0.84,
        wrong_ssim=candidate_wrong_ssim,
        correct_lpips=0.12,
        wrong_lpips=candidate_wrong_lpips,
        residual_mse=candidate_residual_mse,
        residual_ssim=candidate_residual_ssim,
        residual_lpips=candidate_residual_lpips,
        requantized_mse=candidate_requantized_mse,
        requantized_ssim=candidate_requantized_ssim,
        requantized_lpips=candidate_requantized_lpips,
        software_mse=0.018,
        software_ssim=0.88,
        software_lpips=0.09,
        total_parameters=120,
        order_hash=candidate_order_hash,
    )
    baseline_runtime = _write_runtime(
        tmp_path / "baseline_runtime.json",
        total_parameters=100,
        elapsed_seconds=100.0,
    )
    candidate_runtime = _write_runtime(
        tmp_path / "candidate_runtime.json",
        total_parameters=120,
        elapsed_seconds=140.0,
    )
    arguments = [
        "--baseline-vq-summary",
        str(baseline_summary),
        "--baseline-vq-npz",
        str(baseline_npz),
        "--candidate-vq-summary",
        str(candidate_summary),
        "--candidate-vq-npz",
        str(candidate_npz),
        "--baseline-runtime",
        str(baseline_runtime),
        "--candidate-runtime",
        str(candidate_runtime),
        "--output-dir",
        str(tmp_path / "comparison"),
        "--bootstrap-seed",
        "91",
        "--bootstrap-replicates",
        "200",
    ]
    if with_continuous:
        baseline_continuous_summary, baseline_continuous_npz = (
            _write_continuous_evaluation(
                tmp_path / "baseline_continuous",
                mse_level=0.03,
                ssim_level=0.80,
                lpips_level=0.15,
            )
        )
        candidate_continuous_summary, candidate_continuous_npz = (
            _write_continuous_evaluation(
                tmp_path / "candidate_continuous",
                mse_level=0.02,
                ssim_level=0.87,
                lpips_level=0.10,
            )
        )
        arguments.extend(
            [
                "--baseline-continuous-summary",
                str(baseline_continuous_summary),
                "--baseline-continuous-npz",
                str(baseline_continuous_npz),
                "--candidate-continuous-summary",
                str(candidate_continuous_summary),
                "--candidate-continuous-npz",
                str(candidate_continuous_npz),
            ]
        )
    return arguments


def test_comparison_passes_numeric_gate_but_never_auto_promotes(
    tmp_path: Path,
) -> None:
    arguments = _inputs(tmp_path)

    main(arguments)

    output_dir = tmp_path / "comparison"
    payload = json.loads(
        (output_dir / "stage_comparison.json").read_text(encoding="utf-8")
    )
    csv_text = (output_dir / "stage_comparison.csv").read_text(encoding="utf-8")
    assert payload["schema"] == COMPARISON_SCHEMA
    assert payload["sample_identity"]["verified"]
    assert payload["sample_identity"]["identity"]["processed_sample_count"] == 12
    assert payload["vq_metrics"]["correct"]["mse"]["candidate"] < payload[
        "vq_metrics"
    ]["correct"]["mse"]["baseline"]
    assert payload["vq_metrics"]["wrong_all_90_ordered_pairs"]["mse"]
    assert payload["continuous_metrics"]["lpips"]["ci95_entirely_favorable"]
    assert payload["software_metrics"]["mse"]["candidate"] < payload[
        "software_metrics"
    ]["mse"]["baseline"]
    chip_gap = payload["candidate_chip_vs_software_gap"]["metrics"]
    assert chip_gap["mse"]["chip_correct"] > chip_gap["mse"]["software"]
    assert chip_gap["ssim"]["chip_correct"] < chip_gap["ssim"]["software"]
    assert chip_gap["lpips"]["chip_correct"] > chip_gap["lpips"]["software"]
    assert payload["separation"]["wrong_to_correct_mse_ratio"]["candidate"] > 1
    assert payload["r_only_closeness"]["candidate"][
        "r_only_to_full_mse_ratio"
    ] > 1.5
    assert payload["r_only_closeness"]["candidate"][
        "r_only_ssim_performance_retention"
    ] < 0.95
    assert payload["r_only_closeness"]["candidate"][
        "r_only_to_full_lpips_ratio"
    ] > 1.25
    assert payload["requantized_r_only_metrics"]["mse"]["candidate"] == (
        pytest.approx(0.06)
    )
    requantized = payload["requantized_r_only_closeness"]
    assert requantized["candidate"][
        "full_minus_requantized_r_only_psnr_db"
    ] > 3.0
    assert requantized["candidate"][
        "requantized_r_only_ssim_performance_retention"
    ] == pytest.approx(0.55 / 0.84)
    assert requantized["candidate"][
        "requantized_r_only_to_full_lpips_ratio"
    ] == pytest.approx(0.35 / 0.12)
    checks = {
        row["name"]: row["passed"]
        for row in payload["gates"]["numeric_gate"]["checks"]
    }
    for name in (
        "correct_wrong_ssim_gap_positive",
        "correct_wrong_ssim_gap_not_degraded",
        "r_only_mse_not_close_to_full",
        "r_only_ssim_not_close_to_full",
        "r_only_lpips_not_close_to_full",
        "requantized_r_only_psnr_gap_meets_training_selector",
    ):
        assert checks[name]
    assert payload["resources"]["comparison"]["parameters"]["total"][
        "absolute_change_candidate_minus_baseline"
    ] == 20
    assert payload["gates"]["numeric_gate"]["status"] == "pass"
    assert payload["gates"]["visual_artifact_gate"]["status"] == (
        "requires_manual_review"
    )
    assert payload["gates"]["promotion_decision"]["status"] == (
        "pending_manual_review"
    )
    assert not payload["gates"]["promotion_decision"]["auto_promoted"]
    assert "wrong_all_90_ordered_pairs" in csv_text
    assert "software_vq_reconstruction" in csv_text
    assert "candidate_chip_vs_software" in csv_text
    assert "requantized_r_only_reconstruction" in csv_text
    assert "requantized_r_only_closeness" in csv_text
    assert "full_minus_requantized_r_only_psnr_db" in csv_text
    assert "requires_manual_review" in csv_text

    with pytest.raises(FileExistsError, match="already exists"):
        main(arguments)


def test_fingerprint_mismatch_is_rejected_before_output(tmp_path: Path) -> None:
    arguments = _inputs(
        tmp_path,
        candidate_order_hash="d" * 64,
        with_continuous=False,
    )

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        main(arguments)

    assert not (tmp_path / "comparison").exists()


def test_failed_separation_and_r_only_rules_block_numeric_promotion(
    tmp_path: Path,
) -> None:
    arguments = _inputs(
        tmp_path,
        candidate_wrong_mse=0.03,
        candidate_wrong_ssim=0.80,
        candidate_wrong_lpips=0.13,
        candidate_residual_mse=0.027,
        candidate_residual_ssim=0.835,
        candidate_residual_lpips=0.13,
        with_continuous=False,
    )

    main(arguments)

    payload = json.loads(
        (tmp_path / "comparison" / "stage_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    failed = {
        row["name"]
        for row in payload["gates"]["numeric_gate"]["checks"]
        if not row["passed"]
    }
    assert "wrong_correct_mse_separation_not_degraded" in failed
    assert "correct_wrong_lpips_gap_not_degraded" in failed
    assert "correct_wrong_ssim_gap_not_degraded" in failed
    assert "r_only_mse_not_close_to_full" in failed
    assert "r_only_ssim_not_close_to_full" in failed
    assert "r_only_lpips_not_close_to_full" in failed
    assert payload["gates"]["numeric_gate"]["status"] == "fail"
    assert payload["gates"]["promotion_decision"]["status"] == (
        "not_promoted_numeric_gate_failed"
    )
    assert not payload["gates"]["promotion_decision"]["auto_promoted"]


def test_requantized_r_only_psnr_gap_is_a_configurable_hard_gate(
    tmp_path: Path,
) -> None:
    arguments = _inputs(
        tmp_path,
        candidate_requantized_mse=0.045,
        with_continuous=False,
    )

    main(arguments)

    payload = json.loads(
        (tmp_path / "comparison" / "stage_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    checks = {
        row["name"]: row["passed"]
        for row in payload["gates"]["numeric_gate"]["checks"]
    }
    diagnostic = payload["requantized_r_only_closeness"]
    assert diagnostic["candidate"][
        "full_minus_requantized_r_only_psnr_db"
    ] == pytest.approx(10.0 * np.log10(0.045 / 0.025))
    assert diagnostic[
        "candidate_full_minus_requantized_r_only_psnr_db_bootstrap_ci95"
    ]["lower"] < 3.0
    assert not checks["requantized_r_only_psnr_gap_meets_training_selector"]
    assert payload["gates"]["numeric_gate"]["status"] == "fail"

    relaxed = list(arguments)
    relaxed[relaxed.index(str(tmp_path / "comparison"))] = str(
        tmp_path / "comparison_relaxed"
    )
    relaxed.extend(["--min-requantized-r-only-psnr-gap-db", "2.0"])
    main(relaxed)
    relaxed_payload = json.loads(
        (tmp_path / "comparison_relaxed" / "stage_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    relaxed_checks = {
        row["name"]: row["passed"]
        for row in relaxed_payload["gates"]["numeric_gate"]["checks"]
    }
    assert relaxed_checks[
        "requantized_r_only_psnr_gap_meets_training_selector"
    ]
    assert relaxed_payload["gates"]["numeric_gate"]["status"] == "pass"


def test_parser_rejects_invalid_gate_thresholds() -> None:
    required = [
        "--baseline-vq-summary",
        "a.json",
        "--baseline-vq-npz",
        "a.npz",
        "--candidate-vq-summary",
        "b.json",
        "--candidate-vq-npz",
        "b.npz",
        "--baseline-runtime",
        "a-runtime.json",
        "--candidate-runtime",
        "b-runtime.json",
        "--output-dir",
        "out",
    ]
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            required + ["--max-separation-degradation-fraction", "1"]
        )
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            required + ["--min-r-only-to-full-mse-ratio", "0.9"]
        )
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            required + ["--max-r-only-ssim-retention", "1.1"]
        )
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            required + ["--min-r-only-to-full-lpips-ratio", "0.9"]
        )
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            required + ["--min-requantized-r-only-psnr-gap-db", "-0.1"]
        )
