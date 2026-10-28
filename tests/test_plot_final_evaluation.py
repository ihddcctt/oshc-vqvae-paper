from __future__ import annotations

import csv
import json
from pathlib import Path

from lightwave_seeded_vqvae.cli.plot_final_evaluation import (
    FIGURE_SCHEMA,
    compact_state_label,
    generate_figures,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compact_state_label() -> None:
    assert compact_state_label("405nm_10mW") == "405/100"
    assert compact_state_label("640nm_20mW") == "640/200"
    assert compact_state_label("405nm_0mW") == "405/0"


def test_generate_three_png_and_pdf_figures_from_synthetic_evaluation(
    tmp_path: Path,
) -> None:
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    labels = [f"{405 if index < 5 else 640}nm_{(index % 5) * 5}mW" for index in range(10)]
    matrix = [
        [20.0 if row == column else 12.0 + 0.1 * column for column in range(10)]
        for row in range(10)
    ]
    correct = [
        {"state_index": index, "state_id": label, "psnr_db": 20.0 + index * 0.05}
        for index, label in enumerate(labels)
    ]
    usage = [
        {
            "state_index": index,
            "state_id": label,
            "base": {"coverage": 0.80 + index * 0.01},
            "residual": {"coverage": 0.78 + index * 0.01},
        }
        for index, label in enumerate(labels)
    ]
    _write_json(
        evaluation_dir / "evaluation_manifest.json",
        {"scope": "smoke", "schema": "synthetic-evaluation-v1", "outputs": {}},
    )
    _write_json(
        evaluation_dir / "reconstruction_summary.json",
        {
            "scope": "smoke",
            "software": {"overall": {"psnr_db": 20.2}},
            "seeded": {
                "cross_state": {
                    "psnr_db_matrix": matrix,
                    "correct_by_state": correct,
                },
                "usage_by_send_state": usage,
            },
        },
    )
    (evaluation_dir / "cross_state_matrix.csv").write_text("synthetic\n", encoding="utf-8")
    (evaluation_dir / "usage_by_state.csv").write_text("synthetic\n", encoding="utf-8")
    schedule_fields = [
        "schedule",
        "time_shift_frames",
        "synchronized",
        "matched_frame_count",
        "unmatched_frame_count",
        "match_fraction",
        "overall_mse",
        "overall_psnr_db",
        "overall_ssim_mean",
        "matched_only_mse",
        "matched_only_psnr_db",
        "matched_only_ssim_mean",
        "unmatched_only_mse",
        "unmatched_only_psnr_db",
        "unmatched_only_ssim_mean",
    ]
    with (evaluation_dir / "schedule_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=schedule_fields)
        writer.writeheader()
        for schedule in ("fixed", "pseudorandom"):
            for shift in range(10):
                matched = 20 if shift == 0 else (2 if schedule == "pseudorandom" else 0)
                unmatched = 20 - matched
                writer.writerow(
                    {
                        "schedule": schedule,
                        "time_shift_frames": shift,
                        "synchronized": shift == 0,
                        "matched_frame_count": matched,
                        "unmatched_frame_count": unmatched,
                        "match_fraction": matched / 20,
                        "overall_mse": 0.01 if shift == 0 else 0.2,
                        "overall_psnr_db": 20.0 if shift == 0 else 7.0,
                        "overall_ssim_mean": 0.9 if shift == 0 else 0.2,
                        "matched_only_mse": 0.01 if matched else "",
                        "matched_only_psnr_db": 20.0 if matched else "",
                        "matched_only_ssim_mean": 0.9 if matched else "",
                        "unmatched_only_mse": 0.2 if unmatched else "",
                        "unmatched_only_psnr_db": 7.0 if unmatched else "",
                        "unmatched_only_ssim_mean": 0.2 if unmatched else "",
                    }
                )

    output_dir = tmp_path / "figures"
    manifest = generate_figures(
        evaluation_dir, output_dir, dpi=72, annotate_heatmap=False
    )
    assert manifest["schema"] == FIGURE_SCHEMA
    assert manifest["evaluation_scope"] == "smoke"
    assert len(manifest["figures"]) == 3
    assert (output_dir / "figure_manifest.json").is_file()
    assert len(list(output_dir.glob("*.png"))) == 3
    assert len(list(output_dir.glob("*.pdf"))) == 3
