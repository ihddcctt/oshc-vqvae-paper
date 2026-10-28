from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


FIGURE_SCHEMA = "lightwave-seeded-vqvae-figure-manifest-v1"
REQUIRED_INPUTS = (
    "evaluation_manifest.json",
    "reconstruction_summary.json",
    "cross_state_matrix.csv",
    "usage_by_state.csv",
    "schedule_results.csv",
)
COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "gray": "#666666",
}
RC_PARAMS: dict[str, Any] = {
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "axes.titlesize": 11.0,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "savefig.facecolor": "white",
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create compact static figures from a unified final-evaluation directory"
    )
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <evaluation-dir>/figures; the directory must not already exist",
    )
    parser.add_argument("--dpi", type=_positive_int, default=300)
    parser.add_argument("--heatmap-cmap", default="viridis")
    parser.add_argument(
        "--no-cell-values",
        action="store_true",
        help="Do not annotate the 10x10 heatmap cells with numeric PSNR values",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(project_root), "status", "--porcelain"], text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def compact_state_label(state_id: str) -> str:
    """Display a canonical power-coordinate ID as pump nm / Q10 mJ (10 s)."""
    try:
        pump, power = state_id.split("_")
        return f"{int(pump.removesuffix('nm'))}/{10 * int(power.removesuffix('mW'))}"
    except (TypeError, ValueError):
        return state_id


def _scope_prefix(scope: str) -> str:
    return "" if scope == "final" else f"{scope.capitalize()} · "


def _load_schedule_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for source in reader:
            row: dict[str, Any] = {
                "schedule": source["schedule"],
                "time_shift_frames": int(source["time_shift_frames"]),
                "synchronized": source["synchronized"].lower() == "true",
                "matched_frame_count": int(source["matched_frame_count"]),
                "unmatched_frame_count": int(source["unmatched_frame_count"]),
                "match_fraction": float(source["match_fraction"]),
            }
            for key in (
                "overall_mse",
                "overall_psnr_db",
                "overall_ssim_mean",
                "matched_only_mse",
                "matched_only_psnr_db",
                "matched_only_ssim_mean",
                "unmatched_only_mse",
                "unmatched_only_psnr_db",
                "unmatched_only_ssim_mean",
            ):
                row[key] = float(source[key]) if source.get(key, "") != "" else None
            rows.append(row)
    expected = {(name, shift) for name in ("fixed", "pseudorandom") for shift in range(10)}
    observed = {(str(row["schedule"]), int(row["time_shift_frames"])) for row in rows}
    if observed != expected or len(rows) != 20:
        raise ValueError("schedule_results.csv must contain fixed and pseudorandom shifts 0..9")
    return sorted(rows, key=lambda row: (str(row["schedule"]), int(row["time_shift_frames"])))


def _load_plot_data(evaluation_dir: Path) -> dict[str, Any]:
    input_paths = {name: evaluation_dir / name for name in REQUIRED_INPUTS}
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing evaluation inputs: " + ", ".join(missing))
    evaluation_manifest = _read_json(input_paths["evaluation_manifest.json"])
    summary = _read_json(input_paths["reconstruction_summary.json"])
    scope = str(summary.get("scope", evaluation_manifest.get("scope", "unknown")))
    if evaluation_manifest.get("scope") != scope:
        raise ValueError("evaluation manifest and reconstruction summary scopes differ")

    actual_hashes = {name: _sha256_file(path) for name, path in input_paths.items()}
    recorded_hashes = evaluation_manifest.get("outputs", {})
    for name in REQUIRED_INPUTS[1:]:
        recorded = recorded_hashes.get(name)
        if recorded is not None and recorded != actual_hashes[name]:
            raise ValueError(f"evaluation artifact hash mismatch: {name}")

    seeded = summary.get("seeded")
    software = summary.get("software")
    if not isinstance(seeded, dict) or not isinstance(software, dict):
        raise ValueError("summary must contain software and seeded results")
    cross = seeded.get("cross_state")
    usage = seeded.get("usage_by_send_state")
    if not isinstance(cross, dict) or not isinstance(usage, list) or len(usage) != 10:
        raise ValueError("summary does not contain ten-state reconstruction and usage data")
    correct_by_state = cross.get("correct_by_state")
    matrix = np.asarray(cross.get("psnr_db_matrix"), dtype=np.float64)
    if not isinstance(correct_by_state, list) or len(correct_by_state) != 10:
        raise ValueError("correct_by_state must contain ten entries")
    if matrix.shape != (10, 10) or not np.isfinite(matrix).all():
        raise ValueError("psnr_db_matrix must be a finite 10x10 array")

    state_labels = [str(row["state_id"]) for row in correct_by_state]
    usage_labels = [str(row["state_id"]) for row in usage]
    if state_labels != usage_labels or len(set(state_labels)) != 10:
        raise ValueError("state order differs between quality and usage results")
    correct_psnr = np.asarray(
        [float(row["psnr_db"]) for row in correct_by_state], dtype=np.float64
    )
    base_coverage = np.asarray(
        [float(row["base"]["coverage"]) for row in usage], dtype=np.float64
    )
    residual_coverage = np.asarray(
        [float(row["residual"]["coverage"]) for row in usage], dtype=np.float64
    )
    if not (
        np.isfinite(correct_psnr).all()
        and np.isfinite(base_coverage).all()
        and np.isfinite(residual_coverage).all()
    ):
        raise ValueError("quality and usage arrays must be finite")
    if bool(((base_coverage < 0) | (base_coverage > 1)).any()) or bool(
        ((residual_coverage < 0) | (residual_coverage > 1)).any()
    ):
        raise ValueError("coverage values must lie in [0, 1]")

    schedule_rows = _load_schedule_rows(input_paths["schedule_results.csv"])
    return {
        "scope": scope,
        "evaluation_schema": evaluation_manifest.get("schema"),
        "evaluation_manifest": evaluation_manifest,
        "input_paths": input_paths,
        "input_hashes": actual_hashes,
        "state_labels": state_labels,
        "compact_labels": [compact_state_label(label) for label in state_labels],
        "psnr_matrix": matrix,
        "correct_psnr": correct_psnr,
        "software_psnr": float(software["overall"]["psnr_db"]),
        "base_coverage": base_coverage,
        "residual_coverage": residual_coverage,
        "schedule_rows": schedule_rows,
    }


def _save_figure(
    figure: plt.Figure,
    *,
    output_dir: Path,
    stem: str,
    title: str,
    dpi: int,
) -> dict[str, Any]:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={"Title": title, "Creator": "lightwave-plot-final-evaluation"},
    )
    plt.close(figure)
    return {
        "title": title,
        "png": {"path": str(png_path), "sha256": _sha256_file(png_path)},
        "pdf": {"path": str(pdf_path), "sha256": _sha256_file(pdf_path)},
    }


def _plot_psnr_heatmap(
    data: dict[str, Any],
    *,
    output_dir: Path,
    dpi: int,
    cmap: str,
    annotate_cells: bool,
) -> dict[str, Any]:
    matrix = data["psnr_matrix"]
    labels = data["compact_labels"]
    title = f"{_scope_prefix(data['scope'])}Cross-state reconstruction PSNR"
    figure, axis = plt.subplots(figsize=(7.4, 6.3))
    image = axis.imshow(matrix, cmap=cmap, aspect="equal")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("PSNR (dB)")
    axis.set_xticks(np.arange(10), labels=labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(10), labels=labels)
    axis.set_xlabel("Receiver state (pump nm / Q10 mJ, 10 s)")
    axis.set_ylabel("Sender state (pump nm / Q10 mJ, 10 s)")
    axis.set_title(title)
    midpoint = float((matrix.min() + matrix.max()) / 2.0)
    if annotate_cells:
        for row in range(10):
            for column in range(10):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.3,
                    color="white" if value < midpoint else "black",
                )
    for state in range(10):
        axis.add_patch(
            Rectangle(
                (state - 0.5, state - 0.5),
                1,
                1,
                fill=False,
                edgecolor="white",
                linewidth=1.0,
            )
        )
    figure.tight_layout()
    result = _save_figure(
        figure,
        output_dir=output_dir,
        stem="figure_1_cross_state_psnr_heatmap",
        title=title,
        dpi=dpi,
    )
    result["content"] = "10x10 sender-state/receiver-state PSNR matrix; diagonal is outlined"
    result["figure_size_inches"] = [7.4, 6.3]
    return result


def _plot_state_quality_usage(
    data: dict[str, Any], *, output_dir: Path, dpi: int
) -> dict[str, Any]:
    positions = np.arange(10)
    labels = data["compact_labels"]
    title = f"{_scope_prefix(data['scope'])}Correct-state quality and codebook coverage"
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0], "hspace": 0.10},
    )
    quality_axis, usage_axis = axes
    quality_axis.bar(
        positions,
        data["correct_psnr"],
        width=0.72,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.5,
        label="Seeded correct state",
    )
    quality_axis.axhline(
        data["software_psnr"],
        color=COLORS["gray"],
        linestyle="--",
        linewidth=1.2,
        label="Matched software baseline",
    )
    quality_axis.set_ylabel("PSNR (dB)")
    quality_axis.set_title(title)
    quality_axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    quality_axis.legend(frameon=False, loc="best")

    width = 0.36
    usage_axis.bar(
        positions - width / 2,
        data["base_coverage"],
        width=width,
        color=COLORS["blue"],
        label="Frozen base codebook",
    )
    usage_axis.bar(
        positions + width / 2,
        data["residual_coverage"],
        width=width,
        color=COLORS["orange"],
        label="Shared residual codebook",
    )
    usage_axis.set_ylim(0.0, 1.05)
    usage_axis.set_ylabel("Coverage")
    usage_axis.set_xlabel("Sender state (pump nm / Q10 mJ, 10 s)")
    usage_axis.set_xticks(positions, labels=labels, rotation=45, ha="right")
    usage_axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    usage_axis.legend(frameon=False, ncol=2, loc="lower center")
    for axis in axes:
        axis.axvline(4.5, color="#999999", linestyle=":", linewidth=0.9)
    figure.subplots_adjust(
        left=0.11, right=0.98, bottom=0.12, top=0.91, hspace=0.10
    )
    result = _save_figure(
        figure,
        output_dir=output_dir,
        stem="figure_2_state_quality_and_usage",
        title=title,
        dpi=dpi,
    )
    result["content"] = (
        "per-state correct PSNR and per-state base/residual index coverage; "
        "software PSNR shown as a reference"
    )
    result["figure_size_inches"] = [8.0, 6.0]
    return result


def _schedule_series(
    rows: list[dict[str, Any]], schedule: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = sorted(
        (row for row in rows if row["schedule"] == schedule),
        key=lambda row: row["time_shift_frames"],
    )
    if len(selected) != 10:
        raise ValueError(f"schedule {schedule} does not contain shifts 0..9")
    shifts = np.asarray([row["time_shift_frames"] for row in selected], dtype=np.int64)
    psnr = np.asarray(
        [
            row["overall_psnr_db"]
            if row["time_shift_frames"] == 0
            else row["unmatched_only_psnr_db"]
            for row in selected
        ],
        dtype=np.float64,
    )
    match_fraction = np.asarray([row["match_fraction"] for row in selected])
    if not np.isfinite(psnr).all():
        raise ValueError(f"schedule {schedule} lacks unmatched-frame PSNR")
    return shifts, psnr, match_fraction


def _plot_schedule_comparison(
    data: dict[str, Any], *, output_dir: Path, dpi: int
) -> dict[str, Any]:
    fixed_shift, fixed_psnr, fixed_match = _schedule_series(
        data["schedule_rows"], "fixed"
    )
    random_shift, random_psnr, random_match = _schedule_series(
        data["schedule_rows"], "pseudorandom"
    )
    title = f"{_scope_prefix(data['scope'])}Synchronized and time-shifted state schedules"
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.5, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 0.75], "hspace": 0.10},
    )
    quality_axis, match_axis = axes
    quality_axis.plot(
        fixed_shift,
        fixed_psnr,
        marker="o",
        linewidth=1.6,
        color=COLORS["blue"],
        label="Fixed schedule",
    )
    quality_axis.plot(
        random_shift,
        random_psnr,
        marker="s",
        linewidth=1.6,
        color=COLORS["orange"],
        label="Pseudorandom schedule",
    )
    quality_axis.set_ylabel("PSNR (dB)")
    quality_axis.set_title(title)
    quality_axis.grid(axis="both", color="#DDDDDD", linewidth=0.6)
    quality_axis.legend(frameon=False)
    quality_axis.text(
        0.01,
        0.02,
        "shift 0: synchronized; shifts 1–9: unmatched frames only",
        transform=quality_axis.transAxes,
        fontsize=7.5,
        color=COLORS["gray"],
    )

    match_axis.plot(
        fixed_shift,
        fixed_match,
        marker="o",
        linewidth=1.3,
        color=COLORS["blue"],
    )
    match_axis.plot(
        random_shift,
        random_match,
        marker="s",
        linewidth=1.3,
        color=COLORS["orange"],
    )
    match_axis.set_ylim(-0.03, 1.03)
    match_axis.set_ylabel("State-match fraction")
    match_axis.set_xlabel("Receiver schedule shift (frames)")
    match_axis.set_xticks(np.arange(10))
    match_axis.grid(axis="both", color="#DDDDDD", linewidth=0.6)
    figure.subplots_adjust(
        left=0.11, right=0.98, bottom=0.12, top=0.91, hspace=0.10
    )
    result = _save_figure(
        figure,
        output_dir=output_dir,
        stem="figure_3_schedule_sync_and_shift",
        title=title,
        dpi=dpi,
    )
    result["content"] = (
        "fixed/pseudorandom synchronized reconstruction and nine receiver time shifts; "
        "mismatch PSNR excludes accidental state matches"
    )
    result["figure_size_inches"] = [7.5, 5.8]
    return result


def generate_figures(
    evaluation_dir: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 300,
    heatmap_cmap: str = "viridis",
    annotate_heatmap: bool = True,
) -> dict[str, Any]:
    evaluation_path = Path(evaluation_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if not evaluation_path.is_dir():
        raise NotADirectoryError(evaluation_path)
    if output_path.exists():
        raise FileExistsError(f"figure output directory already exists: {output_path}")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    try:
        matplotlib.colormaps[heatmap_cmap]
    except KeyError as error:
        raise ValueError(f"unknown matplotlib colormap: {heatmap_cmap}") from error

    data = _load_plot_data(evaluation_path)
    output_path.mkdir(parents=True, exist_ok=False)
    with plt.rc_context(RC_PARAMS):
        figures = {
            "cross_state_psnr": _plot_psnr_heatmap(
                data,
                output_dir=output_path,
                dpi=dpi,
                cmap=heatmap_cmap,
                annotate_cells=annotate_heatmap,
            ),
            "state_quality_and_usage": _plot_state_quality_usage(
                data, output_dir=output_path, dpi=dpi
            ),
            "schedule_sync_and_shift": _plot_schedule_comparison(
                data, output_dir=output_path, dpi=dpi
            ),
        }

    project_root = Path(__file__).resolve().parents[3]
    manifest = {
        "schema": FIGURE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "evaluation_directory": str(evaluation_path),
        "evaluation_schema": data["evaluation_schema"],
        "evaluation_scope": data["scope"],
        "interpretation_boundary": (
            "Figures summarize reconstruction and state-synchronization behavior; "
            "they do not by themselves establish cryptographic security."
        ),
        "input_sha256": data["input_hashes"],
        "plotting_parameters": {
            "dpi": int(dpi),
            "formats": ["png", "pdf"],
            "heatmap_colormap": heatmap_cmap,
            "heatmap_cell_values": bool(annotate_heatmap),
            "state_tick_units": "pump_nm/Q10_mJ (10 s); Q10_mJ=10*power_mw",
            "schedule_shift_zero_metric": "overall synchronized PSNR",
            "schedule_nonzero_shift_metric": "unmatched-frame-only PSNR",
            "rc_params": RC_PARAMS,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "backend": matplotlib.get_backend(),
        },
        "git": _git_metadata(project_root),
        "figures": figures,
    }
    manifest_path = output_path / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    evaluation_dir = args.evaluation_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else evaluation_dir / "figures"
    )
    manifest = generate_figures(
        evaluation_dir,
        output_dir,
        dpi=args.dpi,
        heatmap_cmap=args.heatmap_cmap,
        annotate_heatmap=not args.no_cell_values,
    )
    print(f"evaluation scope: {manifest['evaluation_scope']}")
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
