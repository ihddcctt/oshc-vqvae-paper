from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from ..data import build_dataset
from ..seed.codebook import load_codebook_bank
from .evaluate_final import (
    _array_sha256,
    _compatibility_report,
    _finish_fingerprint,
    _load_model,
    _new_fingerprint,
    _read_json,
    _sha256_file,
    _state_labels,
    _update_fingerprint,
)
from .plot_final_evaluation import compact_state_label


CIFAR_VISUAL_SCHEMA = "lightwave-seeded-vqvae-cifar-natural-visuals-v2"
CIFAR10_CLASS_NAMES_EN = [
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
CIFAR10_CLASS_NAMES_ZH = [
    "飞机",
    "汽车",
    "鸟",
    "猫",
    "鹿",
    "狗",
    "青蛙",
    "马",
    "船",
    "卡车",
]
ZH_FONT_FAMILY = [
    "Noto Sans CJK JP",
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
RC_PARAMS: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ZH_FONT_FAMILY,
    "font.size": 9.0,
    "axes.titlesize": 13.0,
    "axes.labelsize": 10.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "white",
}
COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "yellow": "#E69F00",
    "red": "#C83E4D",
    "gray": "#666666",
    "light_gray": "#F2F2F2",
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _state_index(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed < 10:
        raise argparse.ArgumentTypeError("state index must lie in [0, 9]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reproducible Chinese PPT figures for the formal CIFAR-10 "
            "lightwave-seeded VQ-VAE evaluation"
        )
    )
    parser.add_argument("--seeded-checkpoint", required=True, type=Path)
    parser.add_argument("--software-checkpoint", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--send-state", type=_state_index, default=2)
    parser.add_argument("--cross-pump-receiver-state", type=_state_index, default=7)
    parser.add_argument("--same-pump-receiver-state", type=_state_index, default=4)
    parser.add_argument("--dpi", type=_positive_int, default=300)
    return parser


def select_median_per_class(
    targets: Sequence[int] | np.ndarray,
    correct_mse: np.ndarray,
    *,
    class_count: int = 10,
) -> list[int]:
    """Choose one deterministic class-median example in official dataset order.

    If ``correct_mse`` is ``N x S``, scores are the mean of all synchronized
    diagonal-state MSE values for each image.  The chosen example minimizes the
    distance to its class median; an exact tie is resolved by the smaller official
    test-set index.
    """

    target_array = np.asarray(targets, dtype=np.int64)
    score_array = np.asarray(correct_mse, dtype=np.float64)
    if score_array.ndim == 2:
        score_array = score_array.mean(axis=1)
    if target_array.ndim != 1 or score_array.ndim != 1:
        raise ValueError("targets and reduced correct_mse must be one-dimensional")
    if target_array.shape[0] != score_array.shape[0]:
        raise ValueError("targets and correct_mse must contain the same number of images")
    if class_count <= 0:
        raise ValueError("class_count must be positive")
    if not np.isfinite(score_array).all() or bool((score_array < 0).any()):
        raise ValueError("correct_mse must contain finite non-negative values")

    selected: list[int] = []
    for class_id in range(class_count):
        candidates = np.flatnonzero(target_array == class_id)
        if candidates.size == 0:
            raise ValueError(f"dataset has no sample for class {class_id}")
        median = float(np.median(score_array[candidates]))
        selected.append(
            min(
                (int(index) for index in candidates),
                key=lambda index: (abs(float(score_array[index]) - median), index),
            )
        )
    return selected


def state_records_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return canonical pump/power metadata in state-index order."""

    labels = _state_labels(manifest)
    states = manifest["states"]
    records: list[dict[str, Any]] = []
    for index, (state, label) in enumerate(zip(states, labels)):
        try:
            pump_nm = int(state["pump_nm"])
            power_mw = int(state["power_mw"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "every codebook state must contain integer pump_nm and power_mw"
            ) from error
        records.append(
            {
                "index": index,
                "state_id": label,
                "pump_nm": pump_nm,
                "power_mw": power_mw,
            }
        )
    if len({(item["pump_nm"], item["power_mw"]) for item in records}) != len(records):
        raise ValueError("pump/power state pairs must be unique")
    return records


def validate_controlled_state_mapping(
    records: Sequence[dict[str, Any]],
    *,
    send_state: int,
    cross_pump_receiver_state: int,
    same_pump_receiver_state: int,
) -> None:
    """Check that CLI examples have the advertised physical relationship."""

    if len(records) != 10:
        raise ValueError("exactly ten state records are required")
    for index in (send_state, cross_pump_receiver_state, same_pump_receiver_state):
        if not 0 <= index < len(records):
            raise ValueError("controlled state indices must lie in [0, 9]")
    sender = records[send_state]
    cross = records[cross_pump_receiver_state]
    same = records[same_pump_receiver_state]
    if send_state in {cross_pump_receiver_state, same_pump_receiver_state}:
        raise ValueError("wrong receiver states must differ from the sender")
    if not (
        cross["pump_nm"] != sender["pump_nm"]
        and cross["power_mw"] == sender["power_mw"]
    ):
        raise ValueError(
            "cross-pump receiver must use a different pump wavelength at the same power"
        )
    if not (
        same["pump_nm"] == sender["pump_nm"]
        and same["power_mw"] != sender["power_mw"]
    ):
        raise ValueError(
            "same-pump receiver must use the same pump wavelength at a different power"
        )


def row_relative_mse(mse_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(mse_matrix, dtype=np.float64)
    if matrix.shape != (10, 10):
        raise ValueError("mse_matrix must have shape (10, 10)")
    if not np.isfinite(matrix).all() or bool((matrix <= 0).any()):
        raise ValueError("mse_matrix must contain finite positive values")
    return matrix / np.diag(matrix)[:, None]


def select_representative_wrong_pair(
    mse_matrix: np.ndarray,
    *,
    send_state: int,
    excluded_pairs: Iterable[tuple[int, int]] = (),
) -> dict[str, float | int]:
    """Pick a typical wrong receiver for one fixed sender from formal MSE.

    The target is the median pooled MSE across all nine wrong receivers in the
    sender's row.  Pre-registered examples may be excluded from the final choice,
    but not from the median definition.  Ties are resolved by receiver index.
    """

    ratios = row_relative_mse(mse_matrix)
    mse = np.asarray(mse_matrix, dtype=np.float64)
    if not 0 <= send_state < 10:
        raise ValueError("send_state must lie in [0, 9]")
    excluded = {(int(sender), int(receiver)) for sender, receiver in excluded_pairs}
    wrong_receivers = [receiver for receiver in range(10) if receiver != send_state]
    median_mse = float(np.median(mse[send_state, wrong_receivers]))
    candidates = [receiver for receiver in wrong_receivers if (send_state, receiver) not in excluded]
    if not candidates:
        raise ValueError("no eligible wrong-state pair remains")
    receiver = min(
        candidates,
        key=lambda item: (abs(float(mse[send_state, item]) - median_mse), item),
    )
    return {
        "send_state": send_state,
        "receiver_state": receiver,
        "pooled_mse": float(mse[send_state, receiver]),
        "row_relative_mse": float(ratios[send_state, receiver]),
        "row_wrong_median_pooled_mse": median_mse,
        "row_wrong_pair_count": len(wrong_receivers),
        "eligible_pair_count": len(candidates),
    }


def _targets_from_dataset(dataset: Any) -> np.ndarray:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        return np.asarray([int(dataset[index][1]) for index in range(len(dataset))])
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    return np.asarray(targets, dtype=np.int64)


def nchw_to_rgb(images: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert NCHW CIFAR images to clipped NHWC RGB using one fixed scale."""

    if torch.is_tensor(images):
        array = images.detach().float().cpu().numpy()
    else:
        array = np.asarray(images, dtype=np.float32)
    if array.ndim != 4 or array.shape[1] != 3:
        raise ValueError("CIFAR display tensors must have shape (N, 3, H, W)")
    return np.clip(np.moveaxis(array, 1, -1), 0.0, 1.0)


def rgb_absolute_error(original: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    """Per-pixel absolute error averaged over RGB channels, fixed to [0, 1]."""

    first = np.asarray(original, dtype=np.float32)
    second = np.asarray(reconstruction, dtype=np.float32)
    if first.shape != second.shape or first.ndim != 4 or first.shape[-1] != 3:
        raise ValueError("original and reconstruction must be equal-shaped NHWC RGB batches")
    return np.clip(np.abs(first - second).mean(axis=-1), 0.0, 1.0)


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _int64_array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


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


def _save_figure(
    figure: plt.Figure,
    *,
    output_dir: Path,
    stem: str,
    title: str,
    dpi: int,
    preserve_canvas: bool = False,
) -> dict[str, Any]:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    canvas_inches = [float(value) for value in figure.get_size_inches()]
    bbox_inches = None if preserve_canvas else "tight"
    figure.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)
    figure.savefig(
        pdf_path,
        bbox_inches=bbox_inches,
        metadata={"Title": title, "Creator": "lightwave-cifar-natural-visuals"},
    )
    plt.close(figure)
    return {
        "title": title,
        "png": {"path": str(png_path), "sha256": _sha256_file(png_path)},
        "pdf": {"path": str(pdf_path), "sha256": _sha256_file(pdf_path)},
        "canvas_inches": canvas_inches,
        "canvas_preserved": preserve_canvas,
    }


def _show_rgb(axis: plt.Axes, image: np.ndarray) -> None:
    axis.imshow(np.clip(image, 0.0, 1.0), vmin=0.0, vmax=1.0, interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _show_error(axis: plt.Axes, error: np.ndarray) -> None:
    axis.imshow(
        np.clip(error, 0.0, 1.0),
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _label_row(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        0.5,
        label,
        transform=axis.transAxes,
        ha="right",
        va="center",
        fontsize=8.6,
        fontweight="bold",
    )


def _state_display(record: dict[str, Any]) -> str:
    return f"{record['pump_nm']} nm / {10 * record['power_mw']} mJ (10 s)"


def plot_cross_state_heatmap(
    mse_matrix: np.ndarray,
    *,
    state_labels: Sequence[str],
    output_dir: Path,
    dpi: int,
    ssim_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    """Plot absolute full-scale RMSE first, with SSIM only when formally available."""

    mse = np.asarray(mse_matrix, dtype=np.float64)
    if mse.shape != (10, 10) or not np.isfinite(mse).all() or bool((mse < 0).any()):
        raise ValueError("mse_matrix must be a finite non-negative (10, 10) array")
    labels = [compact_state_label(item) for item in state_labels]
    use_ssim = ssim_matrix is not None
    panels = 2 if use_ssim else 1
    figure, axes_value = plt.subplots(1, panels, figsize=(13.33 if use_ssim else 7.3, 6.4))
    axes = np.atleast_1d(axes_value)
    rmse_percent = np.sqrt(mse) * 100.0
    rmse_axis = axes[0]
    rmse_image = rmse_axis.imshow(rmse_percent, cmap="magma", aspect="equal")
    colorbar = figure.colorbar(rmse_image, ax=rmse_axis, fraction=0.046, pad=0.04)
    colorbar.set_label("RMSE（占 [0,1] 满量程的百分比，越低越好）")
    rmse_axis.set_title("像素 RMSE（主指标）", fontweight="bold")
    _configure_matrix_axis(rmse_axis, labels)
    _annotate_matrix(rmse_axis, rmse_percent, suffix="%", decimals=1)

    if use_ssim:
        ssim = np.asarray(ssim_matrix, dtype=np.float64)
        if ssim.shape != (10, 10) or not np.isfinite(ssim).all():
            raise ValueError("ssim_matrix must be a finite (10, 10) array")
        ssim_axis = axes[1]
        ssim_image = ssim_axis.imshow(ssim, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
        ssim_bar = figure.colorbar(ssim_image, ax=ssim_axis, fraction=0.046, pad=0.04)
        ssim_bar.set_label("SSIM（越高越好）")
        ssim_axis.set_title("SSIM（辅助指标）", fontweight="bold")
        _configure_matrix_axis(ssim_axis, labels)
        _annotate_matrix(ssim_axis, ssim, suffix="", decimals=2)

    title = "10×10 芯片状态收发错配矩阵：绝对误差，不使用 dB"
    figure.suptitle(title, fontsize=15.5, fontweight="bold", y=0.985)
    figure.text(
        0.5,
        0.012,
        "横轴仅改变接收端码本状态；每一行的发送索引保持不变。绿色框标出收发同步的对角线。",
        ha="center",
        va="bottom",
        color=COLORS["gray"],
        fontsize=9.0,
    )
    figure.subplots_adjust(left=0.08, right=0.96, bottom=0.16, top=0.88, wspace=0.32)
    result = _save_figure(
        figure,
        output_dir=output_dir,
        stem="cifar_01_cross_state_rmse_heatmap_zh",
        title=title,
        dpi=dpi,
    )
    result["content"] = "absolute full-scale RMSE matrix" + (
        " with auxiliary SSIM matrix" if use_ssim else ""
    )
    return result


def _configure_matrix_axis(axis: plt.Axes, labels: Sequence[str]) -> None:
    axis.set_xticks(np.arange(10), labels=labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(10), labels=labels)
    axis.set_xlabel("接收端状态（泵浦 / 功率）")
    axis.set_ylabel("发送端状态（泵浦 / 功率）")
    for index in range(10):
        axis.add_patch(
            Rectangle(
                (index - 0.5, index - 0.5),
                1,
                1,
                fill=False,
                edgecolor=COLORS["green"],
                linewidth=1.6,
            )
        )


def _annotate_matrix(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    suffix: str,
    decimals: int,
) -> None:
    threshold = float(np.nanmedian(values))
    for row in range(10):
        for column in range(10):
            value = float(values[row, column])
            axis.text(
                column,
                row,
                f"{value:.{decimals}f}{suffix}",
                ha="center",
                va="center",
                fontsize=5.8,
                fontweight="bold" if row == column else "normal",
                color="black" if value > threshold else "white",
            )


def plot_controlled_gallery(
    arrays: dict[str, np.ndarray],
    *,
    sample_ids: Sequence[int],
    class_ids: Sequence[int],
    records: Sequence[dict[str, Any]],
    mappings: dict[str, tuple[int, int]],
    pair_ratios: dict[str, float],
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    """Show all ten audited classes in one matched/mismatched gallery."""

    if len(sample_ids) != 10 or len(class_ids) != 10:
        raise ValueError("gallery requires exactly one selected sample for each CIFAR-10 class")
    return _render_controlled_gallery(
        arrays,
        sample_ids=sample_ids,
        class_ids=class_ids,
        column_positions=list(range(10)),
        records=records,
        mappings=mappings,
        pair_ratios=pair_ratios,
        output_dir=output_dir,
        dpi=dpi,
        stem="cifar_02_controlled_mismatch_gallery_zh",
        title="自然图像示例：发送索引不变，只切换接收端芯片状态",
        figsize=(13.33, 10.4),
        preserve_canvas=False,
        purpose="complete ten-class audit gallery",
    )


def plot_controlled_gallery_page(
    arrays: dict[str, np.ndarray],
    *,
    sample_ids: Sequence[int],
    class_ids: Sequence[int],
    class_start: int,
    records: Sequence[dict[str, Any]],
    mappings: dict[str, tuple[int, int]],
    pair_ratios: dict[str, float],
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    """Render one exact-16:9 PPT page for CIFAR classes 0--4 or 5--9."""

    if len(sample_ids) != 10 or len(class_ids) != 10:
        raise ValueError("PPT gallery requires the complete ten-class audit selection")
    if class_start not in {0, 5}:
        raise ValueError("class_start must be 0 or 5")
    column_positions = list(range(class_start, class_start + 5))
    expected_class_ids = list(range(class_start, class_start + 5))
    page_class_ids = [int(class_ids[position]) for position in column_positions]
    if page_class_ids != expected_class_ids:
        raise ValueError("PPT gallery pages require CIFAR class order 0..9")
    page_number = 1 if class_start == 0 else 2
    return _render_controlled_gallery(
        arrays,
        sample_ids=sample_ids,
        class_ids=class_ids,
        column_positions=column_positions,
        records=records,
        mappings=mappings,
        pair_ratios=pair_ratios,
        output_dir=output_dir,
        dpi=dpi,
        stem=(
            f"cifar_02{chr(ord('a') + page_number - 1)}_controlled_mismatch_"
            f"classes_{class_start}_{class_start + 4}_16x9_zh"
        ),
        title=(
            "自然图像状态错配（PPT 分页）："
            f"CIFAR-10 类别 {class_start}–{class_start + 4}"
        ),
        figsize=(40.0 / 3.0, 7.5),
        preserve_canvas=True,
        purpose="16:9 PPT gallery page",
    )


def _controlled_gallery_rows(
    arrays: dict[str, np.ndarray],
) -> list[tuple[str, np.ndarray, bool]]:
    original = np.asarray(arrays["original"])
    return [
        ("原图", original, False),
        ("正确状态", np.asarray(arrays["correct"]), False),
        ("正确状态\n绝对误差", rgb_absolute_error(original, arrays["correct"]), True),
        ("错波长\n同功率", np.asarray(arrays["cross_pump"]), False),
        ("错波长\n绝对误差", rgb_absolute_error(original, arrays["cross_pump"]), True),
        ("同波长\n错功率", np.asarray(arrays["same_pump"]), False),
        ("错功率\n绝对误差", rgb_absolute_error(original, arrays["same_pump"]), True),
        ("代表性错配", np.asarray(arrays["representative"]), False),
        ("代表错配\n绝对误差", rgb_absolute_error(original, arrays["representative"]), True),
    ]


def _controlled_gallery_caption(
    *,
    records: Sequence[dict[str, Any]],
    mappings: dict[str, tuple[int, int]],
    pair_ratios: dict[str, float],
) -> str:
    fixed_sender = mappings["correct"][0]
    return (
        f"固定发送 {_state_display(records[fixed_sender])}，i_B 与 i_R 全程复用；"
        f"跨波长同功率 {pair_ratios['cross_pump']:.2f}×，"
        f"同波长错功率 {pair_ratios['same_pump']:.2f}×，"
        f"代表性错配 {mappings['representative'][0]}→{mappings['representative'][1]} "
        f"为 {pair_ratios['representative']:.2f}×。\n"
        "所有 × 数值均为整套测试集 pooled MSE 倍率（错配 pooled MSE / 同行正确 pooled MSE）；"
        "代表性错配按该发送行 9 种错误状态的 pooled MSE 中位严重度选取。"
        "误差图为 RGB 通道平均绝对误差，全部共用固定 [0,1] 色阶。"
    )


def _add_fixed_error_colorbar(figure: plt.Figure) -> None:
    color_axis = figure.add_axes((0.958, 0.225, 0.012, 0.54))
    scalar = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0),
        cmap="inferno",
    )
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, cax=color_axis, ticks=[0.0, 1.0])
    colorbar.ax.set_yticklabels(["0（黑）", "1（黄）"])
    colorbar.ax.tick_params(labelsize=7.5)
    colorbar.set_label("绝对误差（固定 [0,1]）", fontsize=8.2)


def _render_controlled_gallery(
    arrays: dict[str, np.ndarray],
    *,
    sample_ids: Sequence[int],
    class_ids: Sequence[int],
    column_positions: Sequence[int],
    records: Sequence[dict[str, Any]],
    mappings: dict[str, tuple[int, int]],
    pair_ratios: dict[str, float],
    output_dir: Path,
    dpi: int,
    stem: str,
    title: str,
    figsize: tuple[float, float],
    preserve_canvas: bool,
    purpose: str,
) -> dict[str, Any]:
    rows = _controlled_gallery_rows(arrays)
    original = np.asarray(arrays["original"])
    if original.ndim != 4 or original.shape[0] != 10 or original.shape[-1] != 3:
        raise ValueError("gallery requires ten NHWC RGB images")
    for _, images, is_error in rows:
        expected = (10, original.shape[1], original.shape[2]) if is_error else original.shape
        if images.shape != expected:
            raise ValueError("all gallery arrays must share the selected CIFAR image shape")
    positions = [int(value) for value in column_positions]
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("gallery column positions must be non-empty and unique")
    if min(positions) < 0 or max(positions) >= 10:
        raise ValueError("gallery column positions must lie in [0, 9]")

    figure, axes = plt.subplots(
        len(rows),
        len(positions),
        figsize=figsize,
        squeeze=False,
    )
    for row_index, (row_label, images, is_error) in enumerate(rows):
        for page_column, source_column in enumerate(positions):
            axis = axes[row_index, page_column]
            if is_error:
                _show_error(axis, images[source_column])
            else:
                _show_rgb(axis, images[source_column])
            if row_index == 0:
                class_id = int(class_ids[source_column])
                axis.set_title(
                    f"{CIFAR10_CLASS_NAMES_ZH[class_id]}\n测试集 #{sample_ids[source_column]}",
                    fontsize=8.4 if len(positions) == 5 else 7.8,
                )
        _label_row(axes[row_index, 0], row_label)

    caption = _controlled_gallery_caption(
        records=records,
        mappings=mappings,
        pair_ratios=pair_ratios,
    )
    figure.suptitle(title, fontsize=15.5, fontweight="bold", y=0.982)
    figure.text(
        0.5,
        0.018 if preserve_canvas else 0.006,
        caption,
        ha="center",
        va="bottom",
        fontsize=8.3 if preserve_canvas else 8.1,
        color=COLORS["gray"],
        linespacing=1.35,
    )
    _add_fixed_error_colorbar(figure)
    figure.subplots_adjust(
        left=0.105 if len(positions) == 10 else 0.135,
        right=0.94,
        bottom=0.12 if preserve_canvas else 0.07,
        top=0.92,
        wspace=0.055 if len(positions) == 10 else 0.08,
        hspace=0.065,
    )
    result = _save_figure(
        figure,
        output_dir=output_dir,
        stem=stem,
        title=title,
        dpi=dpi,
        preserve_canvas=preserve_canvas,
    )
    result["content"] = (
        "original, synchronized reconstruction/error, cross-pump same-power "
        "reconstruction/error, same-pump wrong-power reconstruction/error, "
        "representative wrong-pair reconstruction/error"
    )
    result["purpose"] = purpose
    result["class_ids"] = [int(class_ids[position]) for position in positions]
    result["sample_ids"] = [int(sample_ids[position]) for position in positions]
    result["footnote"] = caption
    result["absolute_error_scale"] = {
        "definition": "mean(abs(RGB reconstruction - RGB original), axis=channels)",
        "display_range": [0.0, 1.0],
        "shared_across_all_error_maps": True,
        "colormap": "inferno",
        "legend": "vertical colorbar: black=0, yellow=1",
    }
    if preserve_canvas:
        result["layout"] = {
            "aspect_ratio": "16:9",
            "canvas_inches": [40.0 / 3.0, 7.5],
            "png_pixels": [int(round((40.0 / 3.0) * dpi)), int(round(7.5 * dpi))],
        }
    return result


def extract_component_diagnostics(summary: dict[str, Any]) -> dict[str, Any] | None:
    """Read component and usage evidence; return ``None`` if it is unavailable."""

    try:
        seeded = summary["seeded"]
        cross = seeded["cross_state"]
        correct_mse = float(cross["correct_pooled"]["mse"])
        if not np.isfinite(correct_mse) or correct_mse <= 0:
            raise ValueError("invalid correct-state MSE")
        branches = seeded["branches"]
        metrics = {
            "full_correct": 1.0,
            "software": float(summary["software"]["overall"]["mse"]) / correct_mse,
            "base_only": float(branches["base_only"]["pooled"]["mse"]) / correct_mse,
            "transmitted_residual_only": float(
                branches["transmitted_residual_only"]["pooled"]["mse"]
            )
            / correct_mse,
            "requantized_residual_only": float(
                branches["requantized_residual_only"]["mse"]
            )
            / correct_mse,
            "all_wrong": float(cross["wrong_pooled"]["mse"]) / correct_mse,
        }
        usage = seeded["usage_by_send_state"]
        base_coverage = np.asarray(
            [float(item["base"]["coverage"]) for item in usage], dtype=np.float64
        )
        residual_coverage = np.asarray(
            [float(item["residual"]["coverage"]) for item in usage], dtype=np.float64
        )
        if base_coverage.shape != (10,) or residual_coverage.shape != (10,):
            raise ValueError("usage_by_send_state must contain ten states")
        pooled_residual_coverage = float(seeded["pooled_residual_usage"]["coverage"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    values = np.asarray(list(metrics.values()), dtype=np.float64)
    if not np.isfinite(values).all() or bool((values < 0).any()):
        return None
    if not np.isfinite(base_coverage).all() or not np.isfinite(residual_coverage).all():
        return None
    return {
        "mse_multipliers": metrics,
        "base_coverage_by_state": base_coverage,
        "residual_coverage_by_state": residual_coverage,
        "pooled_residual_coverage": pooled_residual_coverage,
    }


def plot_component_diagnostic(
    diagnostic: dict[str, Any],
    *,
    state_labels: Sequence[str],
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    """Plot the anti-bypass evidence available in the formal evaluation."""

    metrics = diagnostic["mse_multipliers"]
    keys = [
        "software",
        "full_correct",
        "base_only",
        "transmitted_residual_only",
        "requantized_residual_only",
        "all_wrong",
    ]
    labels = [
        "软件基线",
        "完整 $B_s+R$",
        "仅 $B_s$",
        "仅传输的 $R$",
        "仅 $R$（重新量化）",
        "全部错误状态",
    ]
    colors = [
        COLORS["blue"],
        COLORS["green"],
        COLORS["orange"],
        COLORS["yellow"],
        "#CC79A7",
        COLORS["red"],
    ]
    values = np.asarray([float(metrics[key]) for key in keys], dtype=np.float64)
    figure, (metric_axis, usage_axis) = plt.subplots(
        1, 2, figsize=(13.33, 5.6), gridspec_kw={"width_ratios": [1.03, 1.0]}
    )
    positions = np.arange(len(labels))
    bars = metric_axis.barh(positions, values, color=colors, height=0.65)
    metric_axis.set_yticks(positions, labels=labels)
    metric_axis.invert_yaxis()
    metric_axis.set_xlabel("相对完整同步模型的 MSE（1×）")
    metric_axis.set_title("组件消融：重建误差", fontweight="bold")
    metric_axis.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    metric_axis.set_axisbelow(True)
    metric_axis.set_xlim(0, max(1.15, float(values.max()) * 1.2))
    for bar, value in zip(bars, values):
        metric_axis.text(
            value + max(float(values.max()) * 0.015, 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}×",
            va="center",
            ha="left",
            fontsize=9.2,
            fontweight="bold",
        )

    x = np.arange(10)
    base = np.asarray(diagnostic["base_coverage_by_state"], dtype=np.float64) * 100.0
    residual = np.asarray(diagnostic["residual_coverage_by_state"], dtype=np.float64) * 100.0
    usage_axis.plot(x, base, marker="o", color=COLORS["orange"], label="$B_s$ 覆盖率")
    usage_axis.plot(x, residual, marker="s", color=COLORS["blue"], label="$R$ 覆盖率")
    pooled = float(diagnostic["pooled_residual_coverage"]) * 100.0
    usage_axis.axhline(
        pooled,
        color=COLORS["blue"],
        linestyle="--",
        linewidth=1.1,
        label=f"$R$ 汇总覆盖率 {pooled:.1f}%",
    )
    usage_axis.set_xticks(x, labels=[compact_state_label(item) for item in state_labels], rotation=45, ha="right")
    usage_axis.set_ylim(0, 105)
    usage_axis.set_ylabel("码字覆盖率（%）")
    usage_axis.set_xlabel("发送状态（泵浦 / 功率）")
    usage_axis.set_title("索引使用：辅助判断是否由 $R$ 主导", fontweight="bold")
    usage_axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    usage_axis.legend(frameon=False, fontsize=8.5)

    title = "组件与 $R$ 主导诊断（仅报告正式评估中已有证据）"
    figure.suptitle(title, fontsize=15.5, fontweight="bold", y=0.985)
    figure.text(
        0.5,
        0.012,
        "$R$ 的幅值或覆盖率不能单独证明旁路；应联合查看仅 $R$ 退化、错误状态退化及 $B_s$ 使用情况。",
        ha="center",
        va="bottom",
        color=COLORS["gray"],
        fontsize=9.0,
    )
    figure.subplots_adjust(left=0.135, right=0.985, bottom=0.23, top=0.86, wspace=0.32)
    result = _save_figure(
        figure,
        output_dir=output_dir,
        stem="cifar_03_components_and_residual_diagnostic_zh",
        title=title,
        dpi=dpi,
    )
    result["content"] = "component MSE multipliers and state-wise base/residual code coverage"
    return result


def _dataset_fingerprint(dataset: Any, *, batch_size: int = 256) -> dict[str, Any]:
    """Recompute the evaluator's exact full-test fingerprint without shuffling."""

    fingerprint = _new_fingerprint()
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for index in range(len(dataset)):
        image, label = dataset[index]
        images.append(image)
        labels.append(int(label))
        if len(images) == batch_size or index + 1 == len(dataset):
            _update_fingerprint(
                fingerprint,
                torch.stack(images),
                torch.as_tensor(labels, dtype=torch.long),
            )
            images.clear()
            labels.clear()
    return _finish_fingerprint(fingerprint)


def _validate_dataset_fingerprint(
    dataset: Any,
    evaluation_manifest: dict[str, Any],
) -> dict[str, Any]:
    expected = evaluation_manifest.get("dataset", {})
    actual = _dataset_fingerprint(dataset)
    fields = (
        "processed_images_float32_sha256",
        "processed_labels_int64_sha256",
        "processed_sample_order_sha256",
        "processed_sample_count",
    )
    mismatches = {
        field: {"expected": expected.get(field), "actual": actual[field]}
        for field in fields
        if expected.get(field) != actual[field]
    }
    if mismatches:
        raise ValueError(
            "local CIFAR-10 test dataset does not match the formal evaluation: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return actual


@torch.inference_mode()
def run_controlled_inference(
    seeded_model: Any,
    images: torch.Tensor,
    *,
    send_state: int,
    cross_pump_receiver_state: int,
    same_pump_receiver_state: int,
    representative_receiver_state: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Quantize once at the fixed sender, then decode all receiver variants.

    The base and residual indices are intentionally produced in one quantizer call
    and reused verbatim.  Only ``state_ids`` passed to ``decode_indices`` changes.
    """

    images = images.to(device)
    count = int(images.shape[0])
    send_ids = torch.full((count,), send_state, dtype=torch.long, device=device)
    latent = seeded_model.encoder(images)
    transmitted = seeded_model.quantizer(latent, send_ids)

    receiver_states = {
        "correct": send_state,
        "cross_pump": cross_pump_receiver_state,
        "same_pump": same_pump_receiver_state,
        "representative": representative_receiver_state,
    }
    reconstructed: dict[str, torch.Tensor] = {}
    for name, receiver_state in receiver_states.items():
        receive_ids = torch.full(
            (count,), receiver_state, dtype=torch.long, device=device
        )
        reconstructed[name] = seeded_model.decode_indices(
            transmitted.base_indices,
            transmitted.residual_indices,
            receive_ids,
        )

    arrays = {"original": nchw_to_rgb(images)}
    arrays.update({name: nchw_to_rgb(value) for name, value in reconstructed.items()})
    base_indices = np.ascontiguousarray(
        transmitted.base_indices.detach().cpu().numpy(), dtype="<i8"
    )
    residual_indices = np.ascontiguousarray(
        transmitted.residual_indices.detach().cpu().numpy(), dtype="<i8"
    )
    provenance = {
        "sender_quantizer_calls": 1,
        "fixed_send_state": send_state,
        "receiver_states": receiver_states,
        "base_indices_shape": list(base_indices.shape),
        "residual_indices_shape": list(residual_indices.shape),
        "base_indices_int64le_sha256": _int64_array_sha256(base_indices),
        "residual_indices_int64le_sha256": _int64_array_sha256(residual_indices),
        "invariant": (
            "all four reconstructions reuse the exact base/residual index tensors; "
            "only receiver state_ids change"
        ),
    }
    return arrays, provenance


def _resolve_inputs(
    *,
    seeded_checkpoint_path: str | Path,
    software_checkpoint_path: str | Path,
    codebook_path: str | Path,
    manifest_path: str | Path,
    evaluation_dir: str | Path,
) -> dict[str, Path]:
    evaluation = Path(evaluation_dir).expanduser().resolve()
    paths = {
        "seeded_checkpoint": Path(seeded_checkpoint_path).expanduser().resolve(),
        "software_checkpoint": Path(software_checkpoint_path).expanduser().resolve(),
        "codebook": Path(codebook_path).expanduser().resolve(),
        "codebook_manifest": Path(manifest_path).expanduser().resolve(),
        "evaluation_manifest": evaluation / "evaluation_manifest.json",
        "reconstruction_summary": evaluation / "reconstruction_summary.json",
        "per_image_metrics": evaluation / "per_image_metrics.npz",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def _validate_formal_inputs(
    *,
    paths: dict[str, Path],
    evaluation_manifest: dict[str, Any],
    summary: dict[str, Any],
    codebook_manifest: dict[str, Any],
    bank: np.ndarray,
) -> tuple[list[str], list[dict[str, Any]], str]:
    dataset_info = evaluation_manifest.get("dataset", {})
    dataset_name = str(dataset_info.get("name", "")).lower().replace("-", "")
    summary_dataset = str(summary.get("dataset", "")).lower().replace("-", "")
    if evaluation_manifest.get("scope") != "final" or summary.get("scope") != "final":
        raise ValueError("CIFAR visuals require a formal full-test evaluation")
    if dataset_name != "cifar10" or summary_dataset != "cifar10":
        raise ValueError("this natural-image visual package requires CIFAR-10 results")
    if bool(dataset_info.get("test_truncated")) or bool(summary.get("test_truncated")):
        raise ValueError("CIFAR visuals reject truncated or smoke-test results")
    expected = int(dataset_info.get("expected_test_samples", -1))
    processed = int(dataset_info.get("processed_sample_count", -2))
    if expected != 10_000 or processed != expected:
        raise ValueError("formal CIFAR-10 evaluation must contain all 10,000 test images")
    if int(summary.get("processed_test_images", -3)) != processed:
        raise ValueError("evaluation manifest and reconstruction summary counts differ")

    recorded_outputs = evaluation_manifest.get("outputs", {})
    for filename, key in (
        ("reconstruction_summary.json", "reconstruction_summary"),
        ("per_image_metrics.npz", "per_image_metrics"),
    ):
        actual_hash = _sha256_file(paths[key])
        if recorded_outputs.get(filename) != actual_hash:
            raise ValueError(f"formal evaluation artifact hash mismatch: {filename}")

    inputs = evaluation_manifest.get("inputs", {})
    for key in ("seeded_checkpoint", "software_checkpoint"):
        recorded_hash = inputs.get(key, {}).get("sha256")
        actual_hash = _sha256_file(paths[key])
        if recorded_hash != actual_hash:
            raise ValueError(f"formal evaluation and {key} hashes differ")

    state_labels = _state_labels(codebook_manifest)
    records = state_records_from_manifest(codebook_manifest)
    bank_hash = _array_sha256(bank)
    if codebook_manifest.get("bank_array_sha256") != bank_hash:
        raise ValueError("codebook manifest and loaded bank array hashes differ")
    evaluation_codebook = inputs.get("codebook", {})
    if evaluation_codebook.get("array_sha256") != bank_hash:
        raise ValueError("formal evaluation and loaded bank array hashes differ")
    if evaluation_codebook.get("file_sha256") != _sha256_file(paths["codebook"]):
        raise ValueError("formal evaluation and codebook file hashes differ")
    if evaluation_codebook.get("manifest_sha256") != _sha256_file(
        paths["codebook_manifest"]
    ):
        raise ValueError("formal evaluation and codebook manifest hashes differ")
    return state_labels, records, bank_hash


def generate_cifar_natural_visuals(
    *,
    seeded_checkpoint_path: str | Path,
    software_checkpoint_path: str | Path,
    codebook_path: str | Path,
    manifest_path: str | Path,
    evaluation_dir: str | Path,
    output_dir: str | Path,
    device_name: str = "cuda:0",
    send_state: int = 2,
    cross_pump_receiver_state: int = 7,
    same_pump_receiver_state: int = 4,
    dpi: int = 300,
) -> dict[str, Any]:
    """Generate the formal CIFAR-10 presentation package from frozen artifacts."""

    paths = _resolve_inputs(
        seeded_checkpoint_path=seeded_checkpoint_path,
        software_checkpoint_path=software_checkpoint_path,
        codebook_path=codebook_path,
        manifest_path=manifest_path,
        evaluation_dir=evaluation_dir,
    )
    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"output directory already exists: {output_path}")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if send_state != 2:
        raise ValueError("the registered CIFAR PPT comparison fixes send_state=2")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    evaluation_manifest = _read_json(paths["evaluation_manifest"])
    summary = _read_json(paths["reconstruction_summary"])
    codebook_manifest = _read_json(paths["codebook_manifest"])
    bank = load_codebook_bank(paths["codebook"])
    state_labels, records, bank_hash = _validate_formal_inputs(
        paths=paths,
        evaluation_manifest=evaluation_manifest,
        summary=summary,
        codebook_manifest=codebook_manifest,
        bank=bank,
    )
    validate_controlled_state_mapping(
        records,
        send_state=send_state,
        cross_pump_receiver_state=cross_pump_receiver_state,
        same_pump_receiver_state=same_pump_receiver_state,
    )

    seeded_checkpoint = torch.load(paths["seeded_checkpoint"], map_location="cpu")
    software_checkpoint = torch.load(paths["software_checkpoint"], map_location="cpu")
    if seeded_checkpoint.get("base_bank_array_sha256") != bank_hash:
        raise ValueError("seeded checkpoint and loaded bank hashes differ")
    compatibility = _compatibility_report(
        seeded_checkpoint["config"], software_checkpoint["config"]
    )
    if (
        str(seeded_checkpoint["config"]["model"].get("quantizer_type"))
        != "seeded_residual"
        or str(software_checkpoint["config"]["model"].get("quantizer_type"))
        != "trainable_residual"
    ):
        raise ValueError("checkpoint quantizer types do not match seeded/software roles")

    dataset_config = seeded_checkpoint["config"]["dataset"]
    if str(dataset_config["name"]).lower().replace("-", "") != "cifar10":
        raise ValueError("seeded checkpoint is not configured for CIFAR-10")
    dataset = build_dataset(
        str(dataset_config["name"]),
        str(dataset_config["root"]),
        train=False,
        download=bool(dataset_config.get("download", False)),
    )
    fingerprint = _validate_dataset_fingerprint(dataset, evaluation_manifest)
    targets = _targets_from_dataset(dataset)
    if targets.shape != (len(dataset),):
        raise ValueError("CIFAR-10 targets do not match dataset length")

    with np.load(paths["per_image_metrics"], allow_pickle=False) as archive:
        required_keys = {"seeded_cross_mse", "seeded_cross_ssim"}
        missing = sorted(required_keys.difference(archive.files))
        if missing:
            raise ValueError(f"per-image archive is missing keys: {missing}")
        cross_mse = np.asarray(archive["seeded_cross_mse"], dtype=np.float64)
        cross_ssim = np.asarray(archive["seeded_cross_ssim"], dtype=np.float64)
        optional_lpips_available = "seeded_cross_lpips" in archive.files
    expected_shape = (len(dataset), 10, 10)
    if cross_mse.shape != expected_shape or cross_ssim.shape != expected_shape:
        raise ValueError("formal cross-state arrays must have shape (10000, 10, 10)")
    if not np.isfinite(cross_mse).all() or bool((cross_mse < 0).any()):
        raise ValueError("seeded_cross_mse contains invalid values")
    if not np.isfinite(cross_ssim).all():
        raise ValueError("seeded_cross_ssim contains invalid values")

    cross_summary = summary["seeded"]["cross_state"]
    mse_matrix = np.asarray(cross_summary["mse_matrix"], dtype=np.float64)
    ssim_matrix = np.asarray(cross_summary["ssim_mean_matrix"], dtype=np.float64)
    if mse_matrix.shape != (10, 10) or ssim_matrix.shape != (10, 10):
        raise ValueError("formal summary matrices must have shape (10, 10)")
    if not np.allclose(cross_mse.mean(axis=0), mse_matrix, rtol=1e-6, atol=1e-10):
        raise ValueError("per-image and summary MSE matrices are inconsistent")
    if not np.allclose(cross_ssim.mean(axis=0), ssim_matrix, rtol=1e-6, atol=1e-8):
        raise ValueError("per-image and summary SSIM matrices are inconsistent")

    diagonal_mse = np.diagonal(cross_mse, axis1=1, axis2=2)
    sample_ids = select_median_per_class(targets, diagonal_mse, class_count=10)
    samples = [dataset[index] for index in sample_ids]
    images = torch.stack([sample[0] for sample in samples])
    class_ids = [int(sample[1]) for sample in samples]
    if class_ids != list(range(10)):
        raise RuntimeError("class-median selection did not return CIFAR classes 0..9")

    representative = select_representative_wrong_pair(
        mse_matrix,
        send_state=send_state,
        excluded_pairs={
            (send_state, cross_pump_receiver_state),
            (send_state, same_pump_receiver_state),
        },
    )
    representative_receiver = int(representative["receiver_state"])
    seeded_model = _load_model(
        seeded_checkpoint, expected_quantizer_type="seeded_residual", bank=bank
    ).to(device).eval()
    arrays, inference_provenance = run_controlled_inference(
        seeded_model,
        images,
        send_state=send_state,
        cross_pump_receiver_state=cross_pump_receiver_state,
        same_pump_receiver_state=same_pump_receiver_state,
        representative_receiver_state=representative_receiver,
        device=device,
    )

    ratios = row_relative_mse(mse_matrix)
    mappings = {
        "correct": (send_state, send_state),
        "cross_pump": (send_state, cross_pump_receiver_state),
        "same_pump": (send_state, same_pump_receiver_state),
        "representative": (send_state, representative_receiver),
    }
    pair_ratios = {
        key: float(ratios[sender, receiver])
        for key, (sender, receiver) in mappings.items()
    }
    diagnostic = extract_component_diagnostics(summary)

    output_path.mkdir(parents=True, exist_ok=False)
    with plt.rc_context(RC_PARAMS):
        figures: dict[str, Any] = {
            "cross_state_rmse_heatmap": plot_cross_state_heatmap(
                mse_matrix,
                state_labels=state_labels,
                ssim_matrix=ssim_matrix,
                output_dir=output_path,
                dpi=dpi,
            ),
            "controlled_mismatch_gallery": plot_controlled_gallery(
                arrays,
                sample_ids=sample_ids,
                class_ids=class_ids,
                records=records,
                mappings=mappings,
                pair_ratios=pair_ratios,
                output_dir=output_path,
                dpi=dpi,
            ),
            "controlled_mismatch_gallery_ppt_classes_0_4": plot_controlled_gallery_page(
                arrays,
                sample_ids=sample_ids,
                class_ids=class_ids,
                class_start=0,
                records=records,
                mappings=mappings,
                pair_ratios=pair_ratios,
                output_dir=output_path,
                dpi=dpi,
            ),
            "controlled_mismatch_gallery_ppt_classes_5_9": plot_controlled_gallery_page(
                arrays,
                sample_ids=sample_ids,
                class_ids=class_ids,
                class_start=5,
                records=records,
                mappings=mappings,
                pair_ratios=pair_ratios,
                output_dir=output_path,
                dpi=dpi,
            ),
        }
        if diagnostic is not None:
            figures["components_and_residual_diagnostic"] = plot_component_diagnostic(
                diagnostic,
                state_labels=state_labels,
                output_dir=output_path,
                dpi=dpi,
            )

    sample_scores = diagonal_mse.mean(axis=1)
    project_root = Path(__file__).resolve().parents[3]
    manifest: dict[str, Any] = {
        "schema": CIFAR_VISUAL_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(project_root),
        "input_files": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in paths.items()
        },
        "base_bank": {
            "array_sha256": bank_hash,
            "shape": list(bank.shape),
        },
        "formal_evaluation": {
            "scope": evaluation_manifest["scope"],
            "dataset": "cifar10",
            "processed_test_images": len(dataset),
            "dataset_fingerprint": fingerprint,
            "optional_lpips_available": optional_lpips_available,
            "primary_matrix_metric": "pixel MSE and sqrt(MSE) full-scale RMSE",
            "auxiliary_matrix_metric": "SSIM",
        },
        "checkpoint_compatibility": compatibility,
        "selection": {
            "rule": (
                "for each CIFAR-10 class, average the ten synchronized diagonal "
                "per-image MSE values; choose the official-test index closest to "
                "the within-class median; ties choose the smaller official index"
            ),
            "manual_visual_selection": False,
            "sample_ids": sample_ids,
            "class_ids": class_ids,
            "class_names_en": [CIFAR10_CLASS_NAMES_EN[item] for item in class_ids],
            "class_names_zh": [CIFAR10_CLASS_NAMES_ZH[item] for item in class_ids],
            "mean_diagonal_mse_at_selected_ids": [
                float(sample_scores[index]) for index in sample_ids
            ],
        },
        "controlled_mappings": {
            "registered_send_state": send_state,
            "registered_send_state_id": state_labels[send_state],
            "mappings": {
                key: {
                    "send_state": sender,
                    "send_state_id": state_labels[sender],
                    "receiver_state": receiver,
                    "receiver_state_id": state_labels[receiver],
                    "pooled_mse": float(mse_matrix[sender, receiver]),
                    "row_relative_mse": pair_ratios[key],
                }
                for key, (sender, receiver) in mappings.items()
            },
            "representative_rule": {
                "definition": (
                    "within fixed sender state 2, compute the median pooled MSE "
                    "over all nine wrong receivers; after excluding the two "
                    "pre-registered receivers, choose the closest pooled MSE; "
                    "ties choose the smaller receiver index"
                ),
                "excluded_pre_registered_pairs": [
                    [send_state, cross_pump_receiver_state],
                    [send_state, same_pump_receiver_state],
                ],
                **representative,
            },
            "transport_index_reuse": inference_provenance,
        },
        "component_diagnostic": (
            {
                "status": "generated_from_formal_summary",
                "mse_multipliers": diagnostic["mse_multipliers"],
                "base_coverage_by_state": diagnostic[
                    "base_coverage_by_state"
                ].tolist(),
                "residual_coverage_by_state": diagnostic[
                    "residual_coverage_by_state"
                ].tolist(),
                "pooled_residual_coverage": diagnostic["pooled_residual_coverage"],
            }
            if diagnostic is not None
            else {
                "status": "not_generated_missing_formal_component_or_usage_metrics"
            }
        ),
        "rendering": {
            "formats": ["png", "pdf"],
            "dpi": dpi,
            "rgb_display_range": [0.0, 1.0],
            "rgb_display_scaling": "fixed; clip only; no per-image normalization",
            "interpolation": "nearest",
            "absolute_error_definition": "mean(abs(RGB reconstruction - RGB original), axis=channels)",
            "absolute_error_display_range": [0.0, 1.0],
            "absolute_error_scale_shared_across_galleries": True,
            "absolute_error_colormap": "inferno",
            "absolute_error_legend": "vertical colorbar: black=0, yellow=1",
            "ppt_gallery_pages": [
                {
                    "classes": [0, 1, 2, 3, 4],
                    "aspect_ratio": "16:9",
                    "canvas_inches": [40.0 / 3.0, 7.5],
                },
                {
                    "classes": [5, 6, 7, 8, 9],
                    "aspect_ratio": "16:9",
                    "canvas_inches": [40.0 / 3.0, 7.5],
                },
            ],
            "font_fallbacks": ZH_FONT_FAMILY,
        },
        "interpretation_boundary": (
            "The figures visualize formal reconstruction, receiver-state mismatch, "
            "and component ablations. They do not establish entropy, PUF behavior, "
            "unclonability, physical switching speed, or cryptographic security. "
            "Residual-code coverage or magnitude alone is not proof against bypass."
        ),
        "figures": figures,
    }
    manifest["manifest_payload_sha256"] = _json_sha256(manifest)
    output_manifest = output_path / "cifar_visual_manifest.json"
    output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    manifest = generate_cifar_natural_visuals(
        seeded_checkpoint_path=args.seeded_checkpoint,
        software_checkpoint_path=args.software_checkpoint,
        codebook_path=args.codebook,
        manifest_path=args.manifest,
        evaluation_dir=args.evaluation_dir,
        output_dir=args.output_dir,
        device_name=args.device,
        send_state=args.send_state,
        cross_pump_receiver_state=args.cross_pump_receiver_state,
        same_pump_receiver_state=args.same_pump_receiver_state,
        dpi=args.dpi,
    )
    print(f"selected CIFAR-10 test indices: {manifest['selection']['sample_ids']}")
    print(f"saved: {Path(args.output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
