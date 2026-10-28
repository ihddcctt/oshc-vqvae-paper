from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import torch

from ..data import build_dataset
from ..seed.codebook import load_codebook_bank
from .evaluate_final import (
    _array_sha256,
    _compatibility_report,
    _load_model,
    _read_json,
    _sha256_file,
    _state_labels,
)
from .plot_final_evaluation import compact_state_label


EXAMPLE_FIGURE_SCHEMA = "lightwave-seeded-vqvae-ppt-visuals-v1"
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
    "gray": "#666666",
    "light_gray": "#F2F2F2",
    "red": "#C83E4D",
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
            "Generate reproducible Chinese PPT visuals from the frozen final MNIST "
            "checkpoints and evaluation artifacts"
        )
    )
    parser.add_argument("--seeded-checkpoint", required=True, type=Path)
    parser.add_argument("--software-checkpoint", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--aes-result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--send-state", type=_state_index, default=2)
    parser.add_argument("--wrong-receiver-state", type=_state_index, default=7)
    parser.add_argument("--schedule-shift", type=_positive_int, default=1)
    parser.add_argument("--dpi", type=_positive_int, default=300)
    return parser


def select_median_per_class(
    targets: Sequence[int] | np.ndarray,
    correct_mse: np.ndarray,
    *,
    class_count: int = 10,
) -> list[int]:
    """Select a deterministic representative closest to each class median.

    ``correct_mse`` may be one value per image or one value per image/state. In the
    latter case the state values are averaged first. Candidate indices are already
    in official dataset order, so ``argmin`` makes a tie choose the smaller index.
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
        class_scores = score_array[candidates]
        median = float(np.median(class_scores))
        selected.append(int(candidates[np.abs(class_scores - median).argmin()]))
    return selected


def shifted_receiver_schedule(sender_states: Sequence[int], shift: int) -> np.ndarray:
    sender = np.asarray(sender_states, dtype=np.int64)
    if sender.ndim != 1 or sender.size == 0:
        raise ValueError("sender_states must be a non-empty vector")
    if shift <= 0 or shift >= sender.size:
        raise ValueError("shift must lie in [1, len(sender_states)-1]")
    return np.roll(sender, -shift)


def cross_pump_same_power_state(state_index: int) -> int:
    if not 0 <= state_index < 10:
        raise ValueError("state_index must lie in [0, 9]")
    return (state_index + 5) % 10


def row_relative_mse(mse_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(mse_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("mse_matrix must be square")
    if not np.isfinite(matrix).all() or bool((matrix <= 0).any()):
        raise ValueError("mse_matrix must contain finite positive values")
    diagonal = np.diag(matrix)
    return matrix / diagonal[:, None]


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


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _human_state_label(state_id: str) -> str:
    compact = compact_state_label(state_id)
    try:
        pump, power = compact.split("/")
        return f"{pump} nm / {power} mW"
    except ValueError:
        return state_id


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
        metadata={"Title": title, "Creator": "lightwave-ppt-visuals"},
    )
    plt.close(figure)
    return {
        "title": title,
        "png": {"path": str(png_path), "sha256": _sha256_file(png_path)},
        "pdf": {"path": str(pdf_path), "sha256": _sha256_file(pdf_path)},
    }


def _show_image(axis: plt.Axes, image: np.ndarray, *, difference: bool = False) -> None:
    if difference:
        axis.imshow(
            image,
            cmap="inferno",
            vmin=0.0,
            vmax=0.5,
            interpolation="nearest",
        )
    else:
        axis.imshow(
            image,
            cmap="gray",
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
        -0.16,
        0.5,
        label,
        transform=axis.transAxes,
        ha="right",
        va="center",
        fontsize=9.5,
        fontweight="bold",
    )


def _plot_sync_wrong_gallery(
    arrays: dict[str, np.ndarray],
    *,
    sample_ids: Sequence[int],
    digits: Sequence[int],
    send_label: str,
    wrong_label: str,
    pair_ratio: float,
    overall_ratio: float,
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    title = "状态同步时可正常恢复；接收状态错误时重建明显退化"
    rows = [
        ("原始图像", arrays["original"], False),
        ("状态同步", arrays["correct"], False),
        ("状态错配", arrays["wrong"], False),
        ("两种重建的\n绝对差值", np.abs(arrays["correct"] - arrays["wrong"]), True),
    ]
    figure, axes = plt.subplots(4, 10, figsize=(13.33, 6.25))
    for row_index, (row_label, images, is_difference) in enumerate(rows):
        for column in range(10):
            _show_image(axes[row_index, column], images[column], difference=is_difference)
            if row_index == 0:
                axes[row_index, column].set_title(
                    f"数字 {digits[column]}\n测试集 #{sample_ids[column]}", fontsize=8.2
                )
        _label_row(axes[row_index, 0], row_label)
    figure.suptitle(title, fontsize=16, fontweight="bold", y=0.975)
    figure.text(
        0.5,
        0.018,
        (
            f"固定案例：发送 {send_label}；错误接收 {wrong_label}。"
            f"该状态对的完整测试误差为同步的 {pair_ratio:.2f}×；"
            f"全部 90 种错配汇总为 {overall_ratio:.2f}×。差值图统一显示 0–0.5。"
        ),
        ha="center",
        va="bottom",
        fontsize=9.0,
        color=COLORS["gray"],
    )
    figure.subplots_adjust(left=0.10, right=0.995, bottom=0.09, top=0.88, wspace=0.04, hspace=0.08)
    result = _save_figure(
        figure,
        output_dir=output_dir,
        stem="ppt_01_sync_vs_wrong_gallery_zh",
        title=title,
        dpi=dpi,
    )
    result["content"] = "original, synchronized, fixed cross-pump mismatch, absolute difference"
    return result


def _plot_component_gallery(
    arrays: dict[str, np.ndarray],
    *,
    sample_ids: Sequence[int],
    digits: Sequence[int],
    multipliers: dict[str, float],
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    title = "芯片基础码本与共享补码共同完成重建"
    rows = [
        ("原始图像", arrays["original"]),
        ("全可训练\n软件基线", arrays["software"]),
        ("完整 $B_s+R$", arrays["correct"]),
        ("仅 $B_s$", arrays["base_only"]),
        ("仅 $R$\n（重新量化）", arrays["residual_only"]),
    ]
    figure, axes = plt.subplots(5, 10, figsize=(13.33, 7.35))
    for row_index, (row_label, images) in enumerate(rows):
        for column in range(10):
            _show_image(axes[row_index, column], images[column])
            if row_index == 0:
                axes[row_index, column].set_title(
                    f"数字 {digits[column]}\n测试集 #{sample_ids[column]}", fontsize=8.1
                )
        _label_row(axes[row_index, 0], row_label)
    figure.suptitle(title, fontsize=16, fontweight="bold", y=0.975)
    figure.text(
        0.5,
        0.015,
        (
            "完整模型误差设为 1×："
            f"软件基线 {multipliers['software']:.2f}×，"
            f"仅 $B_s$ {multipliers['base_only']:.2f}×，"
            f"仅 $R$（重新量化）{multipliers['residual_only']:.2f}×。"
            "所有面板使用同一批样本和固定灰度范围 [0,1]。"
        ),
        ha="center",
        va="bottom",
        fontsize=9.0,
        color=COLORS["gray"],
    )
    figure.subplots_adjust(left=0.10, right=0.995, bottom=0.075, top=0.89, wspace=0.04, hspace=0.075)
    result = _save_figure(
        figure,
        output_dir=output_dir,
        stem="ppt_02_codebook_components_zh",
        title=title,
        dpi=dpi,
    )
    result["content"] = "original, software baseline, full B_s+R, B_s-only, re-quantized R-only"
    return result


def _add_panel_border(axis: plt.Axes, color: str) -> None:
    axis.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=axis.transAxes,
            fill=False,
            edgecolor=color,
            linewidth=2.0,
            clip_on=False,
        )
    )


def _plot_timing_gallery(
    arrays: dict[str, np.ndarray],
    *,
    digits: Sequence[int],
    state_labels: Sequence[str],
    shifted_states: Sequence[int],
    shift: int,
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    title = f"固定状态时序：接收端错位 {shift} 帧后重建退化（离线软件模拟）"
    rows = [
        ("连续输入", arrays["original"]),
        ("收发状态同步", arrays["timeline_correct"]),
        (f"接收序列错位 {shift} 帧", arrays["timeline_shifted"]),
    ]
    figure, axes = plt.subplots(3, 10, figsize=(13.33, 5.15))
    for row_index, (row_label, images) in enumerate(rows):
        for column in range(10):
            axis = axes[row_index, column]
            _show_image(axis, images[column])
            if row_index == 0:
                axis.set_title(
                    f"t={column} · 数字 {digits[column]}\n发送 {compact_state_label(state_labels[column])}",
                    fontsize=7.8,
                )
            elif row_index == 1:
                _add_panel_border(axis, COLORS["green"])
            else:
                _add_panel_border(axis, COLORS["red"])
                axis.text(
                    0.5,
                    -0.12,
                    f"接收 {compact_state_label(state_labels[int(shifted_states[column])])}",
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=7.1,
                    color=COLORS["red"],
                )
        _label_row(axes[row_index, 0], row_label)
    figure.suptitle(title, fontsize=16, fontweight="bold", y=0.975)
    figure.text(
        0.5,
        0.018,
        "绿色边框：接收端选择同一状态；红色边框：接收端状态序列整体错位。这里只验证状态选择逻辑，不代表实测芯片切换速度。",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color=COLORS["gray"],
    )
    figure.subplots_adjust(left=0.10, right=0.995, bottom=0.11, top=0.86, wspace=0.05, hspace=0.12)
    result = _save_figure(
        figure,
        output_dir=output_dir,
        stem="ppt_03_timing_sync_vs_shift_zh",
        title=title,
        dpi=dpi,
    )
    result["content"] = "ten-state sender sequence, synchronized receiver, shifted receiver"
    return result


def _plot_relative_error_matrix(
    relative_matrix: np.ndarray,
    *,
    state_labels: Sequence[str],
    overall_ratio: float,
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    title = "10 种芯片状态的全部 100 种收发组合"
    labels = [compact_state_label(label) for label in state_labels]
    vmax = float(relative_matrix.max())
    figure, axis = plt.subplots(figsize=(8.1, 6.7))
    image = axis.imshow(
        relative_matrix,
        cmap="RdYlGn_r",
        norm=LogNorm(vmin=min(0.9, float(relative_matrix.min())), vmax=vmax),
        aspect="equal",
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    ticks = [value for value in (1, 2, 5, 10, 20) if value <= vmax]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{value}×" for value in ticks])
    colorbar.set_label("相对于该发送状态同步解码的像素均方误差（越小越好）")
    axis.set_xticks(np.arange(10), labels=labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(10), labels=labels)
    axis.set_xlabel("接收端状态（泵浦 nm / 功率 mW）")
    axis.set_ylabel("发送端状态（泵浦 nm / 功率 mW）")
    axis.set_title(title, fontweight="bold")
    log_midpoint = (np.log10(max(0.9, float(relative_matrix.min()))) + np.log10(vmax)) / 2
    for row in range(10):
        for column in range(10):
            value = float(relative_matrix[row, column])
            text = "1× ✓" if row == column else f"{value:.1f}×"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=6.7,
                fontweight="bold" if row == column else "normal",
                color="white" if np.log10(value) > log_midpoint else "black",
            )
            if row == column:
                axis.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor=COLORS["green"],
                        linewidth=2.0,
                    )
                )
    figure.text(
        0.5,
        0.012,
        f"绿色对角线为状态同步；全部 90 种非对角错配汇总后，均方误差为同步情形的 {overall_ratio:.3f}×。",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color=COLORS["gray"],
    )
    figure.subplots_adjust(left=0.14, right=0.90, bottom=0.15, top=0.91)
    result = _save_figure(
        figure,
        output_dir=output_dir,
        stem="ppt_04_relative_mse_matrix_zh",
        title=title,
        dpi=dpi,
    )
    result["content"] = "row-normalized MSE matrix; every sender-state diagonal is 1x"
    return result


def _rounded_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    text: str,
    fontsize: float = 10.0,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        transform=axis.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.4,
    )
    axis.add_patch(box)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def _plot_plain_summary(
    multipliers: dict[str, float],
    *,
    aes_result: dict[str, Any],
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    title = "两层机制分工：状态相关表示 + 标准认证加密"
    figure, (metric_axis, aes_axis) = plt.subplots(
        1, 2, figsize=(13.33, 5.4), gridspec_kw={"width_ratios": [1.05, 1.0]}
    )
    labels = [
        "全可训练软件基线",
        "完整 $B_s+R$（同步）",
        "仅 $R$（重新量化）",
        "仅 $B_s$",
        "错误状态（90种汇总）",
    ]
    values = np.asarray(
        [
            multipliers["software"],
            1.0,
            multipliers["residual_only"],
            multipliers["base_only"],
            multipliers["wrong"],
        ],
        dtype=np.float64,
    )
    colors = [
        COLORS["blue"],
        COLORS["green"],
        COLORS["yellow"],
        COLORS["orange"],
        COLORS["red"],
    ]
    positions = np.arange(len(labels))
    bars = metric_axis.barh(positions, values, color=colors, height=0.65)
    metric_axis.set_yticks(positions, labels=labels)
    metric_axis.invert_yaxis()
    metric_axis.set_xlabel("相对于完整同步模型的像素均方误差（1×为基准）")
    metric_axis.set_title("表示层：不用理解 dB，直接比较误差倍数", fontweight="bold")
    metric_axis.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    metric_axis.set_axisbelow(True)
    metric_axis.set_xlim(0, max(10.4, float(values.max()) * 1.12))
    for bar, value in zip(bars, values):
        metric_axis.text(
            value + 0.12,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}×",
            va="center",
            ha="left",
            fontsize=10.0,
            fontweight="bold",
        )

    equivalence = aes_result["results"]["equivalence"]
    probes = aes_result["results"]["authentication_probes"]
    aes_axis.set_axis_off()
    aes_axis.set_title("传输层：AES-256-GCM", fontweight="bold")
    _rounded_box(
        aes_axis,
        (0.03, 0.72),
        0.25,
        0.15,
        facecolor="#E8F1F8",
        edgecolor=COLORS["blue"],
        text="两路 VQ 索引\n（74 bytes）",
    )
    _rounded_box(
        aes_axis,
        (0.38, 0.72),
        0.24,
        0.15,
        facecolor="#FFF2D8",
        edgecolor=COLORS["yellow"],
        text="AES-GCM\n加密 + 认证",
    )
    _rounded_box(
        aes_axis,
        (0.72, 0.72),
        0.25,
        0.15,
        facecolor="#E4F4ED",
        edgecolor=COLORS["green"],
        text="认证通过后\n才进入 Decoder",
    )
    for start, end in ((0.28, 0.38), (0.62, 0.72)):
        aes_axis.annotate(
            "",
            xy=(end, 0.795),
            xytext=(start, 0.795),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": COLORS["gray"], "lw": 1.8},
        )
    exact_frames = int(equivalence["aes_off_vs_aes_on_exact_index_frames"])
    probe_total = int(probes["total_attempted"])
    rejected = int(probes["total_authentication_rejections"])
    decoder_calls = int(probes["total_decoder_invocations"])
    _rounded_box(
        aes_axis,
        (0.08, 0.39),
        0.84,
        0.19,
        facecolor="#E4F4ED",
        edgecolor=COLORS["green"],
        text=(
            f"正确密钥且报文完整：{exact_frames:,}/{exact_frames:,} 帧\n"
            "AES 开关前后索引与同一 Decoder 重建完全一致"
        ),
        fontsize=10.5,
    )
    _rounded_box(
        aes_axis,
        (0.08, 0.11),
        0.84,
        0.19,
        facecolor="#FBE9EC",
        edgecolor=COLORS["red"],
        text=(
            f"错误密钥或单比特篡改：{rejected}/{probe_total} 个注册探针被拒绝\n"
            f"Decoder 调用次数 = {decoder_calls}"
        ),
        fontsize=10.5,
    )
    aes_axis.text(
        0.5,
        0.015,
        "AES 密钥独立于芯片信号；码本错配不会被描述成 AES 认证失败。",
        transform=aes_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.0,
        color=COLORS["gray"],
    )
    figure.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    figure.subplots_adjust(left=0.14, right=0.985, bottom=0.15, top=0.86, wspace=0.27)
    result = _save_figure(
        figure,
        output_dir=output_dir,
        stem="ppt_05_error_multipliers_and_aes_zh",
        title=title,
        dpi=dpi,
    )
    result["content"] = "plain-language MSE multipliers and AES-GCM/chip-state responsibility boundary"
    return result


def _targets_from_dataset(dataset: Any) -> np.ndarray:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        return np.asarray([int(dataset[index][1]) for index in range(len(dataset))])
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    return np.asarray(targets, dtype=np.int64)


@torch.inference_mode()
def _run_example_inference(
    seeded_model: Any,
    software_model: Any,
    images: torch.Tensor,
    *,
    send_state: int,
    wrong_receiver_state: int,
    schedule_shift: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    images = images.to(device)
    count = int(images.shape[0])
    fixed_send = torch.full((count,), send_state, dtype=torch.long, device=device)
    fixed_wrong = torch.full(
        (count,), wrong_receiver_state, dtype=torch.long, device=device
    )
    latent = seeded_model.encoder(images)
    quantized = seeded_model.quantizer(latent, fixed_send)
    correct = seeded_model.decode_indices(
        quantized.base_indices, quantized.residual_indices, fixed_send
    )
    wrong = seeded_model.decode_indices(
        quantized.base_indices, quantized.residual_indices, fixed_wrong
    )
    base_only = seeded_model.decode_indices(
        quantized.base_indices,
        quantized.residual_indices,
        fixed_send,
        include_base=True,
        include_residual=False,
    )
    residual_only, _ = seeded_model.reconstruct_residual_only(latent)
    software_state = torch.zeros(count, dtype=torch.long, device=device)
    software = software_model(images, software_state).reconstruction

    timeline_send = torch.arange(count, dtype=torch.long, device=device)
    if count != seeded_model.quantizer.state_count:
        raise ValueError("timeline gallery requires one example for every chip state")
    shifted_numpy = shifted_receiver_schedule(
        timeline_send.cpu().numpy(), schedule_shift
    )
    timeline_receive = torch.as_tensor(
        shifted_numpy, dtype=torch.long, device=device
    )
    timeline_quantized = seeded_model.quantizer(latent, timeline_send)
    timeline_correct = seeded_model.decode_indices(
        timeline_quantized.base_indices,
        timeline_quantized.residual_indices,
        timeline_send,
    )
    timeline_shifted = seeded_model.decode_indices(
        timeline_quantized.base_indices,
        timeline_quantized.residual_indices,
        timeline_receive,
    )

    def images_numpy(tensor: torch.Tensor) -> np.ndarray:
        array = tensor.detach().float().cpu().numpy()
        if array.shape[1] != 1:
            raise ValueError("PPT gallery currently expects single-channel MNIST images")
        return np.clip(array[:, 0], 0.0, 1.0)

    arrays = {
        "original": images_numpy(images),
        "software": images_numpy(software),
        "correct": images_numpy(correct),
        "wrong": images_numpy(wrong),
        "base_only": images_numpy(base_only),
        "residual_only": images_numpy(residual_only),
        "timeline_correct": images_numpy(timeline_correct),
        "timeline_shifted": images_numpy(timeline_shifted),
    }
    return arrays, shifted_numpy


def generate_ppt_visuals(
    *,
    seeded_checkpoint_path: str | Path,
    software_checkpoint_path: str | Path,
    codebook_path: str | Path,
    manifest_path: str | Path,
    evaluation_dir: str | Path,
    aes_result_path: str | Path,
    output_dir: str | Path,
    device_name: str = "cuda:0",
    send_state: int = 2,
    wrong_receiver_state: int = 7,
    schedule_shift: int = 1,
    dpi: int = 300,
) -> dict[str, Any]:
    paths = {
        "seeded_checkpoint": Path(seeded_checkpoint_path).expanduser().resolve(),
        "software_checkpoint": Path(software_checkpoint_path).expanduser().resolve(),
        "codebook": Path(codebook_path).expanduser().resolve(),
        "codebook_manifest": Path(manifest_path).expanduser().resolve(),
        "evaluation_manifest": Path(evaluation_dir).expanduser().resolve()
        / "evaluation_manifest.json",
        "reconstruction_summary": Path(evaluation_dir).expanduser().resolve()
        / "reconstruction_summary.json",
        "per_image_metrics": Path(evaluation_dir).expanduser().resolve()
        / "per_image_metrics.npz",
        "aes_result": Path(aes_result_path).expanduser().resolve(),
    }
    output_path = Path(output_dir).expanduser().resolve()
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(f"output directory already exists: {output_path}")
    if not 0 <= send_state < 10 or not 0 <= wrong_receiver_state < 10:
        raise ValueError("send and wrong receiver states must lie in [0, 9]")
    if send_state == wrong_receiver_state:
        raise ValueError("wrong receiver state must differ from sender state")
    if not 1 <= schedule_shift < 10:
        raise ValueError("schedule_shift must lie in [1, 9]")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    codebook_manifest = _read_json(paths["codebook_manifest"])
    state_labels = _state_labels(codebook_manifest)
    bank = load_codebook_bank(paths["codebook"])
    bank_hash = _array_sha256(bank)
    if codebook_manifest.get("bank_array_sha256") != bank_hash:
        raise ValueError("codebook manifest and bank array hashes differ")

    evaluation_manifest = _read_json(paths["evaluation_manifest"])
    summary = _read_json(paths["reconstruction_summary"])
    aes_result = _read_json(paths["aes_result"])
    if summary.get("scope") != "final" or evaluation_manifest.get("scope") != "final":
        raise ValueError("PPT visuals require the final full-test evaluation")
    recorded_outputs = evaluation_manifest.get("outputs", {})
    for name in ("reconstruction_summary.json", "per_image_metrics.npz"):
        recorded = recorded_outputs.get(name)
        actual = _sha256_file(Path(evaluation_dir).expanduser().resolve() / name)
        if recorded != actual:
            raise ValueError(f"final evaluation artifact hash mismatch: {name}")
    if aes_result.get("scope") != "formal":
        raise ValueError("PPT AES panel requires a formal AES result")
    aes_provenance = aes_result.get("provenance", {})
    aes_results = aes_result.get("results", {})
    aes_checkpoint_hash = aes_provenance.get("checkpoint", {}).get("sha256")
    if aes_checkpoint_hash != _sha256_file(paths["seeded_checkpoint"]):
        raise ValueError("AES result and seeded checkpoint hashes differ")
    aes_bank_hash = aes_provenance.get("base_bank", {}).get("array_sha256")
    if aes_bank_hash != bank_hash:
        raise ValueError("AES result and base-bank array hashes differ")
    evaluation_dataset = evaluation_manifest.get("dataset", {})
    aes_fingerprint = aes_results.get("dataset_fingerprint", {})
    if (
        aes_fingerprint.get("processed_images_float32le_sha256")
        != evaluation_dataset.get("processed_images_float32_sha256")
        or aes_fingerprint.get("processed_labels_int64le_sha256")
        != evaluation_dataset.get("processed_labels_int64_sha256")
    ):
        raise ValueError("AES and reconstruction evaluations used different datasets")
    expected_frames = int(evaluation_dataset.get("processed_sample_count", -1))
    if int(aes_results.get("processed_sample_count", -2)) != expected_frames:
        raise ValueError("AES and reconstruction evaluation frame counts differ")
    equivalence = aes_results.get("equivalence", {})
    if not (
        equivalence.get("aes_off_vs_aes_on_all_indices_bit_identical") is True
        and equivalence.get("same_decoder_all_reconstructions_bit_identical") is True
        and int(equivalence.get("aes_off_vs_aes_on_exact_index_frames", -1))
        == expected_frames
        and int(equivalence.get("same_decoder_exact_reconstruction_frames", -1))
        == expected_frames
        and float(equivalence.get("max_reconstruction_absolute_difference", -1.0))
        == 0.0
    ):
        raise ValueError("AES equivalence checks are incomplete or failed")
    probes = aes_results.get("authentication_probes", {})
    if not (
        int(probes.get("total_attempted", -1)) > 0
        and int(probes.get("total_authentication_rejections", -2))
        == int(probes.get("total_attempted", -1))
        and int(probes.get("total_unexpected_acceptances", -1)) == 0
        and int(probes.get("total_decoder_invocations", -1)) == 0
    ):
        raise ValueError("AES authentication probes are incomplete or failed")

    seeded_checkpoint = torch.load(paths["seeded_checkpoint"], map_location="cpu")
    software_checkpoint = torch.load(paths["software_checkpoint"], map_location="cpu")
    if seeded_checkpoint.get("base_bank_array_sha256") != bank_hash:
        raise ValueError("seeded checkpoint and base bank hashes differ")
    _compatibility_report(seeded_checkpoint["config"], software_checkpoint["config"])
    seeded_model = _load_model(
        seeded_checkpoint, expected_quantizer_type="seeded_residual", bank=bank
    ).to(device).eval()
    software_model = _load_model(
        software_checkpoint, expected_quantizer_type="trainable_residual", bank=None
    ).to(device).eval()

    dataset_config = seeded_checkpoint["config"]["dataset"]
    if str(dataset_config["name"]).lower() != "mnist":
        raise ValueError("this PPT visual package is registered for MNIST")
    dataset = build_dataset(
        str(dataset_config["name"]),
        str(dataset_config["root"]),
        train=False,
        download=bool(dataset_config.get("download", False)),
    )
    targets = _targets_from_dataset(dataset)
    with np.load(paths["per_image_metrics"], allow_pickle=False) as archive:
        cross_mse = np.asarray(archive["seeded_cross_mse"], dtype=np.float64)
    if cross_mse.shape != (len(dataset), 10, 10):
        raise ValueError("seeded_cross_mse does not match the complete MNIST test set")
    diagonal_mse = np.diagonal(cross_mse, axis1=1, axis2=2)
    sample_ids = select_median_per_class(targets, diagonal_mse, class_count=10)
    samples = [dataset[index] for index in sample_ids]
    images = torch.stack([sample[0] for sample in samples])
    digits = [int(sample[1]) for sample in samples]
    if digits != list(range(10)):
        raise RuntimeError("class-median selection did not return digits in 0..9 order")

    arrays, shifted_states = _run_example_inference(
        seeded_model,
        software_model,
        images,
        send_state=send_state,
        wrong_receiver_state=wrong_receiver_state,
        schedule_shift=schedule_shift,
        device=device,
    )
    cross_summary = summary["seeded"]["cross_state"]
    mse_matrix = np.asarray(cross_summary["mse_matrix"], dtype=np.float64)
    correct_mse = float(cross_summary["correct_pooled"]["mse"])
    wrong_mse = float(cross_summary["wrong_pooled"]["mse"])
    pair_ratio = float(mse_matrix[send_state, wrong_receiver_state] / mse_matrix[send_state, send_state])
    multipliers = {
        "software": float(summary["software"]["overall"]["mse"]) / correct_mse,
        "base_only": float(summary["seeded"]["branches"]["base_only"]["pooled"]["mse"])
        / correct_mse,
        "residual_only": float(
            summary["seeded"]["branches"]["requantized_residual_only"]["mse"]
        )
        / correct_mse,
        "wrong": wrong_mse / correct_mse,
    }
    recorded_ratio = float(cross_summary["wrong_to_correct_mse_ratio"])
    if not np.isclose(multipliers["wrong"], recorded_ratio, rtol=1e-10, atol=1e-12):
        raise ValueError("wrong/correct MSE ratio is inconsistent in the final summary")

    output_path.mkdir(parents=True, exist_ok=False)
    send_label = _human_state_label(state_labels[send_state])
    wrong_label = _human_state_label(state_labels[wrong_receiver_state])
    with plt.rc_context(RC_PARAMS):
        figures = {
            "sync_vs_wrong_gallery": _plot_sync_wrong_gallery(
                arrays,
                sample_ids=sample_ids,
                digits=digits,
                send_label=send_label,
                wrong_label=wrong_label,
                pair_ratio=pair_ratio,
                overall_ratio=recorded_ratio,
                output_dir=output_path,
                dpi=dpi,
            ),
            "codebook_components": _plot_component_gallery(
                arrays,
                sample_ids=sample_ids,
                digits=digits,
                multipliers=multipliers,
                output_dir=output_path,
                dpi=dpi,
            ),
            "timing_sync_vs_shift": _plot_timing_gallery(
                arrays,
                digits=digits,
                state_labels=state_labels,
                shifted_states=shifted_states,
                shift=schedule_shift,
                output_dir=output_path,
                dpi=dpi,
            ),
            "relative_mse_matrix": _plot_relative_error_matrix(
                row_relative_mse(mse_matrix),
                state_labels=state_labels,
                overall_ratio=recorded_ratio,
                output_dir=output_path,
                dpi=dpi,
            ),
            "error_multipliers_and_aes": _plot_plain_summary(
                multipliers,
                aes_result=aes_result,
                output_dir=output_path,
                dpi=dpi,
            ),
        }

    project_root = Path(__file__).resolve().parents[3]
    sample_scores = diagonal_mse.mean(axis=1)
    manifest = {
        "schema": EXAMPLE_FIGURE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(project_root),
        "input_files": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in paths.items()
        },
        "base_bank_array_sha256": bank_hash,
        "selection": {
            "rule": (
                "for each MNIST class, choose the official-test index whose mean "
                "correct-state MSE across the ten diagonal states is closest to the "
                "within-class median; ties choose the smaller dataset index"
            ),
            "manual_visual_selection": False,
            "sample_ids": sample_ids,
            "digits": digits,
            "mean_diagonal_mse_at_selected_ids": [
                float(sample_scores[index]) for index in sample_ids
            ],
        },
        "controlled_example": {
            "send_state": send_state,
            "send_state_id": state_labels[send_state],
            "correct_receiver_state": send_state,
            "wrong_receiver_state": wrong_receiver_state,
            "wrong_receiver_state_id": state_labels[wrong_receiver_state],
            "same_power_cross_pump_default": wrong_receiver_state
            == cross_pump_same_power_state(send_state),
            "pair_mse_ratio_wrong_over_correct": pair_ratio,
        },
        "timeline": {
            "sender_states": list(range(10)),
            "receiver_shift_frames": schedule_shift,
            "shifted_receiver_states": shifted_states.tolist(),
            "scope": "offline software state-selection demonstration",
        },
        "formal_full_test_mse_multipliers": {
            "reference": "seeded synchronized B_s+R = 1.0",
            **multipliers,
        },
        "rendering": {
            "dpi": dpi,
            "formats": ["png", "pdf"],
            "image_range": [0.0, 1.0],
            "image_interpolation": "nearest",
            "difference_range": [0.0, 0.5],
            "font_fallbacks": ZH_FONT_FAMILY,
        },
        "interpretation_boundary": (
            "These figures visualize reconstruction and state synchronization. "
            "They do not establish entropy, PUF properties, unclonability, physical "
            "switching speed, or cryptographic security from the codebook mechanism."
        ),
        "figures": figures,
    }
    manifest["manifest_payload_sha256"] = _json_sha256(manifest)
    manifest_path_out = output_path / "ppt_visual_manifest.json"
    manifest_path_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    manifest = generate_ppt_visuals(
        seeded_checkpoint_path=args.seeded_checkpoint,
        software_checkpoint_path=args.software_checkpoint,
        codebook_path=args.codebook,
        manifest_path=args.manifest,
        evaluation_dir=args.evaluation_dir,
        aes_result_path=args.aes_result,
        output_dir=args.output_dir,
        device_name=args.device,
        send_state=args.send_state,
        wrong_receiver_state=args.wrong_receiver_state,
        schedule_shift=args.schedule_shift,
        dpi=args.dpi,
    )
    print(f"selected test indices: {manifest['selection']['sample_ids']}")
    print(f"saved: {Path(args.output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
