from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


SSIM_SPEC: dict[str, Any] = {
    "implementation": "torch-gaussian-valid-v2",
    "denominator_floor": "dtype tiny (underflow protection), not dtype epsilon",
    "data_range": 1.0,
    "window_size": 11,
    "sigma": 1.5,
    "k1": 0.01,
    "k2": 0.03,
    "sample_covariance": False,
    "aggregation": "mean of per-image SSIM values",
}


def psnr_from_mse(mse: float, *, data_range: float = 1.0) -> float:
    if mse < 0.0:
        raise ValueError("mse must be non-negative")
    if data_range <= 0.0:
        raise ValueError("data_range must be positive")
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(data_range * data_range / mse)


def image_mse(reconstruction: Tensor, target: Tensor) -> Tensor:
    if reconstruction.shape != target.shape or reconstruction.ndim != 4:
        raise ValueError("reconstruction and target must share NCHW shape")
    return (reconstruction - target).square().flatten(1).mean(dim=1)


def gaussian_ssim_per_image(
    reconstruction: Tensor,
    target: Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tensor:
    """Return one population-moment, valid-window SSIM value per image."""
    if reconstruction.shape != target.shape or reconstruction.ndim != 4:
        raise ValueError("reconstruction and target must share NCHW shape")
    if window_size <= 1 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer greater than one")
    if min(reconstruction.shape[-2:]) < window_size:
        raise ValueError("images are smaller than the SSIM window")
    if data_range <= 0.0 or sigma <= 0.0 or k1 <= 0.0 or k2 <= 0.0:
        raise ValueError("SSIM constants must be positive")

    dtype = reconstruction.dtype
    device = reconstruction.device
    coordinates = torch.arange(window_size, dtype=dtype, device=device)
    coordinates = coordinates - (window_size - 1) / 2.0
    kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    channels = reconstruction.shape[1]
    kernel = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()

    mean_x = F.conv2d(reconstruction, kernel, groups=channels)
    mean_y = F.conv2d(target, kernel, groups=channels)
    mean_x_square = mean_x.square()
    mean_y_square = mean_y.square()
    mean_xy = mean_x * mean_y
    variance_x = (
        F.conv2d(reconstruction.square(), kernel, groups=channels) - mean_x_square
    ).clamp_min(0.0)
    variance_y = (
        F.conv2d(target.square(), kernel, groups=channels) - mean_y_square
    ).clamp_min(0.0)
    covariance = F.conv2d(reconstruction * target, kernel, groups=channels) - mean_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2.0 * mean_xy + c1) * (2.0 * covariance + c2)
    denominator = (mean_x_square + mean_y_square + c1) * (
        variance_x + variance_y + c2
    )
    # C1*C2 is 9e-8 for [0,1] images, below float32 epsilon. An epsilon
    # floor changes the metric on dark/flat windows and makes black-vs-black
    # SSIM about 0.755 rather than 1. Positive SSIM constants already stabilize
    # the denominator; tiny only guards against numerical underflow.
    score = numerator / denominator.clamp_min(torch.finfo(dtype).tiny)
    return score.mean(dim=(1, 2, 3))


def metric_summary(
    per_observation_mse: np.ndarray,
    per_observation_ssim: np.ndarray,
    *,
    pixels_per_observation: int,
    per_observation_lpips: np.ndarray | None = None,
) -> dict[str, float | int]:
    mse = np.asarray(per_observation_mse, dtype=np.float64).reshape(-1)
    ssim = np.asarray(per_observation_ssim, dtype=np.float64).reshape(-1)
    if mse.shape != ssim.shape or mse.size == 0:
        raise ValueError("MSE and SSIM arrays must be non-empty and equally sized")
    if pixels_per_observation <= 0:
        raise ValueError("pixels_per_observation must be positive")
    pooled_mse = float(mse.mean(dtype=np.float64))
    observation_count = int(mse.size)
    pixel_count = observation_count * int(pixels_per_observation)
    result: dict[str, float | int] = {
        "observation_count": observation_count,
        "pixel_count": pixel_count,
        "squared_error_sum": float(pooled_mse * pixel_count),
        "mse": pooled_mse,
        "rmse": math.sqrt(pooled_mse),
        "psnr_db": psnr_from_mse(pooled_mse),
        "ssim_mean": float(ssim.mean(dtype=np.float64)),
        "ssim_std_across_observations": float(ssim.std(dtype=np.float64)),
    }
    if per_observation_lpips is not None:
        lpips = np.asarray(per_observation_lpips, dtype=np.float64).reshape(-1)
        if lpips.shape != mse.shape:
            raise ValueError("LPIPS and MSE arrays must be equally sized")
        if not np.isfinite(lpips).all():
            raise ValueError("LPIPS values must be finite")
        result.update(
            {
                "lpips_mean": float(lpips.mean(dtype=np.float64)),
                "lpips_std_across_observations": float(lpips.std(dtype=np.float64)),
            }
        )
    return result


def usage_summary(counts: np.ndarray) -> dict[str, Any]:
    histogram = np.asarray(counts, dtype=np.int64)
    if histogram.ndim != 1 or histogram.size == 0:
        raise ValueError("counts must be a non-empty one-dimensional array")
    if bool((histogram < 0).any()):
        raise ValueError("counts must be non-negative")
    token_count = int(histogram.sum())
    if token_count == 0:
        raise ValueError("counts must contain at least one observation")
    probabilities = histogram.astype(np.float64) / token_count
    nonzero = probabilities[probabilities > 0.0]
    entropy_bits = float(-(nonzero * np.log2(nonzero)).sum())
    perplexity = float(2.0**entropy_bits)
    codebook_size = int(histogram.size)
    used_codes = int(np.count_nonzero(histogram))
    return {
        "token_count": token_count,
        "codebook_size": codebook_size,
        "used_codes": used_codes,
        "coverage": float(used_codes / codebook_size),
        "entropy_bits": entropy_bits,
        "perplexity": perplexity,
        "normalized_perplexity": float(perplexity / codebook_size),
        "counts": histogram.tolist(),
    }


def fixed_state_schedule(sample_count: int, state_count: int) -> np.ndarray:
    if sample_count <= 0 or state_count <= 1:
        raise ValueError("sample_count must be positive and state_count must exceed one")
    return np.arange(sample_count, dtype=np.int64) % state_count


def pseudorandom_balanced_schedule(
    sample_count: int,
    state_count: int,
    seed: int,
) -> np.ndarray:
    """Generate deterministic random permutations in state_count-sized blocks."""
    if sample_count <= 0 or state_count <= 1:
        raise ValueError("sample_count must be positive and state_count must exceed one")
    generator = np.random.Generator(np.random.PCG64(seed))
    blocks: list[np.ndarray] = []
    remaining = sample_count
    while remaining > 0:
        permutation = generator.permutation(state_count).astype(np.int64, copy=False)
        take = min(remaining, state_count)
        blocks.append(permutation[:take])
        remaining -= take
    return np.concatenate(blocks)


def schedule_sha256(schedule: np.ndarray) -> str:
    values = np.asarray(schedule, dtype="<i2")
    if values.ndim != 1:
        raise ValueError("schedule must be one-dimensional")
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def summarize_cross_state(
    per_image_mse: np.ndarray,
    per_image_ssim: np.ndarray,
    *,
    state_labels: list[str],
    trained_offset: int | None = None,
    selection_monitored_offset: int | None = None,
    pixels_per_image: int,
    per_image_lpips: np.ndarray | None = None,
) -> dict[str, Any]:
    mse = np.asarray(per_image_mse, dtype=np.float64)
    ssim = np.asarray(per_image_ssim, dtype=np.float64)
    if mse.shape != ssim.shape or mse.ndim != 3 or mse.shape[1] != mse.shape[2]:
        raise ValueError("cross-state arrays must share shape (N, S, S)")
    lpips = None
    if per_image_lpips is not None:
        lpips = np.asarray(per_image_lpips, dtype=np.float64)
        if lpips.shape != mse.shape or not np.isfinite(lpips).all():
            raise ValueError("cross-state LPIPS must be finite and share shape (N, S, S)")
    sample_count, state_count, _ = mse.shape
    if sample_count == 0 or len(state_labels) != state_count:
        raise ValueError("state labels must match a non-empty cross-state array")
    if selection_monitored_offset is None:
        selection_monitored_offset = trained_offset
    elif trained_offset is not None and trained_offset != selection_monitored_offset:
        raise ValueError("deprecated trained_offset conflicts with monitored offset")
    if selection_monitored_offset is None:
        raise ValueError("selection_monitored_offset is required")
    if not 1 <= selection_monitored_offset < state_count:
        raise ValueError("selection_monitored_offset must be in [1, state_count-1]")

    rows = np.arange(state_count)
    correct_mse = mse[:, rows, rows]
    correct_ssim = ssim[:, rows, rows]
    wrong_mask = ~np.eye(state_count, dtype=bool)
    wrong_mse = mse[:, wrong_mask]
    wrong_ssim = ssim[:, wrong_mask]
    correct_lpips = lpips[:, rows, rows] if lpips is not None else None
    wrong_lpips = lpips[:, wrong_mask] if lpips is not None else None
    by_state = []
    for state in range(state_count):
        row = metric_summary(
            mse[:, state, state],
            ssim[:, state, state],
            pixels_per_observation=pixels_per_image,
            per_observation_lpips=(
                lpips[:, state, state] if lpips is not None else None
            ),
        )
        row.update({"state_index": state, "state_id": state_labels[state]})
        by_state.append(row)

    by_offset = []
    for offset in range(1, state_count):
        columns = (rows + offset) % state_count
        row = metric_summary(
            mse[:, rows, columns],
            ssim[:, rows, columns],
            pixels_per_observation=pixels_per_image,
            per_observation_lpips=(
                lpips[:, rows, columns] if lpips is not None else None
            ),
        )
        row.update(
            {
                "state_offset": offset,
                "selection_monitored_offset": offset
                == selection_monitored_offset,
            }
        )
        by_offset.append(row)

    monitored_columns = (rows + selection_monitored_offset) % state_count
    monitored = metric_summary(
        mse[:, rows, monitored_columns],
        ssim[:, rows, monitored_columns],
        pixels_per_observation=pixels_per_image,
        per_observation_lpips=(
            lpips[:, rows, monitored_columns] if lpips is not None else None
        ),
    )
    unmonitored_offsets = [
        offset
        for offset in range(1, state_count)
        if offset != selection_monitored_offset
    ]
    unmonitored_mse = np.concatenate(
        [mse[:, rows, (rows + offset) % state_count] for offset in unmonitored_offsets],
        axis=1,
    )
    unmonitored_ssim = np.concatenate(
        [ssim[:, rows, (rows + offset) % state_count] for offset in unmonitored_offsets],
        axis=1,
    )
    unmonitored_lpips = (
        np.concatenate(
            [
                lpips[:, rows, (rows + offset) % state_count]
                for offset in unmonitored_offsets
            ],
            axis=1,
        )
        if lpips is not None
        else None
    )
    unmonitored = metric_summary(
        unmonitored_mse,
        unmonitored_ssim,
        pixels_per_observation=pixels_per_image,
        per_observation_lpips=unmonitored_lpips,
    )
    correct = metric_summary(
        correct_mse,
        correct_ssim,
        pixels_per_observation=pixels_per_image,
        per_observation_lpips=correct_lpips,
    )
    wrong = metric_summary(
        wrong_mse,
        wrong_ssim,
        pixels_per_observation=pixels_per_image,
        per_observation_lpips=wrong_lpips,
    )
    correct_psnr_by_state = [float(row["psnr_db"]) for row in by_state]
    result = {
        "sample_count": sample_count,
        "state_count": state_count,
        "mse_matrix": mse.mean(axis=0, dtype=np.float64).tolist(),
        "psnr_db_matrix": np.vectorize(psnr_from_mse)(
            mse.mean(axis=0, dtype=np.float64)
        ).tolist(),
        "ssim_mean_matrix": ssim.mean(axis=0, dtype=np.float64).tolist(),
        "correct_pooled": correct,
        "wrong_pooled": wrong,
        "wrong_to_correct_mse_ratio": float(wrong["mse"] / max(correct["mse"], 1e-15)),
        "correct_minus_wrong_psnr_db": float(correct["psnr_db"] - wrong["psnr_db"]),
        "correct_psnr_state_spread_db": float(
            max(correct_psnr_by_state) - min(correct_psnr_by_state)
        ),
        "correct_by_state": by_state,
        "by_state_offset": by_offset,
        "selection_monitored_offset": {
            "state_offset": selection_monitored_offset,
            **monitored,
        },
        "selection_unmonitored_offsets": {
            "state_offsets": unmonitored_offsets,
            **unmonitored,
        },
    }
    if lpips is not None:
        result["lpips_mean_matrix"] = lpips.mean(axis=0, dtype=np.float64).tolist()
    return result


def summarize_schedule(
    per_image_mse: np.ndarray,
    per_image_ssim: np.ndarray,
    *,
    schedule_name: str,
    send_schedule: np.ndarray,
    pixels_per_image: int,
    per_image_lpips: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    mse = np.asarray(per_image_mse, dtype=np.float64)
    ssim = np.asarray(per_image_ssim, dtype=np.float64)
    schedule = np.asarray(send_schedule, dtype=np.int64)
    if mse.shape != ssim.shape or mse.ndim != 3 or mse.shape[1] != mse.shape[2]:
        raise ValueError("cross-state arrays must share shape (N, S, S)")
    lpips = None
    if per_image_lpips is not None:
        lpips = np.asarray(per_image_lpips, dtype=np.float64)
        if lpips.shape != mse.shape or not np.isfinite(lpips).all():
            raise ValueError("cross-state LPIPS must be finite and share shape (N, S, S)")
    sample_count, state_count, _ = mse.shape
    if schedule.shape != (sample_count,):
        raise ValueError("schedule length must equal the image count")
    if bool(((schedule < 0) | (schedule >= state_count)).any()):
        raise ValueError("schedule contains an invalid state")

    images = np.arange(sample_count)
    rows: list[dict[str, Any]] = []
    for shift in range(state_count):
        receiver = np.roll(schedule, -shift)
        selected_mse = mse[images, schedule, receiver]
        selected_ssim = ssim[images, schedule, receiver]
        selected_lpips = lpips[images, schedule, receiver] if lpips is not None else None
        matched = schedule == receiver
        row: dict[str, Any] = {
            "schedule": schedule_name,
            "time_shift_frames": shift,
            "synchronized": shift == 0,
            "receiver_definition": "receiver[t] = sender_schedule[(t + shift) mod N]",
            "matched_frame_count": int(matched.sum()),
            "unmatched_frame_count": int((~matched).sum()),
            "match_fraction": float(matched.mean()),
            "overall": metric_summary(
                selected_mse,
                selected_ssim,
                pixels_per_observation=pixels_per_image,
                per_observation_lpips=selected_lpips,
            ),
            "matched_only": None,
            "unmatched_only": None,
        }
        if bool(matched.any()):
            row["matched_only"] = metric_summary(
                selected_mse[matched],
                selected_ssim[matched],
                pixels_per_observation=pixels_per_image,
                per_observation_lpips=(
                    selected_lpips[matched] if selected_lpips is not None else None
                ),
            )
        if bool((~matched).any()):
            row["unmatched_only"] = metric_summary(
                selected_mse[~matched],
                selected_ssim[~matched],
                pixels_per_observation=pixels_per_image,
                per_observation_lpips=(
                    selected_lpips[~matched] if selected_lpips is not None else None
                ),
            )
        rows.append(row)
    return rows


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    """Summarize a finite one-dimensional observation distribution."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("distribution values must be non-empty and finite")
    q05, q25, q50, q75, q95 = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(array.size),
        "mean": float(array.mean(dtype=np.float64)),
        "std": float(array.std(dtype=np.float64)),
        "min": float(array.min()),
        "p05": float(q05),
        "p25": float(q25),
        "median": float(q50),
        "p75": float(q75),
        "p95": float(q95),
        "max": float(array.max()),
    }


def _image_means(values: np.ndarray, *, name: str) -> np.ndarray:
    """Reduce equally weighted observations within each image.

    The first dimension is always the independent test-image dimension.  Any
    remaining dimensions may represent states or ordered state pairs.  Reducing
    them before resampling prevents pseudo-replication of the same test image.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[0] == 0:
        raise ValueError(f"{name} must have a non-empty image dimension")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.reshape(array.shape[0], -1).mean(axis=1, dtype=np.float64)


def _bootstrap_interval(values: np.ndarray, estimate: float) -> dict[str, float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "estimate": float(estimate),
        "bootstrap_mean": float(values.mean(dtype=np.float64)),
        "ci95_percentile_lower": float(lower),
        "ci95_percentile_upper": float(upper),
    }


def image_metric_bootstrap(
    per_image_mse: np.ndarray,
    per_image_ssim: np.ndarray,
    *,
    per_image_lpips: np.ndarray | None = None,
    seed: int,
    replicates: int,
    chunk_size: int = 64,
) -> dict[str, Any]:
    """Bootstrap pooled reconstruction metrics with the image as sampling unit.

    Inputs may be shaped ``(N,)``, ``(N, states)``, or ``(N, state_pairs)``.
    State/state-pair observations are averaged inside each image before image
    indices are resampled.  Consequently, the intervals quantify held-out image
    sampling uncertainty, not uncertainty across training runs.
    """

    if replicates < 0:
        raise ValueError("replicates must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    mse = _image_means(per_image_mse, name="per_image_mse")
    ssim = _image_means(per_image_ssim, name="per_image_ssim")
    if mse.shape != ssim.shape:
        raise ValueError("MSE and SSIM must share the image dimension")
    if bool((mse < 0.0).any()):
        raise ValueError("per_image_mse must be non-negative")
    lpips = None
    if per_image_lpips is not None:
        lpips = _image_means(per_image_lpips, name="per_image_lpips")
        if lpips.shape != mse.shape:
            raise ValueError("LPIPS must share the image dimension")

    mse_estimate = float(mse.mean(dtype=np.float64))
    estimates: dict[str, float] = {
        "mse": mse_estimate,
        "rmse": math.sqrt(mse_estimate),
        "ssim_mean": float(ssim.mean(dtype=np.float64)),
    }
    if lpips is not None:
        estimates["lpips_mean"] = float(lpips.mean(dtype=np.float64))

    result: dict[str, Any] = {
        "sampling_unit": "test_image",
        "within_image_reduction": "arithmetic mean over supplied states/state pairs",
        "interpretation": "test-image sampling only; not training-run uncertainty",
        "seed": int(seed),
        "replicates": int(replicates),
        "metrics": {
            name: {"estimate": value, "ci95_percentile_lower": None,
                   "ci95_percentile_upper": None}
            for name, value in estimates.items()
        },
    }
    if replicates == 0:
        return result

    sample_count = mse.size
    samples = {
        name: np.empty(replicates, dtype=np.float64) for name in estimates
    }
    generator = np.random.Generator(np.random.PCG64(seed))
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = generator.integers(
            0,
            sample_count,
            size=(stop - start, sample_count),
            dtype=np.int32,
        )
        mse_samples = mse[indices].mean(axis=1, dtype=np.float64)
        samples["mse"][start:stop] = mse_samples
        samples["rmse"][start:stop] = np.sqrt(mse_samples)
        samples["ssim_mean"][start:stop] = ssim[indices].mean(
            axis=1, dtype=np.float64
        )
        if lpips is not None:
            samples["lpips_mean"][start:stop] = lpips[indices].mean(
                axis=1, dtype=np.float64
            )
    result["metrics"] = {
        name: _bootstrap_interval(samples[name], estimate)
        for name, estimate in estimates.items()
    }
    return result


def paired_image_branch_bootstrap(
    correct_mse: np.ndarray,
    wrong_mse: np.ndarray,
    *,
    correct_ssim: np.ndarray,
    wrong_ssim: np.ndarray,
    correct_lpips: np.ndarray | None = None,
    wrong_lpips: np.ndarray | None = None,
    seed: int,
    replicates: int,
    chunk_size: int = 64,
) -> dict[str, Any]:
    """Compare correct and wrong branches using paired image resampling."""

    if replicates < 0:
        raise ValueError("replicates must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    correct_mse_image = _image_means(correct_mse, name="correct_mse")
    wrong_mse_image = _image_means(wrong_mse, name="wrong_mse")
    correct_ssim_image = _image_means(correct_ssim, name="correct_ssim")
    wrong_ssim_image = _image_means(wrong_ssim, name="wrong_ssim")
    arrays = (
        correct_mse_image,
        wrong_mse_image,
        correct_ssim_image,
        wrong_ssim_image,
    )
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("correct and wrong metrics must share the image dimension")
    if bool((correct_mse_image < 0.0).any()) or bool((wrong_mse_image < 0.0).any()):
        raise ValueError("MSE inputs must be non-negative")

    correct_lpips_image = wrong_lpips_image = None
    if (correct_lpips is None) != (wrong_lpips is None):
        raise ValueError("correct_lpips and wrong_lpips must be supplied together")
    if correct_lpips is not None and wrong_lpips is not None:
        correct_lpips_image = _image_means(correct_lpips, name="correct_lpips")
        wrong_lpips_image = _image_means(wrong_lpips, name="wrong_lpips")
        if (
            correct_lpips_image.shape != correct_mse_image.shape
            or wrong_lpips_image.shape != correct_mse_image.shape
        ):
            raise ValueError("LPIPS inputs must share the image dimension")

    tiny = np.finfo(np.float64).tiny

    def point_metrics(indices: np.ndarray | None = None) -> dict[str, float]:
        if indices is None:
            first_mse = float(correct_mse_image.mean(dtype=np.float64))
            second_mse = float(wrong_mse_image.mean(dtype=np.float64))
            first_ssim = float(correct_ssim_image.mean(dtype=np.float64))
            second_ssim = float(wrong_ssim_image.mean(dtype=np.float64))
        else:
            first_mse = float(correct_mse_image[indices].mean(dtype=np.float64))
            second_mse = float(wrong_mse_image[indices].mean(dtype=np.float64))
            first_ssim = float(correct_ssim_image[indices].mean(dtype=np.float64))
            second_ssim = float(wrong_ssim_image[indices].mean(dtype=np.float64))
        first_rmse = math.sqrt(first_mse)
        second_rmse = math.sqrt(second_mse)
        values = {
            "wrong_minus_correct_mse": second_mse - first_mse,
            "wrong_to_correct_mse_ratio": second_mse / max(first_mse, tiny),
            "wrong_minus_correct_rmse": second_rmse - first_rmse,
            "wrong_to_correct_rmse_ratio": second_rmse / max(first_rmse, tiny),
            "correct_minus_wrong_ssim": first_ssim - second_ssim,
        }
        if correct_lpips_image is not None and wrong_lpips_image is not None:
            if indices is None:
                first_lpips = float(correct_lpips_image.mean(dtype=np.float64))
                second_lpips = float(wrong_lpips_image.mean(dtype=np.float64))
            else:
                first_lpips = float(
                    correct_lpips_image[indices].mean(dtype=np.float64)
                )
                second_lpips = float(
                    wrong_lpips_image[indices].mean(dtype=np.float64)
                )
            values["wrong_minus_correct_lpips"] = second_lpips - first_lpips
        return values

    estimates = point_metrics()
    result: dict[str, Any] = {
        "sampling_unit": "test_image",
        "pairing": "same resampled test-image indices for correct and wrong branches",
        "within_image_reduction": "arithmetic mean over supplied states/state pairs",
        "interpretation": "test-image sampling only; not training-run uncertainty",
        "seed": int(seed),
        "replicates": int(replicates),
        "comparisons": {
            name: {"estimate": value, "ci95_percentile_lower": None,
                   "ci95_percentile_upper": None}
            for name, value in estimates.items()
        },
    }
    if replicates == 0:
        return result

    sample_count = correct_mse_image.size
    samples = {
        name: np.empty(replicates, dtype=np.float64) for name in estimates
    }
    generator = np.random.Generator(np.random.PCG64(seed))
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = generator.integers(
            0,
            sample_count,
            size=(stop - start, sample_count),
            dtype=np.int32,
        )
        first_mse = correct_mse_image[indices].mean(axis=1, dtype=np.float64)
        second_mse = wrong_mse_image[indices].mean(axis=1, dtype=np.float64)
        first_rmse = np.sqrt(first_mse)
        second_rmse = np.sqrt(second_mse)
        first_ssim = correct_ssim_image[indices].mean(axis=1, dtype=np.float64)
        second_ssim = wrong_ssim_image[indices].mean(axis=1, dtype=np.float64)
        samples["wrong_minus_correct_mse"][start:stop] = second_mse - first_mse
        samples["wrong_to_correct_mse_ratio"][start:stop] = second_mse / np.maximum(
            first_mse, tiny
        )
        samples["wrong_minus_correct_rmse"][start:stop] = second_rmse - first_rmse
        samples["wrong_to_correct_rmse_ratio"][start:stop] = second_rmse / np.maximum(
            first_rmse, tiny
        )
        samples["correct_minus_wrong_ssim"][start:stop] = first_ssim - second_ssim
        if correct_lpips_image is not None and wrong_lpips_image is not None:
            samples["wrong_minus_correct_lpips"][start:stop] = (
                wrong_lpips_image[indices].mean(axis=1, dtype=np.float64)
                - correct_lpips_image[indices].mean(axis=1, dtype=np.float64)
            )
    result["comparisons"] = {
        name: _bootstrap_interval(samples[name], estimate)
        for name, estimate in estimates.items()
    }
    return result


def paired_correct_wrong_summary(
    per_image_mse: np.ndarray,
    *,
    per_image_lpips: np.ndarray | None,
    per_image_ssim: np.ndarray | None = None,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Compare correct and wrong states while preserving the image pairing.

    The S correct-state and S*(S-1) wrong-state observations are first averaged
    within each test image. Resampling then operates on test-image indices, so the
    resulting intervals describe image-sampling uncertainty only.
    """

    mse = np.asarray(per_image_mse, dtype=np.float64)
    if mse.ndim != 3 or mse.shape[1] != mse.shape[2] or mse.shape[0] == 0:
        raise ValueError("per_image_mse must have non-empty shape (N, S, S)")
    if not np.isfinite(mse).all() or bool((mse < 0.0).any()):
        raise ValueError("per_image_mse must contain finite non-negative values")
    if replicates < 0:
        raise ValueError("replicates must be non-negative")

    sample_count, state_count, _ = mse.shape
    states = np.arange(state_count)
    wrong_mask = ~np.eye(state_count, dtype=bool)
    correct_image_mse = mse[:, states, states].mean(axis=1, dtype=np.float64)
    wrong_image_mse = mse[:, wrong_mask].mean(axis=1, dtype=np.float64)
    denominator = np.maximum(correct_image_mse, np.finfo(np.float64).tiny)
    per_image_mse_ratio = wrong_image_mse / denominator
    per_image_rmse_ratio = np.sqrt(per_image_mse_ratio)
    correct_mean = float(correct_image_mse.mean(dtype=np.float64))
    wrong_mean = float(wrong_image_mse.mean(dtype=np.float64))

    lpips = None
    correct_image_lpips = None
    wrong_image_lpips = None
    if per_image_lpips is not None:
        lpips = np.asarray(per_image_lpips, dtype=np.float64)
        if lpips.shape != mse.shape or not np.isfinite(lpips).all():
            raise ValueError("per_image_lpips must be finite and match per_image_mse")
        correct_image_lpips = lpips[:, states, states].mean(axis=1, dtype=np.float64)
        wrong_image_lpips = lpips[:, wrong_mask].mean(axis=1, dtype=np.float64)

    ssim = None
    correct_image_ssim = None
    wrong_image_ssim = None
    if per_image_ssim is not None:
        ssim = np.asarray(per_image_ssim, dtype=np.float64)
        if ssim.shape != mse.shape or not np.isfinite(ssim).all():
            raise ValueError("per_image_ssim must be finite and match per_image_mse")
        correct_image_ssim = ssim[:, states, states].mean(axis=1, dtype=np.float64)
        wrong_image_ssim = ssim[:, wrong_mask].mean(axis=1, dtype=np.float64)

    result: dict[str, Any] = {
        "sampling_unit": "test image",
        "interpretation": "test-image sampling only; not training-run uncertainty",
        "per_image_reduction": {
            "correct_mse_mean_across_states": distribution_summary(correct_image_mse),
            "wrong_mse_mean_across_state_pairs": distribution_summary(wrong_image_mse),
            "wrong_to_correct_mse_ratio": distribution_summary(per_image_mse_ratio),
            "wrong_to_correct_rmse_ratio": distribution_summary(per_image_rmse_ratio),
        },
        "pooled": {
            "correct_mse": correct_mean,
            "wrong_mse": wrong_mean,
            "wrong_to_correct_mse_ratio": float(
                wrong_mean / max(correct_mean, np.finfo(np.float64).tiny)
            ),
            "wrong_to_correct_rmse_ratio": float(
                math.sqrt(wrong_mean / max(correct_mean, np.finfo(np.float64).tiny))
            ),
            "correct_minus_wrong_psnr_db": float(
                psnr_from_mse(correct_mean) - psnr_from_mse(wrong_mean)
            ),
        },
        "bootstrap": None,
    }
    if correct_image_lpips is not None and wrong_image_lpips is not None:
        lpips_delta = wrong_image_lpips - correct_image_lpips
        result["per_image_reduction"].update(
            {
                "correct_lpips_mean_across_states": distribution_summary(
                    correct_image_lpips
                ),
                "wrong_lpips_mean_across_state_pairs": distribution_summary(
                    wrong_image_lpips
                ),
                "wrong_minus_correct_lpips": distribution_summary(lpips_delta),
            }
        )
        result["pooled"].update(
            {
                "correct_lpips": float(correct_image_lpips.mean(dtype=np.float64)),
                "wrong_lpips": float(wrong_image_lpips.mean(dtype=np.float64)),
                "wrong_minus_correct_lpips": float(lpips_delta.mean(dtype=np.float64)),
            }
        )
    if correct_image_ssim is not None and wrong_image_ssim is not None:
        ssim_delta = correct_image_ssim - wrong_image_ssim
        result["per_image_reduction"].update(
            {
                "correct_ssim_mean_across_states": distribution_summary(
                    correct_image_ssim
                ),
                "wrong_ssim_mean_across_state_pairs": distribution_summary(
                    wrong_image_ssim
                ),
                "correct_minus_wrong_ssim": distribution_summary(ssim_delta),
            }
        )
        result["pooled"].update(
            {
                "correct_ssim": float(correct_image_ssim.mean(dtype=np.float64)),
                "wrong_ssim": float(wrong_image_ssim.mean(dtype=np.float64)),
                "correct_minus_wrong_ssim": float(ssim_delta.mean(dtype=np.float64)),
            }
        )

    if replicates == 0:
        return result

    generator = np.random.Generator(np.random.PCG64(seed))
    mse_ratio_samples = np.empty(replicates, dtype=np.float64)
    rmse_ratio_samples = np.empty(replicates, dtype=np.float64)
    psnr_drop_samples = np.empty(replicates, dtype=np.float64)
    lpips_delta_samples = (
        np.empty(replicates, dtype=np.float64) if correct_image_lpips is not None else None
    )
    ssim_delta_samples = (
        np.empty(replicates, dtype=np.float64) if correct_image_ssim is not None else None
    )
    for replicate in range(replicates):
        indices = generator.integers(0, sample_count, size=sample_count)
        sampled_correct_mse = float(correct_image_mse[indices].mean(dtype=np.float64))
        sampled_wrong_mse = float(wrong_image_mse[indices].mean(dtype=np.float64))
        mse_ratio_samples[replicate] = sampled_wrong_mse / max(
            sampled_correct_mse, np.finfo(np.float64).tiny
        )
        rmse_ratio_samples[replicate] = math.sqrt(mse_ratio_samples[replicate])
        psnr_drop_samples[replicate] = psnr_from_mse(
            sampled_correct_mse
        ) - psnr_from_mse(sampled_wrong_mse)
        if lpips_delta_samples is not None:
            assert correct_image_lpips is not None and wrong_image_lpips is not None
            lpips_delta_samples[replicate] = float(
                wrong_image_lpips[indices].mean(dtype=np.float64)
                - correct_image_lpips[indices].mean(dtype=np.float64)
            )
        if ssim_delta_samples is not None:
            assert correct_image_ssim is not None and wrong_image_ssim is not None
            ssim_delta_samples[replicate] = float(
                correct_image_ssim[indices].mean(dtype=np.float64)
                - wrong_image_ssim[indices].mean(dtype=np.float64)
            )

    def bootstrap_interval(values: np.ndarray, estimate: float) -> dict[str, float]:
        lower, upper = np.quantile(values, [0.025, 0.975])
        return {
            "estimate": float(estimate),
            "bootstrap_mean": float(values.mean(dtype=np.float64)),
            "ci95_percentile_lower": float(lower),
            "ci95_percentile_upper": float(upper),
        }

    bootstrap: dict[str, Any] = {
        "seed": int(seed),
        "replicates": int(replicates),
        "wrong_to_correct_mse_ratio": bootstrap_interval(
            mse_ratio_samples, result["pooled"]["wrong_to_correct_mse_ratio"]
        ),
        "wrong_to_correct_rmse_ratio": bootstrap_interval(
            rmse_ratio_samples, result["pooled"]["wrong_to_correct_rmse_ratio"]
        ),
        "correct_minus_wrong_psnr_db": bootstrap_interval(
            psnr_drop_samples, result["pooled"]["correct_minus_wrong_psnr_db"]
        ),
    }
    if lpips_delta_samples is not None:
        bootstrap["wrong_minus_correct_lpips"] = bootstrap_interval(
            lpips_delta_samples, result["pooled"]["wrong_minus_correct_lpips"]
        )
    if ssim_delta_samples is not None:
        bootstrap["correct_minus_wrong_ssim"] = bootstrap_interval(
            ssim_delta_samples, result["pooled"]["correct_minus_wrong_ssim"]
        )
    result["bootstrap"] = bootstrap
    return result


def paired_image_bootstrap(
    baseline_mse: np.ndarray,
    baseline_ssim: np.ndarray,
    seeded_correct_mse_by_state: np.ndarray,
    seeded_correct_ssim_by_state: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any] | None:
    if replicates == 0:
        return None
    if replicates < 0:
        raise ValueError("replicates must be non-negative")
    base_mse = np.asarray(baseline_mse, dtype=np.float64).reshape(-1)
    base_ssim = np.asarray(baseline_ssim, dtype=np.float64).reshape(-1)
    seeded_mse = np.asarray(seeded_correct_mse_by_state, dtype=np.float64)
    seeded_ssim = np.asarray(seeded_correct_ssim_by_state, dtype=np.float64)
    if (
        seeded_mse.shape != seeded_ssim.shape
        or seeded_mse.ndim != 2
        or seeded_mse.shape[0] != base_mse.size
        or base_mse.shape != base_ssim.shape
        or base_mse.size == 0
    ):
        raise ValueError("bootstrap inputs do not share the image dimension")

    seeded_image_mse = seeded_mse.mean(axis=1, dtype=np.float64)
    seeded_image_ssim = seeded_ssim.mean(axis=1, dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(seed))
    delta_psnr = np.empty(replicates, dtype=np.float64)
    delta_ssim = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        indices = generator.integers(0, base_mse.size, size=base_mse.size)
        delta_psnr[replicate] = psnr_from_mse(float(seeded_image_mse[indices].mean())) - psnr_from_mse(
            float(base_mse[indices].mean())
        )
        delta_ssim[replicate] = float(
            seeded_image_ssim[indices].mean() - base_ssim[indices].mean()
        )

    def interval(values: np.ndarray) -> dict[str, float]:
        lower, upper = np.quantile(values, [0.025, 0.975])
        return {
            "estimate": float(values.mean()),
            "ci95_percentile_lower": float(lower),
            "ci95_percentile_upper": float(upper),
        }

    return {
        "sampling_unit": "test image",
        "interpretation": "test-image sampling only; not training-run uncertainty",
        "seed": int(seed),
        "replicates": int(replicates),
        "seeded_minus_software_psnr_db": interval(delta_psnr),
        "seeded_minus_software_ssim": interval(delta_ssim),
    }
