from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


def freeze_perceptual_metric(metric: nn.Module) -> nn.Module:
    """Put a perceptual metric in inference mode without disabling input gradients."""

    metric.eval()
    metric.requires_grad_(False)
    return metric


def load_lpips_alex_v01(
    *,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Load the validated LPIPS-Alex v0.1 metric for training or evaluation.

    The import is intentionally lazy so importing the project does not initialize
    LPIPS or download its AlexNet weights. Metric parameters are frozen, while
    autograd through the metric inputs remains enabled.
    """

    try:
        import lpips
    except ImportError as error:
        raise RuntimeError(
            "LPIPS-Alex requires the 'lpips' package; install project dependencies"
        ) from error

    metric = lpips.LPIPS(net="alex", version="0.1", verbose=False)
    freeze_perceptual_metric(metric)
    if device is not None:
        metric = metric.to(device)
    return metric


def lpips_alex_per_image(
    reconstruction: Tensor,
    target: Tensor,
    metric: nn.Module,
) -> Tensor:
    """Return one LPIPS-Alex value per image.

    Inputs are expected in full-scale ``[0, 1]`` and are mapped to the LPIPS
    convention ``[-1, 1]``. Grayscale is repeated to RGB and inputs smaller
    than 32 pixels are differentiably resized, matching the validated evaluator.
    No detach or ``no_grad`` context is used, so gradients flow from the
    returned values to ``reconstruction``.
    """

    if reconstruction.shape != target.shape:
        raise ValueError(
            "reconstruction and target must have identical shapes, got "
            f"{tuple(reconstruction.shape)} and {tuple(target.shape)}"
        )
    if reconstruction.ndim != 4 or reconstruction.shape[1] not in {1, 3}:
        raise ValueError(
            "LPIPS-Alex expects NCHW tensors with one or three channels, got "
            f"{tuple(reconstruction.shape)}"
        )
    if reconstruction.shape[0] == 0:
        raise ValueError("LPIPS-Alex requires a non-empty batch")
    if not reconstruction.is_floating_point() or not target.is_floating_point():
        raise TypeError("LPIPS-Alex inputs must be floating-point tensors")

    reconstruction_lpips = reconstruction.mul(2.0).sub(1.0)
    target_lpips = target.mul(2.0).sub(1.0)
    if reconstruction_lpips.shape[1] == 1:
        reconstruction_lpips = reconstruction_lpips.repeat(1, 3, 1, 1)
        target_lpips = target_lpips.repeat(1, 3, 1, 1)
    if min(reconstruction_lpips.shape[-2:]) < 32:
        reconstruction_lpips = torch.nn.functional.interpolate(
            reconstruction_lpips,
            size=(32, 32),
            mode="bilinear",
            align_corners=False,
        )
        target_lpips = torch.nn.functional.interpolate(
            target_lpips,
            size=(32, 32),
            mode="bilinear",
            align_corners=False,
        )
    values: Any = metric(reconstruction_lpips, target_lpips)
    if not isinstance(values, Tensor):
        raise TypeError("LPIPS metric must return a torch.Tensor")
    if values.ndim == 0:
        raise ValueError("LPIPS metric must preserve the batch dimension")
    if values.shape[0] != reconstruction.shape[0]:
        raise ValueError(
            "LPIPS metric must preserve the batch dimension, got "
            f"{tuple(values.shape)} for batch {reconstruction.shape[0]}"
        )
    return values.reshape(values.shape[0], -1).mean(dim=1)
