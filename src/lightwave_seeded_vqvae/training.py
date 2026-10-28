from __future__ import annotations

from collections.abc import Iterable
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .perceptual import lpips_alex_per_image


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def balanced_state_ids(
    batch_size: int,
    state_count: int,
    offset: int,
    device: torch.device,
) -> tuple[Tensor, int]:
    ids = (torch.arange(batch_size, device=device, dtype=torch.long) + offset) % state_count
    return ids, int((offset + batch_size) % state_count)


def make_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    encoder_learning_rate_scale: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    encoder = model.encoder
    backbone = list(encoder.backbone_parameters())
    projection_ids = {id(parameter) for parameter in encoder.projection_parameters()}
    backbone_ids = {id(parameter) for parameter in backbone}
    main = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_ids and id(parameter) not in projection_ids
    ]
    main.extend(list(encoder.projection_parameters()))
    if len({id(parameter) for parameter in backbone + main}) != len(backbone) + len(main):
        raise RuntimeError("optimizer parameter groups overlap")
    return torch.optim.AdamW(
        [
            {
                "params": backbone,
                "lr": learning_rate * encoder_learning_rate_scale,
                "name": "pretrained_encoder",
            },
            {"params": main, "lr": learning_rate, "name": "new_and_quantizer"},
        ],
        weight_decay=weight_decay,
    )


def perplexity_from_indices(indices: Tensor, codebook_size: int) -> tuple[float, float]:
    counts = torch.bincount(indices.reshape(-1).cpu(), minlength=codebook_size).double()
    coverage = float((counts > 0).sum().item() / codebook_size)
    probabilities = counts / counts.sum().clamp_min(1.0)
    nonzero = probabilities[probabilities > 0]
    perplexity = float(torch.exp(-(nonzero * nonzero.log()).sum()).item())
    return coverage, perplexity


def psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def scheduled_weight(epoch: int, *, start_epoch: int, ramp_epochs: int, weight: float) -> float:
    if weight <= 0.0 or epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 1:
        return float(weight)
    progress = min(1.0, (epoch - start_epoch + 1) / ramp_epochs)
    return float(weight * progress)


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def batch_objective(
    reconstruction: Tensor,
    target: Tensor,
    *,
    commitment_loss: Tensor,
    codebook_loss: Tensor,
    residual_norm_loss: Tensor,
    commitment_weight: float,
    codebook_weight: float,
    residual_norm_weight: float,
    wrong_state_mse: Tensor | None = None,
    wrong_state_margin: float = 0.0,
    wrong_state_weight: float = 0.0,
    residual_only_mse: Tensor | None = None,
    residual_only_margin: float = 0.0,
    residual_only_weight: float = 0.0,
    reconstruction_loss_type: str = "mse",
    perceptual_metric: nn.Module | None = None,
    perceptual_weight: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    reconstruction_mse = F.mse_loss(reconstruction, target)
    reconstruction_l1 = F.l1_loss(reconstruction, target)
    reconstruction_lpips = reconstruction_mse.new_zeros(())
    if reconstruction_loss_type == "mse":
        reconstruction_loss = reconstruction_mse
    elif reconstruction_loss_type == "l1_lpips":
        if perceptual_metric is None:
            raise ValueError(
                "perceptual_metric is required when reconstruction_loss_type='l1_lpips'"
            )
        if not math.isfinite(perceptual_weight) or perceptual_weight < 0.0:
            raise ValueError("perceptual_weight must be finite and non-negative")
        reconstruction_lpips = lpips_alex_per_image(
            reconstruction, target, perceptual_metric
        ).mean()
        reconstruction_loss = reconstruction_l1 + perceptual_weight * reconstruction_lpips
    else:
        raise ValueError(
            "reconstruction_loss_type must be either 'mse' or 'l1_lpips'"
        )
    total = (
        reconstruction_loss
        + commitment_weight * commitment_loss
        + codebook_weight * codebook_loss
        + residual_norm_weight * residual_norm_loss
    )
    wrong_state_hinge = reconstruction_mse.new_zeros(())
    if wrong_state_mse is not None and wrong_state_weight > 0.0:
        wrong_state_hinge = F.relu(
            reconstruction_mse.detach() + wrong_state_margin - wrong_state_mse
        )
        total = total + wrong_state_weight * wrong_state_hinge

    residual_only_hinge = reconstruction_mse.new_zeros(())
    if residual_only_mse is not None and residual_only_weight > 0.0:
        residual_only_hinge = F.relu(
            reconstruction_mse.detach() + residual_only_margin - residual_only_mse
        )
        total = total + residual_only_weight * residual_only_hinge

    row = {
        "loss": float(total.detach()),
        "reconstruction_mse": float(reconstruction_mse.detach()),
        "reconstruction_l1": float(reconstruction_l1.detach()),
        "reconstruction_lpips": float(reconstruction_lpips.detach()),
        "reconstruction_loss": float(reconstruction_loss.detach()),
        "commitment_loss": float(commitment_loss.detach()),
        "codebook_loss": float(codebook_loss.detach()),
        "residual_norm_loss": float(residual_norm_loss.detach()),
        "wrong_state_weight": float(wrong_state_weight),
        "wrong_state_mse": float(wrong_state_mse.detach()) if wrong_state_mse is not None else 0.0,
        "wrong_state_hinge": float(wrong_state_hinge.detach()),
        "residual_only_weight": float(residual_only_weight),
        "residual_only_mse": (
            float(residual_only_mse.detach()) if residual_only_mse is not None else 0.0
        ),
        "residual_only_hinge": float(residual_only_hinge.detach()),
    }
    return total, row


def mean_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    return {
        key: float(sum(row[key] for row in rows) / len(rows))
        for key in rows[0]
    }
