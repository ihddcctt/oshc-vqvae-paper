from __future__ import annotations

import torch
from torch import nn
import pytest

from lightwave_seeded_vqvae.training import batch_objective, scheduled_weight


def test_scheduled_weight() -> None:
    assert scheduled_weight(9, start_epoch=10, ramp_epochs=1, weight=0.35) == 0.0
    assert scheduled_weight(10, start_epoch=10, ramp_epochs=1, weight=0.35) == 0.35
    assert scheduled_weight(10, start_epoch=10, ramp_epochs=5, weight=0.5) == 0.1
    assert scheduled_weight(14, start_epoch=10, ramp_epochs=5, weight=0.5) == 0.5


def test_margin_detaches_correct_reconstruction_loss() -> None:
    reconstruction = torch.tensor([0.5], requires_grad=True)
    target = torch.tensor([0.0])
    zero = torch.tensor(0.0)
    wrong = torch.tensor(0.1, requires_grad=True)
    total, metrics = batch_objective(
        reconstruction,
        target,
        commitment_loss=zero,
        codebook_loss=zero,
        residual_norm_loss=zero,
        commitment_weight=0.0,
        codebook_weight=0.0,
        residual_norm_weight=0.0,
        wrong_state_mse=wrong,
        wrong_state_margin=0.02,
        wrong_state_weight=1.0,
    )
    total.backward()
    assert metrics["wrong_state_hinge"] > 0.0
    assert wrong.grad is not None and wrong.grad.item() == -1.0
    # d(MSE(0.5,0))/d0.5 is 1.0; detached margin adds no extra gradient.
    assert reconstruction.grad is not None and reconstruction.grad.item() == 1.0


class _FakePerceptualMetric(nn.Module):
    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return (first - second).square().mean(dim=(1, 2, 3), keepdim=True)


def test_default_mse_objective_is_numerically_backward_compatible() -> None:
    reconstruction = torch.tensor([0.25, 0.75], requires_grad=True)
    target = torch.tensor([0.0, 1.0])
    commitment = torch.tensor(0.2)
    codebook = torch.tensor(0.3)
    residual_norm = torch.tensor(0.4)

    total, metrics = batch_objective(
        reconstruction,
        target,
        commitment_loss=commitment,
        codebook_loss=codebook,
        residual_norm_loss=residual_norm,
        commitment_weight=0.25,
        codebook_weight=1.0,
        residual_norm_weight=0.1,
    )

    raw_mse = torch.nn.functional.mse_loss(reconstruction, target)
    expected = raw_mse + 0.25 * commitment + codebook + 0.1 * residual_norm
    torch.testing.assert_close(total, expected)
    assert metrics["reconstruction_mse"] == float(raw_mse.detach())
    assert metrics["reconstruction_l1"] == 0.25
    assert metrics["reconstruction_lpips"] == 0.0
    assert metrics["reconstruction_loss"] == metrics["reconstruction_mse"]


def test_l1_lpips_objective_logs_components_and_keeps_input_gradient() -> None:
    reconstruction = torch.full((2, 3, 4, 4), 0.75, requires_grad=True)
    target = torch.full_like(reconstruction, 0.25)
    zero = torch.tensor(0.0)

    total, metrics = batch_objective(
        reconstruction,
        target,
        commitment_loss=zero,
        codebook_loss=zero,
        residual_norm_loss=zero,
        commitment_weight=0.0,
        codebook_weight=0.0,
        residual_norm_weight=0.0,
        reconstruction_loss_type="l1_lpips",
        perceptual_metric=_FakePerceptualMetric(),
        perceptual_weight=0.2,
    )

    # L1 is 0.5. Mapping to [-1, 1] makes the fake squared distance 1.0.
    torch.testing.assert_close(total, torch.tensor(0.7))
    assert metrics["reconstruction_mse"] == 0.25
    assert metrics["reconstruction_l1"] == 0.5
    assert metrics["reconstruction_lpips"] == 1.0
    assert abs(metrics["reconstruction_loss"] - 0.7) < 1e-6
    total.backward()
    assert reconstruction.grad is not None
    assert bool(torch.isfinite(reconstruction.grad).all())
    assert float(reconstruction.grad.abs().sum()) > 0.0


def test_l1_lpips_hinges_remain_based_on_raw_mse() -> None:
    reconstruction = torch.full((1, 3, 2, 2), 0.5)
    target = torch.zeros_like(reconstruction)
    zero = torch.tensor(0.0)
    wrong = torch.tensor(0.3)
    residual_only = torch.tensor(0.3)

    _, metrics = batch_objective(
        reconstruction,
        target,
        commitment_loss=zero,
        codebook_loss=zero,
        residual_norm_loss=zero,
        commitment_weight=0.0,
        codebook_weight=0.0,
        residual_norm_weight=0.0,
        wrong_state_mse=wrong,
        wrong_state_margin=0.1,
        wrong_state_weight=1.0,
        residual_only_mse=residual_only,
        residual_only_margin=0.1,
        residual_only_weight=1.0,
        reconstruction_loss_type="l1_lpips",
        perceptual_metric=_FakePerceptualMetric(),
        perceptual_weight=2.0,
    )

    # Raw MSE is 0.25, so each hinge is relu(0.25 + 0.1 - 0.3) = 0.05.
    assert abs(metrics["wrong_state_hinge"] - 0.05) < 1e-6
    assert abs(metrics["residual_only_hinge"] - 0.05) < 1e-6


def test_l1_lpips_requires_metric() -> None:
    zero = torch.tensor(0.0)
    with pytest.raises(ValueError, match="perceptual_metric is required"):
        batch_objective(
            torch.zeros(1, 3, 2, 2),
            torch.zeros(1, 3, 2, 2),
            commitment_loss=zero,
            codebook_loss=zero,
            residual_norm_loss=zero,
            commitment_weight=0.0,
            codebook_weight=0.0,
            residual_norm_weight=0.0,
            reconstruction_loss_type="l1_lpips",
        )
