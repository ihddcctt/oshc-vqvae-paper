from __future__ import annotations

import torch
import torch.nn.functional as F

from lightwave_seeded_vqvae.models.quantizer import (
    SeededResidualVectorQuantizer,
    TrainableResidualVectorQuantizer,
)


def make_bank(states: int = 10, codewords: int = 4, dimension: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(17)
    return F.normalize(
        torch.randn(states, codewords, dimension, generator=generator), dim=-1
    )


def test_quantizer_shapes_gradients_and_frozen_bank() -> None:
    bank = make_bank()
    quantizer = SeededResidualVectorQuantizer(
        bank,
        residual_codebook_size=4,
        latent_dim=8,
    )
    latent = torch.randn(3, 8, 2, 2, requires_grad=True)
    state_ids = torch.tensor([0, 4, 9])
    result = quantizer(latent, state_ids)
    assert result.quantized.shape == latent.shape
    assert result.base_indices.shape == (3, 2, 2)
    assert result.residual_indices.shape == (3, 2, 2)
    loss = result.quantized.square().mean() + result.codebook_loss + result.commitment_loss
    loss.backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert quantizer.residual_codebook.grad is not None
    assert quantizer.base_codebooks.requires_grad is False
    assert "base_codebooks" not in quantizer.state_dict()


def test_decode_uses_only_selected_state() -> None:
    bank = make_bank()
    quantizer = SeededResidualVectorQuantizer(
        bank,
        residual_codebook_size=4,
        latent_dim=8,
    )
    base_indices = torch.zeros(1, 2, 2, dtype=torch.long)
    residual_indices = torch.zeros(1, 2, 2, dtype=torch.long)
    correct = quantizer.decode_indices(base_indices, residual_indices, torch.tensor([0]))
    wrong = quantizer.decode_indices(base_indices, residual_indices, torch.tensor([1]))
    assert not torch.equal(correct, wrong)


def test_strong_residual_only_requantizes_encoder_tokens() -> None:
    bank = make_bank()
    quantizer = SeededResidualVectorQuantizer(
        bank,
        residual_codebook_size=4,
        latent_dim=8,
    )
    latent = torch.randn(2, 8, 3, 3)
    quantized, indices = quantizer.quantize_residual_only(latent)
    assert quantized.shape == latent.shape
    assert indices.shape == (2, 3, 3)
    assert int(indices.min()) >= 0
    assert int(indices.max()) < 4


def test_bank_dimension_is_strict() -> None:
    bank = make_bank(dimension=7)
    try:
        SeededResidualVectorQuantizer(bank, residual_codebook_size=4, latent_dim=8)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("dimension mismatch must fail")


def test_trainable_residual_baseline_trains_both_codebooks() -> None:
    quantizer = TrainableResidualVectorQuantizer(
        base_codebook_size=4,
        residual_codebook_size=4,
        latent_dim=8,
    )
    latent = torch.randn(3, 8, 2, 2, requires_grad=True)
    result = quantizer(latent, torch.zeros(3, dtype=torch.long))
    loss = result.quantized.square().mean() + result.codebook_loss + result.commitment_loss
    loss.backward()
    assert quantizer.base_codebook.grad is not None
    assert quantizer.residual_codebook.grad is not None
    assert torch.isfinite(quantizer.base_codebook.grad).all()
    assert torch.isfinite(quantizer.residual_codebook.grad).all()
    assert "base_codebook" in quantizer.state_dict()


def test_trainable_residual_baseline_rejects_nonzero_state() -> None:
    quantizer = TrainableResidualVectorQuantizer(
        base_codebook_size=4,
        residual_codebook_size=4,
        latent_dim=8,
    )
    latent = torch.randn(1, 8, 2, 2)
    try:
        quantizer(latent, torch.ones(1, dtype=torch.long))
    except ValueError as error:
        assert "only state 0" in str(error)
    else:
        raise AssertionError("software baseline must reject nonzero state IDs")
