from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class QuantizerOutput:
    quantized: Tensor
    base_indices: Tensor
    residual_indices: Tensor
    commitment_loss: Tensor
    codebook_loss: Tensor
    residual_norm_loss: Tensor
    first_stage_error: Tensor
    final_error: Tensor


def _squared_distances(query: Tensor, codebook: Tensor) -> Tensor:
    query_sq = query.square().sum(dim=-1, keepdim=True)
    codebook_sq = codebook.square().sum(dim=-1).unsqueeze(-2)
    cross = torch.matmul(query, codebook.transpose(-1, -2))
    return (query_sq + codebook_sq - 2.0 * cross).clamp_min_(0.0)


def _scaled_vector_mse(first: Tensor, second: Tensor, reference_dim: int) -> Tensor:
    """Mean squared L2 error scaled to a fixed reference dimension.

    At D=reference_dim this is identical to elementwise MSE. Keeping the divisor
    fixed prevents the VQ losses from becoming artificially weaker when D grows.
    """
    return (first - second).square().sum(dim=-1).mean() / reference_dim


class SeededResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        base_codebooks: Tensor | np.ndarray,
        *,
        residual_codebook_size: int,
        latent_dim: int,
        normalize_latents: bool = True,
        loss_reference_dim: int = 64,
    ) -> None:
        super().__init__()
        if residual_codebook_size <= 0:
            raise ValueError("residual_codebook_size must be positive")
        self.latent_dim = latent_dim
        self.normalize_latents = normalize_latents
        if loss_reference_dim <= 0:
            raise ValueError("loss_reference_dim must be positive")
        self.loss_reference_dim = loss_reference_dim
        self.register_buffer("base_codebooks", torch.empty(0), persistent=False)
        self.set_base_codebooks(base_codebooks)

        scale = 0.05 / max(latent_dim, 1) ** 0.5
        self.residual_codebook = nn.Parameter(
            torch.empty(residual_codebook_size, latent_dim).normal_(mean=0.0, std=scale)
        )

    @property
    def state_count(self) -> int:
        return int(self.base_codebooks.shape[0])

    @property
    def base_codebook_size(self) -> int:
        return int(self.base_codebooks.shape[1])

    @property
    def residual_codebook_size(self) -> int:
        return int(self.residual_codebook.shape[0])

    @torch.no_grad()
    def set_base_codebooks(self, base_codebooks: Tensor | np.ndarray) -> None:
        bank = torch.as_tensor(base_codebooks, dtype=torch.float32).detach().clone()
        if bank.ndim != 3 or bank.shape[0] != 10:
            raise ValueError(f"base bank must have shape (10, K_b, D), got {tuple(bank.shape)}")
        if bank.shape[-1] != self.latent_dim:
            raise ValueError(
                f"base bank D={bank.shape[-1]} does not match latent_dim={self.latent_dim}"
            )
        if not torch.isfinite(bank).all():
            raise ValueError("base bank contains non-finite values")
        norms = bank.norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=2e-5, rtol=2e-5):
            raise ValueError("every base codeword must have unit L2 norm")
        self.base_codebooks = bank

    def _tokens(self, latent: Tensor) -> tuple[Tensor, int, int, int]:
        if latent.ndim != 4 or latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent must be NCHW with D={self.latent_dim}, got {tuple(latent.shape)}"
            )
        batch, _, height, width = latent.shape
        tokens = latent.permute(0, 2, 3, 1).reshape(batch, height * width, self.latent_dim)
        if self.normalize_latents:
            tokens = F.normalize(tokens, p=2, dim=-1, eps=1e-8)
        return tokens, batch, height, width

    def forward(self, latent: Tensor, state_ids: Tensor) -> QuantizerOutput:
        tokens, batch, height, width = self._tokens(latent)
        state_ids = state_ids.to(device=latent.device, dtype=torch.long)
        if state_ids.shape != (batch,):
            raise ValueError(f"state_ids must have shape ({batch},), got {tuple(state_ids.shape)}")
        if bool(((state_ids < 0) | (state_ids >= self.state_count)).any()):
            raise ValueError("state_ids contain an out-of-range state")

        active_base = self.base_codebooks[state_ids]
        base_distances = _squared_distances(tokens, active_base)
        base_indices = base_distances.argmin(dim=-1)
        gather_index = base_indices.unsqueeze(-1).expand(-1, -1, self.latent_dim)
        selected_base = active_base.gather(1, gather_index)

        residual_target = tokens - selected_base
        residual_distances = _squared_distances(
            residual_target,
            self.residual_codebook.unsqueeze(0),
        )
        residual_indices = residual_distances.argmin(dim=-1)
        selected_residual = F.embedding(residual_indices, self.residual_codebook)
        quantized = selected_base + selected_residual

        commitment_loss = _scaled_vector_mse(
            tokens, quantized.detach(), self.loss_reference_dim
        )
        codebook_loss = _scaled_vector_mse(
            selected_residual, residual_target.detach(), self.loss_reference_dim
        )
        residual_norm_loss = (
            self.residual_codebook.square().sum(dim=-1).mean()
            / self.loss_reference_dim
        )
        first_stage_error = _scaled_vector_mse(
            tokens, selected_base.detach(), self.loss_reference_dim
        )
        final_error = _scaled_vector_mse(
            tokens, quantized.detach(), self.loss_reference_dim
        )

        straight_through = tokens + (quantized - tokens).detach()
        straight_through = straight_through.reshape(batch, height, width, self.latent_dim)
        straight_through = straight_through.permute(0, 3, 1, 2).contiguous()
        return QuantizerOutput(
            quantized=straight_through,
            base_indices=base_indices.reshape(batch, height, width),
            residual_indices=residual_indices.reshape(batch, height, width),
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            residual_norm_loss=residual_norm_loss,
            first_stage_error=first_stage_error,
            final_error=final_error,
        )

    def quantize_residual_only(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        """Strong R-only control: re-quantize encoder tokens without using B_s."""
        tokens, batch, height, width = self._tokens(latent)
        distances = _squared_distances(tokens, self.residual_codebook.unsqueeze(0))
        indices = distances.argmin(dim=-1)
        vectors = F.embedding(indices, self.residual_codebook)
        quantized = vectors.reshape(batch, height, width, self.latent_dim)
        return quantized.permute(0, 3, 1, 2).contiguous(), indices.reshape(
            batch, height, width
        )

    def decode_indices(
        self,
        base_indices: Tensor,
        residual_indices: Tensor,
        state_ids: Tensor,
        *,
        include_base: bool = True,
        include_residual: bool = True,
    ) -> Tensor:
        if base_indices.shape != residual_indices.shape or base_indices.ndim != 3:
            raise ValueError("base and residual indices must share shape (N, H, W)")
        batch, height, width = base_indices.shape
        state_ids = state_ids.to(device=self.base_codebooks.device, dtype=torch.long)
        if state_ids.shape != (batch,):
            raise ValueError(f"state_ids must have shape ({batch},)")
        if bool(((state_ids < 0) | (state_ids >= self.state_count)).any()):
            raise ValueError("state_ids contain an out-of-range state")

        flat_base = base_indices.to(self.base_codebooks.device, dtype=torch.long).reshape(batch, -1)
        flat_residual = residual_indices.to(
            self.residual_codebook.device, dtype=torch.long
        ).reshape(batch, -1)
        active_base = self.base_codebooks[state_ids]
        base_vectors = active_base.gather(
            1, flat_base.unsqueeze(-1).expand(-1, -1, self.latent_dim)
        )
        residual_vectors = F.embedding(flat_residual, self.residual_codebook)
        vectors = torch.zeros_like(base_vectors)
        if include_base:
            vectors = vectors + base_vectors
        if include_residual:
            vectors = vectors + residual_vectors
        return vectors.reshape(batch, height, width, self.latent_dim).permute(0, 3, 1, 2)


class TrainableResidualVectorQuantizer(nn.Module):
    """Matched software control with trainable, state-independent B and R.

    It emits the same two indices per token as the seeded model. This keeps the
    codebook cardinalities, nominal index rate, and nearest-neighbour stages
    matched while removing the chip-derived frozen bank.
    """

    def __init__(
        self,
        *,
        base_codebook_size: int,
        residual_codebook_size: int,
        latent_dim: int,
        normalize_latents: bool = True,
        loss_reference_dim: int = 64,
    ) -> None:
        super().__init__()
        if base_codebook_size <= 0 or residual_codebook_size <= 0:
            raise ValueError("codebook sizes must be positive")
        self.latent_dim = latent_dim
        self.normalize_latents = normalize_latents
        if loss_reference_dim <= 0:
            raise ValueError("loss_reference_dim must be positive")
        self.loss_reference_dim = loss_reference_dim
        base = F.normalize(torch.randn(base_codebook_size, latent_dim), dim=-1)
        self.base_codebook = nn.Parameter(base)
        scale = 0.05 / max(latent_dim, 1) ** 0.5
        self.residual_codebook = nn.Parameter(
            torch.empty(residual_codebook_size, latent_dim).normal_(mean=0.0, std=scale)
        )

    @property
    def state_count(self) -> int:
        return 1

    @property
    def base_codebook_size(self) -> int:
        return int(self.base_codebook.shape[0])

    @property
    def residual_codebook_size(self) -> int:
        return int(self.residual_codebook.shape[0])

    def _tokens(self, latent: Tensor) -> tuple[Tensor, int, int, int]:
        if latent.ndim != 4 or latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent must be NCHW with D={self.latent_dim}, got {tuple(latent.shape)}"
            )
        batch, _, height, width = latent.shape
        tokens = latent.permute(0, 2, 3, 1).reshape(batch, height * width, self.latent_dim)
        if self.normalize_latents:
            tokens = F.normalize(tokens, p=2, dim=-1, eps=1e-8)
        return tokens, batch, height, width

    def forward(self, latent: Tensor, state_ids: Tensor) -> QuantizerOutput:
        tokens, batch, height, width = self._tokens(latent)
        state_ids = state_ids.to(device=latent.device, dtype=torch.long)
        if state_ids.shape != (batch,):
            raise ValueError(f"state_ids must have shape ({batch},), got {tuple(state_ids.shape)}")
        if bool((state_ids != 0).any()):
            raise ValueError("software baseline has only state 0")

        base_distances = _squared_distances(tokens, self.base_codebook.unsqueeze(0))
        base_indices = base_distances.argmin(dim=-1)
        selected_base = F.embedding(base_indices, self.base_codebook)

        residual_target = tokens - selected_base
        residual_distances = _squared_distances(
            residual_target, self.residual_codebook.unsqueeze(0)
        )
        residual_indices = residual_distances.argmin(dim=-1)
        selected_residual = F.embedding(residual_indices, self.residual_codebook)
        quantized = selected_base + selected_residual

        commitment_loss = _scaled_vector_mse(
            tokens, quantized.detach(), self.loss_reference_dim
        )
        base_codebook_loss = _scaled_vector_mse(
            selected_base, tokens.detach(), self.loss_reference_dim
        )
        residual_codebook_loss = _scaled_vector_mse(
            selected_residual, residual_target.detach(), self.loss_reference_dim
        )
        codebook_loss = base_codebook_loss + residual_codebook_loss
        residual_norm_loss = (
            self.residual_codebook.square().sum(dim=-1).mean()
            / self.loss_reference_dim
        )
        first_stage_error = _scaled_vector_mse(
            tokens, selected_base.detach(), self.loss_reference_dim
        )
        final_error = _scaled_vector_mse(
            tokens, quantized.detach(), self.loss_reference_dim
        )

        straight_through = tokens + (quantized - tokens).detach()
        straight_through = straight_through.reshape(batch, height, width, self.latent_dim)
        straight_through = straight_through.permute(0, 3, 1, 2).contiguous()
        return QuantizerOutput(
            quantized=straight_through,
            base_indices=base_indices.reshape(batch, height, width),
            residual_indices=residual_indices.reshape(batch, height, width),
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            residual_norm_loss=residual_norm_loss,
            first_stage_error=first_stage_error,
            final_error=final_error,
        )

    def quantize_residual_only(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        tokens, batch, height, width = self._tokens(latent)
        distances = _squared_distances(tokens, self.residual_codebook.unsqueeze(0))
        indices = distances.argmin(dim=-1)
        vectors = F.embedding(indices, self.residual_codebook)
        quantized = vectors.reshape(batch, height, width, self.latent_dim)
        return quantized.permute(0, 3, 1, 2).contiguous(), indices.reshape(
            batch, height, width
        )

    def decode_indices(
        self,
        base_indices: Tensor,
        residual_indices: Tensor,
        state_ids: Tensor,
        *,
        include_base: bool = True,
        include_residual: bool = True,
    ) -> Tensor:
        if base_indices.shape != residual_indices.shape or base_indices.ndim != 3:
            raise ValueError("base and residual indices must share shape (N, H, W)")
        batch, height, width = base_indices.shape
        state_ids = state_ids.to(device=self.base_codebook.device, dtype=torch.long)
        if state_ids.shape != (batch,) or bool((state_ids != 0).any()):
            raise ValueError("software baseline state_ids must be all-zero with shape (N,)")
        base_vectors = F.embedding(
            base_indices.to(self.base_codebook.device, dtype=torch.long),
            self.base_codebook,
        )
        residual_vectors = F.embedding(
            residual_indices.to(self.residual_codebook.device, dtype=torch.long),
            self.residual_codebook,
        )
        vectors = torch.zeros_like(base_vectors)
        if include_base:
            vectors = vectors + base_vectors
        if include_residual:
            vectors = vectors + residual_vectors
        return vectors.permute(0, 3, 1, 2).contiguous()
