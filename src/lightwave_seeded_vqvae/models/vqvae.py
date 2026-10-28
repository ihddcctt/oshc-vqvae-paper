from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from torch import Tensor, nn

from .quantizer import (
    QuantizerOutput,
    SeededResidualVectorQuantizer,
    TrainableResidualVectorQuantizer,
)
from .resnet import ResidualImageDecoder, SmallImageResNet18Encoder


@dataclass
class VQVAEOutput:
    reconstruction: Tensor
    quantizer: QuantizerOutput
    encoder_latent: Tensor


@dataclass
class ContinuousAutoencoderOutput:
    """Output of the non-quantized reconstruction ceiling."""

    reconstruction: Tensor
    encoder_latent: Tensor


class ContinuousAutoencoder(nn.Module):
    """Matched encoder/decoder ceiling whose latent bypasses vector quantization."""

    def __init__(
        self,
        *,
        in_channels: int,
        latent_dim: int,
        pretrained_encoder: bool = True,
        imagenet_weights: str = "IMAGENET1K_V1",
        decoder_blocks_per_stage: int = 1,
        decoder_blocks_by_stage: Sequence[int] | None = None,
        decoder_zero_init_extra_residual_blocks: bool = False,
        encoder_reference_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.encoder = SmallImageResNet18Encoder(
            latent_dim=latent_dim,
            in_channels=in_channels,
            pretrained=pretrained_encoder,
            weights_name=imagenet_weights,
            reference_model=encoder_reference_model,
        )
        self.decoder = ResidualImageDecoder(
            latent_dim=latent_dim,
            out_channels=in_channels,
            blocks_per_stage=decoder_blocks_per_stage,
            blocks_by_stage=decoder_blocks_by_stage,
            zero_init_extra_residual_blocks=(
                decoder_zero_init_extra_residual_blocks
            ),
        )

    def forward(self, images: Tensor) -> ContinuousAutoencoderOutput:
        encoder_latent = self.encoder(images)
        reconstruction = self.decoder(encoder_latent)
        return ContinuousAutoencoderOutput(
            reconstruction=reconstruction,
            encoder_latent=encoder_latent,
        )


class SeededResidualVQVAE(nn.Module):
    def __init__(
        self,
        base_codebooks: Tensor | np.ndarray,
        *,
        in_channels: int,
        latent_dim: int,
        residual_codebook_size: int,
        pretrained_encoder: bool = True,
        imagenet_weights: str = "IMAGENET1K_V1",
        normalize_latents: bool = True,
        loss_reference_dim: int = 64,
        decoder_blocks_per_stage: int = 1,
        decoder_blocks_by_stage: Sequence[int] | None = None,
        decoder_zero_init_extra_residual_blocks: bool = False,
        encoder_reference_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.encoder = SmallImageResNet18Encoder(
            latent_dim=latent_dim,
            in_channels=in_channels,
            pretrained=pretrained_encoder,
            weights_name=imagenet_weights,
            reference_model=encoder_reference_model,
        )
        self.quantizer = SeededResidualVectorQuantizer(
            base_codebooks,
            residual_codebook_size=residual_codebook_size,
            latent_dim=latent_dim,
            normalize_latents=normalize_latents,
            loss_reference_dim=loss_reference_dim,
        )
        self.decoder = ResidualImageDecoder(
            latent_dim=latent_dim,
            out_channels=in_channels,
            blocks_per_stage=decoder_blocks_per_stage,
            blocks_by_stage=decoder_blocks_by_stage,
            zero_init_extra_residual_blocks=(
                decoder_zero_init_extra_residual_blocks
            ),
        )

    def forward(self, images: Tensor, state_ids: Tensor) -> VQVAEOutput:
        latent = self.encoder(images)
        quantizer_output = self.quantizer(latent, state_ids)
        reconstruction = self.decoder(quantizer_output.quantized)
        return VQVAEOutput(
            reconstruction=reconstruction,
            quantizer=quantizer_output,
            encoder_latent=latent,
        )

    def reconstruct_residual_only(self, encoder_latent: Tensor) -> tuple[Tensor, Tensor]:
        latent, indices = self.quantizer.quantize_residual_only(encoder_latent)
        return self.decoder(latent), indices

    def decode_indices(
        self,
        base_indices: Tensor,
        residual_indices: Tensor,
        state_ids: Tensor,
        *,
        include_base: bool = True,
        include_residual: bool = True,
    ) -> Tensor:
        latent = self.quantizer.decode_indices(
            base_indices,
            residual_indices,
            state_ids,
            include_base=include_base,
            include_residual=include_residual,
        )
        return self.decoder(latent)


class TrainableResidualVQVAE(nn.Module):
    """State-independent, fully trainable two-stage software baseline."""

    def __init__(
        self,
        *,
        in_channels: int,
        latent_dim: int,
        base_codebook_size: int,
        residual_codebook_size: int,
        pretrained_encoder: bool = True,
        imagenet_weights: str = "IMAGENET1K_V1",
        normalize_latents: bool = True,
        loss_reference_dim: int = 64,
        decoder_blocks_per_stage: int = 1,
        decoder_blocks_by_stage: Sequence[int] | None = None,
        decoder_zero_init_extra_residual_blocks: bool = False,
        encoder_reference_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.encoder = SmallImageResNet18Encoder(
            latent_dim=latent_dim,
            in_channels=in_channels,
            pretrained=pretrained_encoder,
            weights_name=imagenet_weights,
            reference_model=encoder_reference_model,
        )
        self.quantizer = TrainableResidualVectorQuantizer(
            base_codebook_size=base_codebook_size,
            residual_codebook_size=residual_codebook_size,
            latent_dim=latent_dim,
            normalize_latents=normalize_latents,
            loss_reference_dim=loss_reference_dim,
        )
        self.decoder = ResidualImageDecoder(
            latent_dim=latent_dim,
            out_channels=in_channels,
            blocks_per_stage=decoder_blocks_per_stage,
            blocks_by_stage=decoder_blocks_by_stage,
            zero_init_extra_residual_blocks=(
                decoder_zero_init_extra_residual_blocks
            ),
        )

    def forward(self, images: Tensor, state_ids: Tensor) -> VQVAEOutput:
        latent = self.encoder(images)
        quantizer_output = self.quantizer(latent, state_ids)
        reconstruction = self.decoder(quantizer_output.quantized)
        return VQVAEOutput(
            reconstruction=reconstruction,
            quantizer=quantizer_output,
            encoder_latent=latent,
        )

    def reconstruct_residual_only(self, encoder_latent: Tensor) -> tuple[Tensor, Tensor]:
        latent, indices = self.quantizer.quantize_residual_only(encoder_latent)
        return self.decoder(latent), indices

    def decode_indices(
        self,
        base_indices: Tensor,
        residual_indices: Tensor,
        state_ids: Tensor,
        *,
        include_base: bool = True,
        include_residual: bool = True,
    ) -> Tensor:
        latent = self.quantizer.decode_indices(
            base_indices,
            residual_indices,
            state_ids,
            include_base=include_base,
            include_residual=include_residual,
        )
        return self.decoder(latent)
