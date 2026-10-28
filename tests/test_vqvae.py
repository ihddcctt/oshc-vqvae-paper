from __future__ import annotations

import torch
import torch.nn.functional as F

from lightwave_seeded_vqvae.models.vqvae import (
    ContinuousAutoencoder,
    ContinuousAutoencoderOutput,
    SeededResidualVQVAE,
    TrainableResidualVQVAE,
)


def test_end_to_end_mnist_shape_and_checkpoint_excludes_bank() -> None:
    generator = torch.Generator().manual_seed(23)
    bank = F.normalize(torch.randn(10, 8, 64, generator=generator), dim=-1)
    model = SeededResidualVQVAE(
        bank,
        in_channels=1,
        latent_dim=64,
        residual_codebook_size=8,
        pretrained_encoder=False,
    )
    images = torch.rand(2, 1, 28, 28)
    states = torch.tensor([0, 9])
    output = model(images, states)
    assert output.reconstruction.shape == images.shape
    assert output.quantizer.base_indices.shape == (2, 7, 7)
    state_keys = model.state_dict().keys()
    assert not any("base_codebooks" in key for key in state_keys)
    reconstruction = model.decode_indices(
        output.quantizer.base_indices,
        output.quantizer.residual_indices,
        states,
    )
    assert reconstruction.shape == images.shape
    torch.testing.assert_close(reconstruction, output.reconstruction, rtol=0.0, atol=2e-6)


def test_software_baseline_matches_two_index_interface() -> None:
    model = TrainableResidualVQVAE(
        in_channels=1,
        latent_dim=64,
        base_codebook_size=8,
        residual_codebook_size=8,
        pretrained_encoder=False,
    )
    images = torch.rand(2, 1, 28, 28)
    states = torch.zeros(2, dtype=torch.long)
    output = model(images, states)
    assert output.reconstruction.shape == images.shape
    assert output.quantizer.base_indices.shape == (2, 7, 7)
    assert output.quantizer.residual_indices.shape == (2, 7, 7)
    assert "quantizer.base_codebook" in model.state_dict()


def test_end_to_end_cifar_shape_is_32_to_8_to_32() -> None:
    generator = torch.Generator().manual_seed(29)
    bank = F.normalize(torch.randn(10, 8, 64, generator=generator), dim=-1)
    model = SeededResidualVQVAE(
        bank,
        in_channels=3,
        latent_dim=64,
        residual_codebook_size=8,
        pretrained_encoder=False,
    )
    images = torch.rand(2, 3, 32, 32, generator=generator)
    output = model(images, torch.tensor([1, 8]))
    assert output.encoder_latent.shape == (2, 64, 8, 8)
    assert output.quantizer.base_indices.shape == (2, 8, 8)
    assert output.quantizer.residual_indices.shape == (2, 8, 8)
    assert output.reconstruction.shape == images.shape
    assert bool(((output.reconstruction >= 0.0) & (output.reconstruction <= 1.0)).all())


def test_seeded_model_wires_distinct_decoder_block_counts() -> None:
    generator = torch.Generator().manual_seed(31)
    bank = F.normalize(torch.randn(10, 8, 64, generator=generator), dim=-1)
    model = SeededResidualVQVAE(
        bank,
        in_channels=3,
        latent_dim=64,
        residual_codebook_size=8,
        pretrained_encoder=False,
        decoder_blocks_by_stage=[3, 2, 2],
        decoder_zero_init_extra_residual_blocks=True,
    )

    assert model.decoder.blocks_by_stage == (3, 2, 2)
    assert len(model.decoder.pre) == 3
    assert len(model.decoder.up1.blocks) == 2
    assert len(model.decoder.up2.blocks) == 2
    assert model.decoder.zero_init_extra_residual_blocks is True
    assert torch.count_nonzero(model.decoder.pre[1].block[-1].weight).item() == 0


def test_continuous_autoencoder_is_direct_has_no_quantizer_and_backpropagates() -> None:
    model = ContinuousAutoencoder(
        in_channels=3,
        latent_dim=64,
        pretrained_encoder=False,
        decoder_blocks_by_stage=[3, 2, 2],
    ).eval()
    images = torch.rand(1, 3, 32, 32)
    decoder_inputs: list[torch.Tensor] = []

    handle = model.decoder.register_forward_pre_hook(
        lambda _module, inputs: decoder_inputs.append(inputs[0])
    )
    try:
        output = model(images)
    finally:
        handle.remove()

    assert isinstance(output, ContinuousAutoencoderOutput)
    assert output.encoder_latent.shape == (1, 64, 8, 8)
    assert output.reconstruction.shape == images.shape
    assert bool(((output.reconstruction >= 0.0) & (output.reconstruction <= 1.0)).all())
    assert not hasattr(model, "quantizer")
    assert not hasattr(output, "quantizer")
    assert len(decoder_inputs) == 1
    assert decoder_inputs[0] is output.encoder_latent

    output.reconstruction.square().mean().backward()
    encoder_gradient = model.encoder.to_latent.weight.grad
    decoder_gradient = model.decoder.from_latent.weight.grad
    assert encoder_gradient is not None and torch.isfinite(encoder_gradient).all()
    assert decoder_gradient is not None and torch.isfinite(decoder_gradient).all()
    assert bool((encoder_gradient.abs() > 0).any())
    assert bool((decoder_gradient.abs() > 0).any())
