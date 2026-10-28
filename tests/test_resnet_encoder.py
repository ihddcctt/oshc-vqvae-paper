from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torchvision.models import resnet18

from lightwave_seeded_vqvae.models.resnet import (
    DecoderResidualBlock,
    ResidualImageDecoder,
    SmallImageResNet18Encoder,
    adapt_imagenet_stem,
)


def test_adapt_stem_center_crop_and_energy() -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(64, 3, 7, 7, generator=generator)
    weight[0].zero_()
    adapted = adapt_imagenet_stem(weight, 3)
    assert adapted.shape == (64, 3, 3, 3)
    assert torch.isfinite(adapted).all()
    assert torch.allclose(
        adapted[1:].flatten(1).norm(dim=1),
        weight[1:].flatten(1).norm(dim=1),
        atol=1e-5,
        rtol=1e-5,
    )


def test_gray_stem_matches_replicated_rgb_before_input_normalization() -> None:
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn(8, 3, 7, 7, generator=generator)
    rgb_weight = adapt_imagenet_stem(weight, 3)
    gray_weight = adapt_imagenet_stem(weight, 1)
    image = torch.randn(2, 1, 12, 12, generator=generator)
    gray_result = F.conv2d(image, gray_weight, padding=1)
    rgb_result = F.conv2d(image.repeat(1, 3, 1, 1), rgb_weight, padding=1)
    assert torch.allclose(gray_result, rgb_result, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    ("channels", "size", "latent_dim", "expected_spatial"),
    [(1, 28, 64, 7), (1, 28, 128, 7), (3, 32, 64, 8), (3, 32, 128, 8)],
)
def test_small_resnet_shapes(
    channels: int, size: int, latent_dim: int, expected_spatial: int
) -> None:
    model = SmallImageResNet18Encoder(
        latent_dim=latent_dim,
        in_channels=channels,
        pretrained=False,
    ).eval()
    with torch.no_grad():
        output = model(torch.rand(2, channels, size, size))
    assert output.shape == (2, latent_dim, expected_spatial, expected_spatial)


def test_pretrained_transfer_resets_bn_stats_but_preserves_affine() -> None:
    reference = resnet18(weights=None)
    with torch.no_grad():
        reference.conv1.weight.normal_()
        reference.bn1.weight.fill_(1.25)
        reference.bn1.bias.fill_(-0.25)
        reference.bn1.running_mean.fill_(3.0)
        reference.bn1.running_var.fill_(4.0)
    expected_stem = adapt_imagenet_stem(reference.conv1.weight.detach(), 1)
    encoder = SmallImageResNet18Encoder(
        latent_dim=64,
        in_channels=1,
        pretrained=True,
        reference_model=reference,
    )
    assert torch.equal(encoder.conv1.weight, expected_stem)
    assert torch.equal(encoder.bn1.weight, torch.full_like(encoder.bn1.weight, 1.25))
    assert torch.equal(encoder.bn1.bias, torch.full_like(encoder.bn1.bias, -0.25))
    assert torch.equal(encoder.bn1.running_mean, torch.zeros_like(encoder.bn1.running_mean))
    assert torch.equal(encoder.bn1.running_var, torch.ones_like(encoder.bn1.running_var))
    assert encoder.bn1.num_batches_tracked.item() == 0
    keys = encoder.state_dict().keys()
    assert not any(key.startswith("layer4") for key in keys)
    assert not any("fc" in key for key in keys)
    assert encoder.conv1.kernel_size == (3, 3)
    assert encoder.conv1.stride == (1, 1)


def test_rgb_checkpoint_restores_imagenet_normalization_and_output() -> None:
    source = SmallImageResNet18Encoder(
        latent_dim=16,
        in_channels=3,
        pretrained=True,
        reference_model=resnet18(weights=None),
    ).eval()
    restored = SmallImageResNet18Encoder(
        latent_dim=16,
        in_channels=3,
        pretrained=False,
    ).eval()
    restored.load_state_dict(source.state_dict(), strict=True)
    assert torch.equal(source.input_mean, restored.input_mean)
    assert torch.equal(source.input_std, restored.input_std)
    assert torch.allclose(
        restored.input_mean.flatten(), torch.tensor([0.485, 0.456, 0.406])
    )
    images = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(source(images), restored(images))


def test_decoder_legacy_scalar_configuration_keeps_strict_checkpoint_layout() -> None:
    legacy = ResidualImageDecoder(latent_dim=64, out_channels=3)
    restored = ResidualImageDecoder(
        latent_dim=64,
        out_channels=3,
        blocks_per_stage=1,
    )

    assert legacy.blocks_by_stage == (1, 1, 1)
    assert restored.blocks_by_stage == (1, 1, 1)
    assert tuple(legacy.state_dict()) == tuple(restored.state_dict())
    restored.load_state_dict(legacy.state_dict(), strict=True)


def test_decoder_accepts_distinct_block_counts_at_each_resolution() -> None:
    decoder = ResidualImageDecoder(
        latent_dim=64,
        out_channels=3,
        blocks_by_stage=[3, 2, 2],
    )

    assert decoder.blocks_by_stage == (3, 2, 2)
    assert len(decoder.pre) == 3
    assert len(decoder.up1.blocks) == 2
    assert len(decoder.up2.blocks) == 2
    reconstruction = decoder(torch.randn(2, 64, 8, 8))
    assert reconstruction.shape == (2, 3, 32, 32)
    assert bool(((reconstruction >= 0.0) & (reconstruction <= 1.0)).all())


def test_decoder_zero_init_extra_residual_blocks_is_explicit_opt_in() -> None:
    torch.manual_seed(20260907)
    default_decoder = ResidualImageDecoder(
        latent_dim=64,
        out_channels=1,
        blocks_by_stage=[3, 2, 2],
    )
    torch.manual_seed(20260907)
    explicit_default_decoder = ResidualImageDecoder(
        latent_dim=64,
        out_channels=1,
        blocks_by_stage=[3, 2, 2],
        zero_init_extra_residual_blocks=False,
    )

    assert default_decoder.zero_init_extra_residual_blocks is False
    assert explicit_default_decoder.zero_init_extra_residual_blocks is False
    assert tuple(default_decoder.state_dict()) == tuple(
        explicit_default_decoder.state_dict()
    )
    for name, tensor in default_decoder.state_dict().items():
        assert torch.equal(tensor, explicit_default_decoder.state_dict()[name])


def test_decoder_zero_init_makes_only_added_blocks_exact_identities() -> None:
    torch.manual_seed(20260907)
    decoder = ResidualImageDecoder(
        latent_dim=64,
        out_channels=1,
        blocks_by_stage=[3, 2, 2],
        zero_init_extra_residual_blocks=True,
    )
    stages = (decoder.pre, decoder.up1.blocks, decoder.up2.blocks)

    assert decoder.zero_init_extra_residual_blocks is True
    for blocks in stages:
        assert isinstance(blocks[0], DecoderResidualBlock)
        assert blocks[0].zero_init_residual is False
        assert bool((blocks[0].block[-1].weight != 0).any())
        for block in blocks[1:]:
            assert isinstance(block, DecoderResidualBlock)
            assert block.zero_init_residual is True
            assert torch.count_nonzero(block.block[-1].weight).item() == 0
            inputs = torch.randn(2, block.block[-1].out_channels, 5, 5)
            torch.testing.assert_close(block(inputs), inputs, rtol=0.0, atol=0.0)


def test_stabilized_deep_decoder_starts_as_exact_legacy_decoder() -> None:
    torch.manual_seed(20260907)
    legacy = ResidualImageDecoder(
        latent_dim=64,
        out_channels=1,
        blocks_by_stage=[1, 1, 1],
    ).eval()
    torch.manual_seed(20260907)
    stabilized = ResidualImageDecoder(
        latent_dim=64,
        out_channels=1,
        blocks_by_stage=[3, 2, 2],
        zero_init_extra_residual_blocks=True,
    ).eval()

    legacy_state = legacy.state_dict()
    stabilized_state = stabilized.state_dict()
    for name, tensor in legacy_state.items():
        assert torch.equal(tensor, stabilized_state[name])
    inputs = torch.randn(2, 64, 7, 7)
    with torch.no_grad():
        torch.testing.assert_close(
            stabilized(inputs), legacy(inputs), rtol=0.0, atol=0.0
        )


def test_decoder_zero_init_preserves_strict_checkpoint_schema() -> None:
    standard = ResidualImageDecoder(
        latent_dim=64,
        out_channels=3,
        blocks_by_stage=[3, 2, 2],
    )
    stabilized = ResidualImageDecoder(
        latent_dim=64,
        out_channels=3,
        blocks_by_stage=[3, 2, 2],
        zero_init_extra_residual_blocks=True,
    )

    assert tuple(standard.state_dict()) == tuple(stabilized.state_dict())
    standard.load_state_dict(stabilized.state_dict(), strict=True)
    stabilized.load_state_dict(standard.state_dict(), strict=True)


@pytest.mark.parametrize(
    "blocks_by_stage",
    [
        [],
        [1, 2],
        [1, 2, 3, 4],
        [0, 2, 2],
        [-1, 2, 2],
        [True, 2, 2],
        [1.0, 2, 2],
        "322",
    ],
)
def test_decoder_rejects_invalid_per_stage_block_configuration(
    blocks_by_stage: object,
) -> None:
    with pytest.raises(ValueError):
        ResidualImageDecoder(
            latent_dim=64,
            out_channels=3,
            blocks_by_stage=blocks_by_stage,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("blocks_per_stage", [0, -1, True, 1.0])
def test_decoder_rejects_invalid_legacy_block_configuration(
    blocks_per_stage: object,
) -> None:
    with pytest.raises(ValueError):
        ResidualImageDecoder(
            latent_dim=64,
            out_channels=3,
            blocks_per_stage=blocks_per_stage,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("enabled", [0, 1, "true", None])
def test_decoder_rejects_non_boolean_zero_init_option(enabled: object) -> None:
    with pytest.raises(ValueError, match="zero_init_extra_residual_blocks"):
        ResidualImageDecoder(
            latent_dim=64,
            out_channels=3,
            zero_init_extra_residual_blocks=enabled,  # type: ignore[arg-type]
        )
