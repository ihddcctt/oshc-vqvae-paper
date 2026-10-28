from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


def adapt_imagenet_stem(weight_7x7: Tensor, in_channels: int, eps: float = 1e-12) -> Tensor:
    if tuple(weight_7x7.shape[1:]) != (3, 7, 7):
        raise ValueError(f"expected [out, 3, 7, 7], received {tuple(weight_7x7.shape)}")
    if in_channels not in (1, 3):
        raise ValueError("in_channels must be 1 or 3")

    weight_3x3 = weight_7x7[:, :, 2:5, 2:5].clone()
    norm_7x7 = weight_7x7.flatten(1).norm(dim=1)
    norm_3x3 = weight_3x3.flatten(1).norm(dim=1)
    scale = torch.where(
        norm_3x3 > eps,
        norm_7x7 / norm_3x3.clamp_min(eps),
        torch.ones_like(norm_3x3),
    )
    weight_3x3.mul_(scale[:, None, None, None])
    if in_channels == 1:
        weight_3x3 = weight_3x3.sum(dim=1, keepdim=True)
    return weight_3x3


def _weight_enum(name: str) -> ResNet18_Weights:
    if name != "IMAGENET1K_V1":
        raise ValueError("only the fixed IMAGENET1K_V1 weight enum is supported")
    return ResNet18_Weights.IMAGENET1K_V1


class SmallImageResNet18Encoder(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        in_channels: int,
        pretrained: bool = True,
        weights_name: str = "IMAGENET1K_V1",
        reference_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if latent_dim <= 1:
            raise ValueError("latent_dim must exceed one")
        if in_channels not in (1, 3):
            raise ValueError("in_channels must be 1 or 3")

        if reference_model is not None:
            reference = reference_model
            using_pretrained = pretrained
        else:
            weights = _weight_enum(weights_name) if pretrained else None
            reference = resnet18(weights=weights)
            using_pretrained = pretrained

        source_stem = reference.conv1.weight.detach()
        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        if using_pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(adapt_imagenet_stem(source_stem, in_channels))
        else:
            nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")

        self.bn1 = reference.bn1
        self.relu = reference.relu
        self.layer1 = reference.layer1
        self.layer2 = reference.layer2
        self.layer3 = reference.layer3
        self.to_latent = nn.Conv2d(256, latent_dim, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.to_latent.weight)
        nn.init.zeros_(self.to_latent.bias)

        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.reset_running_stats()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.pretrained = using_pretrained
        self.weights_name = weights_name if using_pretrained else "none"

        if in_channels == 3 and using_pretrained:
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        elif in_channels == 3:
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        else:
            mean, std = (0.5,), (0.5,)
        self.register_buffer(
            "input_mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "input_std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=True,
        )

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        modules = (self.conv1, self.bn1, self.layer1, self.layer2, self.layer3)
        for module in modules:
            yield from module.parameters()

    def projection_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.to_latent.parameters()

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != self.in_channels:
            raise ValueError(
                f"expected NCHW input with {self.in_channels} channels, got {tuple(images.shape)}"
            )
        x = (images - self.input_mean.to(images)) / self.input_std.to(images)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.to_latent(x)


class DecoderResidualBlock(nn.Module):
    def __init__(self, channels: int, *, zero_init_residual: bool = False) -> None:
        super().__init__()
        if not isinstance(zero_init_residual, bool):
            raise ValueError("zero_init_residual must be boolean")
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )
        if zero_init_residual:
            # The residual branch initially returns exactly zero, so this block
            # starts as an identity mapping while remaining fully trainable.
            nn.init.zeros_(self.block[-1].weight)
        self.zero_init_residual = zero_init_residual

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class ResizeConvStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        *,
        zero_init_extra_residual_blocks: bool = False,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.blocks = nn.Sequential(
            *(
                DecoderResidualBlock(
                    out_channels,
                    zero_init_residual=(
                        zero_init_extra_residual_blocks and block_index >= 1
                    ),
                )
                for block_index in range(blocks)
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.conv(self.upsample(x)))


class ResidualImageDecoder(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        out_channels: int,
        blocks_per_stage: int = 1,
        blocks_by_stage: Sequence[int] | None = None,
        zero_init_extra_residual_blocks: bool = False,
    ) -> None:
        super().__init__()
        if out_channels not in (1, 3):
            raise ValueError("out_channels must be 1 or 3")
        if not isinstance(zero_init_extra_residual_blocks, bool):
            raise ValueError("zero_init_extra_residual_blocks must be boolean")
        if (
            isinstance(blocks_per_stage, bool)
            or not isinstance(blocks_per_stage, int)
            or blocks_per_stage < 1
        ):
            raise ValueError("blocks_per_stage must be a positive integer")
        if blocks_by_stage is None:
            stage_blocks = (blocks_per_stage,) * 3
        else:
            if isinstance(blocks_by_stage, (str, bytes)):
                raise ValueError("blocks_by_stage must contain exactly three positive integers")
            try:
                stage_blocks = tuple(blocks_by_stage)
            except TypeError as error:
                raise ValueError(
                    "blocks_by_stage must contain exactly three positive integers"
                ) from error
            if len(stage_blocks) != 3:
                raise ValueError(
                    "blocks_by_stage must contain exactly three values: "
                    "[pre8, up16, up32]"
                )
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in stage_blocks
            ):
                raise ValueError("every blocks_by_stage value must be a positive integer")

        self.blocks_by_stage = stage_blocks
        self.zero_init_extra_residual_blocks = zero_init_extra_residual_blocks
        self.from_latent = nn.Conv2d(latent_dim, 256, 1)
        if zero_init_extra_residual_blocks:
            # Build the legacy one-block-per-stage path first. With the same
            # global seed, its parameters and initial function are therefore
            # exactly the same as a [1, 1, 1] decoder. Added capacity is then
            # appended as zero-residual identity mappings.
            self.pre = nn.Sequential(DecoderResidualBlock(256))
            self.up1 = ResizeConvStage(256, 128, 1)
            self.up2 = ResizeConvStage(128, 64, 1)
            self.out = nn.Conv2d(64, out_channels, 3, padding=1)
            for _ in range(stage_blocks[0] - 1):
                self.pre.append(
                    DecoderResidualBlock(256, zero_init_residual=True)
                )
            for _ in range(stage_blocks[1] - 1):
                self.up1.blocks.append(
                    DecoderResidualBlock(128, zero_init_residual=True)
                )
            for _ in range(stage_blocks[2] - 1):
                self.up2.blocks.append(
                    DecoderResidualBlock(64, zero_init_residual=True)
                )
        else:
            self.pre = nn.Sequential(
                *(DecoderResidualBlock(256) for _ in range(stage_blocks[0]))
            )
            self.up1 = ResizeConvStage(256, 128, stage_blocks[1])
            self.up2 = ResizeConvStage(128, 64, stage_blocks[2])
            self.out = nn.Conv2d(64, out_channels, 3, padding=1)

    def forward(self, latent: Tensor) -> Tensor:
        x = self.pre(self.from_latent(latent))
        x = self.up1(x)
        x = self.up2(x)
        return torch.sigmoid(self.out(x))
