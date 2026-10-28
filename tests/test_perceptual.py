from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
from torch import nn
import pytest

from lightwave_seeded_vqvae.perceptual import (
    freeze_perceptual_metric,
    load_lpips_alex_v01,
    lpips_alex_per_image,
)


class _RecordingMetric(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.first: torch.Tensor | None = None
        self.second: torch.Tensor | None = None

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        self.first = first.detach().clone()
        self.second = second.detach().clone()
        return self.scale * (first - second).square().mean(
            dim=(1, 2, 3), keepdim=True
        )


def test_loader_requests_alex_v01_and_freezes_parameters(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    metric = _RecordingMetric()

    def fake_lpips(**kwargs):
        calls.append(kwargs)
        return metric

    monkeypatch.setitem(sys.modules, "lpips", SimpleNamespace(LPIPS=fake_lpips))
    loaded = load_lpips_alex_v01(device="cpu")

    assert loaded is metric
    assert calls == [{"net": "alex", "version": "0.1", "verbose": False}]
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


def test_per_image_mapping_and_reconstruction_gradient() -> None:
    metric = freeze_perceptual_metric(_RecordingMetric())
    reconstruction = torch.stack(
        [torch.zeros(3, 32, 32), torch.full((3, 32, 32), 0.75)]
    ).requires_grad_()
    target = torch.stack(
        [torch.full((3, 32, 32), 0.5), torch.full((3, 32, 32), 0.25)]
    )

    values = lpips_alex_per_image(reconstruction, target, metric)

    assert metric.first is not None and metric.second is not None
    torch.testing.assert_close(metric.first[0], torch.full((3, 32, 32), -1.0))
    torch.testing.assert_close(metric.second[0], torch.zeros(3, 32, 32))
    torch.testing.assert_close(values, torch.tensor([1.0, 1.0]))
    values.mean().backward()
    assert reconstruction.grad is not None
    assert float(reconstruction.grad.abs().sum()) > 0.0
    assert metric.scale.grad is None


def test_per_image_rejects_unsupported_channel_count() -> None:
    metric = _RecordingMetric()
    with pytest.raises(ValueError, match="one or three channels"):
        lpips_alex_per_image(
            torch.zeros(1, 2, 4, 4),
            torch.zeros(1, 2, 4, 4),
            metric,
        )
