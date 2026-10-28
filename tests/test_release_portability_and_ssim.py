from pathlib import Path

import numpy as np
import pytest
from skimage.metrics import structural_similarity
import torch

from lightwave_seeded_vqvae.data import resolve_dataset_root
from lightwave_seeded_vqvae.evaluation import SSIM_SPEC, gaussian_ssim_per_image
from lightwave_seeded_vqvae.cli import (
    evaluate_final,
    plot_mnist_stagec,
    plot_stl10_stagec,
)


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("value", [0.0, 1e-4, 0.003, 0.5, 1.0])
def test_ssim_identity_including_black_and_dark_constant_images(channels, value):
    image = torch.full((2, channels, 28, 28), value, dtype=torch.float32)
    actual = gaussian_ssim_per_image(image, image)
    # Constant high-valued windows can accumulate tiny float32 variance errors.
    torch.testing.assert_close(actual, torch.ones(2), atol=3e-4, rtol=0)
    assert SSIM_SPEC["implementation"] == "torch-gaussian-valid-v2"


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("case", ["random", "dark", "constant", "black"])
def test_ssim_matches_scikit_image_gaussian_population_reference(channels, case):
    rng = np.random.default_rng(219)
    shape = (2, channels, 32, 32)
    if case == "random":
        target = rng.random(shape).astype(np.float32)
        restored = np.clip(target + rng.normal(0, 0.04, shape), 0, 1).astype(np.float32)
    elif case == "dark":
        target = (rng.random(shape) * 0.002).astype(np.float32)
        restored = (target * 0.5).astype(np.float32)
    elif case == "constant":
        target = np.full(shape, 0.003, dtype=np.float32)
        restored = np.full(shape, 0.001, dtype=np.float32)
    else:
        target = np.zeros(shape, dtype=np.float32)
        restored = target.copy()
    expected = np.asarray([
        np.mean([
            structural_similarity(
                image[c], reference[c], data_range=1.0,
                gaussian_weights=True, sigma=1.5, win_size=11,
                use_sample_covariance=False, K1=0.01, K2=0.03,
            )
            for c in range(channels)
        ])
        for image, reference in zip(restored, target)
    ])
    actual = gaussian_ssim_per_image(
        torch.from_numpy(restored), torch.from_numpy(target)
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_dataset_root_override_preserves_recorded_path(tmp_path):
    historical = "/example/historical-data/STL10"
    actual, provenance = resolve_dataset_root(historical, tmp_path)
    assert actual == tmp_path.resolve()
    assert provenance == {
        "recorded_root": historical,
        "effective_root": str(tmp_path.resolve()),
        "override_supplied": True,
    }
    local, no_override = resolve_dataset_root(tmp_path)
    assert local == actual
    assert no_override["override_supplied"] is False


@pytest.mark.parametrize("module", [evaluate_final, plot_mnist_stagec, plot_stl10_stagec])
def test_final_clis_accept_dataset_root_override(module, tmp_path):
    argv = [
        "--seeded-checkpoint", "chip.pt", "--software-checkpoint", "software.pt",
        "--codebook", "bank.npz", "--output-dir", "new-output",
        "--dataset-root", str(tmp_path),
    ]
    if module is evaluate_final:
        argv += ["--manifest", "bank.manifest.json"]
    else:
        argv += [
            "--codebook-manifest", "bank.manifest.json",
            "--registry", "registry.json", "--evaluation-dir", "eval",
        ]
    args = module.build_parser().parse_args(argv)
    assert args.dataset_root == Path(tmp_path)
