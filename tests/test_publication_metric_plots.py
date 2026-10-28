"""Checks for publication pooling and plot options, independent of checkpoints."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import plot_metrics as plots


def _metrics():
    image = np.arange(1, 5, dtype=np.float64)[:, None, None]
    sender = np.arange(1, 11, dtype=np.float64)[None, :, None]
    receiver = np.arange(1, 11, dtype=np.float64)[None, None, :]
    wrong = (~np.eye(10, dtype=bool))[None]
    mse = (image * sender + receiver) / 1000 + wrong * 0.1
    ssim = 1 - mse
    lpips = 2 * mse
    return mse, ssim, lpips


def test_matrix_pooling_preserves_sender_receiver_axes():
    result = plots.compute_plot_statistics(*_metrics())
    assert result["MSE"]["matrix"].shape == (10, 10)
    assert result["MSE"]["matrix"][2, 7] == pytest.approx(0.1155)
    assert result["MSE"]["matrix"][7, 2] == pytest.approx(0.123)
    assert result["SSIM"]["matrix"][2, 7] == pytest.approx(1 - 0.1155)
    assert result["LPIPS"]["matrix"][2, 7] == pytest.approx(2 * 0.1155)


def test_rmse_matrix_is_root_after_image_pooling():
    mse, ssim, lpips = _metrics()
    result = plots.compute_plot_statistics(mse, ssim, lpips)
    expected = np.sqrt(mse.mean(axis=0))
    np.testing.assert_allclose(result["RMSE"]["matrix"], expected, atol=1e-15)
    assert not np.allclose(result["RMSE"]["matrix"], np.sqrt(mse).mean(axis=0))


def test_distributions_have_one_observation_per_image_and_root_after_states():
    mse, ssim, lpips = _metrics()
    result = plots.compute_plot_statistics(mse, ssim, lpips)
    correct = np.array([0.011, 0.0165, 0.022, 0.0275])
    wrong = correct + 0.1
    for name in ("MSE", "RMSE", "SSIM", "LPIPS"):
        assert result[name]["correct_per_image"].shape == (4,)
        assert result[name]["wrong_per_image"].shape == (4,)
    np.testing.assert_allclose(result["MSE"]["correct_per_image"], correct)
    np.testing.assert_allclose(result["MSE"]["wrong_per_image"], wrong)
    np.testing.assert_allclose(result["RMSE"]["correct_per_image"], np.sqrt(correct))
    np.testing.assert_allclose(result["RMSE"]["wrong_per_image"], np.sqrt(wrong))
    np.testing.assert_allclose(result["SSIM"]["correct_per_image"], 1 - correct)
    np.testing.assert_allclose(result["LPIPS"]["wrong_per_image"], 2 * wrong)
    mean_image_rmse = np.sqrt(np.diagonal(mse, axis1=1, axis2=2)).mean(axis=1)
    assert not np.allclose(result["RMSE"]["correct_per_image"], mean_image_rmse)


@pytest.mark.parametrize("kind", ("empty", "shape", "nan", "negative_mse"))
def test_invalid_metric_inputs_are_rejected(kind):
    mse, ssim, lpips = _metrics()
    if kind == "empty":
        mse, ssim, lpips = mse[:0], ssim[:0], lpips[:0]
    elif kind == "shape":
        lpips = lpips[:, :9]
    elif kind == "nan":
        ssim[1, 2, 3] = np.nan
    else:
        mse[0, 1, 2] = -0.001
    with pytest.raises(ValueError):
        plots.compute_plot_statistics(mse, ssim, lpips)


def test_publication_typography_defaults_and_overrides():
    parser = plots.build_parser()
    required = ["--input", "metrics", "--output", "new_figures"]
    defaults = parser.parse_args(required)
    assert defaults.font_family == "Times New Roman"
    assert (defaults.axis_size, defaults.tick_size, defaults.cell_size, defaults.annotation_size) == (54, 48, 32, 36)
    changed = parser.parse_args(required + ["--axis-size", "55", "--tick-size", "49",
                                            "--cell-size", "31", "--annotation-size", "37",
                                            "--font-family", "DejaVu Serif", "--strict-font"])
    assert (changed.axis_size, changed.tick_size, changed.cell_size, changed.annotation_size) == (55, 49, 31, 37)
    assert changed.strict_font


@pytest.mark.parametrize("option,value", (("--dpi", "0"), ("--axis-size", "nan"),
                                          ("--tick-size", "inf"), ("--cell-size", "-1"),
                                          ("--annotation-size", "0")))
def test_font_size_and_dpi_must_be_finite_positive(option, value):
    with pytest.raises(SystemExit):
        plots.build_parser().parse_args(["--input", "metrics", "--output", "new_figures", option, value])


def test_font_fallback_is_explicit_and_strict_mode_rejects_it(monkeypatch):
    monkeypatch.setattr(plots.font_manager.fontManager, "ttflist", [SimpleNamespace(name="DejaVu Serif")])
    with pytest.warns(UserWarning, match="unavailable"):
        assert plots.resolve_font("Times New Roman") == "DejaVu Serif"
    with pytest.raises(ValueError, match="unavailable"):
        plots.resolve_font("Times New Roman", strict=True)
    assert plots.resolve_font("DejaVu Serif", strict=True) == "DejaVu Serif"


def test_titles_distinguish_perceptual_metric_from_native_pixel_metrics():
    assert "grayscale" in plots.matrix_title("MSE", "mnist")
    assert "RGB" in plots.matrix_title("RMSE", "stl10")
    assert "v2" in plots.matrix_title("SSIM", "mnist")
    title = plots.matrix_title("LPIPS", "mnist")
    assert "perceptual" in title and "native" not in title and "[0,1]" not in title
