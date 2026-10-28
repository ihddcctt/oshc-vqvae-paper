from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from lightwave_seeded_vqvae.cli.plot_cifar_natural_examples import (
    extract_component_diagnostics,
    nchw_to_rgb,
    plot_component_diagnostic,
    plot_controlled_gallery,
    plot_controlled_gallery_page,
    plot_cross_state_heatmap,
    rgb_absolute_error,
    run_controlled_inference,
    select_median_per_class,
    select_representative_wrong_pair,
    validate_controlled_state_mapping,
)


def _state_records() -> list[dict[str, int | str]]:
    records: list[dict[str, int | str]] = []
    for pump in (405, 640):
        for power in (0, 5, 10, 15, 20):
            records.append(
                {
                    "index": len(records),
                    "state_id": f"{pump}nm_{power}mW",
                    "pump_nm": pump,
                    "power_mw": power,
                }
            )
    return records


def test_class_median_selection_uses_correct_average_and_fixed_index_tie() -> None:
    targets = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    values = np.asarray([1.0, 2.0, 3.0, 4.0, 40.0, 30.0, 20.0, 10.0])
    correct_mse = np.repeat(values[:, None], 10, axis=1)
    # Class medians are 2.5 and 25.0.  In each exact tie the lower official
    # dataset index must win.
    assert select_median_per_class(targets, correct_mse, class_count=2) == [1, 5]


def test_class_median_selection_rejects_missing_class() -> None:
    with pytest.raises(ValueError, match="no sample for class 1"):
        select_median_per_class([0, 0], np.asarray([0.1, 0.2]), class_count=2)


def test_registered_state_mapping_has_required_physical_relationships() -> None:
    validate_controlled_state_mapping(
        _state_records(),
        send_state=2,
        cross_pump_receiver_state=7,
        same_pump_receiver_state=4,
    )
    with pytest.raises(ValueError, match="different pump wavelength"):
        validate_controlled_state_mapping(
            _state_records(),
            send_state=2,
            cross_pump_receiver_state=4,
            same_pump_receiver_state=3,
        )
    with pytest.raises(ValueError, match="differ from the sender"):
        validate_controlled_state_mapping(
            _state_records(),
            send_state=2,
            cross_pump_receiver_state=7,
            same_pump_receiver_state=2,
        )


def test_representative_pair_is_fixed_sender_row_median_and_excludes_registered() -> None:
    matrix = np.full((10, 10), 2.0, dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)
    receivers = [0, 1, 3, 4, 5, 6, 7, 8, 9]
    matrix[2, receivers] = np.arange(1, 10, dtype=np.float64) * 10.0
    result = select_representative_wrong_pair(
        matrix,
        send_state=2,
        excluded_pairs={(2, 4), (2, 7)},
    )
    assert result["send_state"] == 2
    assert result["receiver_state"] == 5
    assert result["row_wrong_median_pooled_mse"] == 50.0
    assert result["row_wrong_pair_count"] == 9
    assert result["eligible_pair_count"] == 7


def test_rgb_conversion_and_error_use_fixed_zero_one_scale() -> None:
    nchw = np.asarray(
        [[[[1.2, -0.1]], [[0.5, 0.25]], [[0.0, 0.75]]]], dtype=np.float32
    )
    rgb = nchw_to_rgb(nchw)
    assert rgb.shape == (1, 1, 2, 3)
    np.testing.assert_allclose(
        rgb,
        np.asarray([[[[1.0, 0.5, 0.0], [0.0, 0.25, 0.75]]]], dtype=np.float32),
    )
    error = rgb_absolute_error(np.zeros_like(rgb), rgb)
    np.testing.assert_allclose(error, rgb.mean(axis=-1))


class _FakeQuantizer:
    def __init__(self, owner: "_FakeModel") -> None:
        self.owner = owner
        self.calls = 0

    def __call__(self, latent: torch.Tensor, state_ids: torch.Tensor) -> SimpleNamespace:
        self.calls += 1
        self.owner.latent = latent
        self.owner.quantized_send_states = state_ids.detach().cpu().tolist()
        shape = (latent.shape[0], 2, 2)
        return SimpleNamespace(
            base_indices=torch.arange(np.prod(shape), device=latent.device).reshape(shape),
            residual_indices=torch.arange(np.prod(shape), device=latent.device).reshape(shape) + 100,
        )


class _FakeModel:
    def __init__(self) -> None:
        self.quantizer = _FakeQuantizer(self)
        self.decode_calls: list[tuple[int, int, list[int]]] = []
        self.latent = torch.empty(0)
        self.quantized_send_states: list[int] = []

    def encoder(self, images: torch.Tensor) -> torch.Tensor:
        return images

    def decode_indices(
        self,
        base_indices: torch.Tensor,
        residual_indices: torch.Tensor,
        state_ids: torch.Tensor,
    ) -> torch.Tensor:
        self.decode_calls.append(
            (base_indices.data_ptr(), residual_indices.data_ptr(), state_ids.cpu().tolist())
        )
        level = float(state_ids[0].item()) / 10.0
        return torch.full_like(self.latent, level)


def test_controlled_inference_quantizes_once_and_only_switches_receiver_state() -> None:
    model = _FakeModel()
    arrays, provenance = run_controlled_inference(
        model,
        torch.zeros((3, 3, 4, 4), dtype=torch.float32),
        send_state=2,
        cross_pump_receiver_state=7,
        same_pump_receiver_state=4,
        representative_receiver_state=6,
        device=torch.device("cpu"),
    )
    assert model.quantizer.calls == 1
    assert model.quantized_send_states == [2, 2, 2]
    assert [call[2][0] for call in model.decode_calls] == [2, 7, 4, 6]
    assert len({call[0] for call in model.decode_calls}) == 1
    assert len({call[1] for call in model.decode_calls}) == 1
    assert provenance["sender_quantizer_calls"] == 1
    assert arrays["original"].shape == (3, 4, 4, 3)
    np.testing.assert_allclose(arrays["cross_pump"], 0.7)


def _synthetic_summary() -> dict[str, object]:
    usage = [
        {
            "base": {"coverage": 0.7 + state * 0.01},
            "residual": {"coverage": 0.3 + state * 0.01},
        }
        for state in range(10)
    ]
    return {
        "software": {"overall": {"mse": 0.02}},
        "seeded": {
            "cross_state": {
                "correct_pooled": {"mse": 0.01},
                "wrong_pooled": {"mse": 0.08},
            },
            "branches": {
                "base_only": {"pooled": {"mse": 0.05}},
                "transmitted_residual_only": {"pooled": {"mse": 0.06}},
                "requantized_residual_only": {"mse": 0.07},
            },
            "usage_by_send_state": usage,
            "pooled_residual_usage": {"coverage": 0.55},
        },
    }


def test_lightweight_synthetic_plots_write_png_and_pdf(tmp_path) -> None:
    records = _state_records()
    labels = [str(record["state_id"]) for record in records]
    matrix = np.full((10, 10), 0.04, dtype=np.float64)
    np.fill_diagonal(matrix, 0.01)
    original = np.linspace(0.0, 1.0, 10 * 4 * 4 * 3, dtype=np.float32).reshape(
        10, 4, 4, 3
    )
    arrays = {
        "original": original,
        "correct": np.clip(original * 0.95, 0.0, 1.0),
        "cross_pump": np.clip(original * 0.70, 0.0, 1.0),
        "same_pump": np.clip(original * 0.60, 0.0, 1.0),
        "representative": np.clip(original * 0.50, 0.0, 1.0),
    }
    heatmap = plot_cross_state_heatmap(
        matrix,
        state_labels=labels,
        ssim_matrix=np.full((10, 10), 0.75),
        output_dir=tmp_path,
        dpi=40,
    )
    gallery = plot_controlled_gallery(
        arrays,
        sample_ids=list(range(10)),
        class_ids=list(range(10)),
        records=records,
        mappings={
            "correct": (2, 2),
            "cross_pump": (2, 7),
            "same_pump": (2, 4),
            "representative": (2, 6),
        },
        pair_ratios={
            "correct": 1.0,
            "cross_pump": 4.0,
            "same_pump": 4.0,
            "representative": 4.0,
        },
        output_dir=tmp_path,
        dpi=40,
    )
    page_0_4 = plot_controlled_gallery_page(
        arrays,
        sample_ids=list(range(10)),
        class_ids=list(range(10)),
        class_start=0,
        records=records,
        mappings={
            "correct": (2, 2),
            "cross_pump": (2, 7),
            "same_pump": (2, 4),
            "representative": (2, 6),
        },
        pair_ratios={
            "correct": 1.0,
            "cross_pump": 6.84,
            "same_pump": 4.0,
            "representative": 5.0,
        },
        output_dir=tmp_path,
        dpi=40,
    )
    page_5_9 = plot_controlled_gallery_page(
        arrays,
        sample_ids=list(range(10)),
        class_ids=list(range(10)),
        class_start=5,
        records=records,
        mappings={
            "correct": (2, 2),
            "cross_pump": (2, 7),
            "same_pump": (2, 4),
            "representative": (2, 6),
        },
        pair_ratios={
            "correct": 1.0,
            "cross_pump": 6.84,
            "same_pump": 4.0,
            "representative": 5.0,
        },
        output_dir=tmp_path,
        dpi=40,
    )
    diagnostic = extract_component_diagnostics(_synthetic_summary())
    assert diagnostic is not None
    components = plot_component_diagnostic(
        diagnostic,
        state_labels=labels,
        output_dir=tmp_path,
        dpi=40,
    )
    for result in (heatmap, gallery, page_0_4, page_5_9, components):
        for file_type in ("png", "pdf"):
            path = tmp_path / result[file_type]["path"].split(str(tmp_path))[-1].lstrip("/\\")
            assert path.is_file()
            assert result[file_type]["sha256"]
    assert gallery["class_ids"] == list(range(10))
    assert gallery["absolute_error_scale"]["display_range"] == [0.0, 1.0]
    assert gallery["absolute_error_scale"]["shared_across_all_error_maps"] is True
    for result, expected_classes in (
        (page_0_4, list(range(5))),
        (page_5_9, list(range(5, 10))),
    ):
        assert result["class_ids"] == expected_classes
        assert result["layout"]["aspect_ratio"] == "16:9"
        assert result["layout"]["png_pixels"] == [533, 300]
        assert result["canvas_preserved"] is True
        assert "pooled MSE" in result["footnote"]
        assert "中位严重度" in result["footnote"]
        png_path = tmp_path / result["png"]["path"].split(str(tmp_path))[-1].lstrip("/\\")
        png = plt.imread(png_path)
        assert png.shape[:2] == (300, 533)
    plt.close("all")
