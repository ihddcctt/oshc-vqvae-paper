from lightwave_seeded_vqvae.cli.train import _binding_collapse_reasons


def test_collapse_guard_is_inactive_before_a_usable_checkpoint() -> None:
    metrics = {"mse": 1.0, "psnr_db": 0.0, "reconstruction_output_std": 0.0}

    assert _binding_collapse_reasons(
        metrics,
        previous_validation_psnr=30.0,
        best_binding_mse=0.001,
        has_binding_checkpoint=False,
    ) == []


def test_collapse_guard_reports_all_catastrophic_signals() -> None:
    metrics = {"mse": 0.03, "psnr_db": 10.0, "reconstruction_output_std": 0.0}

    assert _binding_collapse_reasons(
        metrics,
        previous_validation_psnr=24.0,
        best_binding_mse=0.003,
        has_binding_checkpoint=True,
    ) == [
        "validation_mse_exceeded_guard",
        "validation_psnr_dropped_over_3db",
        "reconstruction_output_collapsed",
    ]


def test_collapse_guard_allows_normal_validation_variation() -> None:
    metrics = {
        "mse": 0.0035,
        "psnr_db": 24.2,
        "reconstruction_output_std": 0.2,
    }

    assert _binding_collapse_reasons(
        metrics,
        previous_validation_psnr=24.5,
        best_binding_mse=0.003,
        has_binding_checkpoint=True,
    ) == []
