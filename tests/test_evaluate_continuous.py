from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from lightwave_seeded_vqvae.cli import evaluate_continuous
from lightwave_seeded_vqvae.cli.evaluate_continuous import (
    EVALUATION_SCHEMA,
    build_parser,
    evaluation_scope,
    load_continuous_checkpoint,
    lpips_enabled,
    main,
)
from lightwave_seeded_vqvae.models.vqvae import ContinuousAutoencoder


def continuous_config(tmp_path: Path, *, quantizer_type: str = "continuous") -> dict:
    return {
        "_config_path": str(tmp_path / "source.yaml"),
        "experiment": {"name": "continuous_test", "seed": 20260907},
        "dataset": {
            "name": "cifar10",
            "root": str(tmp_path / "data"),
            "download": False,
            "validation_size": 5000,
        },
        "model": {
            "quantizer_type": quantizer_type,
            "latent_dim": 4,
            "pretrained_encoder": False,
            "imagenet_weights": "IMAGENET1K_V1",
            "decoder_blocks_per_stage": 1,
        },
        "training": {"batch_size": 2, "num_workers": 0},
    }


def write_checkpoint(
    tmp_path: Path, *, config: dict | None = None
) -> tuple[Path, ContinuousAutoencoder, dict]:
    resolved = continuous_config(tmp_path) if config is None else config
    model = ContinuousAutoencoder(
        in_channels=3,
        latent_dim=int(resolved["model"]["latent_dim"]),
        pretrained_encoder=False,
        imagenet_weights="IMAGENET1K_V1",
        decoder_blocks_per_stage=int(
            resolved["model"].get("decoder_blocks_per_stage", 1)
        ),
        decoder_blocks_by_stage=resolved["model"].get("decoder_blocks_by_stage"),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    config_path = run_dir / "config.resolved.json"
    config_path.write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    checkpoint_path = run_dir / "checkpoint_best.pt"
    torch.save(
        {
            "epoch": 3,
            "model": model.state_dict(),
            "config": resolved,
            "validation": {"mse": 0.01},
        },
        checkpoint_path,
    )
    return checkpoint_path, model, resolved


def test_parser_scope_and_lpips_policy() -> None:
    common = ["--checkpoint", "continuous.pt", "--output-dir", "results"]
    formal = build_parser().parse_args(common)
    smoke = build_parser().parse_args(common + ["--smoke-max-batches", "1"])
    assert EVALUATION_SCHEMA == "lightwave-seeded-vqvae-continuous-evaluation-v1"
    assert formal.lpips == "auto"
    assert evaluation_scope(formal.smoke_max_batches) == "final"
    assert evaluation_scope(smoke.smoke_max_batches) == "smoke"
    assert lpips_enabled("auto", "CIFAR-10")
    assert not lpips_enabled("off", "cifar10")
    assert lpips_enabled("on", "mnist")
    with pytest.raises(SystemExit):
        build_parser().parse_args(common + ["--smoke-max-batches", "0"])


def test_checkpoint_is_rebuilt_strictly_from_resolved_config(tmp_path: Path) -> None:
    checkpoint_path, original, resolved = write_checkpoint(tmp_path)

    loaded = load_continuous_checkpoint(checkpoint_path)

    assert loaded.config == resolved
    assert loaded.epoch == 3
    assert loaded.checkpoint_sha256
    assert loaded.resolved_config_file_sha256
    assert loaded.model.decoder.blocks_by_stage == (1, 1, 1)
    original_state = original.state_dict()
    loaded_state = loaded.model.state_dict()
    assert original_state.keys() == loaded_state.keys()
    for key in original_state:
        torch.testing.assert_close(original_state[key], loaded_state[key])


def test_checkpoint_rejects_noncontinuous_and_config_mismatch(tmp_path: Path) -> None:
    wrong_config = continuous_config(tmp_path, quantizer_type="trainable_residual")
    checkpoint_path, _, _ = write_checkpoint(tmp_path, config=wrong_config)
    with pytest.raises(ValueError, match="quantizer_type"):
        load_continuous_checkpoint(checkpoint_path)

    other_root = tmp_path / "mismatch"
    checkpoint_path, _, resolved = write_checkpoint(other_root)
    mismatched = json.loads(json.dumps(resolved))
    mismatched["experiment"]["seed"] += 1
    (checkpoint_path.parent / "config.resolved.json").write_text(
        json.dumps(mismatched), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not match"):
        load_continuous_checkpoint(checkpoint_path)


def test_smoke_evaluation_writes_complete_metrics_and_refuses_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path, _, _ = write_checkpoint(tmp_path)
    generator = torch.Generator().manual_seed(91)
    images = torch.rand(3, 3, 32, 32, generator=generator)
    labels = torch.tensor([7, 2, 9], dtype=torch.int64)
    dataset = TensorDataset(images, labels)
    monkeypatch.setattr(evaluate_continuous, "build_dataset", lambda *a, **k: dataset)
    output_dir = tmp_path / "evaluation"
    arguments = [
        "--checkpoint",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
        "--batch-size",
        "2",
        "--num-workers",
        "0",
        "--lpips",
        "off",
        "--smoke-max-batches",
        "1",
    ]

    main(arguments)

    summary = json.loads(
        (output_dir / "reconstruction_summary.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output_dir / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    with np.load(output_dir / "per_image_metrics.npz", allow_pickle=False) as metrics:
        assert set(metrics.files) == {
            "sample_indices",
            "labels",
            "mse",
            "rmse",
            "ssim",
        }
        np.testing.assert_array_equal(metrics["sample_indices"], [0, 1])
        np.testing.assert_array_equal(metrics["labels"], [7, 2])
        np.testing.assert_allclose(np.square(metrics["rmse"]), metrics["mse"])
    assert summary["scope"] == "smoke"
    assert not summary["formal"]
    assert summary["processed_test_images"] == 2
    assert summary["continuous"]["overall"]["observation_count"] == 2
    assert summary["continuous"]["overall"]["rmse"] ** 2 == pytest.approx(
        summary["continuous"]["overall"]["mse"]
    )
    assert "lpips_mean" not in summary["continuous"]["overall"]
    assert summary["metric_space"]["image_range"] == [0.0, 1.0]
    assert summary["model"]["latent_shape_dhw"] == [4, 8, 8]
    assert manifest["checkpoint"]["path"] == str(checkpoint_path.resolve())
    assert manifest["checkpoint"]["sha256"]
    assert manifest["checkpoint"]["resolved_config_file_sha256"]
    assert manifest["dataset"]["shuffle"] is False
    assert manifest["dataset"]["processed_sample_count"] == 2
    assert manifest["model"]["parameters"]["total"] > 0

    with pytest.raises(FileExistsError, match="already exists"):
        main(arguments)


def test_formal_evaluation_rejects_nonofficial_dataset_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path, _, _ = write_checkpoint(tmp_path)
    dataset = TensorDataset(torch.zeros(2, 3, 32, 32), torch.zeros(2, dtype=torch.long))
    monkeypatch.setattr(evaluate_continuous, "build_dataset", lambda *a, **k: dataset)
    output_dir = tmp_path / "formal"

    with pytest.raises(ValueError, match="exactly 10,000"):
        main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--output-dir",
                str(output_dir),
                "--device",
                "cpu",
                "--num-workers",
                "0",
                "--lpips",
                "off",
            ]
        )
    assert not output_dir.exists()
