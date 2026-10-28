"""Contracts for the four released final configurations, not historical trials."""
from __future__ import annotations

import json
from pathlib import Path
import re

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
FINAL_CONFIGS = {
    "mnist": ("mnist_stagec_d64_k128.yaml", "mnist_stagec_d64_k128_software.yaml"),
    "stl10": ("stl10_stagec_d64_k128.yaml", "stl10_stagec_d64_k128_software.yaml"),
}


def load(name):
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def test_only_final_configs_are_distributed():
    assert {p.name for p in CONFIG_ROOT.glob("*.yaml")} == {
        name for pair in FINAL_CONFIGS.values() for name in pair
    }


@pytest.mark.parametrize("dataset", FINAL_CONFIGS)
@pytest.mark.parametrize("variant", [0, 1], ids=["chip", "software"])
def test_final_model_capacity_and_training_contract(dataset, variant):
    config = load(FINAL_CONFIGS[dataset][variant])
    model, training = config["model"], config["training"]
    assert config["experiment"]["seed"] == 20260907
    assert config["dataset"]["name"] == dataset
    assert not Path(config["dataset"]["root"]).is_absolute()
    assert model["latent_dim"] == model["loss_reference_dim"] == 64
    assert model["base_codebook_size"] == model["residual_codebook_size"] == 128
    assert model["decoder_blocks_by_stage"] == [3, 2, 2]
    assert model["pretrained_encoder"] is True
    assert model["imagenet_weights"] == "IMAGENET1K_V1"
    assert model["normalize_latents"] is True
    assert training["epochs"] == 80
    assert training["reconstruction_loss"] == "mse"
    assert training["perceptual_weight"] == 0.0
    assert training["wrong_state_weight"] == 0.0
    assert training["residual_only_weight"] == 0.0
    assert training["commitment_weight"] == 0.25
    assert training["codebook_weight"] == 1.0
    assert training["residual_norm_weight"] == pytest.approx(1e-4)
    assert training["learning_rate"] == pytest.approx(3e-4)
    assert training["encoder_learning_rate_scale"] == 0.1
    if variant == 0:
        assert model["quantizer_type"] == "seeded_residual"
        assert training["state_assignment"] == "balanced_cyclic"
        assert config["codebooks"]["path"] == "artifacts/codebooks/lightwave_bank_d64_k128.npz"
    else:
        assert model["quantizer_type"] == "trainable_residual"
        assert training["state_assignment"] == "single_state"
        assert "codebooks" not in config


@pytest.mark.parametrize("dataset", FINAL_CONFIGS)
def test_software_reference_matches_chip_recipe(dataset):
    chip, software = [load(name) for name in FINAL_CONFIGS[dataset]]
    assert chip["dataset"] == software["dataset"]
    chip_model, software_model = dict(chip["model"]), dict(software["model"])
    chip_model.pop("quantizer_type")
    software_model.pop("quantizer_type")
    assert chip_model == software_model
    chip_training, software_training = dict(chip["training"]), dict(software["training"])
    for training in (chip_training, software_training):
        training.pop("state_assignment")
        # Offset is monitored only in the ten-state chip model; not an objective.
        training.pop("wrong_state_offset", None)
    assert chip_training == software_training


@pytest.mark.parametrize("variant", [0, 1])
def test_native_geometry_initialization_and_recorded_batch_topology(variant):
    mnist = load(FINAL_CONFIGS["mnist"][variant])
    stl = load(FINAL_CONFIGS["stl10"][variant])
    assert mnist["dataset"]["native_resolution"] == [28, 28]
    assert mnist["model"]["decoder_zero_init_extra_residual_blocks"] is True
    assert mnist["training"]["batch_size"] == 256
    assert mnist["training"]["checkpoint_selection"] == "correct_validation_mse"
    assert stl["dataset"]["native_resolution"] == [96, 96]
    assert [stl["dataset"][f"{split}_split"] for split in ("train", "validation", "test")] == ["unlabeled", "train", "test"]
    assert stl["dataset"]["validation_size"] == 5000
    assert stl["dataset"]["train_horizontal_flip_probability"] == 0.5
    assert stl["model"].get("decoder_zero_init_extra_residual_blocks", False) is False
    assert stl["training"]["batch_size"] == 64
    assert stl["training"]["distributed_batch_size_mode"] == "per_rank"
    assert stl["training"]["effective_global_batch_size_4gpu"] == 256
    assert stl["training"]["checkpoint_selection"] == "correct_validation_mse"


def test_k128_manifest_contract_before_bank_generation():
    manifest = json.loads((PROJECT_ROOT / "data/optical/lightwave_bank_d64_k128.manifest.json").read_text(encoding="utf-8"))
    assert manifest["shape"] == [10, 128, 64]
    assert manifest["dtype"] == "float32-little-endian"
    assert manifest["bank_array_sha256"] == "89f7c290012917955f8a911f6df85fdca60c7b7e354061a78020e4e8e5cad199"
    states = manifest["states"]
    assert [state["index"] for state in states] == list(range(10))
    assert [(state["pump_nm"], state["power_mw"]) for state in states] == [
        (pump, power) for pump in (405, 640) for power in (0, 5, 10, 15, 20)
    ]
    for state in states:
        for field in ("codebook_sha256", "seed_sha256", "source_sha256"):
            assert re.fullmatch(r"[0-9a-f]{64}", state[field])
    assert len({state["seed_sha256"] for state in states}) == 10
    assert len({state["codebook_sha256"] for state in states}) == 10
