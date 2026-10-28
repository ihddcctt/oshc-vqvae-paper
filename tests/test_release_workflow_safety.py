"""Exercise release launchers without downloads, training, or study weights."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from lightwave_seeded_vqvae.seed.codebook import build_codebook_bank
from lightwave_seeded_vqvae.seed.spectrum import load_optical_states

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workflow():
    spec = importlib.util.spec_from_file_location(
        "release_reproduce_for_tests", ROOT / "scripts" / "reproduce.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke(workflow, monkeypatch, tmp_path, dataset, stage, *extra):
    monkeypatch.setattr(sys, "argv", [
        "reproduce.py", "--dataset", dataset, "--stage", stage,
        "--data-root", str(tmp_path / "datasets"),
        "--output-root", str(tmp_path / "run"), *extra,
    ])
    workflow.main()


@pytest.mark.parametrize("dataset,stage", [
    (dataset, stage)
    for dataset in ("mnist", "stl10")
    for stage in ("prepare", "train", "evaluate", "figures")
])
def test_dry_run_never_launches_commands_or_creates_outputs(
    workflow, monkeypatch, tmp_path, dataset, stage
):
    def reject(*args, **kwargs):
        raise AssertionError("dry-run launched a subprocess")
    monkeypatch.setattr(workflow.subprocess, "run", reject)
    invoke(workflow, monkeypatch, tmp_path, dataset, stage, "--dry-run")
    assert list(tmp_path.iterdir()) == []


def test_stl_reference_training_refuses_silent_single_gpu_recipe(
    workflow, monkeypatch, tmp_path
):
    import torch
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    commands = []
    monkeypatch.setattr(workflow, "run", lambda *args: commands.append(args))
    with pytest.raises(RuntimeError, match="4 GPUs"):
        invoke(workflow, monkeypatch, tmp_path, "stl10", "train")
    assert commands == []


def test_evaluation_launcher_explicitly_enables_lpips_and_relocated_data(
    workflow, monkeypatch, tmp_path
):
    commands = []
    monkeypatch.setattr(workflow, "run", lambda command, dry_run: commands.append(command))
    invoke(workflow, monkeypatch, tmp_path, "mnist", "evaluate")
    command, = commands
    assert command[command.index("--lpips") + 1] == "on"
    assert command[command.index("--dataset-root") + 1] == (tmp_path / "datasets").resolve()
    assert command[command.index("--output-dir") + 1] == (tmp_path / "run/evaluation").resolve()
    assert "--smoke-max-batches" not in command


def test_prepare_rejects_a_wrong_frozen_bank_before_writing_configs(
    workflow, monkeypatch, tmp_path
):
    repository = tmp_path / "repo"
    bank = repository / "artifacts/codebooks/lightwave_bank_d64_k128.npz"
    bank.parent.mkdir(parents=True)
    np.savez_compressed(bank, codebooks=np.zeros((10, 128, 64), dtype=np.float32))
    monkeypatch.setattr(workflow, "ROOT", repository)
    with pytest.raises(RuntimeError, match="canonical-array hash"):
        invoke(workflow, monkeypatch, tmp_path, "mnist", "prepare")
    assert not (tmp_path / "run").exists()


def test_prepare_preserves_an_existing_different_resolved_configuration(
    workflow, monkeypatch, tmp_path
):
    repository = tmp_path / "repo"
    bank_path = repository / "artifacts/codebooks/lightwave_bank_d64_k128.npz"
    bank_path.parent.mkdir(parents=True)
    optical = ROOT / "data/optical"
    states = load_optical_states(
        optical / "20260907_405nm.xlsx", optical / "20260907_640nm.xlsx"
    )
    bank, _ = build_codebook_bank(states, codebook_size=128, latent_dim=64)
    np.savez_compressed(bank_path, codebooks=bank)
    (repository / "configs").mkdir()
    for name in workflow.CONFIGS["mnist"]:
        (repository / "configs" / name).write_bytes((ROOT / "configs" / name).read_bytes())
    output = tmp_path / "run"
    output.mkdir()
    existing = output / "chip.yaml"
    original = b"user-owned existing configuration\n"
    existing.write_bytes(original)
    monkeypatch.setattr(workflow, "ROOT", repository)
    with pytest.raises(FileExistsError, match="Refusing to change"):
        invoke(workflow, monkeypatch, tmp_path, "mnist", "prepare")
    assert existing.read_bytes() == original
    assert not (output / "software.yaml").exists()


@pytest.mark.parametrize("dataset", ["mnist", "stl10"])
def test_explicit_preserved_registry_is_read_only_and_never_reselected(workflow, monkeypatch, tmp_path, dataset):
    registry = tmp_path / "original-registry.json"
    original = b'{"original": "authoritative bytes"}\n'
    registry.write_bytes(original)
    commands = []
    monkeypatch.setattr(workflow, "run", lambda command, dry_run: commands.append(command))
    invoke(workflow, monkeypatch, tmp_path, dataset, "figures",
           "--registry", str(registry), "--preserve-registered-samples")
    command, = commands
    assert f"lightwave_seeded_vqvae.cli.plot_{dataset}_stagec" in command
    assert "--preserve-registered-samples" in command
    assert command[command.index("--registry") + 1] == registry.resolve()
    assert registry.read_bytes() == original
    assert not (tmp_path / "run/sample_registry.json").exists()


def test_existing_registry_can_be_reused_with_strict_default_validation(workflow, monkeypatch, tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text("{}", encoding="utf-8")
    commands = []
    monkeypatch.setattr(workflow, "run", lambda command, dry_run: commands.append(command))
    invoke(workflow, monkeypatch, tmp_path, "mnist", "figures", "--registry", str(registry))
    command, = commands
    assert "--preserve-registered-samples" not in command
    assert "lightwave_seeded_vqvae.cli.plot_mnist_stagec" in command


def test_default_figures_selects_new_training_registry(workflow, monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(workflow, "run", lambda command, dry_run: commands.append(command))
    invoke(workflow, monkeypatch, tmp_path, "mnist", "figures")
    assert len(commands) == 2
    assert "lightwave_seeded_vqvae.cli.build_mnist_stagec_sample_registry" in commands[0]
    assert "lightwave_seeded_vqvae.cli.plot_mnist_stagec" in commands[1]
    assert all("--preserve-registered-samples" not in command for command in commands)



@pytest.mark.parametrize("dataset,stage,options", [
    ("mnist", "figures", ["--preserve-registered-samples"]),
    ("mnist", "evaluate", ["--registry", "registry.json"]),
    ("stl10", "train", ["--registry", "registry.json", "--preserve-registered-samples"]),
    ("cifar10", "figures", ["--registry", "registry.json"]),
    ("cifar10", "figures", ["--registry", "registry.json", "--preserve-registered-samples"]),
])
def test_unsupported_registry_option_combinations_fail_before_execution(workflow, monkeypatch, tmp_path, dataset, stage, options):
    def reject(*args):
        raise AssertionError("unsupported combination reached a subprocess")
    monkeypatch.setattr(workflow, "run", reject)
    with pytest.raises(SystemExit) as error:
        invoke(workflow, monkeypatch, tmp_path, dataset, stage, *options)
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("occupied", ["figures", "sample_registry.json"])
def test_figures_preflight_never_overwrites_existing_outputs(workflow, monkeypatch, tmp_path, occupied):
    output = tmp_path / "run"
    output.mkdir()
    target = output / occupied
    if occupied == "figures":
        target.mkdir()
        sentinel = target / "original.txt"
    else:
        sentinel = target
    sentinel.write_bytes(b"preserve existing results")
    def reject(*args):
        raise AssertionError("occupied output reached a subprocess")
    monkeypatch.setattr(workflow, "run", reject)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        invoke(workflow, monkeypatch, tmp_path, "mnist", "figures")
    assert sentinel.read_bytes() == b"preserve existing results"
