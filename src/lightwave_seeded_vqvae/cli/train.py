from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ..config import load_config
from ..data import build_dataloaders, dataset_channels
from ..models.vqvae import (
    ContinuousAutoencoder,
    SeededResidualVQVAE,
    TrainableResidualVQVAE,
)
from ..perceptual import load_lpips_alex_v01
from ..seed.codebook import load_codebook_bank
from ..training import (
    atomic_torch_save,
    balanced_state_ids,
    batch_objective,
    make_optimizer,
    mean_metrics,
    perplexity_from_indices,
    psnr_from_mse,
    scheduled_weight,
    set_global_seed,
    write_json,
)


_TORCHRUN_ENVIRONMENT_KEYS = ("RANK", "WORLD_SIZE", "LOCAL_RANK")


@dataclass
class DistributedContext:
    """Process-local execution details for single-device or torchrun training."""

    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str | None = None
    owns_process_group: bool = False
    rank_zero_phase_active: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _torchrun_environment() -> tuple[int, int, int] | None:
    """Return ``(rank, world_size, local_rank)`` when launched by torchrun.

    A partial environment is rejected instead of silently starting independent
    single-GPU jobs that could write to the same output directory.
    """

    present = {key: key in os.environ for key in _TORCHRUN_ENVIRONMENT_KEYS}
    if not any(present.values()):
        return None
    if not all(present.values()):
        missing = [key for key, is_present in present.items() if not is_present]
        raise RuntimeError(
            "incomplete torchrun environment; missing " + ", ".join(missing)
        )
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except ValueError as error:
        raise RuntimeError("torchrun rank variables must be integers") from error
    if world_size < 1:
        raise RuntimeError("WORLD_SIZE must be at least one")
    if not 0 <= rank < world_size:
        raise RuntimeError("RANK must be in [0, WORLD_SIZE)")
    if local_rank < 0:
        raise RuntimeError("LOCAL_RANK must be non-negative")
    return rank, world_size, local_rank


def _initialize_distributed(device_argument: str) -> DistributedContext:
    torchrun = _torchrun_environment()
    if torchrun is None:
        device = torch.device(device_argument)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        return DistributedContext(
            enabled=False,
            rank=0,
            world_size=1,
            local_rank=device.index or 0,
            device=device,
        )

    rank, world_size, local_rank = torchrun
    requested_device = torch.device(device_argument)
    if requested_device.type != "cuda":
        raise RuntimeError("torchrun mode requires a CUDA device and the NCCL backend")
    if not torch.cuda.is_available():
        raise RuntimeError("torchrun mode requires CUDA, but CUDA is unavailable")
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is unavailable in this PyTorch build")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} exceeds the {torch.cuda.device_count()} visible GPUs"
        )

    torch.cuda.set_device(local_rank)
    owns_process_group = False
    try:
        if torch.distributed.is_initialized():
            if torch.distributed.get_backend() != "nccl":
                raise RuntimeError("an existing torchrun process group must use NCCL")
            if torch.distributed.get_rank() != rank:
                raise RuntimeError("existing process-group rank disagrees with RANK")
            if torch.distributed.get_world_size() != world_size:
                raise RuntimeError(
                    "existing process-group world size disagrees with WORLD_SIZE"
                )
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(hours=1),
            )
            owns_process_group = True
    except BaseException:
        if owns_process_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        raise

    return DistributedContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=torch.device("cuda", local_rank),
        backend="nccl",
        owns_process_group=owns_process_group,
    )


def _shutdown_distributed(context: DistributedContext | None) -> None:
    if (
        context is not None
        and context.enabled
        and context.owns_process_group
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.destroy_process_group()


def _barrier(context: DistributedContext) -> None:
    if context.enabled:
        torch.distributed.barrier(device_ids=[context.local_rank])


def _broadcast_rank_zero_phase(
    success: bool,
    stop: bool,
    context: DistributedContext,
) -> tuple[bool, bool]:
    """Release worker ranks after rank-zero validation/checkpoint work."""

    if not context.enabled:
        return success, stop
    payload = torch.tensor(
        [int(success), int(stop)] if context.is_main else [0, 0],
        dtype=torch.int32,
        device=context.device,
    )
    torch.distributed.broadcast(payload, src=0)
    return bool(payload[0].item()), bool(payload[1].item())


def _distributed_batch_size(
    configured_batch_size: int,
    context: DistributedContext,
    mode: str,
) -> int:
    """Resolve a per-rank batch while preserving the configured global default."""

    if configured_batch_size < 1:
        raise ValueError("training.batch_size must be positive")
    if mode not in {"global", "per_rank"}:
        raise ValueError("training.distributed_batch_size_mode must be global or per_rank")
    if not context.enabled or mode == "per_rank":
        return configured_batch_size
    if configured_batch_size % context.world_size != 0:
        raise ValueError(
            "global training.batch_size must be divisible by WORLD_SIZE; "
            f"got {configured_batch_size} and {context.world_size}"
        )
    per_rank = configured_batch_size // context.world_size
    if per_rank < 1:
        raise ValueError("global training.batch_size is smaller than WORLD_SIZE")
    return per_rank


def _with_distributed_sampler(
    loader: DataLoader,
    context: DistributedContext,
    *,
    seed: int,
    per_rank_batch_size: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Clone a training loader with a deterministic rank-sharded sampler."""

    if not context.enabled:
        return loader, None
    sampler = DistributedSampler(
        loader.dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=True,
        seed=seed,
        drop_last=loader.drop_last,
    )
    kwargs: dict[str, Any] = {
        "dataset": loader.dataset,
        "batch_size": per_rank_batch_size,
        "sampler": sampler,
        "shuffle": False,
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn,
        "pin_memory": loader.pin_memory,
        "drop_last": loader.drop_last,
        "timeout": loader.timeout,
        "worker_init_fn": loader.worker_init_fn,
        "multiprocessing_context": loader.multiprocessing_context,
        "generator": torch.Generator().manual_seed(seed + context.rank),
        "persistent_workers": loader.persistent_workers,
    }
    if loader.num_workers > 0:
        kwargs["prefetch_factor"] = loader.prefetch_factor
    pin_memory_device = getattr(loader, "pin_memory_device", "")
    if pin_memory_device:
        kwargs["pin_memory_device"] = pin_memory_device
    return DataLoader(**kwargs), sampler


def _global_mean_metrics(
    rows: list[dict[str, float]], context: DistributedContext
) -> dict[str, float]:
    """Average batch metrics over every rank, weighted by rank batch count."""

    local = mean_metrics(rows)
    if not context.enabled:
        return local
    if not rows:
        raise RuntimeError("distributed training rank observed no batches")
    keys = sorted(local)
    if any(set(row) != set(keys) for row in rows):
        raise RuntimeError("training metric keys changed within an epoch")
    values = [sum(float(row[key]) for row in rows) for key in keys]
    values.append(float(len(rows)))
    payload = torch.tensor(values, dtype=torch.float64, device=context.device)
    torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.SUM)
    global_count = float(payload[-1].item())
    if global_count <= 0:
        raise RuntimeError("distributed training observed no batches globally")
    return {
        key: float(payload[index].item() / global_count)
        for index, key in enumerate(keys)
    }


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _source_local_indices(dataset: torch.utils.data.Dataset) -> np.ndarray:
    indices = getattr(dataset, "indices", range(len(dataset)))
    return np.asarray(indices, dtype="<i8")


def _select_checkpoint(
    checkpoint_selection: str,
    *,
    binding_evaluation_enabled: bool,
    binding_training_enabled: bool,
    has_binding_checkpoint: bool,
    has_post_ramp_checkpoint: bool,
) -> tuple[str, str]:
    if checkpoint_selection == "correct_validation_mse":
        return "checkpoint_best_reconstruction.pt", "best_correct_state_validation_mse"
    if binding_evaluation_enabled and has_binding_checkpoint:
        return (
            "checkpoint_best_binding.pt",
            "best_validation_reconstruction_among_binding_feasible_epochs",
        )
    if binding_evaluation_enabled and not binding_training_enabled:
        return (
            "checkpoint_best_reconstruction.pt",
            "fallback_best_validation_reconstruction_no_binding_feasible_epoch",
        )
    if binding_evaluation_enabled and has_post_ramp_checkpoint:
        return (
            "checkpoint_best_post_ramp.pt",
            "fallback_best_validation_reconstruction_after_full_binding_ramp",
        )
    return "checkpoint_best_reconstruction.pt", "best_validation_reconstruction"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the ten-state seeded residual VQ-VAE")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    return parser


def _git_metadata(project_root: Path) -> dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(project_root), "status", "--porcelain"], text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _resolve_project_path(config_path: Path, candidate: str) -> Path:
    path = Path(candidate).expanduser()
    if path.is_absolute():
        return path
    project_root = config_path.resolve().parent.parent
    return (project_root / path).resolve()


def _binding_collapse_reasons(
    validation_metrics: dict[str, Any],
    *,
    previous_validation_psnr: float | None,
    best_binding_mse: float,
    has_binding_checkpoint: bool,
) -> list[str]:
    """Detect a catastrophic decoder failure after a usable checkpoint exists."""
    if not has_binding_checkpoint:
        return []
    reasons: list[str] = []
    if float(validation_metrics["mse"]) > max(2.0 * best_binding_mse, 0.02):
        reasons.append("validation_mse_exceeded_guard")
    if (
        previous_validation_psnr is not None
        and previous_validation_psnr - float(validation_metrics["psnr_db"]) > 3.0
    ):
        reasons.append("validation_psnr_dropped_over_3db")
    if float(validation_metrics["reconstruction_output_std"]) < 1e-6:
        reasons.append("reconstruction_output_collapsed")
    return reasons


def _selection_collapse_reasons(
    validation_metrics: dict[str, Any],
    *,
    checkpoint_selection: str,
    binding_evaluation_enabled: bool,
    previous_validation_psnr: float | None,
    best_binding_mse: float,
    has_binding_checkpoint: bool,
) -> list[str]:
    """Keep mismatch diagnostics out of correct-MSE-only model selection."""

    if (
        checkpoint_selection != "historical_binding"
        or not binding_evaluation_enabled
    ):
        return []
    return _binding_collapse_reasons(
        validation_metrics,
        previous_validation_psnr=previous_validation_psnr,
        best_binding_mse=best_binding_mse,
        has_binding_checkpoint=has_binding_checkpoint,
    )


@torch.no_grad()
def evaluate(
    model: SeededResidualVQVAE | TrainableResidualVQVAE,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None,
    *,
    measure_binding: bool,
    wrong_state_offset: int,
) -> dict[str, Any]:
    model.eval()
    squared_error = 0.0
    element_count = 0
    base_indices_by_state: list[list[torch.Tensor]] = [
        [] for _ in range(model.quantizer.state_count)
    ]
    residual_indices: list[torch.Tensor] = []
    state_squared_error = np.zeros(model.quantizer.state_count, dtype=np.float64)
    state_element_count = np.zeros(model.quantizer.state_count, dtype=np.int64)
    wrong_squared_error = 0.0
    residual_only_squared_error = 0.0
    residual_only_element_count = 0
    reconstruction_sum = 0.0
    reconstruction_square_sum = 0.0
    reconstruction_element_count = 0
    for batch_index, (images, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        encoder_latent = model.encoder(images)
        for state in range(model.quantizer.state_count):
            state_ids = torch.full(
                (images.shape[0],), state, dtype=torch.long, device=device
            )
            quantizer_output = model.quantizer(encoder_latent, state_ids)
            reconstruction = model.decoder(quantizer_output.quantized)
            sample_squared_error = (reconstruction - images).square()
            error_sum = float(sample_squared_error.sum())
            reconstruction_sum += float(reconstruction.sum())
            reconstruction_square_sum += float(reconstruction.square().sum())
            reconstruction_element_count += reconstruction.numel()
            squared_error += error_sum
            element_count += images.numel()
            state_squared_error[state] += error_sum
            state_element_count[state] += images.numel()
            base_indices_by_state[state].append(quantizer_output.base_indices.cpu())
            residual_indices.append(quantizer_output.residual_indices.cpu())
            if measure_binding:
                wrong_states = (state_ids + wrong_state_offset) % model.quantizer.state_count
                wrong_reconstruction = model.decode_indices(
                    quantizer_output.base_indices,
                    quantizer_output.residual_indices,
                    wrong_states,
                )
                wrong_squared_error += float(
                    F.mse_loss(wrong_reconstruction, images, reduction="sum")
                )
        if measure_binding:
            residual_only_reconstruction, _ = model.reconstruct_residual_only(
                encoder_latent
            )
            residual_only_squared_error += float(
                F.mse_loss(residual_only_reconstruction, images, reduction="sum")
            )
            residual_only_element_count += images.numel()
    mse = squared_error / max(element_count, 1)
    if bool((state_element_count == 0).any()):
        raise RuntimeError("validation did not observe every quantizer state")
    residual = (
        torch.cat(residual_indices) if residual_indices else torch.empty(0, dtype=torch.long)
    )
    base_usage = [
        perplexity_from_indices(
            torch.cat(state_indices), model.quantizer.base_codebook_size
        )
        for state_indices in base_indices_by_state
    ]
    base_coverage_by_state = [value[0] for value in base_usage]
    base_perplexity_by_state = [value[1] for value in base_usage]
    base_coverage = float(np.mean(base_coverage_by_state))
    base_perplexity = float(np.mean(base_perplexity_by_state))
    residual_coverage, residual_perplexity = perplexity_from_indices(
        residual, model.quantizer.residual_codebook_size
    )
    state_mse = np.divide(
        state_squared_error,
        state_element_count,
        out=np.full_like(state_squared_error, np.nan),
        where=state_element_count > 0,
    )
    state_psnr = [psnr_from_mse(float(value)) for value in state_mse]
    metrics: dict[str, Any] = {
        "mse": mse,
        "psnr_db": psnr_from_mse(mse),
        "mse_by_state": state_mse.tolist(),
        "psnr_db_by_state": state_psnr,
        "psnr_db_state_spread": float(max(state_psnr) - min(state_psnr)),
        "base_coverage": base_coverage,
        "base_perplexity": base_perplexity,
        "base_coverage_by_state": base_coverage_by_state,
        "base_perplexity_by_state": base_perplexity_by_state,
        "base_coverage_min": float(min(base_coverage_by_state)),
        "base_perplexity_min": float(min(base_perplexity_by_state)),
        "residual_coverage": residual_coverage,
        "residual_perplexity": residual_perplexity,
        "reconstruction_output_mean": reconstruction_sum
        / max(reconstruction_element_count, 1),
        "reconstruction_output_std": math.sqrt(
            max(
                reconstruction_square_sum / max(reconstruction_element_count, 1)
                - (reconstruction_sum / max(reconstruction_element_count, 1)) ** 2,
                0.0,
            )
        ),
    }
    if measure_binding:
        wrong_mse = wrong_squared_error / max(element_count, 1)
        residual_only_mse = residual_only_squared_error / max(
            residual_only_element_count, 1
        )
        metrics.update(
            {
                "trained_offset_wrong_mse": wrong_mse,
                "trained_offset_wrong_psnr_db": psnr_from_mse(wrong_mse),
                "wrong_to_correct_mse_ratio": wrong_mse / max(mse, 1e-12),
                "requantized_residual_only_mse": residual_only_mse,
                "requantized_residual_only_psnr_db": psnr_from_mse(
                    residual_only_mse
                ),
            }
        )
    return metrics


@torch.no_grad()
def evaluate_continuous(
    model: ContinuousAutoencoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, Any]:
    """Validation metrics for the non-quantized reconstruction ceiling."""

    model.eval()
    squared_error = 0.0
    element_count = 0
    reconstruction_sum = 0.0
    reconstruction_square_sum = 0.0
    for batch_index, (images, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        reconstruction = model(images).reconstruction
        squared_error += float(F.mse_loss(reconstruction, images, reduction="sum"))
        element_count += images.numel()
        reconstruction_sum += float(reconstruction.sum())
        reconstruction_square_sum += float(reconstruction.square().sum())
    if element_count == 0:
        raise RuntimeError("validation did not observe any images")
    mse = squared_error / element_count
    reconstruction_mean = reconstruction_sum / element_count
    return {
        "mse": mse,
        "psnr_db": psnr_from_mse(mse),
        "reconstruction_output_mean": reconstruction_mean,
        "reconstruction_output_std": math.sqrt(
            max(reconstruction_square_sum / element_count - reconstruction_mean**2, 0.0)
        ),
    }


def _run_training(args: argparse.Namespace, distributed: DistributedContext) -> None:
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    experiment = config["experiment"]
    dataset_config = config["dataset"]
    model_config = config["model"]
    training_config = config["training"]
    quantizer_type = str(model_config.get("quantizer_type", "seeded_residual"))

    seed = int(experiment["seed"])
    set_global_seed(seed)
    device = distributed.device

    latent_dim = int(model_config["latent_dim"])
    codebook_path: Path | None = None
    bank: np.ndarray | None = None
    bank_array_sha256: str | None = None
    if quantizer_type == "seeded_residual":
        codebook_config = config["codebooks"]
        codebook_path = _resolve_project_path(config_path, str(codebook_config["path"]))
        bank = load_codebook_bank(codebook_path)
        bank_array_sha256 = hashlib.sha256(
            np.ascontiguousarray(bank, dtype="<f4").tobytes()
        ).hexdigest()
        if bank.shape != (10, int(model_config["base_codebook_size"]), latent_dim):
            raise ValueError(
                f"configured bank shape does not match artifact: expected "
                f"{(10, int(model_config['base_codebook_size']), latent_dim)}, got {bank.shape}"
            )
    elif quantizer_type in {"trainable_residual", "continuous"}:
        if float(training_config.get("wrong_state_weight", 0.0)) != 0.0:
            raise ValueError(f"{quantizer_type} cannot use wrong_state_weight")
        if float(training_config.get("residual_only_weight", 0.0)) != 0.0:
            raise ValueError(f"{quantizer_type} cannot use residual_only_weight")
        if quantizer_type == "continuous" and any(
            float(training_config.get(key, 0.0)) != 0.0
            for key in ("commitment_weight", "codebook_weight", "residual_norm_weight")
        ):
            raise ValueError("continuous autoencoder requires zero VQ loss weights")
    else:
        raise ValueError(f"unsupported quantizer_type: {quantizer_type}")
    expected_assignment = (
        "balanced_cyclic" if quantizer_type == "seeded_residual" else "single_state"
    )
    if str(training_config.get("state_assignment")) != expected_assignment:
        raise ValueError(
            f"{quantizer_type} requires state_assignment={expected_assignment}"
        )

    channels = dataset_channels(str(dataset_config["name"]))
    decoder_blocks_by_stage = model_config.get("decoder_blocks_by_stage")
    decoder_zero_init_extra_residual_blocks = model_config.get(
        "decoder_zero_init_extra_residual_blocks", False
    )
    if not isinstance(decoder_zero_init_extra_residual_blocks, bool):
        raise ValueError(
            "model.decoder_zero_init_extra_residual_blocks must be boolean"
        )
    shared_model_arguments = {
        "in_channels": channels,
        "latent_dim": latent_dim,
        "pretrained_encoder": bool(model_config["pretrained_encoder"]),
        "imagenet_weights": str(model_config["imagenet_weights"]),
        "decoder_blocks_per_stage": int(model_config.get("decoder_blocks_per_stage", 1)),
        "decoder_blocks_by_stage": decoder_blocks_by_stage,
        "decoder_zero_init_extra_residual_blocks": (
            decoder_zero_init_extra_residual_blocks
        ),
    }
    if quantizer_type == "seeded_residual":
        assert bank is not None
        model = SeededResidualVQVAE(
            bank,
            residual_codebook_size=int(model_config["residual_codebook_size"]),
            normalize_latents=bool(model_config.get("normalize_latents", True)),
            loss_reference_dim=int(model_config.get("loss_reference_dim", 64)),
            **shared_model_arguments,
        )
    elif quantizer_type == "trainable_residual":
        model = TrainableResidualVQVAE(
            base_codebook_size=int(model_config["base_codebook_size"]),
            residual_codebook_size=int(model_config["residual_codebook_size"]),
            normalize_latents=bool(model_config.get("normalize_latents", True)),
            loss_reference_dim=int(model_config.get("loss_reference_dim", 64)),
            **shared_model_arguments,
        )
    else:
        model = ContinuousAutoencoder(**shared_model_arguments)
    model = model.to(device)
    training_model: torch.nn.Module = (
        DistributedDataParallel(
            model,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
            broadcast_buffers=True,
        )
        if distributed.enabled
        else model
    )
    train_loader, validation_loader, test_loader = build_dataloaders(
        dataset_config, training_config, split_seed=seed
    )
    configured_batch_size = int(training_config["batch_size"])
    distributed_batch_size_mode = str(
        training_config.get("distributed_batch_size_mode", "per_rank")
    )
    per_rank_batch_size = _distributed_batch_size(
        configured_batch_size,
        distributed,
        distributed_batch_size_mode,
    )
    train_loader, train_sampler = _with_distributed_sampler(
        train_loader,
        distributed,
        seed=seed + 1,
        per_rank_batch_size=per_rank_batch_size,
    )
    train_indices = _source_local_indices(train_loader.dataset)
    validation_indices = _source_local_indices(validation_loader.dataset)
    test_indices = _source_local_indices(test_loader.dataset)
    optimizer = make_optimizer(
        model,
        learning_rate=float(training_config["learning_rate"]),
        encoder_learning_rate_scale=float(training_config["encoder_learning_rate_scale"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    amp_enabled = bool(training_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    reconstruction_loss_type = str(training_config.get("reconstruction_loss", "mse"))
    perceptual_weight = float(training_config.get("perceptual_weight", 0.0))
    checkpoint_selection = str(
        training_config.get("checkpoint_selection", "historical_binding")
    )
    if checkpoint_selection not in {"historical_binding", "correct_validation_mse"}:
        raise ValueError(
            "training.checkpoint_selection must be historical_binding or "
            "correct_validation_mse"
        )
    if reconstruction_loss_type not in {"mse", "l1_lpips"}:
        raise ValueError("training.reconstruction_loss must be 'mse' or 'l1_lpips'")
    perceptual_metric = (
        load_lpips_alex_v01(device=device)
        if reconstruction_loss_type == "l1_lpips"
        else None
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    project_root = config_path.parent.parent
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir: Path | None = None
    if distributed.is_main:
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else project_root / "outputs" / f"{experiment['name']}_{timestamp}"
        )
        output_dir.mkdir(parents=True, exist_ok=False)
    resolved_config = json.loads(json.dumps(config))
    if distributed.is_main:
        assert output_dir is not None
        write_json(resolved_config, output_dir / "config.resolved.json")
    parameter_counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "encoder_total": sum(parameter.numel() for parameter in model.encoder.parameters()),
        "decoder_total": sum(parameter.numel() for parameter in model.decoder.parameters()),
        "quantizer_total": (
            sum(parameter.numel() for parameter in model.quantizer.parameters())
            if hasattr(model, "quantizer")
            else 0
        ),
    }
    run_metadata: dict[str, Any] = {
        "created_utc": timestamp,
        "git": _git_metadata(project_root),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "quantizer_type": quantizer_type,
        "base_bank_shape": list(bank.shape) if bank is not None else None,
        "base_bank_array_sha256": bank_array_sha256,
        "parameter_counts": parameter_counts,
        "reconstruction_loss": {
            "type": reconstruction_loss_type,
            "perceptual_metric": (
                "LPIPS-Alex-v0.1" if perceptual_metric is not None else None
            ),
            "perceptual_weight": perceptual_weight,
        },
        "checkpoint_selection": checkpoint_selection,
        "decoder_initialization": {
            "zero_init_extra_residual_blocks": (
                decoder_zero_init_extra_residual_blocks
            ),
            "scope": "blocks_after_the_first_block_in_each_decoder_stage",
        },
        "distributed": {
            "enabled": distributed.enabled,
            "backend": distributed.backend,
            "rank": distributed.rank,
            "local_rank": distributed.local_rank,
            "world_size": distributed.world_size,
            "configured_batch_size": configured_batch_size,
            "batch_size_mode": distributed_batch_size_mode,
            "per_rank_batch_size": per_rank_batch_size,
            "effective_global_batch_size": per_rank_batch_size
            * distributed.world_size,
            "validation_and_io_rank": 0,
            "train_metric_reduction": "global_all_rank_batch_mean",
        },
        "data_split": {
            "split_seed": seed,
            "train_split": dataset_config.get("train_split"),
            "validation_split": dataset_config.get("validation_split"),
            "test_split": dataset_config.get("test_split"),
            "augmentation": {
                "train_horizontal_flip_probability": float(
                    dataset_config.get("train_horizontal_flip_probability", 0.0)
                ),
                "validation_random_augmentation": False,
                "test_random_augmentation": False,
            },
            "train_samples": len(train_loader.dataset),
            "validation_samples": len(validation_loader.dataset),
            "test_samples_held_out": len(test_loader.dataset),
            "train_indices_sha256": hashlib.sha256(train_indices.tobytes()).hexdigest(),
            "validation_indices_sha256": hashlib.sha256(
                validation_indices.tobytes()
            ).hexdigest(),
            "test_indices_sha256": hashlib.sha256(test_indices.tobytes()).hexdigest(),
            "index_hash_scope": "source-local ordered int64-little-endian indices",
        },
        "cli_overrides": {
            "epochs": args.epochs,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
        },
    }
    if distributed.is_main:
        assert output_dir is not None
        write_json(run_metadata, output_dir / "run_metadata.json")

    epochs = args.epochs if args.epochs is not None else int(training_config["epochs"])
    run_started = time.perf_counter()
    history: list[dict[str, Any]] = []
    best_reconstruction_mse = float("inf")
    best_post_ramp_mse = float("inf")
    best_binding_mse = float("inf")
    has_post_ramp_checkpoint = False
    has_binding_checkpoint = False
    binding_evaluation_enabled = quantizer_type == "seeded_residual"
    binding_training_enabled = binding_evaluation_enabled and (
        float(training_config.get("wrong_state_weight", 0.0)) > 0.0
        or float(training_config.get("residual_only_weight", 0.0)) > 0.0
    )
    if binding_training_enabled:
        binding_ready_epoch = max(
            int(training_config.get("wrong_state_start_epoch", 1))
            + max(int(training_config.get("wrong_state_ramp_epochs", 1)), 1)
            - 1,
            int(training_config.get("residual_only_start_epoch", 1))
            + max(int(training_config.get("residual_only_ramp_epochs", 1)), 1)
            - 1,
        )
    else:
        binding_ready_epoch = 1
    previous_validation_psnr: float | None = None
    early_stop_payload: dict[str, Any] | None = None
    state_offset = 0
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        training_model.train()
        batch_rows: list[dict[str, float]] = []
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{epochs}",
            leave=False,
            disable=not distributed.is_main,
        )
        for batch_index, (images, _) in enumerate(progress):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            images = images.to(device, non_blocking=True)
            if quantizer_type == "continuous":
                state_ids = None
            else:
                state_ids, state_offset = balanced_state_ids(
                    images.shape[0], model.quantizer.state_count, state_offset, device
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = (
                    training_model(images)
                    if state_ids is None
                    else training_model(images, state_ids)
                )
                wrong_weight = scheduled_weight(
                    epoch,
                    start_epoch=int(training_config.get("wrong_state_start_epoch", 1)),
                    ramp_epochs=int(training_config.get("wrong_state_ramp_epochs", 1)),
                    weight=float(training_config.get("wrong_state_weight", 0.0)),
                )
                residual_only_weight = scheduled_weight(
                    epoch,
                    start_epoch=int(training_config.get("residual_only_start_epoch", 1)),
                    ramp_epochs=int(training_config.get("residual_only_ramp_epochs", 1)),
                    weight=float(training_config.get("residual_only_weight", 0.0)),
                )
                wrong_mse = None
                if wrong_weight > 0.0:
                    wrong_offset = int(training_config.get("wrong_state_offset", 1))
                    if not 1 <= wrong_offset < model.quantizer.state_count:
                        raise ValueError("wrong_state_offset must be in [1, state_count-1]")
                    assert state_ids is not None
                    wrong_states = (state_ids + wrong_offset) % model.quantizer.state_count
                    wrong_reconstruction = model.decode_indices(
                        output.quantizer.base_indices,
                        output.quantizer.residual_indices,
                        wrong_states,
                    )
                    wrong_mse = F.mse_loss(wrong_reconstruction, images)
                residual_only_mse = None
                if residual_only_weight > 0.0:
                    residual_only_reconstruction, _ = model.reconstruct_residual_only(
                        output.encoder_latent
                    )
                    residual_only_mse = F.mse_loss(residual_only_reconstruction, images)
                if quantizer_type == "continuous":
                    zero = output.reconstruction.new_zeros(())
                    commitment_loss = codebook_loss = residual_norm_loss = zero
                else:
                    commitment_loss = output.quantizer.commitment_loss
                    codebook_loss = output.quantizer.codebook_loss
                    residual_norm_loss = output.quantizer.residual_norm_loss
            objective_autocast_enabled = amp_enabled and reconstruction_loss_type == "mse"
            if reconstruction_loss_type == "l1_lpips":
                objective_reconstruction = output.reconstruction.float()
                objective_images = images.float()
                commitment_loss = commitment_loss.float()
                codebook_loss = codebook_loss.float()
                residual_norm_loss = residual_norm_loss.float()
                wrong_mse = wrong_mse.float() if wrong_mse is not None else None
                residual_only_mse = (
                    residual_only_mse.float()
                    if residual_only_mse is not None
                    else None
                )
            else:
                objective_reconstruction = output.reconstruction
                objective_images = images
            with torch.cuda.amp.autocast(enabled=objective_autocast_enabled):
                total_loss, row = batch_objective(
                    objective_reconstruction,
                    objective_images,
                    commitment_loss=commitment_loss,
                    codebook_loss=codebook_loss,
                    residual_norm_loss=residual_norm_loss,
                    commitment_weight=float(training_config["commitment_weight"]),
                    codebook_weight=float(training_config["codebook_weight"]),
                    residual_norm_weight=float(training_config["residual_norm_weight"]),
                    wrong_state_mse=wrong_mse,
                    wrong_state_margin=float(training_config.get("wrong_state_margin", 0.02)),
                    wrong_state_weight=wrong_weight,
                    residual_only_mse=residual_only_mse,
                    residual_only_margin=float(
                        training_config.get("residual_only_margin", 0.02)
                    ),
                    residual_only_weight=residual_only_weight,
                    reconstruction_loss_type=reconstruction_loss_type,
                    perceptual_metric=perceptual_metric,
                    perceptual_weight=perceptual_weight,
                )
            if quantizer_type == "continuous":
                row["first_stage_error"] = 0.0
                row["final_quantization_error"] = 0.0
            else:
                row["first_stage_error"] = float(
                    output.quantizer.first_stage_error.detach()
                )
                row["final_quantization_error"] = float(
                    output.quantizer.final_error.detach()
                )
            scaler.scale(total_loss).backward()
            clip_norm = float(training_config.get("binding_gradient_clip_norm", 0.0))
            clip_active = (
                clip_norm > 0.0
                and (wrong_weight > 0.0 or residual_only_weight > 0.0)
            )
            row["binding_gradient_clip_active"] = float(clip_active)
            row["gradient_norm_pre_clip"] = 0.0
            row["decoder_gradient_norm_pre_clip"] = 0.0
            if clip_active:
                scaler.unscale_(optimizer)
                decoder_squared_norm = sum(
                    float(parameter.grad.detach().float().square().sum())
                    for parameter in model.decoder.parameters()
                    if parameter.grad is not None
                )
                row["decoder_gradient_norm_pre_clip"] = math.sqrt(
                    decoder_squared_norm
                )
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_norm
                )
                row["gradient_norm_pre_clip"] = float(gradient_norm.detach())
            scaler.step(optimizer)
            scaler.update()
            batch_rows.append(row)
            progress.set_postfix(loss=f"{row['loss']:.4f}", mse=f"{row['reconstruction_mse']:.4f}")

        train_metrics = _global_mean_metrics(batch_rows, distributed)
        _barrier(distributed)
        if not distributed.is_main:
            phase_succeeded, should_stop = _broadcast_rank_zero_phase(
                False, False, distributed
            )
            if not phase_succeeded:
                raise RuntimeError(
                    "rank zero failed during validation or checkpoint writing"
                )
            if should_stop:
                break
            continue

        distributed.rank_zero_phase_active = distributed.enabled
        assert output_dir is not None
        if quantizer_type == "continuous":
            validation_metrics = evaluate_continuous(
                model, validation_loader, device, args.max_val_batches
            )
        else:
            validation_metrics = evaluate(
                model,
                validation_loader,
                device,
                args.max_val_batches,
                measure_binding=binding_evaluation_enabled,
                wrong_state_offset=int(training_config.get("wrong_state_offset", 1)),
            )
        wrong_target_met = False
        residual_only_target_met = False
        state_spread_target_met = (
            validation_metrics["psnr_db_state_spread"] <= 1.5
            if binding_evaluation_enabled
            else True
        )
        binding_feasible = False
        if binding_evaluation_enabled:
            wrong_psnr_drop = (
                validation_metrics["psnr_db"]
                - validation_metrics["trained_offset_wrong_psnr_db"]
            )
            residual_only_psnr_drop = (
                validation_metrics["psnr_db"]
                - validation_metrics["requantized_residual_only_psnr_db"]
            )
            wrong_target_met = (
                wrong_psnr_drop >= 5.0
                or validation_metrics["wrong_to_correct_mse_ratio"] >= 2.0
            )
            residual_only_target_met = residual_only_psnr_drop >= 3.0
            binding_feasible = (
                epoch >= binding_ready_epoch
                and wrong_target_met
                and residual_only_target_met
                and state_spread_target_met
            )
        epoch_row: dict[str, Any] = {
            "epoch": epoch,
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "train": train_metrics,
            "validation": validation_metrics,
            "selection": {
                "binding_ready_epoch": (
                    binding_ready_epoch if binding_evaluation_enabled else None
                ),
                "binding_feasible": binding_feasible,
                "wrong_target_met": (
                    wrong_target_met if binding_evaluation_enabled else None
                ),
                "residual_only_target_met": (
                    residual_only_target_met if binding_evaluation_enabled else None
                ),
                "state_spread_target_met": state_spread_target_met,
            },
        }
        collapse_reasons = _selection_collapse_reasons(
            validation_metrics,
            checkpoint_selection=checkpoint_selection,
            binding_evaluation_enabled=binding_evaluation_enabled,
            previous_validation_psnr=previous_validation_psnr,
            best_binding_mse=best_binding_mse,
            has_binding_checkpoint=has_binding_checkpoint,
        )
        epoch_row["selection"]["collapse_detected"] = bool(collapse_reasons)
        history.append(epoch_row)
        write_json({"epochs": history}, output_dir / "history.json")
        checkpoint = {
            "epoch": epoch,
            "model": _unwrap_model(training_model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": resolved_config,
            "base_bank_path": str(codebook_path) if codebook_path is not None else None,
            "base_bank_array_sha256": bank_array_sha256,
            "validation": validation_metrics,
        }
        if quantizer_type == "seeded_residual":
            current_bank_sha256 = hashlib.sha256(
                np.ascontiguousarray(
                    model.quantizer.base_codebooks.detach().cpu().numpy(), dtype="<f4"
                ).tobytes()
            ).hexdigest()
            if current_bank_sha256 != bank_array_sha256:
                raise RuntimeError("frozen base bank changed before checkpoint save")
        atomic_torch_save(checkpoint, output_dir / "checkpoint_last.pt")
        if validation_metrics["mse"] < best_reconstruction_mse:
            best_reconstruction_mse = float(validation_metrics["mse"])
            atomic_torch_save(
                checkpoint, output_dir / "checkpoint_best_reconstruction.pt"
            )
            write_json(
                {
                    "epoch": epoch,
                    "validation_mse": best_reconstruction_mse,
                    "validation_psnr_db": validation_metrics["psnr_db"],
                },
                output_dir / "best_reconstruction_metric.json",
            )
        if binding_evaluation_enabled and epoch >= binding_ready_epoch:
            has_post_ramp_checkpoint = True
            if validation_metrics["mse"] < best_post_ramp_mse:
                best_post_ramp_mse = float(validation_metrics["mse"])
                atomic_torch_save(
                    checkpoint, output_dir / "checkpoint_best_post_ramp.pt"
                )
                write_json(
                    {
                        "epoch": epoch,
                        "validation_mse": best_post_ramp_mse,
                        "validation_psnr_db": validation_metrics["psnr_db"],
                        "binding_feasible": binding_feasible,
                    },
                    output_dir / "best_post_ramp_metric.json",
                )
        if binding_feasible and validation_metrics["mse"] < best_binding_mse:
            has_binding_checkpoint = True
            best_binding_mse = float(validation_metrics["mse"])
            atomic_torch_save(checkpoint, output_dir / "checkpoint_best_binding.pt")
            write_json(
                {
                    "epoch": epoch,
                    "validation_mse": best_binding_mse,
                    "validation_psnr_db": validation_metrics["psnr_db"],
                    "wrong_to_correct_mse_ratio": validation_metrics[
                        "wrong_to_correct_mse_ratio"
                    ],
                    "residual_only_psnr_drop": validation_metrics["psnr_db"]
                    - validation_metrics["requantized_residual_only_psnr_db"],
                    "psnr_db_state_spread": validation_metrics[
                        "psnr_db_state_spread"
                    ],
                },
                output_dir / "best_binding_metric.json",
            )
        print(json.dumps(epoch_row, ensure_ascii=False))
        previous_validation_psnr = float(validation_metrics["psnr_db"])
        should_stop = bool(collapse_reasons)
        if collapse_reasons:
            early_stop_payload = {
                "epoch": epoch,
                "reasons": collapse_reasons,
                "validation_mse": validation_metrics["mse"],
                "validation_psnr_db": validation_metrics["psnr_db"],
                "best_binding_mse_before_collapse": best_binding_mse,
            }
            write_json(early_stop_payload, output_dir / "early_stop.json")
        distributed.rank_zero_phase_active = False
        phase_succeeded, should_stop = _broadcast_rank_zero_phase(
            True, should_stop, distributed
        )
        assert phase_succeeded
        if should_stop:
            break

    if not distributed.is_main:
        phase_succeeded, _ = _broadcast_rank_zero_phase(False, False, distributed)
        if not phase_succeeded:
            raise RuntimeError("rank zero failed during final checkpoint writing")
        return

    distributed.rank_zero_phase_active = distributed.enabled
    assert output_dir is not None
    if quantizer_type == "seeded_residual":
        trained_bank_sha256 = hashlib.sha256(
            np.ascontiguousarray(
                model.quantizer.base_codebooks.detach().cpu().numpy(), dtype="<f4"
            ).tobytes()
        ).hexdigest()
        if trained_bank_sha256 != bank_array_sha256:
            raise RuntimeError("frozen base bank changed during training")
    selected_name, selection_reason = _select_checkpoint(
        checkpoint_selection,
        binding_evaluation_enabled=binding_evaluation_enabled,
        binding_training_enabled=binding_training_enabled,
        has_binding_checkpoint=has_binding_checkpoint,
        has_post_ramp_checkpoint=has_post_ramp_checkpoint,
    )
    selected_checkpoint = torch.load(output_dir / selected_name, map_location="cpu")
    atomic_torch_save(selected_checkpoint, output_dir / "checkpoint_best.pt")
    write_json(
        {
            "selected_checkpoint": selected_name,
            "selection_reason": selection_reason,
            "checkpoint_selection": checkpoint_selection,
            "binding_evaluation_enabled": binding_evaluation_enabled,
            "binding_training_enabled": binding_training_enabled,
            "binding_ready_epoch": (
                binding_ready_epoch if binding_evaluation_enabled else None
            ),
            "binding_target_satisfied": (
                has_binding_checkpoint if binding_evaluation_enabled else None
            ),
            "completed_epochs": len(history),
            "early_stop": early_stop_payload,
        },
        output_dir / "selected_checkpoint.json",
    )
    completed_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runtime_payload = {
        "started_utc": timestamp,
        "completed_utc": completed_utc,
        "elapsed_seconds": time.perf_counter() - run_started,
        "completed_epochs": len(history),
        "seconds_by_epoch": [row["elapsed_seconds"] for row in history],
        "parameter_counts": parameter_counts,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }
    write_json(runtime_payload, output_dir / "training_runtime.json")
    run_metadata["completed_utc"] = completed_utc
    run_metadata["runtime_file"] = "training_runtime.json"
    write_json(run_metadata, output_dir / "run_metadata.json")
    print(f"run directory: {output_dir}")
    distributed.rank_zero_phase_active = False
    phase_succeeded, _ = _broadcast_rank_zero_phase(True, False, distributed)
    assert phase_succeeded


def main(arguments: list[str] | None = None) -> None:
    args = build_parser().parse_args(arguments)
    distributed: DistributedContext | None = None
    try:
        distributed = _initialize_distributed(args.device)
        if args.local_rank is not None:
            if not distributed.enabled:
                raise RuntimeError("--local-rank requires a torchrun environment")
            if args.local_rank != distributed.local_rank:
                raise RuntimeError("--local-rank disagrees with LOCAL_RANK")
        _run_training(args, distributed)
    except BaseException:
        if (
            distributed is not None
            and distributed.enabled
            and distributed.is_main
            and distributed.rank_zero_phase_active
            and torch.distributed.is_initialized()
        ):
            distributed.rank_zero_phase_active = False
            try:
                _broadcast_rank_zero_phase(False, True, distributed)
            except BaseException:
                # Preserve the original training exception; teardown below still
                # closes the communicator if notification itself cannot complete.
                pass
        raise
    finally:
        _shutdown_distributed(distributed)


if __name__ == "__main__":
    main()
