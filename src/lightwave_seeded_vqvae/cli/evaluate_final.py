from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

from ..data import build_dataset, dataset_channels, resolve_dataset_root
from ..evaluation import (
    SSIM_SPEC,
    fixed_state_schedule,
    gaussian_ssim_per_image,
    image_mse,
    image_metric_bootstrap,
    metric_summary,
    paired_correct_wrong_summary,
    paired_image_branch_bootstrap,
    paired_image_bootstrap,
    pseudorandom_balanced_schedule,
    schedule_sha256,
    summarize_cross_state,
    summarize_schedule,
    usage_summary,
)
from ..models.vqvae import SeededResidualVQVAE, TrainableResidualVQVAE
from ..seed.codebook import load_codebook_bank
from ..training import write_json


EVALUATION_SCHEMA = "lightwave-seeded-vqvae-final-evaluation-v2"
SCHEDULE_ALGORITHM = "numpy-pcg64-blockwise-permutation-v1"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified held-out evaluation for the matched software baseline and "
            "ten-state seeded residual VQ-VAE"
        )
    )
    parser.add_argument("--seeded-checkpoint", required=True, type=Path)
    parser.add_argument("--software-checkpoint", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--dataset-root", type=Path,
        help="Read a relocated dataset here; retain checkpoint paths in provenance.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=8)
    parser.add_argument("--schedule-seed", type=int, default=20260907)
    parser.add_argument("--bootstrap-seed", type=int, default=20260908)
    parser.add_argument("--bootstrap-replicates", type=_nonnegative_int, default=2000)
    parser.add_argument(
        "--lpips",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "LPIPS-Alex v0.1 policy. 'auto' enables it for CIFAR-10/STL-10 and "
            "disables it for MNIST; 'on' and 'off' override that dataset-aware "
            "default."
        ),
    )
    parser.add_argument(
        "--smoke-max-batches",
        type=_positive_int,
        help=(
            "Evaluate only this many batches and mark every artifact as smoke. "
            "Omit for the formal full-test evaluation."
        ),
    )
    return parser


def evaluation_scope(smoke_max_batches: int | None) -> str:
    return "smoke" if smoke_max_batches is not None else "final"


def _lpips_enabled(policy: str, dataset_name: str) -> bool:
    if policy not in {"auto", "on", "off"}:
        raise ValueError(f"unsupported LPIPS policy: {policy}")
    if policy == "on":
        return True
    if policy == "off":
        return False
    normalized = dataset_name.strip().lower().replace("-", "")
    return normalized in {"cifar10", "stl10"}


def _lpips_per_image(
    metric: torch.nn.Module,
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Evaluate LPIPS after the documented [0, 1] -> [-1, 1] mapping."""

    if reconstruction.shape != target.shape or reconstruction.ndim != 4:
        raise ValueError("LPIPS reconstruction and target must share NCHW shape")
    if reconstruction.shape[1] not in {1, 3}:
        raise ValueError("LPIPS supports one-channel or three-channel evaluator inputs")
    first = reconstruction.mul(2.0).sub(1.0)
    second = target.mul(2.0).sub(1.0)
    if first.shape[1] == 1:
        first = first.repeat(1, 3, 1, 1)
        second = second.repeat(1, 3, 1, 1)
    if min(first.shape[-2:]) < 32:
        first = torch.nn.functional.interpolate(
            first, size=(32, 32), mode="bilinear", align_corners=False
        )
        second = torch.nn.functional.interpolate(
            second, size=(32, 32), mode="bilinear", align_corners=False
        )
    scores = metric(first, second)
    if not isinstance(scores, torch.Tensor) or scores.shape[0] != first.shape[0]:
        raise RuntimeError("LPIPS implementation returned an unexpected result shape")
    return scores.reshape(scores.shape[0], -1).mean(dim=1)


def _build_lpips_alex(device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError(
            "LPIPS is enabled but the 'lpips' package is unavailable; install the "
            "project dependencies or rerun with --lpips off"
        ) from error
    metric = lpips.LPIPS(net="alex", version="0.1", verbose=False)
    metric.to(device).eval()
    for parameter in metric.parameters():
        parameter.requires_grad_(False)
    try:
        package_version = importlib_metadata.version("lpips")
    except importlib_metadata.PackageNotFoundError:
        package_version = getattr(lpips, "__version__", None)
    provenance = {
        "enabled": True,
        "metric": "LPIPS",
        "network": "alex",
        "version": "0.1",
        "package_version": package_version,
        "input_mapping": "x_lpips = 2*x_[0,1] - 1",
        "spatial": False,
        "grayscale_policy": "repeat to three channels; resize to 32x32 if needed",
    }
    return metric, provenance


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    little_endian = np.ascontiguousarray(array, dtype="<f4")
    return hashlib.sha256(little_endian.tobytes()).hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must contain an object: {path}")
    return payload


def _optional_json(path: Path) -> dict[str, Any] | None:
    return _read_json(path) if path.is_file() else None


def _git_metadata(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(project_root), "status", "--porcelain"], text=True
        ).strip()
        return {"commit": commit, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _state_labels(manifest: dict[str, Any]) -> list[str]:
    states = manifest.get("states")
    if not isinstance(states, list) or len(states) != 10:
        raise ValueError("codebook manifest must contain exactly ten states")
    labels: list[str] = []
    for expected_index, state in enumerate(states):
        if not isinstance(state, dict) or int(state.get("index", -1)) != expected_index:
            raise ValueError("manifest states must be stored in canonical index order")
        label = state.get("state_id")
        if not isinstance(label, str) or not label:
            raise ValueError("every manifest state must have a state_id")
        labels.append(label)
    if len(set(labels)) != 10:
        raise ValueError("manifest state labels must be distinct")
    return labels


PHYSICAL_MISMATCH_GROUPS = (
    "cross_pump_same_power",
    "same_pump_wrong_power",
    "cross_pump_wrong_power",
)
PHYSICAL_MISMATCH_MARGINALS = (
    "all_cross_pump",
    "all_wrong_power",
)


def physical_mismatch_groups_from_manifest(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build the three disjoint physical groups for all 90 ordered mismatches."""

    labels = _state_labels(manifest)
    states = manifest["states"]
    records: list[dict[str, Any]] = []
    physical_keys: set[tuple[int, int]] = set()
    for index, (state, label) in enumerate(zip(states, labels)):
        pump_value = state.get("pump_nm")
        power_value = state.get("power_mw")
        if isinstance(pump_value, bool) or isinstance(power_value, bool):
            raise ValueError("pump_nm and power_mw must be integer physical values")
        try:
            pump_nm = int(pump_value)
            power_mw = int(power_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "every manifest state must contain integer pump_nm and power_mw"
            ) from error
        if (pump_nm, power_mw) in physical_keys:
            raise ValueError("manifest contains duplicate pump/power states")
        physical_keys.add((pump_nm, power_mw))
        records.append(
            {
                "state_index": index,
                "state_id": label,
                "pump_nm": pump_nm,
                "power_mw": power_mw,
            }
        )

    exclusive: dict[str, list[dict[str, Any]]] = {
        name: [] for name in PHYSICAL_MISMATCH_GROUPS
    }
    marginal: dict[str, list[dict[str, Any]]] = {
        name: [] for name in PHYSICAL_MISMATCH_MARGINALS
    }
    all_pairs: list[dict[str, Any]] = []
    for send in records:
        for receive in records:
            if receive["state_index"] == send["state_index"]:
                continue
            pump_changed = receive["pump_nm"] != send["pump_nm"]
            power_changed = receive["power_mw"] != send["power_mw"]
            if pump_changed and not power_changed:
                group = "cross_pump_same_power"
            elif not pump_changed and power_changed:
                group = "same_pump_wrong_power"
            elif pump_changed and power_changed:
                group = "cross_pump_wrong_power"
            else:
                raise RuntimeError("distinct state indices share one physical state")
            pair = {
                "send_state_index": send["state_index"],
                "send_state_id": send["state_id"],
                "send_pump_nm": send["pump_nm"],
                "send_power_mw": send["power_mw"],
                "receive_state_index": receive["state_index"],
                "receive_state_id": receive["state_id"],
                "receive_pump_nm": receive["pump_nm"],
                "receive_power_mw": receive["power_mw"],
                "exclusive_group": group,
                "cross_pump": pump_changed,
                "wrong_power": power_changed,
            }
            exclusive[group].append(pair)
            if pump_changed:
                marginal["all_cross_pump"].append(pair)
            if power_changed:
                marginal["all_wrong_power"].append(pair)
            all_pairs.append(pair)

    expected_exclusive = {
        "cross_pump_same_power": 10,
        "same_pump_wrong_power": 40,
        "cross_pump_wrong_power": 40,
    }
    observed_exclusive = {name: len(pairs) for name, pairs in exclusive.items()}
    if len(all_pairs) != 90 or observed_exclusive != expected_exclusive:
        raise ValueError(
            "the ten-state physical grid must yield 90 ordered mismatches split "
            "as 10 cross-pump/same-power, 40 same-pump/wrong-power, and 40 "
            "cross-pump/wrong-power pairs"
        )
    observed_marginal = {name: len(pairs) for name, pairs in marginal.items()}
    if observed_marginal != {"all_cross_pump": 50, "all_wrong_power": 80}:
        raise ValueError("physical mismatch marginal counts are inconsistent")

    return {
        "state_records": records,
        "ordered_wrong_pair_count": len(all_pairs),
        "exclusive_group_order": list(PHYSICAL_MISMATCH_GROUPS),
        "marginal_group_order": list(PHYSICAL_MISMATCH_MARGINALS),
        "exclusive_groups": {
            name: {"pair_count": len(pairs), "pairs": pairs}
            for name, pairs in exclusive.items()
        },
        "marginal_groups": {
            name: {"pair_count": len(pairs), "pairs": pairs}
            for name, pairs in marginal.items()
        },
        "definitions": {
            "ordered_pair": "sender and receiver roles are directional",
            "cross_pump_same_power": "pump differs and power is unchanged",
            "same_pump_wrong_power": "pump is unchanged and power differs",
            "cross_pump_wrong_power": "both pump and power differ",
            "all_cross_pump": (
                "marginal group; union of cross_pump_same_power and "
                "cross_pump_wrong_power"
            ),
            "all_wrong_power": (
                "marginal group; union of same_pump_wrong_power and "
                "cross_pump_wrong_power"
            ),
        },
    }


def _cross_values_for_pairs(
    values: np.ndarray, pairs: list[dict[str, Any]]
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 3 or array.shape[1:] != (10, 10):
        raise ValueError("cross-state metric array must have shape (N, 10, 10)")
    return np.stack(
        [
            array[:, int(pair["send_state_index"]), int(pair["receive_state_index"])]
            for pair in pairs
        ],
        axis=1,
    )


def _correct_values_for_pairs(
    values: np.ndarray, pairs: list[dict[str, Any]]
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 3 or array.shape[1:] != (10, 10):
        raise ValueError("cross-state metric array must have shape (N, 10, 10)")
    return np.stack(
        [
            array[:, int(pair["send_state_index"]), int(pair["send_state_index"])]
            for pair in pairs
        ],
        axis=1,
    )


def summarize_physical_mismatch_groups(
    per_image_mse: np.ndarray,
    per_image_ssim: np.ndarray,
    *,
    grouping: dict[str, Any],
    pixels_per_image: int,
    per_image_lpips: np.ndarray | None,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    """Summarize exclusive and marginal mismatch groups with image-level CIs."""

    mse = np.asarray(per_image_mse)
    ssim = np.asarray(per_image_ssim)
    if mse.shape != ssim.shape or mse.ndim != 3 or mse.shape[1:] != (10, 10):
        raise ValueError("physical summaries require equal (N, 10, 10) MSE/SSIM arrays")
    lpips = None if per_image_lpips is None else np.asarray(per_image_lpips)
    if lpips is not None and lpips.shape != mse.shape:
        raise ValueError("physical-summary LPIPS must match the cross-state arrays")

    def summarize_collection(collection: dict[str, Any]) -> dict[str, Any]:
        summarized: dict[str, Any] = {}
        for name, definition in collection.items():
            pairs = definition["pairs"]
            wrong_mse = _cross_values_for_pairs(mse, pairs)
            wrong_ssim = _cross_values_for_pairs(ssim, pairs)
            correct_mse = _correct_values_for_pairs(mse, pairs)
            correct_ssim = _correct_values_for_pairs(ssim, pairs)
            wrong_lpips = (
                _cross_values_for_pairs(lpips, pairs) if lpips is not None else None
            )
            correct_lpips = (
                _correct_values_for_pairs(lpips, pairs) if lpips is not None else None
            )
            summarized[name] = {
                "pair_count": int(definition["pair_count"]),
                "wrong_metrics": metric_summary(
                    wrong_mse,
                    wrong_ssim,
                    pixels_per_observation=pixels_per_image,
                    per_observation_lpips=wrong_lpips,
                ),
                "wrong_metric_confidence_intervals": image_metric_bootstrap(
                    wrong_mse,
                    wrong_ssim,
                    per_image_lpips=wrong_lpips,
                    seed=bootstrap_seed,
                    replicates=bootstrap_replicates,
                ),
                "paired_against_sender_correct": paired_image_branch_bootstrap(
                    correct_mse,
                    wrong_mse,
                    correct_ssim=correct_ssim,
                    wrong_ssim=wrong_ssim,
                    correct_lpips=correct_lpips,
                    wrong_lpips=wrong_lpips,
                    seed=bootstrap_seed,
                    replicates=bootstrap_replicates,
                ),
            }
        return summarized

    return {
        "ordered_wrong_pair_count": int(grouping["ordered_wrong_pair_count"]),
        "definitions": grouping["definitions"],
        "exclusive_groups": summarize_collection(grouping["exclusive_groups"]),
        "marginal_groups": summarize_collection(grouping["marginal_groups"]),
        "marginal_overlap_note": (
            "all_cross_pump and all_wrong_power overlap on the 40 "
            "cross_pump_wrong_power ordered pairs"
        ),
    }


def cross_state_metric_confidence_intervals(
    per_image_mse: np.ndarray,
    per_image_ssim: np.ndarray,
    *,
    state_labels: list[str],
    per_image_lpips: np.ndarray | None,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    """Return pooled and per-correct-state image-bootstrap intervals."""

    mse = np.asarray(per_image_mse)
    ssim = np.asarray(per_image_ssim)
    if mse.shape != ssim.shape or mse.ndim != 3 or mse.shape[1:] != (10, 10):
        raise ValueError("confidence intervals require equal (N, 10, 10) arrays")
    if len(state_labels) != 10:
        raise ValueError("state_labels must contain ten entries")
    lpips = None if per_image_lpips is None else np.asarray(per_image_lpips)
    if lpips is not None and lpips.shape != mse.shape:
        raise ValueError("LPIPS must match the cross-state arrays")
    states = np.arange(10)
    wrong_mask = ~np.eye(10, dtype=bool)
    result = {
        "correct_pooled": image_metric_bootstrap(
            mse[:, states, states],
            ssim[:, states, states],
            per_image_lpips=(lpips[:, states, states] if lpips is not None else None),
            seed=bootstrap_seed,
            replicates=bootstrap_replicates,
        ),
        "wrong_pooled": image_metric_bootstrap(
            mse[:, wrong_mask],
            ssim[:, wrong_mask],
            per_image_lpips=(lpips[:, wrong_mask] if lpips is not None else None),
            seed=bootstrap_seed,
            replicates=bootstrap_replicates,
        ),
        "correct_by_state": [],
    }
    result["paired_wrong_against_correct"] = paired_image_branch_bootstrap(
        mse[:, states, states],
        mse[:, wrong_mask],
        correct_ssim=ssim[:, states, states],
        wrong_ssim=ssim[:, wrong_mask],
        correct_lpips=(lpips[:, states, states] if lpips is not None else None),
        wrong_lpips=(lpips[:, wrong_mask] if lpips is not None else None),
        seed=bootstrap_seed + 1,
        replicates=bootstrap_replicates,
    )
    for state, label in enumerate(state_labels):
        row = image_metric_bootstrap(
            mse[:, state, state],
            ssim[:, state, state],
            per_image_lpips=(lpips[:, state, state] if lpips is not None else None),
            seed=bootstrap_seed,
            replicates=bootstrap_replicates,
        )
        row.update({"state_index": state, "state_id": label})
        result["correct_by_state"].append(row)
    return result


def _model_arguments(config: dict[str, Any]) -> dict[str, Any]:
    dataset_config = config["dataset"]
    model_config = config["model"]
    encoder_backbone = _encoder_backbone(model_config)
    if encoder_backbone != "resnet18":
        raise ValueError(
            "the current formal evaluator supports encoder_backbone=resnet18 only"
        )
    return {
        "in_channels": dataset_channels(str(dataset_config["name"])),
        "latent_dim": int(model_config["latent_dim"]),
        "residual_codebook_size": int(model_config["residual_codebook_size"]),
        "pretrained_encoder": False,
        "imagenet_weights": str(model_config["imagenet_weights"]),
        "normalize_latents": bool(model_config.get("normalize_latents", True)),
        "loss_reference_dim": int(model_config.get("loss_reference_dim", 64)),
        "decoder_blocks_per_stage": int(
            model_config.get("decoder_blocks_per_stage", 1)
        ),
        "decoder_blocks_by_stage": _decoder_blocks_by_stage(model_config),
        "decoder_zero_init_extra_residual_blocks": model_config.get(
            "decoder_zero_init_extra_residual_blocks", False
        ),
    }


def _encoder_backbone(model_config: dict[str, Any]) -> str:
    return str(model_config.get("encoder_backbone", "resnet18")).strip().lower()


def _decoder_blocks_by_stage(model_config: dict[str, Any]) -> tuple[int, int, int]:
    configured = model_config.get("decoder_blocks_by_stage")
    if configured is None:
        scalar = model_config.get("decoder_blocks_per_stage", 1)
        if isinstance(scalar, bool) or not isinstance(scalar, int) or scalar < 1:
            raise ValueError("decoder_blocks_per_stage must be a positive integer")
        return (scalar, scalar, scalar)
    if isinstance(configured, (str, bytes)):
        raise ValueError("decoder_blocks_by_stage must contain [pre8, up16, up32]")
    try:
        values = tuple(configured)
    except TypeError as error:
        raise ValueError(
            "decoder_blocks_by_stage must contain [pre8, up16, up32]"
        ) from error
    if len(values) != 3:
        raise ValueError("decoder_blocks_by_stage must contain [pre8, up16, up32]")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in values
    ):
        raise ValueError("decoder_blocks_by_stage values must be positive integers")
    return values


def _reconstruction_loss(config: dict[str, Any]) -> str:
    training_config = config.get("training", {})
    return str(
        training_config.get(
            "reconstruction_loss",
            training_config.get("reconstruction_loss_type", "mse"),
        )
    ).strip().lower()


def _perceptual_weight(config: dict[str, Any]) -> float:
    return float(config.get("training", {}).get("perceptual_weight", 0.0))


def _load_model(
    checkpoint: dict[str, Any],
    *,
    expected_quantizer_type: str,
    bank: np.ndarray | None,
) -> SeededResidualVQVAE | TrainableResidualVQVAE:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint does not contain a resolved configuration")
    model_config = config["model"]
    quantizer_type = str(model_config.get("quantizer_type", "seeded_residual"))
    if quantizer_type != expected_quantizer_type:
        raise ValueError(
            f"expected {expected_quantizer_type} checkpoint, received {quantizer_type}"
        )
    arguments = _model_arguments(config)
    if quantizer_type == "seeded_residual":
        if bank is None:
            raise ValueError("seeded model requires a base codebook bank")
        model: SeededResidualVQVAE | TrainableResidualVQVAE = SeededResidualVQVAE(
            bank, **arguments
        )
    else:
        model = TrainableResidualVQVAE(
            base_codebook_size=int(model_config["base_codebook_size"]),
            **arguments,
        )
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint does not contain model weights")
    model.load_state_dict(state_dict, strict=True)
    return model


def _compatibility_report(
    seeded_config: dict[str, Any], software_config: dict[str, Any]
) -> dict[str, Any]:
    seeded_dataset = seeded_config["dataset"]
    software_dataset = software_config["dataset"]
    comparisons = {
        "dataset.name": (
            str(seeded_dataset["name"]).lower(),
            str(software_dataset["name"]).lower(),
        ),
        "dataset.validation_size": (
            int(seeded_dataset.get("validation_size", 5000)),
            int(software_dataset.get("validation_size", 5000)),
        ),
        "dataset.root": (
            str(Path(str(seeded_dataset["root"])).expanduser().resolve()),
            str(Path(str(software_dataset["root"])).expanduser().resolve()),
        ),
        "dataset.train_split": (
            seeded_dataset.get("train_split"),
            software_dataset.get("train_split"),
        ),
        "dataset.validation_split": (
            seeded_dataset.get("validation_split"),
            software_dataset.get("validation_split"),
        ),
        "dataset.test_split": (
            seeded_dataset.get("test_split"),
            software_dataset.get("test_split"),
        ),
        "dataset.train_horizontal_flip_probability": (
            float(seeded_dataset.get("train_horizontal_flip_probability", 0.0)),
            float(software_dataset.get("train_horizontal_flip_probability", 0.0)),
        ),
        "experiment.seed": (
            int(seeded_config["experiment"]["seed"]),
            int(software_config["experiment"]["seed"]),
        ),
        "model.encoder_backbone": (
            _encoder_backbone(seeded_config["model"]),
            _encoder_backbone(software_config["model"]),
        ),
        "model.decoder_blocks_by_stage": (
            _decoder_blocks_by_stage(seeded_config["model"]),
            _decoder_blocks_by_stage(software_config["model"]),
        ),
        "model.decoder_zero_init_extra_residual_blocks": (
            seeded_config["model"].get(
                "decoder_zero_init_extra_residual_blocks", False
            ),
            software_config["model"].get(
                "decoder_zero_init_extra_residual_blocks", False
            ),
        ),
        "training.reconstruction_loss": (
            _reconstruction_loss(seeded_config),
            _reconstruction_loss(software_config),
        ),
        "training.perceptual_weight": (
            _perceptual_weight(seeded_config),
            _perceptual_weight(software_config),
        ),
    }
    for key in (
        "latent_dim",
        "base_codebook_size",
        "residual_codebook_size",
        "pretrained_encoder",
        "imagenet_weights",
        "normalize_latents",
        "loss_reference_dim",
        "decoder_blocks_per_stage",
    ):
        comparisons[f"model.{key}"] = (
            seeded_config["model"].get(
                key, 1 if key == "decoder_blocks_per_stage" else None
            ),
            software_config["model"].get(
                key, 1 if key == "decoder_blocks_per_stage" else None
            ),
        )
    mismatches = {
        name: {"seeded": values[0], "software": values[1]}
        for name, values in comparisons.items()
        if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(
            "software and seeded checkpoints are not matched: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return {"matched_fields": sorted(comparisons), "mismatches": {}}


def _new_fingerprint() -> dict[str, Any]:
    return {
        "images": hashlib.sha256(),
        "labels": hashlib.sha256(),
        "sample_count": 0,
    }


def _update_fingerprint(
    fingerprint: dict[str, Any], images: torch.Tensor, labels: torch.Tensor
) -> None:
    image_array = np.ascontiguousarray(images.numpy(), dtype="<f4")
    label_array = np.ascontiguousarray(torch.as_tensor(labels).numpy(), dtype="<i8")
    fingerprint["images"].update(image_array.tobytes())
    fingerprint["labels"].update(label_array.tobytes())
    fingerprint["sample_count"] += int(images.shape[0])


def _finish_fingerprint(fingerprint: dict[str, Any]) -> dict[str, Any]:
    image_hash = fingerprint["images"].hexdigest()
    label_hash = fingerprint["labels"].hexdigest()
    combined = hashlib.sha256(bytes.fromhex(image_hash) + bytes.fromhex(label_hash)).hexdigest()
    return {
        "processed_images_float32_sha256": image_hash,
        "processed_labels_int64_sha256": label_hash,
        "processed_sample_order_sha256": combined,
        "processed_sample_count": int(fingerprint["sample_count"]),
    }


def _new_moment_accumulator(state_count: int) -> dict[str, np.ndarray]:
    return {
        "count": np.zeros(state_count, dtype=np.int64),
        "sum": np.zeros(state_count, dtype=np.float64),
        "sum_square": np.zeros(state_count, dtype=np.float64),
        "min": np.full(state_count, np.inf, dtype=np.float64),
        "max": np.full(state_count, -np.inf, dtype=np.float64),
    }


def _update_moments(
    accumulator: dict[str, np.ndarray], state: int, values: torch.Tensor
) -> None:
    flat = values.detach().reshape(-1).to(dtype=torch.float64)
    if flat.numel() == 0:
        raise ValueError("cannot accumulate an empty tensor")
    reduced = torch.stack(
        (flat.sum(), flat.square().sum(), flat.min(), flat.max())
    ).cpu().numpy()
    accumulator["count"][state] += flat.numel()
    accumulator["sum"][state] += reduced[0]
    accumulator["sum_square"][state] += reduced[1]
    accumulator["min"][state] = min(accumulator["min"][state], reduced[2])
    accumulator["max"][state] = max(accumulator["max"][state], reduced[3])


def _new_quantization_geometry(
    state_count: int, latent_dim: int
) -> dict[str, Any]:
    return {
        "latent_dim": int(latent_dim),
        "token_count": np.zeros(state_count, dtype=np.int64),
        "first_stage_squared_l2_sum": np.zeros(state_count, dtype=np.float64),
        "final_squared_l2_sum": np.zeros(state_count, dtype=np.float64),
        "residual_norm_greater_than_base_count": np.zeros(
            state_count, dtype=np.int64
        ),
        "moments": {
            name: _new_moment_accumulator(state_count)
            for name in (
                "selected_base_l2_norm",
                "selected_residual_l2_norm",
                "selected_base_plus_residual_l2_norm",
                "selected_residual_to_base_norm_ratio",
            )
        },
    }


def _update_quantization_geometry(
    accumulator: dict[str, Any],
    model: SeededResidualVQVAE,
    latent: torch.Tensor,
    quantized: Any,
    send_state: int,
) -> None:
    tokens, _, _, _ = model.quantizer._tokens(latent)
    token_vectors = tokens.reshape(-1, model.quantizer.latent_dim)
    base_indices = quantized.base_indices.reshape(-1)
    residual_indices = quantized.residual_indices.reshape(-1)
    selected_base = torch.nn.functional.embedding(
        base_indices, model.quantizer.base_codebooks[send_state]
    )
    selected_residual = torch.nn.functional.embedding(
        residual_indices, model.quantizer.residual_codebook
    )
    selected_final = selected_base + selected_residual
    base_norm = selected_base.norm(dim=-1)
    residual_norm = selected_residual.norm(dim=-1)
    final_norm = selected_final.norm(dim=-1)
    residual_to_base = residual_norm / base_norm.clamp_min(
        torch.finfo(base_norm.dtype).tiny
    )
    for name, values in (
        ("selected_base_l2_norm", base_norm),
        ("selected_residual_l2_norm", residual_norm),
        ("selected_base_plus_residual_l2_norm", final_norm),
        ("selected_residual_to_base_norm_ratio", residual_to_base),
    ):
        _update_moments(accumulator["moments"][name], send_state, values)

    first_squared_l2 = (token_vectors - selected_base).square().sum(dim=-1)
    final_squared_l2 = (token_vectors - selected_final).square().sum(dim=-1)
    accumulator["token_count"][send_state] += token_vectors.shape[0]
    accumulator["first_stage_squared_l2_sum"][send_state] += float(
        first_squared_l2.sum(dtype=torch.float64).item()
    )
    accumulator["final_squared_l2_sum"][send_state] += float(
        final_squared_l2.sum(dtype=torch.float64).item()
    )
    accumulator["residual_norm_greater_than_base_count"][send_state] += int(
        (residual_norm > base_norm).sum().item()
    )


def _moment_summary(
    accumulator: dict[str, np.ndarray], state: int | None = None
) -> dict[str, float | int]:
    if state is None:
        count = int(accumulator["count"].sum())
        total = float(accumulator["sum"].sum())
        total_square = float(accumulator["sum_square"].sum())
        minimum = float(accumulator["min"].min())
        maximum = float(accumulator["max"].max())
    else:
        count = int(accumulator["count"][state])
        total = float(accumulator["sum"][state])
        total_square = float(accumulator["sum_square"][state])
        minimum = float(accumulator["min"][state])
        maximum = float(accumulator["max"][state])
    if count <= 0:
        raise ValueError("moment accumulator is empty")
    mean = total / count
    variance = max(total_square / count - mean * mean, 0.0)
    return {
        "count": count,
        "mean": mean,
        "std": math.sqrt(variance),
        "rms": math.sqrt(max(total_square / count, 0.0)),
        "min": minimum,
        "max": maximum,
    }


def _tensor_norm_summary(vectors: torch.Tensor) -> dict[str, float | int]:
    norms = vectors.detach().norm(dim=-1).reshape(-1).to(dtype=torch.float64)
    if norms.numel() == 0:
        raise ValueError("cannot summarize an empty vector collection")
    return {
        "count": norms.numel(),
        "mean": float(norms.mean().item()),
        "std": float(norms.std(unbiased=False).item()),
        "rms": float(norms.square().mean().sqrt().item()),
        "min": float(norms.min().item()),
        "max": float(norms.max().item()),
    }


def _model_parameter_summary(model: torch.nn.Module) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "non_trainable_parameters": int(total - trainable),
        "definition": "number of torch.nn.Parameter elements; buffers are excluded",
    }


def _array_distribution(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("distance distribution must contain finite values")
    return {
        "count": int(array.size),
        "mean": float(array.mean(dtype=np.float64)),
        "std": float(array.std(dtype=np.float64)),
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
    }


def _base_codebook_state_distances(
    base_codebooks: torch.Tensor | np.ndarray,
    state_labels: list[str],
) -> dict[str, Any]:
    if torch.is_tensor(base_codebooks):
        bank = base_codebooks.detach().cpu().numpy()
    else:
        bank = np.asarray(base_codebooks)
    bank = np.asarray(bank, dtype=np.float64)
    if bank.ndim != 3 or bank.shape[0] < 2 or bank.shape[1] < 1 or bank.shape[2] < 1:
        raise ValueError("base codebooks must have shape (states, K, D)")
    state_count, codebook_size, latent_dim = bank.shape
    if len(state_labels) != state_count or len(set(state_labels)) != state_count:
        raise ValueError("state_labels must be distinct and match the base bank")
    if not np.isfinite(bank).all():
        raise ValueError("base codebooks contain non-finite values")

    distances = np.linalg.norm(
        bank[:, None, :, :] - bank[None, :, :, :], axis=-1
    )
    mean_matrix = distances.mean(axis=-1, dtype=np.float64)
    different_state_mask = ~np.eye(state_count, dtype=bool)
    by_state_pair: list[dict[str, Any]] = []
    for first_state, first_label in enumerate(state_labels):
        for second_state, second_label in enumerate(state_labels):
            row = _array_distribution(distances[first_state, second_state])
            row.update(
                {
                    "first_state_index": first_state,
                    "first_state_id": first_label,
                    "second_state_index": second_state,
                    "second_state_id": second_label,
                    "same_state": first_state == second_state,
                }
            )
            by_state_pair.append(row)
    return {
        "definition": (
            "L2 distance ||B_s[k]-B_t[k]|| between equal code indices k; "
            "the base bank is frozen and no assignment matching is performed"
        ),
        "state_count": int(state_count),
        "codebook_size": int(codebook_size),
        "latent_dim": int(latent_dim),
        "ordered_state_pair_count": int(state_count * state_count),
        "different_state_ordered_pair_count": int(state_count * (state_count - 1)),
        "mean_l2_matrix": mean_matrix.tolist(),
        "different_state_pooled": _array_distribution(
            distances[different_state_mask]
        ),
        "including_same_state_pooled": _array_distribution(distances),
        "by_state_pair": by_state_pair,
    }


def _quantization_error_summary(
    accumulator: dict[str, Any], state: int | None = None
) -> dict[str, float | int | None]:
    latent_dim = int(accumulator["latent_dim"])
    if state is None:
        token_count = int(accumulator["token_count"].sum())
        first_sum = float(accumulator["first_stage_squared_l2_sum"].sum())
        final_sum = float(accumulator["final_squared_l2_sum"].sum())
    else:
        token_count = int(accumulator["token_count"][state])
        first_sum = float(accumulator["first_stage_squared_l2_sum"][state])
        final_sum = float(accumulator["final_squared_l2_sum"][state])
    if token_count <= 0:
        raise ValueError("quantization-error accumulator is empty")
    first_per_token = first_sum / token_count
    final_per_token = final_sum / token_count
    if first_sum > 0.0:
        final_to_first = final_sum / first_sum
        reduction = 1.0 - final_to_first
        improvement_db = 10.0 * math.log10(first_sum / max(final_sum, 1e-300))
    else:
        final_to_first = None
        reduction = None
        improvement_db = None
    return {
        "token_count": token_count,
        "first_stage_squared_l2_sum": first_sum,
        "final_squared_l2_sum": final_sum,
        "first_stage_mean_squared_l2_per_token": first_per_token,
        "final_mean_squared_l2_per_token": final_per_token,
        "first_stage_elementwise_mse": first_per_token / latent_dim,
        "final_elementwise_mse": final_per_token / latent_dim,
        "first_stage_elementwise_rmse": math.sqrt(first_per_token / latent_dim),
        "final_elementwise_rmse": math.sqrt(final_per_token / latent_dim),
        "final_to_first_error_ratio": final_to_first,
        "first_to_final_reduction_fraction": reduction,
        "first_to_final_improvement_db": improvement_db,
    }


def _finalize_quantization_geometry(
    model: SeededResidualVQVAE,
    accumulator: dict[str, Any],
    state_labels: list[str],
) -> dict[str, Any]:
    base_bank = model.quantizer.base_codebooks.detach()
    residual_bank = model.quantizer.residual_codebook.detach()
    codebook_by_state = []
    selected_by_state = []
    for state, label in enumerate(state_labels):
        all_sums = base_bank[state, :, None, :] + residual_bank[None, :, :]
        codebook_by_state.append(
            {
                "state_index": state,
                "state_id": label,
                "base": _tensor_norm_summary(base_bank[state]),
                "base_plus_residual_all_pairs": _tensor_norm_summary(all_sums),
            }
        )
        token_count = int(accumulator["token_count"][state])
        selected_by_state.append(
            {
                "state_index": state,
                "state_id": label,
                "norms": {
                    name: _moment_summary(moment, state)
                    for name, moment in accumulator["moments"].items()
                },
                "fraction_selected_residual_norm_greater_than_base": float(
                    accumulator["residual_norm_greater_than_base_count"][state]
                    / token_count
                ),
                "quantization_error": _quantization_error_summary(
                    accumulator, state
                ),
            }
        )

    all_sums = base_bank[:, :, None, :] + residual_bank[None, None, :, :]
    pooled_token_count = int(accumulator["token_count"].sum())
    return {
        "definitions": {
            "B": "frozen state-specific chip-seeded base vector",
            "R": "shared trainable residual vector selected after B",
            "B_plus_R": "final transmitted-index reconstruction vector B_s[i] + R[j]",
            "first_stage_error": "squared L2 distance from normalized encoder token to selected B",
            "final_error": "squared L2 distance from normalized encoder token to selected B+R",
            "aggregation": "held-out selected-token statistics are token pooled, not batch averaged",
        },
        "selected_on_held_out_test": {
            "pooled": {
                "norms": {
                    name: _moment_summary(moment)
                    for name, moment in accumulator["moments"].items()
                },
                "fraction_selected_residual_norm_greater_than_base": float(
                    accumulator["residual_norm_greater_than_base_count"].sum()
                    / pooled_token_count
                ),
                "quantization_error": _quantization_error_summary(accumulator),
            },
            "by_state": selected_by_state,
        },
        "codebook_unweighted_supplement": {
            "base_pooled": _tensor_norm_summary(base_bank),
            "shared_residual": _tensor_norm_summary(residual_bank),
            "base_plus_residual_all_state_code_pairs": _tensor_norm_summary(
                all_sums
            ),
            "by_state": codebook_by_state,
            "interpretation": (
                "Cartesian all-pairs geometry is supplementary; selected-token "
                "held-out statistics are the primary diagnostic."
            ),
        },
    }


@torch.inference_mode()
def _evaluate_software(
    model: TrainableResidualVQVAE,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None,
    perceptual_metric: torch.nn.Module | None = None,
) -> dict[str, Any]:
    model.eval()
    mse_chunks: list[np.ndarray] = []
    ssim_chunks: list[np.ndarray] = []
    lpips_chunks: list[np.ndarray] = []
    base_counts = np.zeros(model.quantizer.base_codebook_size, dtype=np.int64)
    residual_counts = np.zeros(model.quantizer.residual_codebook_size, dtype=np.int64)
    fingerprint = _new_fingerprint()
    latent_shape: tuple[int, int] | None = None
    progress = tqdm(loader, desc="software baseline", leave=False)
    for batch_index, (images, labels) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        _update_fingerprint(fingerprint, images, labels)
        images = images.to(device, non_blocking=True)
        state_ids = torch.zeros(images.shape[0], dtype=torch.long, device=device)
        output = model(images, state_ids)
        current_latent_shape = tuple(
            int(value) for value in output.quantizer.base_indices.shape[-2:]
        )
        if latent_shape is None:
            latent_shape = current_latent_shape
        elif current_latent_shape != latent_shape:
            raise RuntimeError("software latent spatial shape changed during evaluation")
        mse_chunks.append(image_mse(output.reconstruction, images).cpu().numpy())
        ssim_chunks.append(
            gaussian_ssim_per_image(output.reconstruction, images).cpu().numpy()
        )
        if perceptual_metric is not None:
            lpips_chunks.append(
                _lpips_per_image(
                    perceptual_metric, output.reconstruction, images
                ).cpu().numpy()
            )
        base_counts += torch.bincount(
            output.quantizer.base_indices.reshape(-1),
            minlength=model.quantizer.base_codebook_size,
        ).cpu().numpy()
        residual_counts += torch.bincount(
            output.quantizer.residual_indices.reshape(-1),
            minlength=model.quantizer.residual_codebook_size,
        ).cpu().numpy()
    if not mse_chunks:
        raise RuntimeError("software evaluation processed no samples")
    assert latent_shape is not None
    return {
        "mse": np.concatenate(mse_chunks).astype(np.float32, copy=False),
        "ssim": np.concatenate(ssim_chunks).astype(np.float32, copy=False),
        "lpips": (
            np.concatenate(lpips_chunks).astype(np.float32, copy=False)
            if perceptual_metric is not None
            else None
        ),
        "base_counts": base_counts,
        "residual_counts": residual_counts,
        "latent_shape": list(latent_shape),
        "dataset_fingerprint": _finish_fingerprint(fingerprint),
    }


@torch.inference_mode()
def _evaluate_seeded(
    model: SeededResidualVQVAE,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None,
    perceptual_metric: torch.nn.Module | None = None,
    state_labels: list[str] | None = None,
) -> dict[str, Any]:
    model.eval()
    state_count = model.quantizer.state_count
    if state_labels is None:
        state_labels = [str(index) for index in range(state_count)]
    if len(state_labels) != state_count:
        raise ValueError("state_labels must match the seeded model state count")
    base_counts = np.zeros(
        (state_count, model.quantizer.base_codebook_size), dtype=np.int64
    )
    residual_counts = np.zeros(
        (state_count, model.quantizer.residual_codebook_size), dtype=np.int64
    )
    cross_mse_chunks: list[np.ndarray] = []
    cross_ssim_chunks: list[np.ndarray] = []
    cross_lpips_chunks: list[np.ndarray] = []
    base_only_mse_chunks: list[np.ndarray] = []
    base_only_ssim_chunks: list[np.ndarray] = []
    base_only_lpips_chunks: list[np.ndarray] = []
    transmitted_residual_mse_chunks: list[np.ndarray] = []
    transmitted_residual_ssim_chunks: list[np.ndarray] = []
    transmitted_residual_lpips_chunks: list[np.ndarray] = []
    requantized_residual_mse_chunks: list[np.ndarray] = []
    requantized_residual_ssim_chunks: list[np.ndarray] = []
    requantized_residual_lpips_chunks: list[np.ndarray] = []
    fingerprint = _new_fingerprint()
    quantization_geometry = _new_quantization_geometry(
        state_count, model.quantizer.latent_dim
    )
    latent_shape: tuple[int, int] | None = None

    progress = tqdm(loader, desc="seeded 10x10 evaluation", leave=False)
    for batch_index, (images, labels) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        _update_fingerprint(fingerprint, images, labels)
        images = images.to(device, non_blocking=True)
        batch_size = images.shape[0]
        latent = model.encoder(images)
        current_latent_shape = tuple(int(value) for value in latent.shape[-2:])
        if latent_shape is None:
            latent_shape = current_latent_shape
        elif current_latent_shape != latent_shape:
            raise RuntimeError("seeded latent spatial shape changed during evaluation")
        cross_mse = np.empty((batch_size, state_count, state_count), dtype=np.float32)
        cross_ssim = np.empty_like(cross_mse)
        cross_lpips = (
            np.empty_like(cross_mse) if perceptual_metric is not None else None
        )
        base_only_mse = np.empty((batch_size, state_count), dtype=np.float32)
        base_only_ssim = np.empty_like(base_only_mse)
        base_only_lpips = (
            np.empty_like(base_only_mse) if perceptual_metric is not None else None
        )
        transmitted_residual_mse = np.empty_like(base_only_mse)
        transmitted_residual_ssim = np.empty_like(base_only_mse)
        transmitted_residual_lpips = (
            np.empty_like(base_only_mse) if perceptual_metric is not None else None
        )

        requantized_residual, _ = model.reconstruct_residual_only(latent)
        requantized_residual_mse_chunks.append(
            image_mse(requantized_residual, images).cpu().numpy()
        )
        requantized_residual_ssim_chunks.append(
            gaussian_ssim_per_image(requantized_residual, images).cpu().numpy()
        )
        if perceptual_metric is not None:
            requantized_residual_lpips_chunks.append(
                _lpips_per_image(
                    perceptual_metric, requantized_residual, images
                ).cpu().numpy()
            )

        for send_state in range(state_count):
            send_ids = torch.full(
                (batch_size,), send_state, dtype=torch.long, device=device
            )
            quantized = model.quantizer(latent, send_ids)
            _update_quantization_geometry(
                quantization_geometry, model, latent, quantized, send_state
            )
            base_counts[send_state] += torch.bincount(
                quantized.base_indices.reshape(-1),
                minlength=model.quantizer.base_codebook_size,
            ).cpu().numpy()
            residual_counts[send_state] += torch.bincount(
                quantized.residual_indices.reshape(-1),
                minlength=model.quantizer.residual_codebook_size,
            ).cpu().numpy()

            for receive_state in range(state_count):
                receive_ids = torch.full_like(send_ids, receive_state)
                reconstruction = model.decode_indices(
                    quantized.base_indices, quantized.residual_indices, receive_ids
                )
                cross_mse[:, send_state, receive_state] = image_mse(
                    reconstruction, images
                ).cpu().numpy()
                cross_ssim[:, send_state, receive_state] = gaussian_ssim_per_image(
                    reconstruction, images
                ).cpu().numpy()
                if cross_lpips is not None:
                    cross_lpips[:, send_state, receive_state] = _lpips_per_image(
                        perceptual_metric, reconstruction, images
                    ).cpu().numpy()

            base_reconstruction = model.decode_indices(
                quantized.base_indices,
                quantized.residual_indices,
                send_ids,
                include_base=True,
                include_residual=False,
            )
            residual_reconstruction = model.decode_indices(
                quantized.base_indices,
                quantized.residual_indices,
                send_ids,
                include_base=False,
                include_residual=True,
            )
            base_only_mse[:, send_state] = image_mse(
                base_reconstruction, images
            ).cpu().numpy()
            base_only_ssim[:, send_state] = gaussian_ssim_per_image(
                base_reconstruction, images
            ).cpu().numpy()
            if base_only_lpips is not None:
                base_only_lpips[:, send_state] = _lpips_per_image(
                    perceptual_metric, base_reconstruction, images
                ).cpu().numpy()
            transmitted_residual_mse[:, send_state] = image_mse(
                residual_reconstruction, images
            ).cpu().numpy()
            transmitted_residual_ssim[:, send_state] = gaussian_ssim_per_image(
                residual_reconstruction, images
            ).cpu().numpy()
            if transmitted_residual_lpips is not None:
                transmitted_residual_lpips[:, send_state] = _lpips_per_image(
                    perceptual_metric, residual_reconstruction, images
                ).cpu().numpy()

        cross_mse_chunks.append(cross_mse)
        cross_ssim_chunks.append(cross_ssim)
        if cross_lpips is not None:
            cross_lpips_chunks.append(cross_lpips)
        base_only_mse_chunks.append(base_only_mse)
        base_only_ssim_chunks.append(base_only_ssim)
        if base_only_lpips is not None:
            base_only_lpips_chunks.append(base_only_lpips)
        transmitted_residual_mse_chunks.append(transmitted_residual_mse)
        transmitted_residual_ssim_chunks.append(transmitted_residual_ssim)
        if transmitted_residual_lpips is not None:
            transmitted_residual_lpips_chunks.append(transmitted_residual_lpips)

    if not cross_mse_chunks:
        raise RuntimeError("seeded evaluation processed no samples")
    assert latent_shape is not None
    return {
        "cross_mse": np.concatenate(cross_mse_chunks, axis=0),
        "cross_ssim": np.concatenate(cross_ssim_chunks, axis=0),
        "cross_lpips": (
            np.concatenate(cross_lpips_chunks, axis=0)
            if perceptual_metric is not None
            else None
        ),
        "base_only_mse": np.concatenate(base_only_mse_chunks, axis=0),
        "base_only_ssim": np.concatenate(base_only_ssim_chunks, axis=0),
        "base_only_lpips": (
            np.concatenate(base_only_lpips_chunks, axis=0)
            if perceptual_metric is not None
            else None
        ),
        "transmitted_residual_mse": np.concatenate(
            transmitted_residual_mse_chunks, axis=0
        ),
        "transmitted_residual_ssim": np.concatenate(
            transmitted_residual_ssim_chunks, axis=0
        ),
        "transmitted_residual_lpips": (
            np.concatenate(transmitted_residual_lpips_chunks, axis=0)
            if perceptual_metric is not None
            else None
        ),
        "requantized_residual_mse": np.concatenate(
            requantized_residual_mse_chunks
        ).astype(np.float32, copy=False),
        "requantized_residual_ssim": np.concatenate(
            requantized_residual_ssim_chunks
        ).astype(np.float32, copy=False),
        "requantized_residual_lpips": (
            np.concatenate(requantized_residual_lpips_chunks).astype(
                np.float32, copy=False
            )
            if perceptual_metric is not None
            else None
        ),
        "base_counts": base_counts,
        "residual_counts": residual_counts,
        "latent_shape": list(latent_shape),
        "dataset_fingerprint": _finish_fingerprint(fingerprint),
        "quantization_geometry": _finalize_quantization_geometry(
            model, quantization_geometry, state_labels
        ),
    }


def _summarize_state_branch(
    mse: np.ndarray,
    ssim: np.ndarray,
    *,
    state_labels: list[str],
    pixels_per_image: int,
    lpips: np.ndarray | None = None,
) -> dict[str, Any]:
    if mse.shape != ssim.shape or mse.ndim != 2 or mse.shape[1] != len(state_labels):
        raise ValueError("branch arrays must share shape (N, state_count)")
    if lpips is not None and lpips.shape != mse.shape:
        raise ValueError("branch LPIPS must share shape (N, state_count)")
    by_state = []
    for state, label in enumerate(state_labels):
        row = metric_summary(
            mse[:, state],
            ssim[:, state],
            pixels_per_observation=pixels_per_image,
            per_observation_lpips=(lpips[:, state] if lpips is not None else None),
        )
        row.update({"state_index": state, "state_id": label})
        by_state.append(row)
    return {
        "pooled": metric_summary(
            mse,
            ssim,
            pixels_per_observation=pixels_per_image,
            per_observation_lpips=lpips,
        ),
        "by_state": by_state,
    }


def _nominal_bits(
    base_size: int,
    residual_size: int,
    *,
    latent_shape: list[int] | tuple[int, int] | None = None,
    image_shape: list[int] | tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    if base_size <= 0 or residual_size <= 0:
        raise ValueError("codebook sizes must be positive")
    base_bits = math.ceil(math.log2(base_size))
    residual_bits = math.ceil(math.log2(residual_size))
    result: dict[str, Any] = {
        "base_index_bits_per_token": base_bits,
        "residual_index_bits_per_token": residual_bits,
        "total_index_bits_per_token": base_bits + residual_bits,
    }
    if latent_shape is None and image_shape is None:
        return result
    if latent_shape is None or image_shape is None:
        raise ValueError("latent_shape and image_shape must be provided together")
    if len(latent_shape) != 2 or len(image_shape) != 3:
        raise ValueError("expected latent HW and image CHW shapes")
    latent_height, latent_width = (int(value) for value in latent_shape)
    image_channels, image_height, image_width = (int(value) for value in image_shape)
    if (
        min(latent_height, latent_width, image_channels, image_height, image_width)
        <= 0
    ):
        raise ValueError("latent and image dimensions must be positive")
    token_count = latent_height * latent_width
    bits_per_image = token_count * (base_bits + residual_bits)
    raw_bits_per_pixel = image_channels * 8
    raw_bits_per_image = image_height * image_width * raw_bits_per_pixel
    result.update(
        {
            "latent_spatial_shape": [latent_height, latent_width],
            "latent_positions_per_image": token_count,
            "nominal_index_bits_per_image": bits_per_image,
            "nominal_index_bits_per_spatial_pixel": float(
                bits_per_image / (image_height * image_width)
            ),
            "raw_image_shape_chw": [image_channels, image_height, image_width],
            "raw_reference_bits_per_channel": 8,
            "raw_reference_bits_per_pixel": raw_bits_per_pixel,
            "raw_reference_bits_per_image": raw_bits_per_image,
            "nominal_raw_to_index_compression_ratio": (
                float(raw_bits_per_image / bits_per_image)
                if bits_per_image > 0
                else None
            ),
            "definition": (
                "index payload only; ceil(log2(K_B))+ceil(log2(K_R)) bits per "
                "latent position, compared with uncompressed 8-bit-per-channel input"
            ),
        }
    )
    return result


def _write_base_distance_csv(path: Path, distances: dict[str, Any]) -> None:
    fields = [
        "first_state_index",
        "first_state_id",
        "second_state_index",
        "second_state_id",
        "same_state",
        "count",
        "mean",
        "std",
        "min",
        "q25",
        "median",
        "q75",
        "max",
    ]
    rows = distances.get("by_state_pair")
    if not isinstance(rows, list):
        raise ValueError("base distance summary lacks by_state_pair rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _write_matrix_csv(
    path: Path,
    cross: dict[str, Any],
    state_labels: list[str],
    selection_monitored_offset: int,
) -> None:
    fields = [
        "send_state_index",
        "send_state_id",
        "receive_state_index",
        "receive_state_id",
        "same_state",
        "state_offset",
        "selection_monitored_offset",
        "image_count",
        "mse",
        "rmse",
        "psnr_db",
        "ssim_mean",
        "lpips_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for send_state, send_label in enumerate(state_labels):
            for receive_state, receive_label in enumerate(state_labels):
                offset = (receive_state - send_state) % len(state_labels)
                writer.writerow(
                    {
                        "send_state_index": send_state,
                        "send_state_id": send_label,
                        "receive_state_index": receive_state,
                        "receive_state_id": receive_label,
                        "same_state": receive_state == send_state,
                        "state_offset": offset,
                        "selection_monitored_offset": offset
                        == selection_monitored_offset,
                        "image_count": cross["sample_count"],
                        "mse": cross["mse_matrix"][send_state][receive_state],
                        "rmse": math.sqrt(
                            cross["mse_matrix"][send_state][receive_state]
                        ),
                        "psnr_db": cross["psnr_db_matrix"][send_state][receive_state],
                        "ssim_mean": cross["ssim_mean_matrix"][send_state][
                            receive_state
                        ],
                        "lpips_mean": (
                            cross["lpips_mean_matrix"][send_state][receive_state]
                            if "lpips_mean_matrix" in cross
                            else ""
                        ),
                    }
                )


def _write_physical_mismatch_pairs_csv(
    path: Path,
    grouping: dict[str, Any],
    cross: dict[str, Any],
) -> None:
    """Write one auditable row for each of the 90 ordered wrong-state pairs."""

    fields = [
        "send_state_index",
        "send_state_id",
        "send_pump_nm",
        "send_power_mw",
        "receive_state_index",
        "receive_state_id",
        "receive_pump_nm",
        "receive_power_mw",
        "exclusive_group",
        "cross_pump",
        "wrong_power",
        "mse",
        "rmse",
        "psnr_db",
        "ssim_mean",
        "lpips_mean",
    ]
    pairs: list[dict[str, Any]] = []
    for name in grouping["exclusive_group_order"]:
        pairs.extend(grouping["exclusive_groups"][name]["pairs"])
    if len(pairs) != 90 or len(
        {
            (pair["send_state_index"], pair["receive_state_index"])
            for pair in pairs
        }
    ) != 90:
        raise ValueError("physical mismatch pair table must contain 90 unique pairs")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in sorted(
            pairs,
            key=lambda item: (
                int(item["send_state_index"]),
                int(item["receive_state_index"]),
            ),
        ):
            send = int(pair["send_state_index"])
            receive = int(pair["receive_state_index"])
            mse = float(cross["mse_matrix"][send][receive])
            writer.writerow(
                {
                    **pair,
                    "mse": mse,
                    "rmse": math.sqrt(mse),
                    "psnr_db": cross["psnr_db_matrix"][send][receive],
                    "ssim_mean": cross["ssim_mean_matrix"][send][receive],
                    "lpips_mean": (
                        cross["lpips_mean_matrix"][send][receive]
                        if "lpips_mean_matrix" in cross
                        else ""
                    ),
                }
            )


def _write_metric_confidence_intervals_csv(
    path: Path,
    *,
    software: dict[str, Any],
    cross_state: dict[str, Any],
) -> None:
    fields = [
        "population",
        "state_index",
        "state_id",
        "metric",
        "estimate",
        "bootstrap_mean",
        "ci95_percentile_lower",
        "ci95_percentile_upper",
        "sampling_unit",
        "bootstrap_seed",
        "bootstrap_replicates",
    ]
    rows: list[dict[str, Any]] = []

    def add(
        population: str,
        summary: dict[str, Any],
        *,
        state_index: int | str = "",
        state_id: str = "",
    ) -> None:
        for metric, interval in summary["metrics"].items():
            rows.append(
                {
                    "population": population,
                    "state_index": state_index,
                    "state_id": state_id,
                    "metric": metric,
                    "estimate": interval["estimate"],
                    "bootstrap_mean": interval.get("bootstrap_mean", ""),
                    "ci95_percentile_lower": interval.get(
                        "ci95_percentile_lower", ""
                    ),
                    "ci95_percentile_upper": interval.get(
                        "ci95_percentile_upper", ""
                    ),
                    "sampling_unit": summary["sampling_unit"],
                    "bootstrap_seed": summary["seed"],
                    "bootstrap_replicates": summary["replicates"],
                }
            )

    add("software_overall", software)
    add("seeded_correct_pooled", cross_state["correct_pooled"])
    add("seeded_wrong_pooled", cross_state["wrong_pooled"])
    for state in cross_state["correct_by_state"]:
        add(
            "seeded_correct_state",
            state,
            state_index=int(state["state_index"]),
            state_id=str(state["state_id"]),
        )
    paired = cross_state["paired_wrong_against_correct"]
    for metric, interval in paired["comparisons"].items():
        rows.append(
            {
                "population": "seeded_wrong_vs_correct",
                "state_index": "",
                "state_id": "",
                "metric": metric,
                "estimate": interval["estimate"],
                "bootstrap_mean": interval.get("bootstrap_mean", ""),
                "ci95_percentile_lower": interval.get(
                    "ci95_percentile_lower", ""
                ),
                "ci95_percentile_upper": interval.get(
                    "ci95_percentile_upper", ""
                ),
                "sampling_unit": paired["sampling_unit"],
                "bootstrap_seed": paired["seed"],
                "bootstrap_replicates": paired["replicates"],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_physical_mismatch_summary_csv(
    path: Path, physical: dict[str, Any]
) -> None:
    fields = [
        "group_family",
        "group",
        "pair_count",
        "metric",
        "estimate",
        "bootstrap_mean",
        "ci95_percentile_lower",
        "ci95_percentile_upper",
        "sampling_unit",
        "bootstrap_seed",
        "bootstrap_replicates",
    ]
    rows: list[dict[str, Any]] = []
    for family_key, family_label in (
        ("exclusive_groups", "exclusive"),
        ("marginal_groups", "marginal_overlapping"),
    ):
        for name, group in physical[family_key].items():
            confidence = group["wrong_metric_confidence_intervals"]
            for metric, interval in confidence["metrics"].items():
                rows.append(
                    {
                        "group_family": family_label,
                        "group": name,
                        "pair_count": group["pair_count"],
                        "metric": metric,
                        "estimate": interval["estimate"],
                        "bootstrap_mean": interval.get("bootstrap_mean", ""),
                        "ci95_percentile_lower": interval.get(
                            "ci95_percentile_lower", ""
                        ),
                        "ci95_percentile_upper": interval.get(
                            "ci95_percentile_upper", ""
                        ),
                        "sampling_unit": confidence["sampling_unit"],
                        "bootstrap_seed": confidence["seed"],
                        "bootstrap_replicates": confidence["replicates"],
                    }
                )
            paired = group["paired_against_sender_correct"]
            for metric, interval in paired["comparisons"].items():
                rows.append(
                    {
                        "group_family": family_label,
                        "group": name,
                        "pair_count": group["pair_count"],
                        "metric": metric,
                        "estimate": interval["estimate"],
                        "bootstrap_mean": interval.get("bootstrap_mean", ""),
                        "ci95_percentile_lower": interval.get(
                            "ci95_percentile_lower", ""
                        ),
                        "ci95_percentile_upper": interval.get(
                            "ci95_percentile_upper", ""
                        ),
                        "sampling_unit": paired["sampling_unit"],
                        "bootstrap_seed": paired["seed"],
                        "bootstrap_replicates": paired["replicates"],
                    }
                )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _usage_csv_row(
    model: str,
    state_index: int | str,
    state_id: str,
    stage: str,
    usage: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "state_index": state_index,
        "state_id": state_id,
        "stage": stage,
        **{key: value for key, value in usage.items() if key != "counts"},
    }


def _write_usage_csv(
    path: Path,
    software_usage: dict[str, Any],
    seeded_usage: list[dict[str, Any]],
    pooled_residual_usage: dict[str, Any],
) -> None:
    rows = [
        _usage_csv_row("software", 0, "state_independent", "base", software_usage["base"]),
        _usage_csv_row(
            "software", 0, "state_independent", "residual", software_usage["residual"]
        ),
    ]
    for state in seeded_usage:
        rows.append(
            _usage_csv_row(
                "seeded", state["state_index"], state["state_id"], "base", state["base"]
            )
        )
        rows.append(
            _usage_csv_row(
                "seeded",
                state["state_index"],
                state["state_id"],
                "residual",
                state["residual"],
            )
        )
    rows.append(
        _usage_csv_row(
            "seeded", "pooled", "all_send_states", "residual", pooled_residual_usage
        )
    )
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_field(metric: dict[str, Any] | None, key: str) -> Any:
    return "" if metric is None else metric.get(key, "")


def _write_schedule_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "schedule",
        "time_shift_frames",
        "synchronized",
        "matched_frame_count",
        "unmatched_frame_count",
        "match_fraction",
        "overall_mse",
        "overall_rmse",
        "overall_psnr_db",
        "overall_ssim_mean",
        "overall_lpips_mean",
        "matched_only_mse",
        "matched_only_rmse",
        "matched_only_psnr_db",
        "matched_only_ssim_mean",
        "matched_only_lpips_mean",
        "unmatched_only_mse",
        "unmatched_only_rmse",
        "unmatched_only_psnr_db",
        "unmatched_only_ssim_mean",
        "unmatched_only_lpips_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "schedule": row["schedule"],
                    "time_shift_frames": row["time_shift_frames"],
                    "synchronized": row["synchronized"],
                    "matched_frame_count": row["matched_frame_count"],
                    "unmatched_frame_count": row["unmatched_frame_count"],
                    "match_fraction": row["match_fraction"],
                    "overall_mse": row["overall"]["mse"],
                    "overall_rmse": row["overall"]["rmse"],
                    "overall_psnr_db": row["overall"]["psnr_db"],
                    "overall_ssim_mean": row["overall"]["ssim_mean"],
                    "overall_lpips_mean": row["overall"].get("lpips_mean", ""),
                    "matched_only_mse": _metric_field(row["matched_only"], "mse"),
                    "matched_only_rmse": _metric_field(row["matched_only"], "rmse"),
                    "matched_only_psnr_db": _metric_field(
                        row["matched_only"], "psnr_db"
                    ),
                    "matched_only_ssim_mean": _metric_field(
                        row["matched_only"], "ssim_mean"
                    ),
                    "matched_only_lpips_mean": _metric_field(
                        row["matched_only"], "lpips_mean"
                    ),
                    "unmatched_only_mse": _metric_field(row["unmatched_only"], "mse"),
                    "unmatched_only_rmse": _metric_field(
                        row["unmatched_only"], "rmse"
                    ),
                    "unmatched_only_psnr_db": _metric_field(
                        row["unmatched_only"], "psnr_db"
                    ),
                    "unmatched_only_ssim_mean": _metric_field(
                        row["unmatched_only"], "ssim_mean"
                    ),
                    "unmatched_only_lpips_mean": _metric_field(
                        row["unmatched_only"], "lpips_mean"
                    ),
                }
            )


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _weight_provenance(config: dict[str, Any]) -> dict[str, Any]:
    pretrained = bool(config["model"].get("pretrained_encoder", False))
    name = str(config["model"].get("imagenet_weights", "none"))
    result: dict[str, Any] = {"pretrained_encoder": pretrained, "weight_enum": name}
    if not pretrained:
        return result
    if name != "IMAGENET1K_V1":
        result["cached_weight_file"] = None
        result["cached_weight_file_sha256"] = None
        return result
    weights = ResNet18_Weights.IMAGENET1K_V1
    cached = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    result.update(
        {
            "url": weights.url,
            "cached_weight_file": str(cached) if cached.is_file() else None,
            "cached_weight_file_sha256": _sha256_file(cached) if cached.is_file() else None,
        }
    )
    return result


def _residual_dominance_diagnostic(
    full: dict[str, Any],
    base_only: dict[str, Any],
    transmitted_residual_only: dict[str, Any],
    requantized_residual_only: dict[str, Any],
    quantization_geometry: dict[str, Any],
    pooled_residual_usage: dict[str, Any],
) -> dict[str, Any]:
    selected = quantization_geometry["selected_on_held_out_test"]["pooled"]

    def metric_projection(metric: dict[str, Any]) -> dict[str, float]:
        keys = ("mse", "rmse", "psnr_db", "ssim_mean", "lpips_mean")
        return {key: float(metric[key]) for key in keys if key in metric}

    reconstruction = {
        "component_metrics": {
            "full_B_plus_R": metric_projection(full),
            "base_only_B": metric_projection(base_only),
            "transmitted_residual_only_R_from_full_indices": metric_projection(
                transmitted_residual_only
            ),
            "requantized_residual_only_R_reselected_without_B": metric_projection(
                requantized_residual_only
            ),
        },
        "base_only_to_full_mse_ratio": float(
            base_only["mse"] / max(full["mse"], 1e-300)
        ),
        "transmitted_residual_only_to_full_mse_ratio": float(
            transmitted_residual_only["mse"] / max(full["mse"], 1e-300)
        ),
        "requantized_residual_only_to_full_mse_ratio": float(
            requantized_residual_only["mse"] / max(full["mse"], 1e-300)
        ),
        "full_minus_base_only_psnr_db": float(
            full["psnr_db"] - base_only["psnr_db"]
        ),
        "full_minus_transmitted_residual_only_psnr_db": float(
            full["psnr_db"] - transmitted_residual_only["psnr_db"]
        ),
        "full_minus_requantized_residual_only_psnr_db": float(
            full["psnr_db"] - requantized_residual_only["psnr_db"]
        ),
        "transmitted_residual_only_to_base_only_mse_ratio": float(
            transmitted_residual_only["mse"] / max(base_only["mse"], 1e-300)
        ),
        "transmitted_residual_only_to_base_only_rmse_ratio": float(
            transmitted_residual_only["rmse"] / max(base_only["rmse"], 1e-300)
        ),
        "transmitted_residual_only_minus_base_only_psnr_db": float(
            transmitted_residual_only["psnr_db"] - base_only["psnr_db"]
        ),
        "transmitted_residual_only_minus_base_only_ssim": float(
            transmitted_residual_only["ssim_mean"] - base_only["ssim_mean"]
        ),
    }
    if "lpips_mean" in full:
        for name, branch in (
            ("base_only", base_only),
            ("transmitted_residual_only", transmitted_residual_only),
            ("requantized_residual_only", requantized_residual_only),
        ):
            reconstruction[f"{name}_to_full_lpips_ratio"] = float(
                branch["lpips_mean"] / max(full["lpips_mean"], 1e-300)
            )
            reconstruction[f"{name}_minus_full_lpips"] = float(
                branch["lpips_mean"] - full["lpips_mean"]
            )
        reconstruction[
            "transmitted_residual_only_minus_base_only_lpips"
        ] = float(
            transmitted_residual_only["lpips_mean"] - base_only["lpips_mean"]
        )
    return {
        "status": "evidence_only_insufficient_for_causal_or_security_claim",
        "selected_vector_geometry": {
            "mean_residual_to_base_norm_ratio": selected["norms"][
                "selected_residual_to_base_norm_ratio"
            ]["mean"],
            "fraction_selected_residual_norm_greater_than_base": selected[
                "fraction_selected_residual_norm_greater_than_base"
            ],
        },
        "first_to_final_quantization": selected["quantization_error"],
        "shared_residual_usage": {
            key: value
            for key, value in pooled_residual_usage.items()
            if key != "counts"
        },
        "reconstruction_controls": reconstruction,
        "interpretation_boundary": (
            "Norm ratios and component-only reconstruction quality diagnose whether R "
            "can dominate the representation; they do not establish cryptographic "
            "security or causal attribution by themselves."
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    scope = evaluation_scope(args.smoke_max_batches)
    seeded_checkpoint_path = args.seeded_checkpoint.expanduser().resolve()
    software_checkpoint_path = args.software_checkpoint.expanduser().resolve()
    codebook_path = args.codebook.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for source in (
        seeded_checkpoint_path,
        software_checkpoint_path,
        codebook_path,
        manifest_path,
    ):
        if not source.is_file():
            raise FileNotFoundError(source)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    manifest = _read_json(manifest_path)
    state_labels = _state_labels(manifest)
    bank = load_codebook_bank(codebook_path)
    bank_array_sha256 = _array_sha256(bank)
    if manifest.get("bank_array_sha256") != bank_array_sha256:
        raise ValueError("manifest and loaded base-bank array hashes differ")

    seeded_checkpoint = torch.load(seeded_checkpoint_path, map_location="cpu")
    software_checkpoint = torch.load(software_checkpoint_path, map_location="cpu")
    seeded_expected_hash = seeded_checkpoint.get("base_bank_array_sha256")
    if seeded_expected_hash != bank_array_sha256:
        raise ValueError("seeded checkpoint and loaded base-bank array hashes differ")
    seeded_config = seeded_checkpoint["config"]
    software_config = software_checkpoint["config"]
    compatibility = _compatibility_report(seeded_config, software_config)
    seeded_model = _load_model(
        seeded_checkpoint, expected_quantizer_type="seeded_residual", bank=bank
    )
    software_model = _load_model(
        software_checkpoint, expected_quantizer_type="trainable_residual", bank=None
    )
    if seeded_model.quantizer.state_count != len(state_labels):
        raise ValueError("model state count does not match the manifest")
    seeded_model.to(device).eval()
    software_model.to(device).eval()

    dataset_config = seeded_config["dataset"]
    dataset_name = str(dataset_config["name"])
    dataset_root, dataset_path_provenance = resolve_dataset_root(
        str(dataset_config["root"]), args.dataset_root
    )
    use_lpips = _lpips_enabled(args.lpips, dataset_name)
    if use_lpips:
        perceptual_metric, lpips_provenance = _build_lpips_alex(device)
    else:
        perceptual_metric = None
        lpips_provenance = {
            "enabled": False,
            "policy": args.lpips,
            "dataset": dataset_name,
            "reason": (
                "disabled explicitly"
                if args.lpips == "off"
                else (
                    "dataset-aware auto policy enables LPIPS only for CIFAR-10 "
                    "and STL-10"
                )
            ),
        }
    lpips_provenance["policy"] = args.lpips
    lpips_provenance["dataset"] = dataset_name
    dataset = build_dataset(
        dataset_name,
        str(dataset_root),
        train=False,
        download=bool(dataset_config.get("download", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    example_image = dataset[0][0]
    image_shape = [int(value) for value in example_image.shape]
    pixels_per_image = int(np.prod(image_shape))

    software_raw = _evaluate_software(
        software_model,
        loader,
        device=device,
        max_batches=args.smoke_max_batches,
        perceptual_metric=perceptual_metric,
    )
    seeded_raw = _evaluate_seeded(
        seeded_model,
        loader,
        device=device,
        max_batches=args.smoke_max_batches,
        perceptual_metric=perceptual_metric,
        state_labels=state_labels,
    )
    sample_count = int(software_raw["mse"].shape[0])
    if seeded_raw["cross_mse"].shape[0] != sample_count:
        raise RuntimeError("software and seeded evaluation sample counts differ")
    if software_raw["dataset_fingerprint"] != seeded_raw["dataset_fingerprint"]:
        raise RuntimeError("software and seeded evaluations observed different test streams")
    if software_raw["latent_shape"] != seeded_raw["latent_shape"]:
        raise RuntimeError("software and seeded latent spatial shapes differ")
    if scope == "final" and sample_count != len(dataset):
        raise RuntimeError("formal evaluation did not process the complete held-out test set")

    selection_monitored_offset = int(
        seeded_config["training"].get("wrong_state_offset", 1)
    )
    wrong_state_weight = float(
        seeded_config["training"].get("wrong_state_weight", 0.0)
    )
    explicit_wrong_state_training = wrong_state_weight > 0.0
    cross_summary = summarize_cross_state(
        seeded_raw["cross_mse"],
        seeded_raw["cross_ssim"],
        state_labels=state_labels,
        selection_monitored_offset=selection_monitored_offset,
        pixels_per_image=pixels_per_image,
        per_image_lpips=seeded_raw["cross_lpips"],
    )
    software_usage = {
        "base": usage_summary(software_raw["base_counts"]),
        "residual": usage_summary(software_raw["residual_counts"]),
    }
    software_token_count = int(software_raw["base_counts"].sum())
    if int(software_raw["residual_counts"].sum()) != software_token_count:
        raise RuntimeError("software base and residual token counts differ")
    seeded_usage = []
    for state, label in enumerate(state_labels):
        state_base_tokens = int(seeded_raw["base_counts"][state].sum())
        state_residual_tokens = int(seeded_raw["residual_counts"][state].sum())
        if (
            state_base_tokens != software_token_count
            or state_residual_tokens != state_base_tokens
        ):
            raise RuntimeError(
                f"state {state} usage count does not match the common test token count"
            )
        seeded_usage.append(
            {
                "state_index": state,
                "state_id": label,
                "base": usage_summary(seeded_raw["base_counts"][state]),
                "residual": usage_summary(seeded_raw["residual_counts"][state]),
            }
        )
    pooled_residual_usage = usage_summary(seeded_raw["residual_counts"].sum(axis=0))

    fixed_schedule = fixed_state_schedule(sample_count, len(state_labels))
    pseudorandom_schedule = pseudorandom_balanced_schedule(
        sample_count, len(state_labels), args.schedule_seed
    )
    schedule_rows = summarize_schedule(
        seeded_raw["cross_mse"],
        seeded_raw["cross_ssim"],
        schedule_name="fixed",
        send_schedule=fixed_schedule,
        pixels_per_image=pixels_per_image,
        per_image_lpips=seeded_raw["cross_lpips"],
    ) + summarize_schedule(
        seeded_raw["cross_mse"],
        seeded_raw["cross_ssim"],
        schedule_name="pseudorandom",
        send_schedule=pseudorandom_schedule,
        pixels_per_image=pixels_per_image,
        per_image_lpips=seeded_raw["cross_lpips"],
    )

    state_indices = np.arange(len(state_labels))
    correct_mse_by_state = seeded_raw["cross_mse"][:, state_indices, state_indices]
    correct_ssim_by_state = seeded_raw["cross_ssim"][:, state_indices, state_indices]
    software_overall = metric_summary(
        software_raw["mse"],
        software_raw["ssim"],
        pixels_per_observation=pixels_per_image,
        per_observation_lpips=software_raw["lpips"],
    )
    software_confidence_intervals = image_metric_bootstrap(
        software_raw["mse"],
        software_raw["ssim"],
        per_image_lpips=software_raw["lpips"],
        seed=args.bootstrap_seed,
        replicates=args.bootstrap_replicates,
    )
    cross_confidence_intervals = cross_state_metric_confidence_intervals(
        seeded_raw["cross_mse"],
        seeded_raw["cross_ssim"],
        state_labels=state_labels,
        per_image_lpips=seeded_raw["cross_lpips"],
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    cross_summary["confidence_intervals"] = cross_confidence_intervals
    physical_grouping = physical_mismatch_groups_from_manifest(manifest)
    physical_mismatch_summary = summarize_physical_mismatch_groups(
        seeded_raw["cross_mse"],
        seeded_raw["cross_ssim"],
        grouping=physical_grouping,
        pixels_per_image=pixels_per_image,
        per_image_lpips=seeded_raw["cross_lpips"],
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    paired_bootstrap = paired_image_bootstrap(
        software_raw["mse"],
        software_raw["ssim"],
        correct_mse_by_state,
        correct_ssim_by_state,
        seed=args.bootstrap_seed,
        replicates=args.bootstrap_replicates,
    )
    branches = {
        "base_only": _summarize_state_branch(
            seeded_raw["base_only_mse"],
            seeded_raw["base_only_ssim"],
            state_labels=state_labels,
            pixels_per_image=pixels_per_image,
            lpips=seeded_raw["base_only_lpips"],
        ),
        "transmitted_residual_only": _summarize_state_branch(
            seeded_raw["transmitted_residual_mse"],
            seeded_raw["transmitted_residual_ssim"],
            state_labels=state_labels,
            pixels_per_image=pixels_per_image,
            lpips=seeded_raw["transmitted_residual_lpips"],
        ),
        "requantized_residual_only": metric_summary(
            seeded_raw["requantized_residual_mse"],
            seeded_raw["requantized_residual_ssim"],
            pixels_per_observation=pixels_per_image,
            per_observation_lpips=seeded_raw["requantized_residual_lpips"],
        ),
    }
    wrong_correct_summary = paired_correct_wrong_summary(
        seeded_raw["cross_mse"],
        per_image_lpips=seeded_raw["cross_lpips"],
        per_image_ssim=seeded_raw["cross_ssim"],
        seed=args.bootstrap_seed + 1,
        replicates=args.bootstrap_replicates,
    )
    residual_dominance = _residual_dominance_diagnostic(
        cross_summary["correct_pooled"],
        branches["base_only"]["pooled"],
        branches["transmitted_residual_only"]["pooled"],
        branches["requantized_residual_only"],
        seeded_raw["quantization_geometry"],
        pooled_residual_usage,
    )
    software_parameters = _model_parameter_summary(software_model)
    seeded_parameters = _model_parameter_summary(seeded_model)
    base_state_distances = _base_codebook_state_distances(bank, state_labels)
    software_nominal_rate = _nominal_bits(
        software_model.quantizer.base_codebook_size,
        software_model.quantizer.residual_codebook_size,
        latent_shape=software_raw["latent_shape"],
        image_shape=image_shape,
    )
    seeded_nominal_rate = _nominal_bits(
        seeded_model.quantizer.base_codebook_size,
        seeded_model.quantizer.residual_codebook_size,
        latent_shape=seeded_raw["latent_shape"],
        image_shape=image_shape,
    )
    fixed_sync = schedule_rows[0]["overall"]
    pseudorandom_sync = schedule_rows[len(state_labels)]["overall"]
    reconstruction_summary: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "scope": scope,
        "dataset": str(dataset_config["name"]),
        "expected_test_images": len(dataset),
        "processed_test_images": sample_count,
        "test_truncated": sample_count != len(dataset),
        "software": {
            "overall": software_overall,
            "confidence_intervals": software_confidence_intervals,
            "usage": software_usage,
            "model_parameters": software_parameters,
            "nominal_index_rate": software_nominal_rate,
        },
        "seeded": {
            "wrong_state_training": {
                "explicit_wrong_state_training": explicit_wrong_state_training,
                "wrong_state_weight": wrong_state_weight,
                "selection_monitored_offset": selection_monitored_offset,
                "offset_role": (
                    "explicit wrong-state loss and selection monitoring"
                    if explicit_wrong_state_training
                    else "validation/checkpoint-selection monitoring only"
                ),
            },
            "cross_state": cross_summary,
            "physical_mismatch_groups": physical_mismatch_summary,
            "usage_by_send_state": seeded_usage,
            "pooled_residual_usage": pooled_residual_usage,
            "branches": branches,
            "quantization_geometry": seeded_raw["quantization_geometry"],
            "base_codebook_corresponding_index_state_distances": base_state_distances,
            "residual_dominance_diagnostic": residual_dominance,
            "paired_wrong_vs_correct": wrong_correct_summary,
            "model_parameters": seeded_parameters,
            "nominal_index_rate": seeded_nominal_rate,
            "schedules": {
                "fixed": schedule_rows[: len(state_labels)],
                "pseudorandom": schedule_rows[len(state_labels) :],
                "synchronized_fixed_minus_pseudorandom_psnr_db": float(
                    fixed_sync["psnr_db"] - pseudorandom_sync["psnr_db"]
                ),
            },
        },
        "comparison": {
            "seeded_minus_software_psnr_db": float(
                cross_summary["correct_pooled"]["psnr_db"] - software_overall["psnr_db"]
            ),
            "seeded_minus_software_ssim": float(
                cross_summary["correct_pooled"]["ssim_mean"]
                - software_overall["ssim_mean"]
            ),
            "paired_image_bootstrap": paired_bootstrap,
        },
    }
    if "lpips_mean" in software_overall:
        reconstruction_summary["comparison"]["seeded_minus_software_lpips"] = float(
            cross_summary["correct_pooled"]["lpips_mean"]
            - software_overall["lpips_mean"]
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "reconstruction_summary.json"
    matrix_path = output_dir / "cross_state_matrix.csv"
    base_distance_path = output_dir / "base_codebook_state_distances.csv"
    usage_path = output_dir / "usage_by_state.csv"
    schedules_path = output_dir / "schedule_results.csv"
    metric_confidence_path = output_dir / "metric_confidence_intervals.csv"
    physical_pairs_path = output_dir / "physical_mismatch_pairs.csv"
    physical_summary_path = output_dir / "physical_mismatch_summary.csv"
    arrays_path = output_dir / "per_image_metrics.npz"
    write_json(reconstruction_summary, summary_path)
    _write_matrix_csv(
        matrix_path, cross_summary, state_labels, selection_monitored_offset
    )
    _write_base_distance_csv(base_distance_path, base_state_distances)
    _write_usage_csv(usage_path, software_usage, seeded_usage, pooled_residual_usage)
    _write_schedule_csv(schedules_path, schedule_rows)
    _write_metric_confidence_intervals_csv(
        metric_confidence_path,
        software=software_confidence_intervals,
        cross_state=cross_confidence_intervals,
    )
    _write_physical_mismatch_pairs_csv(
        physical_pairs_path, physical_grouping, cross_summary
    )
    _write_physical_mismatch_summary_csv(
        physical_summary_path, physical_mismatch_summary
    )
    per_image_arrays = {
        "software_mse": software_raw["mse"],
        "software_ssim": software_raw["ssim"],
        "seeded_cross_mse": seeded_raw["cross_mse"],
        "seeded_cross_ssim": seeded_raw["cross_ssim"],
        "seeded_base_only_mse": seeded_raw["base_only_mse"],
        "seeded_base_only_ssim": seeded_raw["base_only_ssim"],
        "seeded_transmitted_residual_only_mse": seeded_raw[
            "transmitted_residual_mse"
        ],
        "seeded_transmitted_residual_only_ssim": seeded_raw[
            "transmitted_residual_ssim"
        ],
        "seeded_requantized_residual_only_mse": seeded_raw[
            "requantized_residual_mse"
        ],
        "seeded_requantized_residual_only_ssim": seeded_raw[
            "requantized_residual_ssim"
        ],
        "fixed_schedule": fixed_schedule.astype("<i2"),
        "pseudorandom_schedule": pseudorandom_schedule.astype("<i2"),
    }
    if use_lpips:
        per_image_arrays.update(
            {
                "software_lpips": software_raw["lpips"],
                "seeded_cross_lpips": seeded_raw["cross_lpips"],
                "seeded_base_only_lpips": seeded_raw["base_only_lpips"],
                "seeded_transmitted_residual_only_lpips": seeded_raw[
                    "transmitted_residual_lpips"
                ],
                "seeded_requantized_residual_only_lpips": seeded_raw[
                    "requantized_residual_lpips"
                ],
            }
        )
    _atomic_save_npz(arrays_path, **per_image_arrays)

    project_root = Path(__file__).resolve().parents[3]
    source_hashes = {
        str(state["source_file"]): str(state["source_sha256"])
        for state in manifest["states"]
    }
    created_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_output: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "scope": scope,
        "created_utc": created_utc,
        "cli": sys.argv,
        "git": _git_metadata(project_root),
        "inputs": {
            "seeded_checkpoint": {
                "path": str(seeded_checkpoint_path),
                "sha256": _sha256_file(seeded_checkpoint_path),
                "epoch": int(seeded_checkpoint["epoch"]),
                "selection": _optional_json(
                    seeded_checkpoint_path.parent / "selected_checkpoint.json"
                ),
                "resolved_config_sha256": _json_sha256(seeded_config),
            },
            "software_checkpoint": {
                "path": str(software_checkpoint_path),
                "sha256": _sha256_file(software_checkpoint_path),
                "epoch": int(software_checkpoint["epoch"]),
                "selection": _optional_json(
                    software_checkpoint_path.parent / "selected_checkpoint.json"
                ),
                "resolved_config_sha256": _json_sha256(software_config),
            },
            "codebook": {
                "path": str(codebook_path),
                "file_sha256": _sha256_file(codebook_path),
                "array_sha256": bank_array_sha256,
                "shape": list(bank.shape),
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256_file(manifest_path),
                "manifest_schema": manifest.get("schema"),
                "generator_version": manifest.get("generator_version"),
                "source_workbook_sha256": source_hashes,
                "states": manifest["states"],
            },
            "imagenet_weights": _weight_provenance(seeded_config),
        },
        "compatibility": compatibility,
        "models": {
            "seeded": {
                "parameters": seeded_parameters,
                "nominal_index_rate": seeded_nominal_rate,
            },
            "software": {
                "parameters": software_parameters,
                "nominal_index_rate": software_nominal_rate,
            },
        },
        "wrong_state_training": {
            "explicit_wrong_state_training": explicit_wrong_state_training,
            "wrong_state_weight": wrong_state_weight,
            "selection_monitored_offset": selection_monitored_offset,
            "selection_unmonitored_offsets": [
                offset
                for offset in range(1, len(state_labels))
                if offset != selection_monitored_offset
            ],
        },
        "dataset": {
            "name": str(dataset_config["name"]),
            "root": str(dataset_root),
            "path_provenance": dataset_path_provenance,
            "expected_test_samples": len(dataset),
            "processed_test_samples": sample_count,
            "test_truncated": sample_count != len(dataset),
            "shuffle": False,
            "augmentation": "ToTensor only",
            **software_raw["dataset_fingerprint"],
        },
        "state_order": [
            {"index": index, "state_id": label}
            for index, label in enumerate(state_labels)
        ],
        "physical_mismatch_grouping": {
            "ordered_wrong_pair_count": physical_grouping[
                "ordered_wrong_pair_count"
            ],
            "exclusive_pair_counts": {
                name: physical_grouping["exclusive_groups"][name]["pair_count"]
                for name in physical_grouping["exclusive_group_order"]
            },
            "marginal_pair_counts": {
                name: physical_grouping["marginal_groups"][name]["pair_count"]
                for name in physical_grouping["marginal_group_order"]
            },
            "definitions": physical_grouping["definitions"],
        },
        "schedules": {
            "algorithm": SCHEDULE_ALGORITHM,
            "schedule_seed": int(args.schedule_seed),
            "receiver_shift_definition": (
                "receiver[t] = sender_schedule[(t + shift) mod N]"
            ),
            "fixed_sha256": schedule_sha256(fixed_schedule),
            "pseudorandom_sha256": schedule_sha256(pseudorandom_schedule),
            "fixed_state_counts": np.bincount(
                fixed_schedule, minlength=len(state_labels)
            ).tolist(),
            "pseudorandom_state_counts": np.bincount(
                pseudorandom_schedule, minlength=len(state_labels)
            ).tolist(),
        },
        "metrics": {
            "mse": "pooled squared-error sum divided by pooled pixel count",
            "rmse": "sqrt(pooled MSE); per-observation RMSE values are never averaged",
            "psnr": "10*log10(1/MSE); PSNR values are never averaged",
            "ssim": SSIM_SPEC,
            "lpips": lpips_provenance,
            "bootstrap": {
                "seed": int(args.bootstrap_seed),
                "replicates": int(args.bootstrap_replicates),
                "sampling_unit": "test image",
                "metric_interval_reduction": (
                    "average supplied states/state pairs within each image, then "
                    "resample test-image indices"
                ),
                "wrong_vs_correct_seed": int(args.bootstrap_seed + 1),
                "wrong_vs_correct_reduction": (
                    "mean correct-state and wrong-state-pair metrics within each "
                    "image before resampling images"
                ),
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "evaluation_dtype": "float32 model inference; float64 metric aggregation",
            "amp": False,
            "model_eval_mode": True,
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "smoke_max_batches": args.smoke_max_batches,
        },
        "outputs": {
            path.name: _sha256_file(path)
            for path in (
                summary_path,
                matrix_path,
                base_distance_path,
                usage_path,
                schedules_path,
                metric_confidence_path,
                physical_pairs_path,
                physical_summary_path,
                arrays_path,
            )
        },
    }
    manifest_output_path = output_dir / "evaluation_manifest.json"
    write_json(manifest_output, manifest_output_path)
    print(json.dumps(reconstruction_summary["comparison"], ensure_ascii=False, indent=2))
    print(f"scope: {scope}")
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
