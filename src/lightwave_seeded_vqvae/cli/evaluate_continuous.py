from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..data import build_dataset, dataset_channels
from ..evaluation import (
    SSIM_SPEC,
    distribution_summary,
    gaussian_ssim_per_image,
    image_mse,
    metric_summary,
)
from ..models.vqvae import ContinuousAutoencoder
from ..perceptual import load_lpips_alex_v01, lpips_alex_per_image
from ..training import write_json


EVALUATION_SCHEMA = "lightwave-seeded-vqvae-continuous-evaluation-v1"
EXPECTED_CIFAR10_TEST_IMAGES = 10_000


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
            "Evaluate a strictly reconstructed Continuous Autoencoder on the "
            "held-out CIFAR-10 test set"
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=8)
    parser.add_argument(
        "--lpips",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "LPIPS-Alex v0.1 policy. auto enables LPIPS for CIFAR-10; on/off "
            "provide explicit overrides."
        ),
    )
    parser.add_argument(
        "--smoke-max-batches",
        type=_positive_int,
        help="Evaluate only this many leading test batches and mark outputs as smoke.",
    )
    return parser


def evaluation_scope(smoke_max_batches: int | None) -> str:
    return "smoke" if smoke_max_batches is not None else "final"


def lpips_enabled(policy: str, dataset_name: str) -> bool:
    if policy not in {"auto", "on", "off"}:
        raise ValueError(f"unsupported LPIPS policy: {policy}")
    if policy == "on":
        return True
    if policy == "off":
        return False
    return dataset_name.strip().lower().replace("-", "") == "cifar10"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must contain an object: {path}")
    return payload


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _strict_model_arguments(config: dict[str, Any]) -> dict[str, Any]:
    dataset_config = _require_mapping(config.get("dataset"), "config.dataset")
    model_config = _require_mapping(config.get("model"), "config.model")
    dataset_name = str(dataset_config.get("name", "")).lower().replace("-", "")
    if dataset_name != "cifar10":
        raise ValueError(
            "continuous evaluator is registered for the held-out CIFAR-10 test set"
        )
    if model_config.get("quantizer_type") != "continuous":
        raise ValueError("checkpoint model.quantizer_type must be exactly 'continuous'")
    encoder_name = str(model_config.get("encoder_arch", "resnet18")).lower()
    if encoder_name not in {"resnet18", "small_resnet18"}:
        raise ValueError(f"unsupported continuous encoder architecture: {encoder_name}")
    if not isinstance(model_config.get("pretrained_encoder"), bool):
        raise ValueError("model.pretrained_encoder must be boolean")
    latent_dim = model_config.get("latent_dim")
    if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 1:
        raise ValueError("model.latent_dim must be an integer greater than one")
    weights_name = model_config.get("imagenet_weights")
    if not isinstance(weights_name, str) or not weights_name:
        raise ValueError("model.imagenet_weights must be a non-empty string")
    blocks_per_stage = model_config.get("decoder_blocks_per_stage", 1)
    if (
        isinstance(blocks_per_stage, bool)
        or not isinstance(blocks_per_stage, int)
        or blocks_per_stage <= 0
    ):
        raise ValueError("model.decoder_blocks_per_stage must be a positive integer")
    blocks_by_stage = model_config.get("decoder_blocks_by_stage")
    if blocks_by_stage is not None:
        if (
            not isinstance(blocks_by_stage, list)
            or len(blocks_by_stage) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in blocks_by_stage
            )
        ):
            raise ValueError(
                "model.decoder_blocks_by_stage must contain three positive integers"
            )
    return {
        "in_channels": dataset_channels(str(dataset_config["name"])),
        "latent_dim": latent_dim,
        # Checkpoint weights replace the initialization, so disabling the external
        # ImageNet download here does not alter the reconstructed state dict.
        "pretrained_encoder": False,
        "imagenet_weights": weights_name,
        "decoder_blocks_per_stage": blocks_per_stage,
        "decoder_blocks_by_stage": blocks_by_stage,
        "decoder_zero_init_extra_residual_blocks": model_config.get(
            "decoder_zero_init_extra_residual_blocks", False
        ),
    }


@dataclass(frozen=True)
class LoadedContinuousCheckpoint:
    model: ContinuousAutoencoder
    config: dict[str, Any]
    epoch: int
    checkpoint_path: Path
    checkpoint_sha256: str
    resolved_config_path: Path
    resolved_config_file_sha256: str
    resolved_config_payload_sha256: str
    selection: dict[str, Any] | None
    selection_path: Path | None
    selection_sha256: str | None


def load_continuous_checkpoint(path: str | Path) -> LoadedContinuousCheckpoint:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    resolved_config_path = checkpoint_path.parent / "config.resolved.json"
    if not resolved_config_path.is_file():
        raise FileNotFoundError(
            f"continuous checkpoint requires sibling config.resolved.json: "
            f"{resolved_config_path}"
        )

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    checkpoint = _require_mapping(checkpoint, "checkpoint")
    embedded_config = _require_mapping(checkpoint.get("config"), "checkpoint.config")
    file_config = _read_json_object(resolved_config_path)
    if embedded_config != file_config:
        raise ValueError(
            "checkpoint embedded configuration does not match config.resolved.json"
        )
    arguments = _strict_model_arguments(file_config)
    model = ContinuousAutoencoder(**arguments)
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint.model must contain a state dictionary")
    model.load_state_dict(state_dict, strict=True)

    epoch = checkpoint.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("checkpoint.epoch must be a positive integer")
    selection_path = checkpoint_path.parent / "selected_checkpoint.json"
    selection = (
        _read_json_object(selection_path) if selection_path.is_file() else None
    )
    return LoadedContinuousCheckpoint(
        model=model,
        config=file_config,
        epoch=epoch,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=_sha256_file(checkpoint_path),
        resolved_config_path=resolved_config_path.resolve(),
        resolved_config_file_sha256=_sha256_file(resolved_config_path),
        resolved_config_payload_sha256=_json_sha256(file_config),
        selection=selection,
        selection_path=selection_path.resolve() if selection is not None else None,
        selection_sha256=(
            _sha256_file(selection_path) if selection is not None else None
        ),
    )


def parameter_summary(model: ContinuousAutoencoder) -> dict[str, int]:
    def count(module: nn.Module) -> int:
        return int(sum(parameter.numel() for parameter in module.parameters()))

    return {
        "total": count(model),
        "encoder": count(model.encoder),
        "decoder": count(model.decoder),
        "trainable_at_load": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }


def _new_fingerprint() -> dict[str, Any]:
    return {"images": hashlib.sha256(), "labels": hashlib.sha256(), "count": 0}


def _update_fingerprint(
    accumulator: dict[str, Any], images: torch.Tensor, labels: torch.Tensor
) -> None:
    image_array = np.ascontiguousarray(images.numpy(), dtype="<f4")
    label_array = np.ascontiguousarray(labels.numpy(), dtype="<i8")
    accumulator["images"].update(image_array.tobytes())
    accumulator["labels"].update(label_array.tobytes())
    accumulator["count"] += int(images.shape[0])


def _finish_fingerprint(accumulator: dict[str, Any]) -> dict[str, Any]:
    image_hash = accumulator["images"].hexdigest()
    label_hash = accumulator["labels"].hexdigest()
    return {
        "processed_images_float32_sha256": image_hash,
        "processed_labels_int64_sha256": label_hash,
        "processed_sample_order_sha256": hashlib.sha256(
            bytes.fromhex(image_hash) + bytes.fromhex(label_hash)
        ).hexdigest(),
        "processed_sample_count": int(accumulator["count"]),
    }


def _validate_unit_interval(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain finite floating-point values")
    tolerance = 1e-6
    if float(tensor.min()) < -tolerance or float(tensor.max()) > 1.0 + tolerance:
        raise ValueError(f"{name} must be in the original [0,1] image space")


def evaluate_continuous_model(
    model: ContinuousAutoencoder,
    dataset: Dataset[Any],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    lpips_metric: nn.Module | None,
    smoke_max_batches: int | None,
    expected_latent_dim: int,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model = model.to(device).eval()
    if lpips_metric is not None:
        lpips_metric = lpips_metric.to(device).eval()

    mse_rows: list[np.ndarray] = []
    ssim_rows: list[np.ndarray] = []
    lpips_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    fingerprint = _new_fingerprint()
    latent_shape: tuple[int, int, int] | None = None
    input_shape: tuple[int, int, int] | None = None

    with torch.inference_mode():
        for batch_index, (images_cpu, labels_cpu) in enumerate(loader):
            if smoke_max_batches is not None and batch_index >= smoke_max_batches:
                break
            images_cpu = torch.as_tensor(images_cpu, dtype=torch.float32).cpu()
            labels_cpu = torch.as_tensor(labels_cpu, dtype=torch.int64).cpu()
            _validate_unit_interval(images_cpu, "held-out images")
            _update_fingerprint(fingerprint, images_cpu, labels_cpu)
            images = images_cpu.to(device, non_blocking=device.type == "cuda")
            output = model(images)
            reconstruction = output.reconstruction
            latent = output.encoder_latent
            if reconstruction.shape != images.shape:
                raise ValueError(
                    "continuous reconstruction shape does not match the input: "
                    f"{tuple(reconstruction.shape)} vs {tuple(images.shape)}"
                )
            if latent.ndim != 4 or latent.shape[1] != expected_latent_dim:
                raise ValueError(
                    "continuous latent does not match resolved D: "
                    f"{tuple(latent.shape)} vs D={expected_latent_dim}"
                )
            _validate_unit_interval(reconstruction, "continuous reconstruction")
            current_latent_shape = tuple(int(value) for value in latent.shape[1:])
            current_input_shape = tuple(int(value) for value in images.shape[1:])
            if latent_shape is None:
                latent_shape = current_latent_shape
                input_shape = current_input_shape
            elif latent_shape != current_latent_shape or input_shape != current_input_shape:
                raise ValueError("input or latent shape changed within evaluation")

            batch_mse = image_mse(reconstruction, images)
            batch_ssim = gaussian_ssim_per_image(reconstruction, images)
            mse_rows.append(batch_mse.detach().cpu().numpy().astype("<f8", copy=False))
            ssim_rows.append(batch_ssim.detach().cpu().numpy().astype("<f8", copy=False))
            label_rows.append(labels_cpu.numpy().astype("<i8", copy=False))
            if lpips_metric is not None:
                batch_lpips = lpips_alex_per_image(
                    reconstruction, images, lpips_metric
                )
                lpips_rows.append(
                    batch_lpips.detach().cpu().numpy().astype("<f8", copy=False)
                )

    if not mse_rows or latent_shape is None or input_shape is None:
        raise ValueError("continuous evaluation processed no images")
    mse = np.ascontiguousarray(np.concatenate(mse_rows), dtype="<f8")
    rmse = np.ascontiguousarray(np.sqrt(mse), dtype="<f8")
    ssim = np.ascontiguousarray(np.concatenate(ssim_rows), dtype="<f8")
    labels = np.ascontiguousarray(np.concatenate(label_rows), dtype="<i8")
    lpips = (
        np.ascontiguousarray(np.concatenate(lpips_rows), dtype="<f8")
        if lpips_metric is not None
        else None
    )
    if not (mse.shape == rmse.shape == ssim.shape == labels.shape):
        raise RuntimeError("continuous per-image arrays do not share one sample axis")
    if lpips is not None and lpips.shape != mse.shape:
        raise RuntimeError("continuous LPIPS array does not share the sample axis")
    pixels_per_image = int(np.prod(input_shape))
    overall = metric_summary(
        mse,
        ssim,
        pixels_per_observation=pixels_per_image,
        per_observation_lpips=lpips,
    )
    distributions: dict[str, Any] = {
        "mse": distribution_summary(mse),
        "rmse": distribution_summary(rmse),
        "ssim": distribution_summary(ssim),
        "lpips": distribution_summary(lpips) if lpips is not None else None,
    }
    return {
        "mse": mse,
        "rmse": rmse,
        "ssim": ssim,
        "lpips": lpips,
        "labels": labels,
        "sample_indices": np.arange(mse.size, dtype="<i8"),
        "overall": overall,
        "distributions": distributions,
        "input_shape": list(input_shape),
        "latent_shape": list(latent_shape),
        "dataset_fingerprint": _finish_fingerprint(fingerprint),
    }


def _git_metadata(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", str(project_root), "branch", "--show-current"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(project_root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return {"commit": commit, "branch": branch, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def _lpips_provenance(policy: str, enabled: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "policy": policy,
        "enabled": enabled,
        "metric": "LPIPS" if enabled else None,
        "network": "alex" if enabled else None,
        "version": "0.1" if enabled else None,
        "input_mapping": "x_lpips = 2*x_[0,1] - 1" if enabled else None,
    }
    if enabled:
        try:
            payload["package_version"] = importlib_metadata.version("lpips")
        except importlib_metadata.PackageNotFoundError:
            payload["package_version"] = None
    return payload


def _resolved_decoder_blocks(model: ContinuousAutoencoder) -> list[int]:
    return [int(value) for value in model.decoder.blocks_by_stage]


def main(argv: Sequence[str] | None = None) -> None:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(raw_argv)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"evaluation output directory already exists: {output_dir}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    loaded = load_continuous_checkpoint(args.checkpoint)
    dataset_config = _require_mapping(loaded.config.get("dataset"), "config.dataset")
    model_config = _require_mapping(loaded.config.get("model"), "config.model")
    dataset_name = str(dataset_config["name"])
    dataset = build_dataset(
        dataset_name,
        str(dataset_config["root"]),
        train=False,
        download=bool(dataset_config.get("download", False)),
    )
    scope = evaluation_scope(args.smoke_max_batches)
    if scope == "final" and len(dataset) != EXPECTED_CIFAR10_TEST_IMAGES:
        raise ValueError(
            "formal CIFAR-10 evaluation requires exactly 10,000 held-out images, "
            f"received {len(dataset)}"
        )
    use_lpips = lpips_enabled(args.lpips, dataset_name)
    lpips_metric = load_lpips_alex_v01(device=device) if use_lpips else None
    parameters = parameter_summary(loaded.model)
    evaluated = evaluate_continuous_model(
        loaded.model,
        dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lpips_metric=lpips_metric,
        smoke_max_batches=args.smoke_max_batches,
        expected_latent_dim=int(model_config["latent_dim"]),
    )
    processed_images = int(evaluated["mse"].size)
    if scope == "final" and processed_images != EXPECTED_CIFAR10_TEST_IMAGES:
        raise RuntimeError(
            "formal continuous evaluation did not process the full CIFAR-10 test set"
        )

    summary = {
        "schema": EVALUATION_SCHEMA,
        "scope": scope,
        "formal": scope == "final",
        "dataset": "cifar10",
        "expected_test_images": EXPECTED_CIFAR10_TEST_IMAGES,
        "available_test_images": len(dataset),
        "processed_test_images": processed_images,
        "test_truncated": processed_images < len(dataset),
        "continuous": {
            "overall": evaluated["overall"],
            "per_image_distribution": evaluated["distributions"],
            "transport_applicable": False,
            "role": "non-quantized reconstruction upper bound",
        },
        "model": {
            "architecture": "ContinuousAutoencoder",
            "encoder": "small-image ResNet18",
            "decoder": "ResidualImageDecoder",
            "input_shape_chw": evaluated["input_shape"],
            "latent_shape_dhw": evaluated["latent_shape"],
            "latent_dim": int(model_config["latent_dim"]),
            "decoder_blocks_by_stage": _resolved_decoder_blocks(loaded.model),
            "parameters": parameters,
        },
        "metric_space": {
            "image_range": [0.0, 1.0],
            "mse_rmse": "original RGB [0,1] space",
            "ssim": SSIM_SPEC,
            "lpips": _lpips_provenance(args.lpips, use_lpips),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "reconstruction_summary.json"
    metrics_path = output_dir / "per_image_metrics.npz"
    manifest_path = output_dir / "evaluation_manifest.json"
    write_json(summary, summary_path)
    metric_arrays = {
        "sample_indices": evaluated["sample_indices"],
        "labels": evaluated["labels"],
        "mse": evaluated["mse"],
        "rmse": evaluated["rmse"],
        "ssim": evaluated["ssim"],
    }
    if evaluated["lpips"] is not None:
        metric_arrays["lpips"] = evaluated["lpips"]
    np.savez_compressed(metrics_path, **metric_arrays)

    project_root = Path(__file__).resolve().parents[3]
    manifest = {
        "schema": EVALUATION_SCHEMA,
        "scope": scope,
        "formal": scope == "final",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cli": [str(Path(__file__).resolve()), *raw_argv],
        "git": _git_metadata(project_root),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "checkpoint": {
            "path": str(loaded.checkpoint_path),
            "sha256": loaded.checkpoint_sha256,
            "epoch": loaded.epoch,
            "resolved_config_path": str(loaded.resolved_config_path),
            "resolved_config_file_sha256": loaded.resolved_config_file_sha256,
            "resolved_config_payload_sha256": loaded.resolved_config_payload_sha256,
            "selection_path": (
                str(loaded.selection_path) if loaded.selection_path is not None else None
            ),
            "selection_sha256": loaded.selection_sha256,
            "selection": loaded.selection,
        },
        "dataset": {
            "name": "cifar10",
            "root": str(Path(str(dataset_config["root"])).expanduser().resolve()),
            "augmentation": "ToTensor only",
            "shuffle": False,
            "available_test_images": len(dataset),
            "expected_test_images": EXPECTED_CIFAR10_TEST_IMAGES,
            "processed_test_images": processed_images,
            "smoke_max_batches": args.smoke_max_batches,
            **evaluated["dataset_fingerprint"],
        },
        "model": summary["model"],
        "metrics": summary["metric_space"],
        "outputs": {
            "reconstruction_summary": {
                "path": str(summary_path),
                "sha256": _sha256_file(summary_path),
            },
            "per_image_metrics": {
                "path": str(metrics_path),
                "sha256": _sha256_file(metrics_path),
                "keys": sorted(metric_arrays),
            },
        },
    }
    write_json(manifest, manifest_path)
    print(json.dumps(summary["continuous"]["overall"], sort_keys=True))
    print(f"evaluation directory: {output_dir}")


if __name__ == "__main__":
    main()
