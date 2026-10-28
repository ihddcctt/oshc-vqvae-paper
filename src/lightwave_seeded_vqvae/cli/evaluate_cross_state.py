from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import build_dataset, dataset_channels
from ..models.vqvae import SeededResidualVQVAE
from ..seed.codebook import load_codebook_bank
from ..training import psnr_from_mse, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the ten-by-ten codebook-state matrix")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--codebook",
        type=Path,
        help="Override a relocated base-bank artifact; its array hash must match the checkpoint",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output", type=Path)
    return parser


def _state_labels(codebook_path: Path) -> list[str]:
    with np.load(codebook_path, allow_pickle=False) as archive:
        pumps = archive["pump_nm"].tolist()
        powers = archive["power_mw"].tolist()
    return [f"{int(pump)}nm_{int(power)}mW" for pump, power in zip(pumps, powers)]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    if str(config["model"].get("quantizer_type", "seeded_residual")) != "seeded_residual":
        raise ValueError("cross-state evaluation requires a seeded_residual checkpoint")
    dataset_config = config["dataset"]
    model_config = config["model"]
    stored_bank_path = checkpoint.get("base_bank_path")
    if args.codebook is not None:
        bank_path = args.codebook.expanduser().resolve()
    elif stored_bank_path is not None:
        bank_path = Path(stored_bank_path).expanduser().resolve()
    else:
        raise ValueError("checkpoint does not identify a base-bank artifact")
    bank = load_codebook_bank(bank_path)
    bank_array_sha256 = hashlib.sha256(
        np.ascontiguousarray(bank, dtype="<f4").tobytes()
    ).hexdigest()
    expected_bank_sha256 = checkpoint.get("base_bank_array_sha256")
    if not expected_bank_sha256:
        raise ValueError("checkpoint does not contain a base-bank array hash")
    if bank_array_sha256 != expected_bank_sha256:
        raise ValueError(
            "base-bank hash mismatch: checkpoint expects "
            f"{expected_bank_sha256}, loaded artifact is {bank_array_sha256}"
        )
    labels = _state_labels(bank_path)

    channels = dataset_channels(str(dataset_config["name"]))
    model = SeededResidualVQVAE(
        bank,
        in_channels=channels,
        latent_dim=int(model_config["latent_dim"]),
        residual_codebook_size=int(model_config["residual_codebook_size"]),
        pretrained_encoder=False,
        imagenet_weights=str(model_config["imagenet_weights"]),
        normalize_latents=bool(model_config.get("normalize_latents", True)),
        loss_reference_dim=int(model_config.get("loss_reference_dim", 64)),
        decoder_blocks_per_stage=int(model_config.get("decoder_blocks_per_stage", 1)),
        decoder_blocks_by_stage=model_config.get("decoder_blocks_by_stage"),
        decoder_zero_init_extra_residual_blocks=model_config.get(
            "decoder_zero_init_extra_residual_blocks", False
        ),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    dataset = build_dataset(
        str(dataset_config["name"]),
        str(dataset_config["root"]),
        train=False,
        download=bool(dataset_config.get("download", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    state_count = model.quantizer.state_count
    squared_error = np.zeros((state_count, state_count), dtype=np.float64)
    element_count = np.zeros((state_count, state_count), dtype=np.int64)
    base_only_error = np.zeros(state_count, dtype=np.float64)
    transmitted_residual_only_error = np.zeros(state_count, dtype=np.float64)
    requantized_residual_only_error = np.zeros(state_count, dtype=np.float64)
    branch_element_count = np.zeros(state_count, dtype=np.int64)

    progress = tqdm(loader, desc="cross-state evaluation")
    for batch_index, (images, _) in enumerate(progress):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        images = images.to(device, non_blocking=True)
        elements = images.numel()
        for send_state in range(state_count):
            send_ids = torch.full(
                (images.shape[0],), send_state, dtype=torch.long, device=device
            )
            encoded = model(images, send_ids)
            base_indices = encoded.quantizer.base_indices
            residual_indices = encoded.quantizer.residual_indices
            for receive_state in range(state_count):
                if receive_state == send_state:
                    reconstruction = encoded.reconstruction
                else:
                    receive_ids = torch.full_like(send_ids, receive_state)
                    reconstruction = model.decode_indices(
                        base_indices, residual_indices, receive_ids
                    )
                squared_error[send_state, receive_state] += float(
                    F.mse_loss(reconstruction, images, reduction="sum")
                )
                element_count[send_state, receive_state] += elements

            base_only = model.decode_indices(
                base_indices,
                residual_indices,
                send_ids,
                include_base=True,
                include_residual=False,
            )
            transmitted_residual_only = model.decode_indices(
                base_indices,
                residual_indices,
                send_ids,
                include_base=False,
                include_residual=True,
            )
            requantized_residual_only, _ = model.reconstruct_residual_only(
                encoded.encoder_latent
            )
            base_only_error[send_state] += float(
                F.mse_loss(base_only, images, reduction="sum")
            )
            transmitted_residual_only_error[send_state] += float(
                F.mse_loss(transmitted_residual_only, images, reduction="sum")
            )
            requantized_residual_only_error[send_state] += float(
                F.mse_loss(requantized_residual_only, images, reduction="sum")
            )
            branch_element_count[send_state] += elements

    if not element_count.all():
        raise RuntimeError("evaluation processed no samples for one or more state pairs")
    mse_matrix = squared_error / element_count
    psnr_matrix = np.vectorize(psnr_from_mse)(mse_matrix)
    correct_mse = [float(mse_matrix[index, index]) for index in range(state_count)]
    wrong_mse = [
        float(mse_matrix[row, column])
        for row in range(state_count)
        for column in range(state_count)
        if row != column
    ]
    base_only_mse = (base_only_error / branch_element_count).tolist()
    transmitted_residual_only_mse = (
        transmitted_residual_only_error / branch_element_count
    ).tolist()
    requantized_residual_only_mse = (
        requantized_residual_only_error / branch_element_count
    ).tolist()
    trained_offset = int(config["training"].get("wrong_state_offset", 1))
    trained_offset_mse = [
        float(mse_matrix[state, (state + trained_offset) % state_count])
        for state in range(state_count)
    ]
    unseen_wrong_mse = [
        float(mse_matrix[row, column])
        for row in range(state_count)
        for column in range(state_count)
        if column != row and column != (row + trained_offset) % state_count
    ]
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "base_bank_path": str(bank_path),
        "base_bank_array_sha256": bank_array_sha256,
        "state_labels": labels,
        "mse_matrix": mse_matrix.tolist(),
        "psnr_db_matrix": psnr_matrix.tolist(),
        "summary": {
            "correct_mean_mse": _mean(correct_mse),
            "correct_mean_psnr_db": _mean([psnr_from_mse(value) for value in correct_mse]),
            "wrong_mean_mse": _mean(wrong_mse),
            "wrong_mean_psnr_db": _mean([psnr_from_mse(value) for value in wrong_mse]),
            "wrong_to_correct_mse_ratio": _mean(wrong_mse) / _mean(correct_mse),
            "base_only_mean_mse": _mean(base_only_mse),
            "base_only_mean_psnr_db": _mean(
                [psnr_from_mse(value) for value in base_only_mse]
            ),
            "trained_offset": trained_offset,
            "trained_offset_wrong_mean_mse": _mean(trained_offset_mse),
            "trained_offset_wrong_mean_psnr_db": _mean(
                [psnr_from_mse(value) for value in trained_offset_mse]
            ),
            "unseen_wrong_mean_mse": _mean(unseen_wrong_mse),
            "unseen_wrong_mean_psnr_db": _mean(
                [psnr_from_mse(value) for value in unseen_wrong_mse]
            ),
            "transmitted_residual_only_mean_mse": _mean(
                transmitted_residual_only_mse
            ),
            "transmitted_residual_only_mean_psnr_db": _mean(
                [psnr_from_mse(value) for value in transmitted_residual_only_mse]
            ),
            "requantized_residual_only_mean_mse": _mean(
                requantized_residual_only_mse
            ),
            "requantized_residual_only_mean_psnr_db": _mean(
                [psnr_from_mse(value) for value in requantized_residual_only_mse]
            ),
        },
        "base_only_mse_by_state": base_only_mse,
        "transmitted_residual_only_mse_by_state": transmitted_residual_only_mse,
        "requantized_residual_only_mse_by_state": requantized_residual_only_mse,
    }
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else checkpoint_path.parent / "cross_state_evaluation.json"
    )
    write_json(result, output_path)
    print(json.dumps(result["summary"], indent=2))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
