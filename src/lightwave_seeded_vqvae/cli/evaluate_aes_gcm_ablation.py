from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import build_dataset, dataset_channels
from ..models.vqvae import SeededResidualVQVAE
from ..seed.codebook import GENERATOR_VERSION, load_codebook_bank
from ..training import write_json
from ..transport import (
    AuthenticationError,
    GCM_TAG_SIZE,
    HEADER_SIZE,
    INDEX_BITS,
    PROTOCOL_VERSION,
    IndexPacketReceiver,
    IndexPacketSender,
    derive_model_id,
    pack_indices_6bit,
    unpack_indices_6bit,
)


EVALUATION_SCHEMA = "lightwave-seeded-vqvae-aes-gcm-ablation-v1"
TRANSPORT_GENERATOR_VERSION = 1
REGISTERED_PROBES_PER_CASE = 100
REGISTERED_PROBE_CASES = ("wrong_key", "aad_bit_flip", "ciphertext_bit_flip", "tag_bit_flip")
EXPECTED_FORMAL_TEST_SAMPLES = 10_000


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
        description="Held-out AES-GCM ablation for one seeded D64 checkpoint"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=8)
    parser.add_argument(
        "--smoke-max-batches",
        type=_positive_int,
        help=(
            "Process only this many test batches and label the result as smoke. "
            "Omit only for the registered 10,000-image formal run."
        ),
    )
    return parser


def evaluation_scope(smoke_max_batches: int | None) -> str:
    return "smoke" if smoke_max_batches is not None else "formal"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype="<f4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must contain an object: {path}")
    return payload


def frame_format_summary(
    *,
    latent_height: int,
    latent_width: int,
    base_codebook_size: int,
    residual_codebook_size: int,
) -> dict[str, Any]:
    if latent_height <= 0 or latent_width <= 0:
        raise ValueError("latent dimensions must be positive")
    if not 1 <= base_codebook_size <= 64 or not 1 <= residual_codebook_size <= 64:
        raise ValueError("six-bit transport requires codebook sizes in [1, 64]")
    token_count = latent_height * latent_width
    nominal_bits_per_stream = token_count * INDEX_BITS
    packed_bytes_per_stream = (nominal_bits_per_stream + 7) // 8
    padding_bits_per_stream = packed_bytes_per_stream * 8 - nominal_bits_per_stream
    plaintext_bytes = 2 * packed_bytes_per_stream
    packet_bytes = HEADER_SIZE + plaintext_bytes + GCM_TAG_SIZE
    return {
        "latent_shape": [latent_height, latent_width],
        "tokens_per_frame": token_count,
        "index_bits": INDEX_BITS,
        "nominal_index_bits_per_frame": 2 * nominal_bits_per_stream,
        "base_packed_bytes": packed_bytes_per_stream,
        "residual_packed_bytes": packed_bytes_per_stream,
        "padding_bits_per_stream": padding_bits_per_stream,
        "plaintext_bytes": plaintext_bytes,
        "aad_header_bytes": HEADER_SIZE,
        "ciphertext_bytes_excluding_tag": plaintext_bytes,
        "gcm_tag_bytes": GCM_TAG_SIZE,
        "authenticated_packet_bytes": packet_bytes,
        "authenticated_overhead_bytes": HEADER_SIZE + GCM_TAG_SIZE,
        "authenticated_overhead_fraction": float(
            (HEADER_SIZE + GCM_TAG_SIZE) / packet_bytes
        ),
    }


def timing_summary(total_ns: int, frame_count: int) -> dict[str, float | int]:
    if total_ns < 0 or frame_count <= 0:
        raise ValueError("timing totals must be non-negative for at least one frame")
    total_seconds = total_ns / 1_000_000_000.0
    return {
        "frame_count": int(frame_count),
        "total_nanoseconds": int(total_ns),
        "total_seconds": total_seconds,
        "mean_microseconds_per_frame": float(total_ns / frame_count / 1_000.0),
        "frames_per_second": (
            float(frame_count / total_seconds) if total_seconds > 0.0 else float("inf")
        ),
    }


def validate_artifact_compatibility(
    checkpoint: dict[str, Any],
    bank: np.ndarray,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint does not contain a resolved configuration")
    model_config = config.get("model")
    dataset_config = config.get("dataset")
    if not isinstance(model_config, dict) or not isinstance(dataset_config, dict):
        raise ValueError("checkpoint configuration lacks model or dataset settings")
    if str(model_config.get("quantizer_type", "seeded_residual")) != "seeded_residual":
        raise ValueError("AES-GCM ablation requires a seeded_residual checkpoint")

    latent_dim = int(model_config["latent_dim"])
    base_size = int(model_config["base_codebook_size"])
    residual_size = int(model_config["residual_codebook_size"])
    expected_shape = (10, base_size, latent_dim)
    if latent_dim != 64:
        raise ValueError(f"registered AES-GCM ablation requires D64, got D{latent_dim}")
    if base_size != 64 or residual_size != 64:
        raise ValueError("registered six-bit ablation requires K_b=K_r=64")
    if tuple(bank.shape) != expected_shape:
        raise ValueError(
            f"base bank shape mismatch: expected {expected_shape}, got {tuple(bank.shape)}"
        )

    bank_hash = array_sha256(bank)
    if checkpoint.get("base_bank_array_sha256") != bank_hash:
        raise ValueError("checkpoint and loaded base-bank array hashes differ")
    if manifest.get("bank_array_sha256") != bank_hash:
        raise ValueError("manifest and loaded base-bank array hashes differ")
    if manifest.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("manifest uses an unsupported codebook generator version")
    if manifest.get("shape") != list(expected_shape):
        raise ValueError("manifest shape does not match checkpoint and bank")
    states = manifest.get("states")
    if not isinstance(states, list) or len(states) != 10:
        raise ValueError("manifest must contain exactly ten states")
    for expected_index, state in enumerate(states):
        if not isinstance(state, dict) or int(state.get("index", -1)) != expected_index:
            raise ValueError("manifest states are not in canonical index order")
    if not isinstance(checkpoint.get("model"), dict):
        raise ValueError("checkpoint does not contain model weights")
    return {
        "config": config,
        "dataset_config": dataset_config,
        "model_config": model_config,
        "bank_array_sha256": bank_hash,
        "bank_shape": list(bank.shape),
        "base_codebook_size": base_size,
        "residual_codebook_size": residual_size,
        "latent_dim": latent_dim,
    }


def _receiver(
    *,
    master_key: bytes,
    model_id: bytes,
    session_id: bytes,
    base_codebook_size: int,
    residual_codebook_size: int,
    state_count: int,
) -> IndexPacketReceiver:
    return IndexPacketReceiver(
        transport_master_key=master_key,
        expected_model_id=model_id,
        expected_session_id=session_id,
        expected_generator_version=TRANSPORT_GENERATOR_VERSION,
        expected_base_codebook_size=base_codebook_size,
        expected_residual_codebook_size=residual_codebook_size,
        state_count=state_count,
    )


def _different_random_key(master_key: bytes) -> bytes:
    while True:
        candidate = os.urandom(32)
        if candidate != master_key:
            return candidate


def _one_bit_mutation(packet: bytes, bit_number: int, start: int, length: int) -> bytes:
    if length <= 0:
        raise ValueError("mutation region must be non-empty")
    relative_bit = bit_number % (length * 8)
    location = start + relative_bit // 8
    mutated = bytearray(packet)
    mutated[location] ^= 1 << (relative_bit % 8)
    return bytes(mutated)


def run_authentication_probes(
    packet: bytes,
    *,
    master_key: bytes,
    model_id: bytes,
    session_id: bytes,
    base_codebook_size: int,
    residual_codebook_size: int,
    state_count: int,
    probes_per_case: int = REGISTERED_PROBES_PER_CASE,
) -> dict[str, Any]:
    """Run fail-closed probes.  This function has no decoder call path."""

    if probes_per_case <= 0:
        raise ValueError("probes_per_case must be positive")
    ciphertext_bytes = len(packet) - HEADER_SIZE - GCM_TAG_SIZE
    if ciphertext_bytes <= 0:
        raise ValueError("probe packet has no ciphertext")
    results: dict[str, dict[str, int]] = {}
    started = time.perf_counter_ns()

    def require_authentication_failure(
        case: str, receiver: IndexPacketReceiver, candidate: bytes
    ) -> None:
        try:
            receiver.open(candidate)
        except AuthenticationError:
            return
        raise RuntimeError(f"security probe {case} was not rejected by authentication")

    for case in REGISTERED_PROBE_CASES:
        rejected = 0
        shared_receiver = None
        if case != "wrong_key":
            shared_receiver = _receiver(
                master_key=master_key,
                model_id=model_id,
                session_id=session_id,
                base_codebook_size=base_codebook_size,
                residual_codebook_size=residual_codebook_size,
                state_count=state_count,
            )
        for probe in range(probes_per_case):
            if case == "wrong_key":
                candidate = packet
                receiver = _receiver(
                    master_key=_different_random_key(master_key),
                    model_id=model_id,
                    session_id=session_id,
                    base_codebook_size=base_codebook_size,
                    residual_codebook_size=residual_codebook_size,
                    state_count=state_count,
                )
            elif case == "aad_bit_flip":
                # Mutate only model_id bytes.  Structural version parsing remains
                # valid, so rejection must be an AES-GCM authentication failure.
                candidate = _one_bit_mutation(packet, probe, 8, 16)
                assert shared_receiver is not None
                receiver = shared_receiver
            elif case == "ciphertext_bit_flip":
                candidate = _one_bit_mutation(
                    packet, probe, HEADER_SIZE, ciphertext_bytes
                )
                assert shared_receiver is not None
                receiver = shared_receiver
            else:
                candidate = _one_bit_mutation(
                    packet, probe, len(packet) - GCM_TAG_SIZE, GCM_TAG_SIZE
                )
                assert shared_receiver is not None
                receiver = shared_receiver
            require_authentication_failure(case, receiver, candidate)
            rejected += 1
        results[case] = {
            "attempted": probes_per_case,
            "authentication_rejections": rejected,
            "unexpected_acceptances": 0,
            "decoder_invocations": 0,
        }

    elapsed = time.perf_counter_ns() - started
    total = probes_per_case * len(REGISTERED_PROBE_CASES)
    return {
        "registered_probes_per_case": probes_per_case,
        "registered_cases": list(REGISTERED_PROBE_CASES),
        "one_bit_mutation": True,
        "cases": results,
        "total_attempted": total,
        "total_authentication_rejections": total,
        "total_unexpected_acceptances": 0,
        "total_decoder_invocations": 0,
        "timing": timing_summary(elapsed, total),
    }


@dataclass(frozen=True, slots=True)
class FrameRoundTrip:
    packet: bytes
    off_base_indices: tuple[int, ...]
    off_residual_indices: tuple[int, ...]
    on_base_indices: tuple[int, ...]
    on_residual_indices: tuple[int, ...]
    aes_off_ns: int
    aes_seal_ns: int
    aes_open_ns: int


def roundtrip_index_frame(
    base_indices: tuple[int, ...],
    residual_indices: tuple[int, ...],
    *,
    sender: IndexPacketSender,
    receiver: IndexPacketReceiver,
    state_id: int,
    latent_height: int,
    latent_width: int,
    stream_id: int,
    frame_seq: int,
) -> FrameRoundTrip:
    """Run AES-off and AES-on serialization for one already-quantized frame."""

    started = time.perf_counter_ns()
    base_packed = pack_indices_6bit(base_indices)
    residual_packed = pack_indices_6bit(residual_indices)
    off_base = unpack_indices_6bit(base_packed, len(base_indices))
    off_residual = unpack_indices_6bit(residual_packed, len(residual_indices))
    aes_off_ns = time.perf_counter_ns() - started

    started = time.perf_counter_ns()
    packet = sender.seal(
        base_indices,
        residual_indices,
        state_id=state_id,
        latent_height=latent_height,
        latent_width=latent_width,
        stream_id=stream_id,
        frame_seq=frame_seq,
    )
    aes_seal_ns = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    decoded = receiver.open(packet)
    aes_open_ns = time.perf_counter_ns() - started

    on_payload = pack_indices_6bit(decoded.base_indices) + pack_indices_6bit(
        decoded.residual_indices
    )
    if on_payload != base_packed + residual_packed:
        raise RuntimeError("AES on/off packed index bitstreams differ")
    return FrameRoundTrip(
        packet=packet,
        off_base_indices=off_base,
        off_residual_indices=off_residual,
        on_base_indices=decoded.base_indices,
        on_residual_indices=decoded.residual_indices,
        aes_off_ns=aes_off_ns,
        aes_seal_ns=aes_seal_ns,
        aes_open_ns=aes_open_ns,
    )


def _load_model(checkpoint: dict[str, Any], bank: np.ndarray) -> SeededResidualVQVAE:
    config = checkpoint["config"]
    dataset_config = config["dataset"]
    model_config = config["model"]
    model = SeededResidualVQVAE(
        bank,
        in_channels=dataset_channels(str(dataset_config["name"])),
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
    return model


def _git_metadata(project_root: Path) -> dict[str, Any]:
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


def _fresh_nonzero_bytes(length: int) -> bytes:
    while True:
        value = os.urandom(length)
        if any(value):
            return value


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate_transport(
    model: SeededResidualVQVAE,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None,
    master_key: bytes,
    session_id: bytes,
    model_id: bytes,
) -> dict[str, Any]:
    model.eval()
    state_count = model.quantizer.state_count
    base_size = model.quantizer.base_codebook_size
    residual_size = model.quantizer.residual_codebook_size
    stream_id = int.from_bytes(os.urandom(4), "big")
    sender = IndexPacketSender(
        transport_master_key=master_key,
        model_id=model_id,
        session_id=session_id,
        generator_version=TRANSPORT_GENERATOR_VERSION,
        base_codebook_size=base_size,
        residual_codebook_size=residual_size,
    )
    receiver = _receiver(
        master_key=master_key,
        model_id=model_id,
        session_id=session_id,
        base_codebook_size=base_size,
        residual_codebook_size=residual_size,
        state_count=state_count,
    )

    processed = 0
    state_counts = np.zeros(state_count, dtype=np.int64)
    image_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    schedule_digest = hashlib.sha256()
    aes_off_ns = 0
    aes_seal_ns = 0
    aes_open_ns = 0
    encoder_quantizer_ns = 0
    decoder_off_ns = 0
    decoder_on_ns = 0
    exact_index_frames = 0
    exact_reconstruction_frames = 0
    max_reconstruction_absolute_difference = 0.0
    format_report: dict[str, Any] | None = None
    representative_packet: bytes | None = None
    started_wall = time.perf_counter_ns()

    progress = tqdm(loader, desc="AES-GCM ablation", leave=False)
    for batch_index, (cpu_images, cpu_labels) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        image_array = np.ascontiguousarray(cpu_images.numpy(), dtype="<f4")
        label_array = np.ascontiguousarray(torch.as_tensor(cpu_labels).numpy(), dtype="<i8")
        image_digest.update(image_array.tobytes())
        label_digest.update(label_array.tobytes())

        images = cpu_images.to(device, non_blocking=True)
        batch_size = images.shape[0]
        state_values = np.arange(processed, processed + batch_size, dtype=np.int64) % state_count
        state_counts += np.bincount(state_values, minlength=state_count)
        schedule_digest.update(np.ascontiguousarray(state_values, dtype="<i2").tobytes())
        state_ids = torch.as_tensor(state_values, dtype=torch.long, device=device)

        _synchronize(device)
        started = time.perf_counter_ns()
        latent = model.encoder(images)
        quantized = model.quantizer(latent, state_ids)
        _synchronize(device)
        encoder_quantizer_ns += time.perf_counter_ns() - started

        height, width = quantized.base_indices.shape[-2:]
        current_format = frame_format_summary(
            latent_height=height,
            latent_width=width,
            base_codebook_size=base_size,
            residual_codebook_size=residual_size,
        )
        if format_report is None:
            format_report = current_format
        elif current_format != format_report:
            raise RuntimeError("latent/packet format changed within one evaluation")

        original_base = quantized.base_indices.reshape(batch_size, -1).cpu().tolist()
        original_residual = quantized.residual_indices.reshape(batch_size, -1).cpu().tolist()
        off_base_rows: list[tuple[int, ...]] = []
        off_residual_rows: list[tuple[int, ...]] = []
        on_base_rows: list[tuple[int, ...]] = []
        on_residual_rows: list[tuple[int, ...]] = []
        for local_index in range(batch_size):
            base_tuple = tuple(int(value) for value in original_base[local_index])
            residual_tuple = tuple(int(value) for value in original_residual[local_index])
            frame_sequence = processed + local_index
            roundtrip = roundtrip_index_frame(
                base_tuple,
                residual_tuple,
                sender=sender,
                receiver=receiver,
                state_id=int(state_values[local_index]),
                latent_height=height,
                latent_width=width,
                stream_id=stream_id,
                frame_seq=frame_sequence,
            )
            aes_off_ns += roundtrip.aes_off_ns
            aes_seal_ns += roundtrip.aes_seal_ns
            aes_open_ns += roundtrip.aes_open_ns
            if len(roundtrip.packet) != current_format["authenticated_packet_bytes"]:
                raise RuntimeError("authenticated packet length differs from registered format")
            if representative_packet is None:
                representative_packet = roundtrip.packet
            if not (
                roundtrip.off_base_indices
                == roundtrip.on_base_indices
                == base_tuple
                and roundtrip.off_residual_indices
                == roundtrip.on_residual_indices
                == residual_tuple
            ):
                raise RuntimeError("AES off/on indices are not bit-identical")
            exact_index_frames += 1
            off_base_rows.append(roundtrip.off_base_indices)
            off_residual_rows.append(roundtrip.off_residual_indices)
            on_base_rows.append(roundtrip.on_base_indices)
            on_residual_rows.append(roundtrip.on_residual_indices)

        off_base_tensor = torch.tensor(off_base_rows, dtype=torch.long, device=device).reshape(
            batch_size, height, width
        )
        off_residual_tensor = torch.tensor(
            off_residual_rows, dtype=torch.long, device=device
        ).reshape(batch_size, height, width)
        on_base_tensor = torch.tensor(on_base_rows, dtype=torch.long, device=device).reshape(
            batch_size, height, width
        )
        on_residual_tensor = torch.tensor(
            on_residual_rows, dtype=torch.long, device=device
        ).reshape(batch_size, height, width)

        _synchronize(device)
        started = time.perf_counter_ns()
        off_reconstruction = model.decode_indices(
            off_base_tensor, off_residual_tensor, state_ids
        )
        _synchronize(device)
        decoder_off_ns += time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        on_reconstruction = model.decode_indices(
            on_base_tensor, on_residual_tensor, state_ids
        )
        _synchronize(device)
        decoder_on_ns += time.perf_counter_ns() - started

        per_image_equal = (off_reconstruction == on_reconstruction).flatten(1).all(dim=1)
        if not bool(per_image_equal.all()):
            raise RuntimeError("same-decoder AES off/on reconstructions are not bit-identical")
        exact_reconstruction_frames += int(per_image_equal.sum())
        difference = float((off_reconstruction - on_reconstruction).abs().max())
        max_reconstruction_absolute_difference = max(
            max_reconstruction_absolute_difference, difference
        )
        processed += batch_size

    wall_ns = time.perf_counter_ns() - started_wall
    if processed == 0 or format_report is None or representative_packet is None:
        raise RuntimeError("AES-GCM evaluation processed no test images")
    probes = run_authentication_probes(
        representative_packet,
        master_key=master_key,
        model_id=model_id,
        session_id=session_id,
        base_codebook_size=base_size,
        residual_codebook_size=residual_size,
        state_count=state_count,
    )
    return {
        "processed_sample_count": processed,
        "state_assignment": {
            "algorithm": "state[t] = t mod 10",
            "state_counts": state_counts.tolist(),
            "schedule_int16le_sha256": schedule_digest.hexdigest(),
        },
        "dataset_fingerprint": {
            "processed_images_float32le_sha256": image_digest.hexdigest(),
            "processed_labels_int64le_sha256": label_digest.hexdigest(),
        },
        "frame_format": format_report,
        "equivalence": {
            "aes_off_vs_aes_on_exact_index_frames": exact_index_frames,
            "aes_off_vs_aes_on_all_indices_bit_identical": exact_index_frames == processed,
            "same_decoder_exact_reconstruction_frames": exact_reconstruction_frames,
            "same_decoder_all_reconstructions_bit_identical": (
                exact_reconstruction_frames == processed
            ),
            "max_reconstruction_absolute_difference": (
                max_reconstruction_absolute_difference
            ),
        },
        "transport_timing": {
            "clock": "time.perf_counter_ns",
            "aes_off_pack_and_unpack": timing_summary(aes_off_ns, processed),
            "aes_on_seal": timing_summary(aes_seal_ns, processed),
            "aes_on_open": timing_summary(aes_open_ns, processed),
            "aes_on_seal_and_open": timing_summary(aes_seal_ns + aes_open_ns, processed),
            "encoder_and_quantizer": timing_summary(encoder_quantizer_ns, processed),
            "same_decoder_aes_off": timing_summary(decoder_off_ns, processed),
            "same_decoder_aes_on": timing_summary(decoder_on_ns, processed),
            "whole_ablation_loop": timing_summary(wall_ns, processed),
        },
        "authentication_probes": probes,
    }


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    codebook_path = args.codebook.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite AES-GCM result: {output_path}")
    for path in (checkpoint_path, codebook_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint payload must be a mapping")
    bank = load_codebook_bank(codebook_path)
    manifest = read_json_object(manifest_path)
    validated = validate_artifact_compatibility(checkpoint, bank, manifest)
    bank_sha256 = validated["bank_array_sha256"]
    model_id = derive_model_id(
        checkpoint_sha256=checkpoint_sha256,
        base_bank_sha256=bank_sha256,
    )
    if not any(model_id):
        raise RuntimeError("derived model_id is unexpectedly all-zero")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    model = _load_model(checkpoint, bank).to(device).eval()
    dataset_config = validated["dataset_config"]
    dataset = build_dataset(
        str(dataset_config["name"]),
        str(dataset_config["root"]),
        train=False,
        download=bool(dataset_config.get("download", False)),
    )
    if len(dataset) != EXPECTED_FORMAL_TEST_SAMPLES:
        raise ValueError(
            "registered formal AES-GCM ablation requires the official "
            f"{EXPECTED_FORMAL_TEST_SAMPLES}-image test split; got {len(dataset)}"
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

    # These values are ephemeral cryptographic material and are intentionally
    # absent from the result object and output JSON.
    master_key = _fresh_nonzero_bytes(32)
    session_id = _fresh_nonzero_bytes(16)
    started_utc = datetime.now(timezone.utc).isoformat()
    result = evaluate_transport(
        model,
        loader,
        device=device,
        max_batches=args.smoke_max_batches,
        master_key=master_key,
        session_id=session_id,
        model_id=model_id,
    )
    scope = evaluation_scope(args.smoke_max_batches)
    if scope == "formal" and result["processed_sample_count"] != EXPECTED_FORMAL_TEST_SAMPLES:
        raise RuntimeError("formal AES-GCM ablation did not process all 10,000 test images")

    project_root = Path(__file__).resolve().parents[3]
    payload: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "scope": scope,
        "formal": scope == "formal",
        "smoke_max_batches": args.smoke_max_batches,
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "version": PROTOCOL_VERSION,
            "generator_version_id": TRANSPORT_GENERATOR_VERSION,
            "codebook_generator_version": GENERATOR_VERSION,
            "model_id_hex": model_id.hex(),
            "model_id_derivation": (
                "SHA256(domain || checkpoint_sha256_bytes || "
                "base_bank_array_sha256_bytes)[:16]"
            ),
            "nonce_layout": "uint32be(stream_id) || uint64be(frame_seq)",
            "session_key_kdf": "HKDF-SHA256 to 32-byte AES-256 key",
            "master_key_source": "ephemeral OS CSPRNG; value not recorded",
            "session_id_source": "ephemeral OS CSPRNG; value not recorded",
            "secret_or_session_material_recorded": False,
        },
        "provenance": {
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
                "epoch": int(checkpoint.get("epoch", -1)),
                "validation": checkpoint.get("validation"),
            },
            "base_bank": {
                "path": str(codebook_path),
                "array_sha256": bank_sha256,
                "shape": validated["bank_shape"],
            },
            "codebook_manifest": {
                "path": str(manifest_path),
                "file_sha256": sha256_file(manifest_path),
                "schema": manifest.get("schema"),
                "generator_version": manifest.get("generator_version"),
            },
            "resolved_config_sha256": canonical_json_sha256(validated["config"]),
            "dataset": {
                "name": str(dataset_config["name"]),
                "root": str(Path(str(dataset_config["root"])).expanduser().resolve()),
                "split": "official_test",
                "official_test_size": len(dataset),
                "shuffle": False,
            },
            "state_assignment": "state[t] = t mod 10",
            "git": _git_metadata(project_root),
            "command": sys.argv,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "results": result,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(payload, output_path)
    print(json.dumps({"scope": scope, **result["equivalence"]}, indent=2))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
