from __future__ import annotations

from hashlib import sha256, shake_256
import json
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from typing import Sequence

import numpy as np

from .derive import canonical_state_record, derive_state_seed
from .spectrum import OpticalState


GENERATOR_VERSION = "shake256-uniform-center-l2-f32le-v1"
CODEBOOK_DOMAIN = b"LIGHTWAVE_SEEDED_VQVAE/BASE_CODEBOOK/V1\x00"


def _array_sha256(array: np.ndarray) -> str:
    little_endian = np.ascontiguousarray(array, dtype="<f4")
    return sha256(little_endian.tobytes(order="C")).hexdigest()


def generate_codebook(seed: bytes, codebook_size: int, latent_dim: int) -> np.ndarray:
    if len(seed) != 32:
        raise ValueError("seed must contain exactly 32 bytes")
    if codebook_size <= 0 or latent_dim <= 1:
        raise ValueError("codebook_size must be positive and latent_dim must exceed one")

    descriptor = (
        CODEBOOK_DOMAIN
        + GENERATOR_VERSION.encode("ascii")
        + b"\x00"
        + struct.pack(">II", codebook_size, latent_dim)
        + seed
    )
    raw = shake_256(descriptor).digest(codebook_size * latent_dim * 2)
    unsigned = np.frombuffer(raw, dtype=">u2").astype(np.float64)
    values = ((unsigned + 0.5) / 65536.0) * 2.0 - 1.0
    values = values.reshape(codebook_size, latent_dim)
    values -= values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise RuntimeError("deterministic generator produced a degenerate row")
    values /= norms
    return np.ascontiguousarray(values, dtype="<f4")


def build_codebook_bank(
    states: Sequence[OpticalState],
    *,
    codebook_size: int,
    latent_dim: int,
    experiment_id: str = "20260907_ZHY",
) -> tuple[np.ndarray, dict[str, object]]:
    if len(states) != 10:
        raise ValueError(f"expected ten optical states, received {len(states)}")

    state_entries: list[dict[str, object]] = []
    codebooks: list[np.ndarray] = []
    for index, state in enumerate(states):
        seed = derive_state_seed(state, experiment_id=experiment_id)
        codebook = generate_codebook(seed, codebook_size, latent_dim)
        codebooks.append(codebook)
        state_entries.append(
            {
                "index": index,
                "state_id": state.state_id,
                "pump_nm": state.pump_nm,
                "power_mw": state.power_mw,
                "source_file": Path(state.source_path).name,
                "source_sha256": state.source_sha256,
                "source_row": state.source_row,
                "canonical_record": canonical_state_record(
                    state, experiment_id=experiment_id
                ),
                "seed_sha256": seed.hex(),
                "codebook_sha256": _array_sha256(codebook),
            }
        )

    bank = np.ascontiguousarray(np.stack(codebooks, axis=0), dtype="<f4")
    seed_set = {entry["seed_sha256"] for entry in state_entries}
    codebook_set = {entry["codebook_sha256"] for entry in state_entries}
    if len(seed_set) != 10 or len(codebook_set) != 10:
        raise RuntimeError("state seeds and codebooks must be mutually distinct")

    manifest: dict[str, object] = {
        "schema": "lightwave-seeded-vqvae-codebook-manifest-v1",
        "experiment_id": experiment_id,
        "generator_version": GENERATOR_VERSION,
        "dtype": "float32-little-endian",
        "shape": [10, codebook_size, latent_dim],
        "normalization": "row-centered then row-L2-normalized",
        "bank_array_sha256": _array_sha256(bank),
        "states": state_entries,
    }
    return bank, manifest


def save_codebook_bank(
    output_path: str | Path,
    bank: np.ndarray,
    manifest: dict[str, object],
) -> tuple[Path, Path]:
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() != ".npz":
        raise ValueError("output path must end in .npz")

    states = manifest["states"]
    if not isinstance(states, list):
        raise ValueError("manifest states must be a list")
    manifest_path = destination.with_suffix(".manifest.json")
    destination.parent.mkdir(parents=True, exist_ok=True)

    occupied = [
        path
        for path in (destination, manifest_path)
        if path.exists() or path.is_symlink()
    ]
    if occupied:
        names = ", ".join(str(path) for path in occupied)
        raise FileExistsError(f"refusing to overwrite existing codebook output: {names}")

    manifest_text = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    with TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.stem}.staging-"
    ) as staging_directory:
        staged_destination = Path(staging_directory) / destination.name
        staged_manifest = Path(staging_directory) / manifest_path.name
        np.savez_compressed(
            staged_destination,
            codebooks=np.ascontiguousarray(bank, dtype="<f4"),
            pump_nm=np.asarray(
                [entry["pump_nm"] for entry in states], dtype=np.int32
            ),
            power_mw=np.asarray(
                [entry["power_mw"] for entry in states], dtype=np.int32
            ),
        )
        staged_manifest.write_text(manifest_text, encoding="utf-8")

        installed: list[Path] = []
        try:
            # Hard-link installation is exclusive: unlike replace/rename, it cannot
            # overwrite a target that appeared after the preflight check.
            os.link(staged_destination, destination)
            installed.append(destination)
            os.link(staged_manifest, manifest_path)
            installed.append(manifest_path)
        except BaseException:
            for path in reversed(installed):
                path.unlink(missing_ok=True)
            raise

    return destination, manifest_path


def load_codebook_bank(path: str | Path) -> np.ndarray:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        bank = np.asarray(archive["codebooks"], dtype=np.float32)
    if bank.ndim != 3 or bank.shape[0] != 10:
        raise ValueError(f"expected codebook bank shape (10, K, D), received {bank.shape}")
    if not np.isfinite(bank).all():
        raise ValueError("codebook bank contains non-finite values")
    return np.ascontiguousarray(bank)
