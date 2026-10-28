from __future__ import annotations

import argparse
from pathlib import Path

from ..seed.codebook import build_codebook_bank, save_codebook_bank
from ..seed.spectrum import load_optical_states


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ten deterministic chip-seeded codebooks")
    parser.add_argument("--input-405", required=True, type=Path)
    parser.add_argument("--input-640", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--latent-dim", required=True, type=int)
    parser.add_argument("--base-codebook-size", type=int, default=64)
    parser.add_argument("--experiment-id", default="20260907_ZHY")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    states = load_optical_states(args.input_405, args.input_640)
    bank, manifest = build_codebook_bank(
        states,
        codebook_size=args.base_codebook_size,
        latent_dim=args.latent_dim,
        experiment_id=args.experiment_id,
    )
    bank_path, manifest_path = save_codebook_bank(args.output, bank, manifest)
    print(f"saved {bank.shape} bank: {bank_path}")
    print(f"bank array sha256: {manifest['bank_array_sha256']}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()

