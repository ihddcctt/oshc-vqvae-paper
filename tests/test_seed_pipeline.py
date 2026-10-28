from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
import os
from pathlib import Path

import numpy as np
import pytest

from lightwave_seeded_vqvae.seed import codebook as codebook_module
from lightwave_seeded_vqvae.seed.codebook import (
    build_codebook_bank,
    generate_codebook,
    save_codebook_bank,
)
from lightwave_seeded_vqvae.seed.derive import canonical_state_bytes, derive_state_seed
from lightwave_seeded_vqvae.seed.spectrum import OpticalState


def make_state(pump: int, power: int, offset: int = 0) -> OpticalState:
    return OpticalState(
        pump_nm=pump,
        power_mw=power,
        lambda1_nm=Decimal("1548.752") + Decimal(offset) / Decimal("1000"),
        lambda2_nm=Decimal("1549.493") + Decimal(offset) / Decimal("1000"),
        extinction1_db=Decimal("8.1501") + Decimal(offset) / Decimal("10000"),
        extinction2_db=Decimal("9.0964") + Decimal(offset) / Decimal("10000"),
        source_path=f"{pump}.xlsx",
        source_sha256=f"{pump:064d}"[-64:],
        source_row=power // 5 + 2,
    )


def test_canonical_state_is_typed_and_deterministic() -> None:
    state = make_state(405, 0)
    first = canonical_state_bytes(state)
    second = canonical_state_bytes(state)
    assert first == second
    assert b'"wavelength_pm":1548752' in first
    assert b'"extinction_db_1e4":81501' in first
    assert derive_state_seed(state) == derive_state_seed(state)
    changed = replace(state, extinction1_db=Decimal("8.1502"))
    assert derive_state_seed(changed) != derive_state_seed(state)


def test_codebook_generator_is_deterministic_and_unit_norm() -> None:
    seed = bytes(range(32))
    first = generate_codebook(seed, codebook_size=64, latent_dim=64)
    second = generate_codebook(seed, codebook_size=64, latent_dim=64)
    assert first.dtype == np.dtype("float32")
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=2e-6)
    assert not np.array_equal(first, generate_codebook(bytes(reversed(seed)), 64, 64))


def test_ten_state_bank_contract() -> None:
    states = [
        make_state(pump, power, offset=index)
        for index, (pump, power) in enumerate(
            (pump, power) for pump in (405, 640) for power in (0, 5, 10, 15, 20)
        )
    ]
    bank, manifest = build_codebook_bank(states, codebook_size=8, latent_dim=16)
    assert bank.shape == (10, 8, 16)
    assert len({entry["seed_sha256"] for entry in manifest["states"]}) == 10
    assert len({entry["codebook_sha256"] for entry in manifest["states"]}) == 10


def make_bank_and_manifest() -> tuple[np.ndarray, dict[str, object]]:
    states = [
        make_state(pump, power, offset=index)
        for index, (pump, power) in enumerate(
            (pump, power) for pump in (405, 640) for power in (0, 5, 10, 15, 20)
        )
    ]
    return build_codebook_bank(states, codebook_size=8, latent_dim=16)


def test_save_codebook_bank_preserves_first_save_format(tmp_path: Path) -> None:
    bank, manifest = make_bank_and_manifest()
    output = tmp_path / "bank.npz"

    saved_bank, saved_manifest = save_codebook_bank(output, bank, manifest)

    assert saved_bank == output.resolve()
    assert saved_manifest == output.with_suffix(".manifest.json").resolve()
    with np.load(saved_bank, allow_pickle=False) as archive:
        assert np.array_equal(archive["codebooks"], bank)
        assert np.array_equal(archive["pump_nm"], [405] * 5 + [640] * 5)
        assert np.array_equal(archive["power_mw"], [0, 5, 10, 15, 20] * 2)
    assert saved_manifest.read_text(encoding="utf-8") == (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


@pytest.mark.parametrize("existing", ["bank", "manifest", "both"])
def test_save_codebook_bank_refuses_any_existing_target(
    tmp_path: Path, existing: str
) -> None:
    bank, manifest = make_bank_and_manifest()
    output = tmp_path / "bank.npz"
    manifest_path = output.with_suffix(".manifest.json")
    if existing in {"bank", "both"}:
        output.write_bytes(b"existing bank")
    if existing in {"manifest", "both"}:
        manifest_path.write_bytes(b"existing manifest")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_codebook_bank(output, bank, manifest)

    if existing in {"bank", "both"}:
        assert output.read_bytes() == b"existing bank"
    else:
        assert not output.exists()
    if existing in {"manifest", "both"}:
        assert manifest_path.read_bytes() == b"existing manifest"
    else:
        assert not manifest_path.exists()


def test_save_codebook_bank_rolls_back_if_second_install_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank, manifest = make_bank_and_manifest()
    output = tmp_path / "bank.npz"
    manifest_path = output.with_suffix(".manifest.json")
    original_link = os.link

    def fail_manifest_link(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == manifest_path.resolve():
            raise OSError("simulated manifest installation failure")
        original_link(source, destination)

    monkeypatch.setattr(codebook_module.os, "link", fail_manifest_link)
    with pytest.raises(OSError, match="simulated manifest installation failure"):
        save_codebook_bank(output, bank, manifest)

    assert not output.exists()
    assert not manifest_path.exists()
