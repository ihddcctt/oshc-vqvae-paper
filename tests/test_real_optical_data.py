from __future__ import annotations

import os
from pathlib import Path

from lightwave_seeded_vqvae.seed.spectrum import load_optical_states


DATA_DIRECTORY = Path(os.environ.get(
    "LIGHTWAVE_OPTIC_DATA_DIR",
    str(Path(__file__).resolve().parents[1] / "data" / "optical"),
))


def test_registered_20260907_workbooks() -> None:
    directory = Path(DATA_DIRECTORY)
    states = load_optical_states(
        directory / "20260907_405nm.xlsx",
        directory / "20260907_640nm.xlsx",
    )
    assert len(states) == 10
    assert [(state.pump_nm, state.power_mw) for state in states] == [
        (pump, power)
        for pump in (405, 640)
        for power in (0, 5, 10, 15, 20)
    ]
    assert str(states[0].lambda1_nm) == "1548.752"
    assert str(states[4].lambda2_nm) == "1549.709"
    assert str(states[5].extinction1_db) == "9.0548"
    assert str(states[9].extinction2_db) == "8.9882"

