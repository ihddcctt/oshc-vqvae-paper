from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN
from hashlib import sha256
import json

from .spectrum import OpticalState


STATE_SCHEMA = "lightwave-seeded-vqvae-state-v1"
SEED_DOMAIN = b"LIGHTWAVE_SEEDED_VQVAE/STATE_SEED/V1\x00"
LAMBDA_SCALE = Decimal("1000")
EXTINCTION_SCALE = Decimal("10000")


def _fixed_integer(value: Decimal, scale: Decimal) -> int:
    return int((value * scale).to_integral_value(rounding=ROUND_HALF_EVEN))


def canonical_state_record(
    state: OpticalState,
    *,
    experiment_id: str = "20260907_ZHY",
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "peaks": [
            {
                "extinction_db_1e4": _fixed_integer(state.extinction1_db, EXTINCTION_SCALE),
                "peak": 1,
                "wavelength_pm": _fixed_integer(state.lambda1_nm, LAMBDA_SCALE),
            },
            {
                "extinction_db_1e4": _fixed_integer(state.extinction2_db, EXTINCTION_SCALE),
                "peak": 2,
                "wavelength_pm": _fixed_integer(state.lambda2_nm, LAMBDA_SCALE),
            },
        ],
        "power_mw": state.power_mw,
        "pump_nm": state.pump_nm,
        "schema": STATE_SCHEMA,
    }


def canonical_state_bytes(
    state: OpticalState,
    *,
    experiment_id: str = "20260907_ZHY",
) -> bytes:
    record = canonical_state_record(state, experiment_id=experiment_id)
    return json.dumps(
        record,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def derive_state_seed(
    state: OpticalState,
    *,
    experiment_id: str = "20260907_ZHY",
) -> bytes:
    return sha256(SEED_DOMAIN + canonical_state_bytes(state, experiment_id=experiment_id)).digest()


def derive_state_seed_hex(
    state: OpticalState,
    *,
    experiment_id: str = "20260907_ZHY",
) -> str:
    return derive_state_seed(state, experiment_id=experiment_id).hex()

