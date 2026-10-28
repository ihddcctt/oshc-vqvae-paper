"""Publication payload checks that need no task weights or downloaded images."""
import importlib.util
import json
from pathlib import Path

from lightwave_seeded_vqvae.seed.codebook import build_codebook_bank
from lightwave_seeded_vqvae.seed.spectrum import load_optical_states

ROOT = Path(__file__).resolve().parents[1]


def test_public_spectra_regenerate_all_reference_seeds_and_codewords():
    folder = ROOT / 'data/optical'
    states = load_optical_states(folder / '20260907_405nm.xlsx', folder / '20260907_640nm.xlsx')
    bank, regenerated = build_codebook_bank(states, codebook_size=128, latent_dim=64,
                                           experiment_id='20260907_ZHY')
    reference = json.loads((folder / 'lightwave_bank_d64_k128.manifest.json').read_text(encoding='utf-8'))
    assert bank.shape == (10, 128, 64)
    assert regenerated['bank_array_sha256'] == reference['bank_array_sha256']
    for actual, expected in zip(regenerated['states'], reference['states']):
        for key in ('canonical_record', 'seed_sha256', 'codebook_sha256', 'source_sha256'):
            assert actual[key] == expected[key]


def test_saved_final_transport_evidence_matches_full_test_observations():
    spec = importlib.util.spec_from_file_location('public_aes_verifier', ROOT / 'scripts/verify_aes_results.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for dataset in ('mnist', 'stl10'):
        report = module.verify_dataset(ROOT, dataset)
        assert report['matched_cases'] == 300
        assert report['negative_probes'] == 1800
