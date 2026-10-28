"""Read-only verification of public AES evidence; no weights or private paths."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_dataset(root, dataset):
    directory = root / 'results/aes_gcm_k128' / dataset
    summary = load(directory / 'summary.json')
    registry = load(directory / 'sample_registry_used.json')
    rows = [json.loads(line) for line in (directory / 'per_sample_state.jsonl').read_text(encoding='utf-8').splitlines()]
    with (directory / 'per_sample_state.csv').open(encoding='utf-8', newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(rows) == len(csv_rows) == 330, f'{dataset}: row count')
    ids = {item['sample_id'] for item in registry['samples']}
    require(len(ids) == 30, f'{dataset}: sample count')
    matched = [r for r in rows if r['condition'] == 'matched']
    wrong = [r for r in rows if r['condition'] == 'registered_mismatch']
    expected = {(sample, state) for sample in ids for state in range(10)}
    require(len(matched) == 300 and {(r['sample_id'], r['sender_state']) for r in matched} == expected,
            f'{dataset}: matched coverage')
    require(all(r['sender_state'] == r['receiver_state'] for r in matched), f'{dataset}: diagonal')
    require(len(wrong) == 30 and {r['sample_id'] for r in wrong} == ids, f'{dataset}: mismatch coverage')
    pair = registry['registered_mismatch']
    require(all((r['sender_state'], r['receiver_state']) == (pair['send_state'], pair['receiver_state'])
                and r['sender_state'] != r['receiver_state'] for r in wrong), f'{dataset}: fixed mismatch')
    lookup = {(r['sample_id'], r['sender_state']): r for r in matched}
    for row in wrong:
        reference = lookup[row['sample_id'], row['sender_state']]
        require(all(row[key] == reference[key] for key in ('base_indices_sha256', 'residual_indices_sha256')),
                f'{dataset}: mismatch changed transmitted indices')
    budget = (686, 2, 86, 178) if dataset == 'mnist' else (8064, 0, 1008, 1100)
    require(tuple(summary['transport'][key] for key in ('valid_index_bits', 'alignment_padding_bits',
            'plaintext_bytes', 'packet_bytes')) == budget, f'{dataset}: packet budget')
    for row, csv_row in zip(rows, csv_rows):
        require(all(str(value) == csv_row[key] for key, value in row.items()), f'{dataset}: CSV/JSONL discrepancy')
        require(row['base_indices_equal'] and row['residual_indices_equal'], f'{dataset}: unequal indices')
        require(row['base_indices_sha256'] == row['recovered_base_indices_sha256'] and
                row['residual_indices_sha256'] == row['recovered_residual_indices_sha256'], f'{dataset}: index hashes')
        require(row['image_elementwise_equal'] and row['image_max_abs_difference'] == 0 and
                row['image_before_sha256'] == row['image_after_sha256'], f'{dataset}: reconstruction mismatch')
        require(row['packet_bytes'] == budget[-1], f'{dataset}: per-row packet budget')
        for metric in ('mse', 'rmse', 'ssim', 'lpips'):
            require(np.isfinite(row[metric + '_before']) and
                    row[metric + '_before'] == row[metric + '_after'] and row[metric + '_delta'] == 0,
                    f'{dataset}: {metric} transport difference')
    rejected = load(directory / 'authentication_rejections.json')
    require(len(rejected) == 300 and {(r['sample_id'], r['sender_state']) for r in rejected} == expected,
            f'{dataset}: rejection coverage')
    rejection_types = dict(wrong_key='AuthenticationError', ciphertext='AuthenticationError',
                           tag='AuthenticationError', authenticated_state='AuthenticationError',
                           nonce='AuthenticationError', truncated='ProtocolError')
    require(all(r['decoder_invocations'] == 0 and r['results'] == rejection_types for r in rejected),
            f'{dataset}: failure did not reject before reconstruction')
    require(summary['matched_cases'] == 300 and summary['registered_mismatch_cases'] == 30 and
            summary['rejected_packets'] == 1800 and summary['all_indices_exact'] and summary['all_images_exact'],
            f'{dataset}: summary discrepancy')
    with np.load(root / f'results/recomputed/{dataset}/per_image_metrics.npz', allow_pickle=False) as previous:
        for metric in ('mse', 'ssim', 'lpips'):
            differences = [abs(r[metric + '_before'] - float(previous['seeded_cross_' + metric]
                [r['sample_id'], r['sender_state'], r['receiver_state']])) for r in rows]
            require(max(differences) == 0, f'{dataset}: differs from saved full-test {metric}')
    # Only renderers may receive publication-label edits. Computational paths
    # must match the sources recorded in the original transport evaluation.
    for name, expected_hash in summary['source_sha256'].items():
        if Path(name).name.startswith('plot_'):
            continue
        actual = hashlib.sha256((root / 'src/lightwave_seeded_vqvae' / name).read_bytes()).hexdigest()
        require(actual == expected_hash, f'{dataset}: computational source changed: {name}')
    return dict(dataset=dataset, unique_images=30, matched_cases=300, mismatch_cases=30,
                negative_probes=1800, exact_indices=True, exact_images=True, zero_metric_deltas=True,
                saved_full_test_observations_match=True)


def main():
    reports = [verify_dataset(ROOT, name) for name in ('mnist', 'stl10')]
    print(json.dumps(dict(ok=True, scope='saved evidence readback, not inference rerun',
                         datasets=reports), indent=2))


if __name__ == '__main__':
    main()
