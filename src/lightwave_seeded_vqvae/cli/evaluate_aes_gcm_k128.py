"""Fixed registered paper samples only; no training, reselection or old writes.

Run with python -m lightwave_seeded_vqvae.cli.evaluate_aes_gcm_k128 --help.
Original batch-size-128 context is reconstructed, but only the 30 registered
images are transported/scored. No image arrays, key material or weights saved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from .evaluate_final import (_load_model, _build_lpips_alex, _lpips_per_image,
                             _sha256_file, _json_sha256, _array_sha256)
from ..data import build_dataset
from ..evaluation import image_mse, gaussian_ssim_per_image, SSIM_SPEC
from ..seed.codebook import load_codebook_bank
from ..transport.protocol_v2 import (SessionSender, ProtocolError, AuthenticationError,
                                     model_binding, pack_indices, unpack_indices, budget,
                                     HEADER_SIZE, NONCE_SIZE)


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def digest(array, dtype):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def metrics(model, reconstruction, target):
    mse = image_mse(reconstruction, target)
    return dict(mse=mse.cpu().numpy(), rmse=mse.double().sqrt().cpu().numpy(),
                ssim=gaussian_ssim_per_image(reconstruction, target).cpu().numpy(),
                lpips=_lpips_per_image(model, reconstruction, target).cpu().numpy())


def negative_probes(sender, wrong_receiver, packet):
    """Actual gate counter: no parser output or decoder callback on failure."""
    cases = [('wrong_key', packet, wrong_receiver)]
    for label, position in [('ciphertext', HEADER_SIZE + NONCE_SIZE), ('tag', -1),
                            ('authenticated_state', 7), ('nonce', HEADER_SIZE)]:
        modified = bytearray(packet)
        modified[position] ^= 1
        cases.append((label, bytes(modified), sender.receiver()))
    cases.append(('truncated', packet[:20], sender.receiver()))
    outcomes = {}
    for label, candidate, receiver in cases:
        decoder_invocations = 0
        def decode_after_auth():
            nonlocal decoder_invocations
            receiver.open(candidate)
            decoder_invocations += 1
        try:
            decode_after_auth()
        except ProtocolError as error:
            if label != 'truncated' and not isinstance(error, AuthenticationError):
                raise RuntimeError(f'{label} was not an authentication rejection') from error
            outcomes[label] = type(error).__name__
        else:
            raise RuntimeError(f'{label} was accepted')
        if decoder_invocations:
            raise RuntimeError('decoder gate crossed on rejection')
    return outcomes


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'codebook', 'evaluation-manifest', 'registry',
                 'dataset-root', 'output-dir']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    return p


@torch.inference_mode()
def main(argv=None):
    args = parser().parse_args(argv)
    started = time.perf_counter()
    paths = {k: v.resolve() for k, v in vars(args).items() if isinstance(v, Path)}
    manifest = json.loads(paths['evaluation_manifest'].read_text(encoding='utf-8'))
    registry = json.loads(paths['registry'].read_text(encoding='utf-8'))
    name = manifest['dataset']['name']
    if name not in ('mnist', 'stl10'):
        raise ValueError('validation is restricted to MNIST/STL-10')
    if manifest['schema'] != 'lightwave-seeded-vqvae-final-evaluation-v2':
        raise ValueError('SSIM v2 evaluation manifest required')
    cp_hash = _sha256_file(paths['checkpoint'])
    bank_hash = _sha256_file(paths['codebook'])
    if cp_hash != manifest['inputs']['seeded_checkpoint']['sha256']:
        raise ValueError('checkpoint fingerprint does not match final evaluation')
    if bank_hash != manifest['inputs']['codebook']['file_sha256']:
        raise ValueError('bank fingerprint does not match final evaluation')
    payload = torch.load(paths['checkpoint'], map_location='cpu')
    config = payload['config']
    if _json_sha256(config) != manifest['inputs']['seeded_checkpoint']['resolved_config_sha256']:
        raise ValueError('resolved config fingerprint differs from final evaluation')
    mc = config['model']
    if (mc['base_codebook_size'], mc['residual_codebook_size'], mc['latent_dim']) != (128, 128, 64):
        raise ValueError('only D64/K128 final models are allowed')
    if config['dataset']['name'] != name or registry['dataset']['name'].lower().replace('-', '') != name:
        raise ValueError('dataset mismatch')
    bank = load_codebook_bank(paths['codebook'])
    if _array_sha256(bank) != manifest['inputs']['codebook']['array_sha256']:
        raise ValueError('frozen B tensor fingerprint mismatch')
    selected = []
    for entry in registry['classes']:
        for quantile in ('q25', 'q50', 'q75'):
            selected.append(dict(sample_id=int(entry['quantiles'][quantile]['sample_id']),
                class_id=int(entry['class_id']), quantile=quantile))
    if len(selected) != 30 or len({x['sample_id'] for x in selected}) != 30:
        raise ValueError('expected 30 distinct pre-registered paper samples')
    if {x['class_id'] for x in selected} != set(range(10)):
        raise ValueError('all ten classes are required')
    state_order = registry['state_order']
    if len(state_order) != 10 or [s['state_id'] for s in state_order] != [s['state_id'] for s in manifest['inputs']['codebook']['states']]:
        raise ValueError('state order differs from frozen B manifest')
    batch_size = int(manifest['runtime']['batch_size'])
    if batch_size != 128:
        raise ValueError('this validation preserves the published batch-size-128 context')
    dataset = build_dataset(name, paths['dataset_root'], train=False, download=False, split='test')
    if len(dataset) != manifest['dataset']['expected_test_samples']:
        raise ValueError('official test split length mismatch')
    out = paths['output_dir']
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    # Match the published evaluator's PyTorch 2.1 defaults. Enabling the global
    # deterministic switch changes MNIST's 28->32 LPIPS resize numerics; the
    # diagnostic run is preserved separately. Exact paired output is asserted.
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    model = _load_model(payload, expected_quantizer_type='seeded_residual', bank=bank).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    perceptual, lpips_spec = _build_lpips_alex(device)
    binding = model_binding(cp_hash, bank_hash)
    sender = SessionSender(model_id=binding, dataset_id={'mnist': 1, 'stl10': 2}[name])
    receiver = sender.receiver()
    wrong_receiver = SessionSender(model_id=binding, dataset_id={'mnist': 1, 'stl10': 2}[name]).receiver()
    shape = (7, 7) if name == 'mnist' else (24, 24)
    lookup = {x['sample_id']: x for x in selected}
    batch_starts = sorted({i // batch_size * batch_size for i in lookup})
    pair = registry['representative_selection']
    rows, failures, input_fingerprints = [], [], []
    packet_nonces = set()
    context_count = 0
    print(f'{name}: {len(selected)} registered samples, 10 send states, {len(batch_starts)} original batch contexts', flush=True)
    for batch_number, start in enumerate(batch_starts):
        ids = list(range(start, min(start + batch_size, len(dataset))))
        examples = [dataset[i] for i in ids]
        images = torch.stack([x[0] for x in examples]).to(device)
        context_count += len(ids)
        positions = [p for p, sid in enumerate(ids) if sid in lookup]
        for position in positions:
            sid = ids[position]
            if int(examples[position][1]) != lookup[sid]['class_id']:
                raise ValueError('registered sample class does not match input dataset')
            input_fingerprints.append(dict(**lookup[sid], image_float32_sha256=digest(examples[position][0].numpy(), '<f4')))
        latent = model.encoder(images)
        for state in range(10):
            states = torch.full((len(ids),), state, dtype=torch.long, device=device)
            quantized = model.quantizer(latent, states)
            base, residual = quantized.base_indices, quantized.residual_indices
            if tuple(base.shape[1:]) != shape:
                raise RuntimeError('unexpected latent shape')
            transported_base, transported_residual = base.clone(), residual.clone()
            evidence = {}
            for position in positions:
                b = base[position].cpu().numpy()
                r = residual[position].cpu().numpy()
                plaintext = pack_indices(b, r)
                direct_b, direct_r = unpack_indices(plaintext, shape)
                if not np.array_equal(direct_b, b) or not np.array_equal(direct_r, r):
                    raise RuntimeError('plain serialization changed indices')
                packet = sender.seal(b, r, sender_state=state)
                decoded = receiver.open(packet)
                same_b, same_r = np.array_equal(decoded.base_indices, b), np.array_equal(decoded.residual_indices, r)
                if not same_b or not same_r or decoded.sender_state != state:
                    raise RuntimeError('authenticated transport changed indices/state')
                if decoded.nonce in packet_nonces:
                    raise RuntimeError('nonce reuse detected')
                packet_nonces.add(decoded.nonce)
                if pack_indices(decoded.base_indices, decoded.residual_indices) != plaintext:
                    raise RuntimeError('recovered plaintext differs')
                transported_base[position] = torch.from_numpy(decoded.base_indices).to(device)
                transported_residual[position] = torch.from_numpy(decoded.residual_indices).to(device)
                probes = negative_probes(sender, wrong_receiver, packet)
                failures.append(dict(sample_id=ids[position], sender_state=state, results=probes, decoder_invocations=0))
                evidence[position] = dict(base_indices_equal=bool(same_b), residual_indices_equal=bool(same_r),
                    base_indices_sha256=digest(b, '<i8'), residual_indices_sha256=digest(r, '<i8'),
                    recovered_base_indices_sha256=digest(decoded.base_indices, '<i8'),
                    recovered_residual_indices_sha256=digest(decoded.residual_indices, '<i8'),
                    plaintext_sha256=hashlib.sha256(plaintext).hexdigest(), packet_bytes=len(packet),
                    aes_authentication='accepted', latent_height=shape[0], latent_width=shape[1])
            receive_states = [(state, 'matched')]
            if state == int(pair['send_state']):
                receive_states.append((int(pair['receiver_state']), 'registered_mismatch'))
            for receive_state, condition in receive_states:
                receive_ids = torch.full_like(states, receive_state)
                # Same existing final-evaluation path: q = B_receiver[i_B] + R[i_R].
                off = model.decode_indices(base, residual, receive_ids)
                on = model.decode_indices(transported_base, transported_residual, receive_ids)
                if not torch.equal(off, on):
                    repeat = model.decode_indices(base, residual, receive_ids)
                    write_json(out / 'nonzero_difference_diagnostic.json', dict(
                        state=state, receiver_state=receive_state,
                        max_abs=float((off-on).abs().max()), repeat_exact=torch.equal(off, repeat),
                        interpretation='stop; investigate inference determinism before claiming equivalence'))
                    raise RuntimeError('nonzero reconstruction difference; see diagnostic')
                off_metrics, on_metrics = metrics(perceptual, off, images), metrics(perceptual, on, images)
                for position in positions:
                    row = dict(dataset=name, **lookup[ids[position]], sender_state=state,
                        sender_state_id=state_order[state]['state_id'], receiver_state=receive_state,
                        receiver_state_id=state_order[receive_state]['state_id'], condition=condition,
                        **evidence[position], image_elementwise_equal=bool(torch.equal(off[position], on[position])),
                        image_max_abs_difference=float((off[position]-on[position]).abs().max()),
                        image_before_sha256=digest(off[position].cpu().numpy(), '<f4'),
                        image_after_sha256=digest(on[position].cpu().numpy(), '<f4'),
                        context_batch_start=start, context_batch_size=len(ids))
                    for metric in off_metrics:
                        a, b = float(off_metrics[metric][position]), float(on_metrics[metric][position])
                        if a != b or not np.isfinite([a,b]).all():
                            raise RuntimeError(f'non-identical or non-finite {metric}')
                        row[metric + '_before'] = a
                        row[metric + '_after'] = b
                        row[metric + '_delta'] = b - a
                    rows.append(row)
        print(f'{name}: context {batch_number+1}/{len(batch_starts)}; rows={len(rows)}; elapsed={time.perf_counter()-started:.1f}s', flush=True)
    if len(rows) != 330 or len(packet_nonces) != 300:
        raise RuntimeError('incomplete validation coverage')
    for path, expected in [(paths['checkpoint'], cp_hash), (paths['codebook'], bank_hash)]:
        if _sha256_file(path) != expected:
            raise RuntimeError('protected input changed during validation')
    rows.sort(key=lambda r: (r['sample_id'], r['sender_state'], r['condition']))
    with (out / 'per_sample_state.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / 'per_sample_state.jsonl').open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    write_json(out / 'authentication_rejections.json', failures)
    write_json(out / 'sample_registry_used.json', dict(source=str(paths['registry']), sha256=_sha256_file(paths['registry']),
               samples=selected, input_fingerprints=input_fingerprints, registered_mismatch=pair))
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {str(p.relative_to(source_root)): _sha256_file(p) for p in source_root.rglob('*.py')}
    result = dict(schema='aes-gcm-k128-fixed-paper-samples-v2', dataset=name, scope='fixed registered samples only',
        full_test_set_aes_test=False, physical_optical_link=False, sample_count=30, matched_cases=300,
        registered_mismatch_cases=30, send_states=list(range(10)), packet_count=300, unique_nonces=300,
        rejected_packets=len(failures)*6, rejected_packet_decoder_invocations=0,
        all_indices_exact=True, all_images_exact=True, max_image_abs_difference=0.0,
        maximum_absolute_metric_deltas={k: max(abs(r[k+'_delta']) for r in rows) for k in off_metrics},
        transport=budget(shape), algorithm='AES-256-GCM', aes_key_generation='AESGCM.generate_key(256); OS CSPRNG; fresh per session; memory only; independent of optical seeds',
        nonce_generation='random fixed 32-bit session prefix || uint64be invocation counter; one sender per fresh key',
        aad='entire 64-byte header', tag_bits=128, paths={k:str(v) for k,v in paths.items()},
        protected_inputs_unchanged=True, checkpoint_sha256=cp_hash, codebook_sha256=bank_hash,
        canonical_config_sha256=_json_sha256(config), source_sha256=source_hashes,
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
            packages={k:version(k) for k in ['torch','torchvision','numpy','cryptography','lpips']},
            cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), device=str(device),
            device_name=torch.cuda.get_device_name(device), model_eval=True, amp=False, dtype='float32',
            deterministic_algorithms=False, cudnn_deterministic=False, cudnn_benchmark=False,
            matmul_tf32=False, cudnn_tf32=True),
        context=dict(batch_size=batch_size, context_batches=len(batch_starts), encoded_context_images=context_count,
            transported_unique_images=30, note='unselected context entries only preserve the original batch execution; not counted as AES-tested samples'),
        metrics=dict(ssim=SSIM_SPEC, lpips=lpips_spec, rmse='sqrt(per-image MSE), reported per observation'),
        elapsed_seconds=time.perf_counter()-started)
    write_json(out / 'summary.json', result)
    print(json.dumps({k:result[k] for k in ['dataset','matched_cases','registered_mismatch_cases','all_images_exact','max_image_abs_difference','elapsed_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
