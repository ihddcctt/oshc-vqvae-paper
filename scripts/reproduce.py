"""Portable, final-configuration workflow; no authors' trained weights required.

All generated files go into a new user-chosen run directory. Run from anywhere.
STL10 uses the recorded four-GPU DDP training, not a silently changed batch size.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    'mnist': ('mnist_stagec_d64_k128.yaml', 'mnist_stagec_d64_k128_software.yaml'),
    'stl10': ('stl10_stagec_d64_k128.yaml', 'stl10_stagec_d64_k128_software.yaml'),
}
EXPECTED_BANK_HASH = '89f7c290012917955f8a911f6df85fdca60c7b7e354061a78020e4e8e5cad199'


def run(command, dry_run=False):
    print(subprocess.list2cmdline([str(v) for v in command]), flush=True)
    if not dry_run:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
        subprocess.run([str(v) for v in command], cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=CONFIGS, required=True)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='torchvision root for the selected dataset')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare', 'train', 'evaluate', 'figures'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--registry', type=Path,
                        help='Figures only: read an existing MNIST/STL10 sample registry without replacing it')
    parser.add_argument('--preserve-registered-samples', action='store_true',
                        help='Figures only, requires --registry: retain original-run IDs/states and annotate current metrics')
    args = parser.parse_args()
    if (args.registry is not None or args.preserve_registered_samples) and args.stage != 'figures':
        parser.error('--registry and --preserve-registered-samples are supported only with --stage figures')
    if args.preserve_registered_samples and args.registry is None:
        parser.error('--preserve-registered-samples requires an explicit --registry')
    data = args.data_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    bank = ROOT / 'artifacts/codebooks/lightwave_bank_d64_k128.npz'
    manifest = bank.with_suffix('.manifest.json')
    python = sys.executable
    prefix = [python, '-m', 'lightwave_seeded_vqvae.cli.']

    def module(name, *parts):
        return [python, '-m', prefix[-1] + name, *parts]

    if args.stage == 'prepare':
        if not bank.exists():
            run(module('build_codebooks', '--input-405', ROOT / 'data/optical/20260907_405nm.xlsx',
                       '--input-640', ROOT / 'data/optical/20260907_640nm.xlsx',
                       '--output', bank, '--latent-dim', 64, '--base-codebook-size', 128), args.dry_run)
        if not args.dry_run:
            import numpy as np
            with np.load(bank, allow_pickle=False) as saved:
                actual = hashlib.sha256(np.ascontiguousarray(saved['codebooks'], dtype='<f4').tobytes()).hexdigest()
            if actual != EXPECTED_BANK_HASH:
                raise RuntimeError('Frozen bank fails the published canonical-array hash')
            if args.download:
                from torchvision.datasets import MNIST, STL10
                if args.dataset == 'stl10':
                    for split in ('unlabeled', 'train', 'test'):
                        STL10(str(data), split=split, download=True)
                else:
                    for train in (True, False):
                        MNIST(str(data), train=train, download=True)
            output.mkdir(parents=True, exist_ok=True)
            import yaml
            for label, original in zip(('chip', 'software'), CONFIGS[args.dataset]):
                config = yaml.safe_load((ROOT / 'configs' / original).read_text(encoding='utf-8'))
                config['dataset']['root'] = str(data)
                if 'codebooks' in config:
                    config['codebooks']['path'] = str(bank)
                config['training']['num_workers'] = args.num_workers
                target = output / f'{label}.yaml'
                text = yaml.safe_dump(config, sort_keys=False)
                if target.exists() and target.read_text(encoding='utf-8') != text:
                    raise FileExistsError(f'Refusing to change existing resolved config: {target}')
                if not target.exists():
                    target.write_text(text, encoding='utf-8')
            print('Prepared frozen bank and resolved configurations. No model was trained.')
        return
    if args.stage == 'train':
        if args.dataset == 'stl10' and not args.dry_run:
            import torch
            if sys.platform == 'win32' or torch.cuda.device_count() < 4:
                raise RuntimeError('Recorded STL10 recipe requires Linux/NCCL and 4 GPUs (64 per GPU). '
                                   'A changed-device recipe is a new experiment; do not silently reuse its label.')
        for label in ('chip', 'software'):
            parts = ['--config', output / f'{label}.yaml', '--output-dir', output / label]
            if args.dataset == 'stl10':
                command = [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                           '-m', 'lightwave_seeded_vqvae.cli.train', *parts]
            else:
                command = module('train', *parts, '--device', args.device)
            run(command, args.dry_run)
        return
    common = ['--seeded-checkpoint', output / 'chip/checkpoint_best.pt',
              '--software-checkpoint', output / 'software/checkpoint_best.pt', '--codebook', bank]
    if args.stage == 'evaluate':
        run(module('evaluate_final', *common, '--manifest', manifest, '--output-dir', output / 'evaluation',
                   '--dataset-root', data, '--batch-size', 128, '--num-workers', args.num_workers,
                   '--device', args.device, '--lpips', 'on', '--bootstrap-seed', 20260907,
                   '--bootstrap-replicates', 2000), args.dry_run)
        return
    figures = output / 'figures'
    if figures.exists() or figures.is_symlink():
        raise FileExistsError(f'Refusing to overwrite existing figure output: {figures}')
    registry = (
        args.registry.expanduser().resolve()
        if args.registry is not None else output / 'sample_registry.json'
    )
    if args.registry is None and (registry.exists() or registry.is_symlink()):
        raise FileExistsError(
            f'Refusing to overwrite an existing sample registry: {registry}; '
            'use --registry to read it explicitly'
        )
    if args.registry is not None and not args.dry_run and not registry.is_file():
        raise FileNotFoundError(registry)
    registry_module = 'build_mnist_stagec_sample_registry' if args.dataset == 'mnist' else 'build_stl10_sample_registry'
    plot_module = 'plot_mnist_stagec' if args.dataset == 'mnist' else 'plot_stl10_stagec'
    if args.registry is None:
        run(module(registry_module, '--per-image-metrics', output / 'evaluation/per_image_metrics.npz',
                   '--dataset-root', data, '--codebook-manifest', manifest, '--output', registry), args.dry_run)
    preservation = ['--preserve-registered-samples'] if args.preserve_registered_samples else []
    run(module(plot_module, *common, '--codebook-manifest', manifest, '--registry', registry,
               '--evaluation-dir', output / 'evaluation', '--output-dir', figures,
               '--dataset-root', data, '--device', args.device, *preservation), args.dry_run)


if __name__ == '__main__':
    main()
