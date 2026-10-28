# Reproduction guide

## Numerical source-data verification

```bash
python scripts/verify_release.py --require-manifest
python scripts/summarize_results.py
python scripts/verify_aes_results.py
python scripts/plot_metrics.py --input results/recomputed/stl10 --output runs/stl10-metrics
```

These commands verify distributed files, recompute full-test metrics from per-image arrays, inspect saved transport evidence and plot numerical results. The AES checker reads the records; rerunning model inference requires a trained checkpoint.

## Seed and frozen-bank regeneration

The `prepare` stage below regenerates the bank from the two supplied optical feature tables. Its shape is (10,128,64). SHA-256 over contiguous C-order little-endian float32 bank bytes must be:

```text
89f7c290012917955f8a911f6df85fdca60c7b7e354061a78020e4e8e5cad199
```

The compressed NPZ file hash is not the regeneration criterion. Exact canonical bytes and known-answer examples are specified in [S2–S3](supporting_information.md#s2-optical-input-contract-and-physical-interpretation). The experiment identifier and `power_mw` field are part of this numerical contract.

## Independent training

Install the environment as described in the [README](../README.md), and obtain MNIST/STL-10 and upstream ImageNet/LPIPS weights from their official sources. All output paths must be new.

```bash
python scripts/reproduce.py --dataset stl10 --data-root ./datasets/STL10 --output-root ./runs/stl10 --stage prepare --download
python scripts/reproduce.py --dataset stl10 --data-root ./datasets/STL10 --output-root ./runs/stl10 --stage train
python scripts/reproduce.py --dataset stl10 --data-root ./datasets/STL10 --output-root ./runs/stl10 --stage evaluate
python scripts/reproduce.py --dataset stl10 --data-root ./datasets/STL10 --output-root ./runs/stl10 --stage figures
```

For MNIST, use `--dataset mnist --data-root ./datasets --output-root ./runs/mnist`. The four final configurations are under `configs/`. `--dry-run` previews the commands without creating outputs or starting training.

STL-10 uses four ranks, local BatchNorm and batch 64 per rank; MNIST uses one GPU with batch 256. A different GPU count or batch distribution changes the numerical recipe. Training used Python 3.10.14, PyTorch 2.1.0, torchvision 0.16.0 and CUDA 12.1. Final evaluation used Python 3.10.6, PyTorch 2.1.2+cu121, torchvision 0.16.2+cu121, NumPy 1.26.4 and LPIPS 0.1.4. `environment.yml` specifies a reference environment rather than a complete historical lockfile.

Evaluation uses float32, batch 128 and an unshuffled official test stream. ImageNet normalization is internal to the encoder; targets and sigmoid outputs already lie in [0,1]. Do not apply inverse ImageNet normalization to these arrays. LPIPS has a separate metric input mapping. Numerical kernels and batch context can affect nearest-codeword decisions near a boundary.

## Fixed-sample galleries

The default `figures` stage selects quantiles from the new run. To retain the publication's sample identities and receiver choices:

```bash
python scripts/reproduce.py --dataset stl10 --data-root ./datasets/STL10 --output-root ./runs/stl10 --stage figures --registry artifacts/fixed_samples/stl10_stagec_d64_k128_q2575_87f5b36.json --preserve-registered-samples
```

For MNIST use `artifacts/fixed_samples/mnist_stagec_sample_registry.json`. The registries were fixed after the reference evaluation. Q25/Q50/Q75 labels refer to that selection; they need not remain exact quantiles after retraining. Existing registries and output directories are protected against overwriting. The `stagec` strings in filenames are retained interface identifiers for the final configuration.

## Reproduction scope

The release supports exact seed/bank regeneration, reanalysis of the supplied numerical source data and method-level retraining. Task weights are not supplied, so the authors' checkpoint-specific inference images cannot be generated directly. Checkpoint hashes identify the reference artifacts but do not substitute for parameters. Independent retraining agreement has not been measured, and a fixed seed is not a guarantee of identical tensors or results.

One training run is reported per final configuration. Bootstrap intervals describe image sampling conditional on that trained model. The [technical protocol](supporting_information.md) records the evaluation design, sample selection and interpretation limits. Use the explicit K128 commands in [AES-GCM validation](aes_gcm.md) for a new transport inference test.
