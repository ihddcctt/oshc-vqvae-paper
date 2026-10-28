# OSHC-VQ-VAE

**Optical-State-Seeded Hybrid-Codebook VQ-VAE**: code and numerical source data for a waveguide-chip image-reconstruction demonstration. [中文说明](README_zh.md).

This release contains the final **MNIST (28×28 grayscale)** and **STL-10 (96×96 RGB)** experiments, with D=64 and K_B=K_R=128, and their K128 AES-GCM software-loopback validation. Cite version **v1.0.0** at https://github.com/ihddcctt/oshc-vqvae-paper.

## Method

Ten measured records cover the 405/640 nm pump series and Q₁₀=0,50,100,150,200 mJ over 10 s. The two zero-energy records are the respective scan references. P₁₀=Q₁₀/(10 s), stored as `power_mw`, two peak wavelengths, two extinction ratios and a fixed experiment identifier form the canonical input. SHA-256 and SHAKE256 generate frozen state-dependent codebooks B_s.

Within each dataset, an ImageNet-initialized ResNet18 encoder, residual decoder and trainable residual codebook R are shared across states:

```text
i_B = argmin_k ||z - B_s[k]||²
i_R = argmin_j ||z - B_s[i_B] - R[j]||²
q   = B_s[i_B] + R[i_R]
```

Both index maps are transmitted. Mismatch evaluation keeps the indices fixed and substitutes only the receiver bank B_s′. The two datasets use independently trained networks and the same frozen bank. Exact serialization, architecture, losses and metrics are specified in the [technical protocol](docs/supporting_information.md).

## Results

| Dataset | Matched pooled MSE | Mismatched pooled MSE | Wrong/correct ratio |
|---|---:|---:|---:|
| MNIST, 10,000 test images | 0.00175440 | 0.03261143 | 18.5884 |
| STL-10, 8,000 test images | 0.00311294 | 0.03371626 | 10.8310 |

Evaluation covers ten matches and 90 ordered mismatches per image. [Full results](docs/final_results.md) include RMSE, SSIM, LPIPS, software VQ, component diagnostics and confidence intervals. State-mismatch scores describe reconstruction dependence; authenticated transport is evaluated separately in the [AES-GCM supplement](docs/aes_gcm.md).

## Install and verify

```bash
conda env create -f environment.yml
conda activate oshc-vqvae
python -m pip install --no-deps -e .
python -m pytest tests -q
python scripts/verify_release.py --require-manifest
python scripts/verify_aes_results.py
python scripts/summarize_results.py
```

The three verification scripts require no GPU, image datasets or task weights. To regenerate metric figures in a new directory:

```bash
python scripts/plot_metrics.py --input results/recomputed/stl10 --output runs/stl10-metrics
```

## Train and evaluate

Obtain datasets and upstream initialization weights from their official sources. STL-10 uses 100,000 unlabeled images for optimization, 5,000 labeled training images for validation, and all 8,000 test images. Its reference training recipe uses four GPUs with local BatchNorm and batch 64 per GPU.

```bash
python scripts/reproduce.py --dataset stl10 --data-root ./datasets/STL10 --output-root ./runs/stl10 --stage prepare --download
python scripts/reproduce.py --dataset stl10 --data-root ./datasets/STL10 --output-root ./runs/stl10 --stage train
python scripts/reproduce.py --dataset stl10 --data-root ./datasets/STL10 --output-root ./runs/stl10 --stage evaluate
python scripts/reproduce.py --dataset stl10 --data-root ./datasets/STL10 --output-root ./runs/stl10 --stage figures
```

For MNIST, substitute `--dataset mnist --data-root ./datasets --output-root ./runs/mnist`; training uses one GPU and a 55,000/5,000/10,000 split. `--dry-run` previews each stage. [Reproduction instructions](docs/reproducibility.md) describe bank verification, original fixed-sample galleries and numerical settings. Use the explicit K128 entry point in the [AES instructions](docs/aes_gcm.md).

## Included material and access

- `src/`, `configs/`, `tests/`, `scripts/`: implementation, four final configurations and verification tools.
- `data/optical/`, `artifacts/fixed_samples/`: measured peak tables, seed/bank manifest and fixed sample IDs.
- `results/`, `figures/`: full-test numerical source data, training provenance, transport records and metric plots.
- `docs/`: technical protocol, results, reproduction instructions and references.

**Task-specific trained weights are not included.** The release supports exact seed/bank regeneration, numerical reanalysis and independent retraining; reproducing the authors' checkpoint-specific inference images requires their task weights. One training run is reported per final configuration; evaluation scope is stated in the technical protocol.

Dataset images are obtained from their original distributions. No code licence has yet been assigned; third-party terms apply separately. See [data and code availability](docs/data_and_code_availability.md).
