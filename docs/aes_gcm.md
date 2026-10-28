# K=128 authenticated index-transport validation

## Scope and result

This supplement uses the final MNIST and STL-10 chip models, D=64 and K_B=K_R=128. Thirty previously fixed Q25/Q50/Q75 sample IDs per dataset were transported at all ten sender states: 60 distinct images and 600 matched-state cases. The existing registered representative mismatch was also evaluated once per image, giving 60 mismatch cases. The fixed sample selection is defined in the technical protocol.

Both index maps and decoded image arrays were elementwise identical before and after successful AES transport. Maximum image difference and MSE/RMSE/SSIM/LPIPS differences were all zero. Six negative packet probes per matched case (wrong AES key, modified ciphertext, tag, authenticated state header, nonce, or truncated packet) gave 3,600 rejections, with no reconstruction after failure. Transport is software loopback; there was no physical-link experiment or full-test-set AES run.

The transmitted index maps remain fixed in the optical-state mismatch test. Only B_s is replaced with B_s′ after successful authentication. Wrong optical-state selection and wrong AES-key authentication are different experiments.

## Protocol

The base index map is flattened row-major, followed by the residual index map. Each value in [0,127] uses exactly seven bits, most-significant bit first. Zero padding is appended only at the end to reach a byte boundary. The parser validates range, count, shape and padding.

| Payload field | MNIST | STL-10 |
|---|---:|---:|
| Latent grid | 7×7 | 24×24 |
| Valid index bits | 686 | 8,064 |
| Alignment padding bits | 2 | 0 |
| Plaintext bytes | 86 | 1,008 |
| Authenticated header bytes | 64 | 64 |
| Nonce bytes | 12 | 12 |
| Authentication tag bytes | 16 | 16 |
| Complete application packet bytes | 178 | 1,100 |

The protocol uses AES-256-GCM with a fresh independently generated random 256-bit session key, independent of optical seeds. The 96-bit nonce comprises a random fixed 32-bit session prefix and a monotonically increasing 64-bit invocation counter under that key. It must not be reused. The 64-byte header is authenticated as associated data and includes protocol metadata, bit width, dataset and sender state, latent shape/count, model/bank binding and session identifier. Authentication and validation precede index release. The exact wire layout and parser are in `src/lightwave_seeded_vqvae/transport/protocol_v2.py`. Use the explicit `protocol_v2` API or `evaluate_aes_gcm_k128` command for these experiments; the package-level legacy transport exports use a different bit width.

The packed indices, AES keys and raw packets are not saved in the public evidence. Per-case records contain numerical outcomes and hashes, not secret keys. Standard reference: Dworkin, *NIST SP 800-38D* (2007), [doi:10.6028/NIST.SP.800-38D](https://doi.org/10.6028/NIST.SP.800-38D).

## Inspect saved evidence without weights

```bash
python scripts/verify_aes_results.py
```

The checker verifies matched coverage, fixed mismatch coverage, before/after hashes, all four metric deltas, rejection outcomes, packet budgets and agreement with corresponding saved full-test observations. It does not load a checkpoint or claim to rerun image inference. Published evidence is under `results/aes_gcm_k128/mnist/` and `stl10/`.

## Rerun transport on a newly trained checkpoint

First complete `prepare`, `train` and `evaluate` using `scripts/reproduce.py`. Then use a checkpoint, bank and evaluation manifest from that same new run, along with the appropriate fixed sample registry:

```bash
python -m lightwave_seeded_vqvae.cli.evaluate_aes_gcm_k128 \
  --checkpoint runs/mnist/chip/checkpoint_best.pt \
  --codebook artifacts/codebooks/lightwave_bank_d64_k128.npz \
  --evaluation-manifest runs/mnist/evaluation/evaluation_manifest.json \
  --registry artifacts/fixed_samples/mnist_stagec_sample_registry.json \
  --dataset-root datasets --output-dir runs/mnist/aes-validation --device cuda:0
```

For STL-10 substitute the STL-10 run paths and `artifacts/fixed_samples/stl10_stagec_d64_k128_q2575_87f5b36.json`. The recorded evaluation uses batch 128. Non-selected entries in each batch may be computed to preserve the batch context, but only the selected 30 image IDs count as AES-tested samples. The routine checks checkpoint/config/bank fingerprints against the supplied evaluation manifest; do not combine a new checkpoint with the authors' published manifest or assume that a regenerated NPZ container has identical file bytes.

The result demonstrates lossless authenticated transport of the quantized indices in this implementation. It is not a new AES security proof, a key-distribution protocol, or a comprehensive attack assessment of the chip-conditioned codec.
