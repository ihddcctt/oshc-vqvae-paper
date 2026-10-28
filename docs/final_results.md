# Final evaluation results

All results use `torch-gaussian-valid-v2` and the complete official test splits. MSE is measured in native [0,1] image space; pooled RMSE is the square root of pooled MSE. SSIM and LPIPS are means of per-observation scores. Chip evaluation includes ten matched and 90 ordered mismatched states per image.

| Dataset | Condition | Pooled MSE | Pooled RMSE | Mean SSIM ↑ | Mean LPIPS ↓ |
|---|---|---:|---:|---:|---:|
| MNIST, 10,000 images | Matched | 0.00175440 | 0.041886 | 0.979402 | 0.002637 |
| MNIST | Mismatched | 0.03261143 | 0.180586 | 0.687721 | 0.041427 |
| MNIST | Software VQ | 0.00088030 | 0.029670 | 0.990069 | 0.001017 |
| STL-10, 8,000 images | Matched | 0.00311294 | 0.055794 | 0.812340 | 0.141596 |
| STL-10 | Mismatched | 0.03371626 | 0.183620 | 0.370839 | 0.431635 |
| STL-10 | Software VQ | 0.00185606 | 0.043082 | 0.889938 | 0.063753 |

| Dataset | Wrong/correct MSE | Paired image-bootstrap 95% interval | Wrong-minus-correct LPIPS |
|---|---:|---|---:|
| MNIST | 18.5884 | 18.3980–18.7783 | 0.038790 |
| STL-10 | 10.8310 | 10.7029–10.9630 | 0.290040 |

## Component diagnostics

All diagnostics use the same trained chip decoder. Transmitted R-only retains i_R from the full model; reselected R-only chooses its own nearest entry to z and therefore changes the diagnostic index rule.

| Dataset | Full B_s+R MSE | B-only MSE | Transmitted R-only MSE | Reselected R-only MSE | Latent residual-error reduction | Selected mean R/B norm ratio |
|---|---:|---:|---:|---:|---:|---:|
| MNIST | 0.00175440 | 0.02538717 | 0.00762124 | 0.03379112 | 52.44% | 0.771 |
| STL-10 | 0.00311294 | 0.01624953 | 0.02302449 | 0.00776438 | 33.57% | 0.616 |

The selected R norm exceeds the selected B norm for 22.02% of MNIST tokens and 2.50% of STL-10 tokens. Norms and residual-error reduction do not measure an information fraction. Component-removal results are descriptive diagnostics rather than optimized recovery attacks.

Per-image arrays, per-state values, all 10×10 cells, confidence intervals, codebook occupancy/perplexity and bank-distance statistics are in [`results/recomputed/`](../results/recomputed/). [The verified summary](../results/audit/verified_metrics_summary.json) contains unrounded values and input hashes. Run `python scripts/summarize_results.py` to check it against the arrays and evaluation manifests.

Natural-image examples show colour, texture and structure effects at native resolution. Cross-dataset SSIM/LPIPS and mismatch ratios are not a common security scale. Bootstrap intervals are conditional on one trained model. See the [technical protocol](supporting_information.md) for metric definitions and evaluation scope, and [AES-GCM validation](aes_gcm.md) for the separate transport test.
