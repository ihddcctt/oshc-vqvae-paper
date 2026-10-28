# Technical protocol: OSHC-VQ-VAE

This protocol specifies the final independently trained MNIST and native-resolution STL-10 models, which share the same frozen optical bank. It defines the numerical procedure, full-test evaluation and fixed-sample figures. See the [reproduction guide](reproducibility.md) for commands and [AES-GCM validation](aes_gcm.md) for authenticated transport.

All distributed results use `torch-gaussian-valid-v2` on the full official test splits. Unrounded metrics and input hashes are in the [verified summary](../results/audit/verified_metrics_summary.json).

## S1. Experimental scope and dataset roles

The experiment demonstrates state-dependent reconstruction in a controlled software transmission protocol. It uses a finite library generated from ten optical measurement records. It does not demonstrate a physical optical communication link or establish cryptographic confidentiality. STL-10 is the principal natural-image example; MNIST shows the same codebook design on a simpler image distribution. These are separate training runs, not transfer of one trained image model between datasets.

| Setting | MNIST | STL-10 |
| --- | ---: | ---: |
| Input | 1×28×28 grayscale | 3×96×96 RGB |
| Optimization images | 55,000 | 100,000 unlabeled |
| Validation images | 5,000 | 5,000 official labeled train |
| Test images | 10,000 | 8,000 |
| Latent tensor, H×W×D | 7×7×64 | 24×24×64 |
| K_B / K_R | 128 / 128 | 128 / 128 |
| Encoder-to-decoder downsampling | 4 | 4 |
| Decoder residual blocks by scale | 3, 2, 2 | 3, 2, 2 |
| Extra decoder blocks zero initialized | Yes | No |
| Training GPUs | 1 | 4 |

The MNIST validation subset was drawn from its official training set with `torch.utils.data.random_split` and a generator seeded with 20260907. STL-10 used its complete fixed official splits: `unlabeled` for optimization, `train` for validation, and `test` for evaluation. Labels were not used by the reconstruction objective. STL-10 training applied random horizontal reflection with probability 0.5; MNIST used `ToTensor` only. Validation and test were unshuffled and had no random transformations. Images were never resized to a common cross-dataset resolution. [MNIST source](https://yann.lecun.com/exdb/mnist/), [STL-10 source](https://cs.stanford.edu/~acoates/stl10/)

The official STL-10 binary archive has MD5 `91f7769df0f17e558f3565bffb0c7dfb`; the retained archive SHA-256 is `f31fd99273a1acb8609c8db427cebb1de3f71de77758cdc0e22956e1289b9866`. Ordered sample indices and processed test-stream fingerprints are recorded in run/evaluation manifests. Users should obtain image datasets from their official distributions rather than substitute repackaged or resized variants.

STL-10 images originate from ImageNet. The encoder uses ImageNet1K pretraining, and overlap between the original pretraining images and STL-10 has not been audited image by image. “Held out” therefore refers to task-specific optimization and epoch selection, not proven absence from external pretraining. No from-scratch or uncontaminated-pretraining benchmark claim is made.

## S2. Optical input contract and physical interpretation

The public source files are `data/optical/20260907_405nm.xlsx` and `data/optical/20260907_640nm.xlsx`. Their SHA-256 values are:

| File | SHA-256 |
| --- | --- |
| 20260907_405nm.xlsx | `14d1c741f5a2b51025058620804d831c45858231698643278333b4a327e4ce6d` |
| 20260907_640nm.xlsx | `6b227c7f11a5e36bebfa828d918739b3d4cb8d10a0751d19d852bf9d34c5d165` |

Each workbook is read with `openpyxl.load_workbook(read_only=True, data_only=False)`. The parser requires one worksheet and used range A1:E6. After NFC normalization and trimming whitespace, A1 identifies the nominal pump wavelength and B1:E1 are `λ1 (nm)`, `λ2 (nm)`, `E1 (dB)`, `E2 (dB)`. The five power rows are 0, 5, 10, 15, and 20 mW in that order. Measurement cells must be finite numeric values. Decimal conversion starts from `Decimal(str(cell_value))`. Peak identity is inherited from the columns: peak 1 is B/D and peak 2 is C/E. The parser requires λ1<λ2 and does not reorder the peaks.

The physical state labels use the fixed-duration energy Q₁₀=0,50,100,150,200 mJ. Before serialization, P₁₀=Q₁₀/τ with τ=10 s is expressed numerically in mW in the `power_mw` field, giving 0,5,10,15,20. The workbook row coordinate and internal IDs use this duration-normalized value. Q₁₀ and τ are not extra canonical hash fields. This input convention neither integrates the four spectral descriptors nor independently establishes measurement stability.

The 405/640 values are pump wavelengths; the measured peak wavelengths are approximately 1549 nm. E denotes the recorded extinction ratio in dB, not linear intensity. Its serialized numerical value is not combined with wavelength by a raw weighted sum. The workbooks contain extracted two-peak features, not full instrument spectral traces; reproducibility here begins with those recorded feature tables.

States are ordered by `(pump_nm,power_mw)`. The two zero-energy records are the respective reference measurements of the 405 and 640 nm scan series.

| Index | State ID | λ1, integer pm | E1, integer 10^-4 dB | λ2, integer pm | E2, integer 10^-4 dB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | 405nm_0mW | 1548752 | 81501 | 1549493 | 90964 |
| 1 | 405nm_5mW | 1548775 | 88232 | 1549503 | 75386 |
| 2 | 405nm_10mW | 1548835 | 91613 | 1549563 | 63134 |
| 3 | 405nm_15mW | 1548906 | 82830 | 1549640 | 68645 |
| 4 | 405nm_20mW | 1549009 | 80364 | 1549709 | 73616 |
| 5 | 640nm_0mW | 1548991 | 90548 | 1549690 | 86424 |
| 6 | 640nm_5mW | 1548896 | 102331 | 1549620 | 75486 |
| 7 | 640nm_10mW | 1548824 | 112275 | 1549552 | 95208 |
| 8 | 640nm_15mW | 1548779 | 94762 | 1549484 | 86428 |
| 9 | 640nm_20mW | 1548759 | 101964 | 1549480 | 89882 |

## S3. Canonical record, deterministic seed, and frozen base bank

### S3.1 Fixed-point conversion and byte serialization

For peak p of record s, with round-half-to-even denoted RHE,

\[
L_{s,p}=\operatorname{RHE}(10^3\lambda_{s,p}^{\rm nm}),\qquad
X_{s,p}=\operatorname{RHE}(10^4E_{s,p}^{\rm dB}).
\]

The different physical units are serialized in separate fields. The chosen scales represent decimal storage precision, not measured instrumental accuracy, quantization-noise tolerance, or optical repeatability. The canonical record contains:

```json
{"experiment_id":"20260907_ZHY","peaks":[{"extinction_db_1e4":91613,"peak":1,"wavelength_pm":1548835},{"extinction_db_1e4":63134,"peak":2,"wavelength_pm":1549563}],"power_mw":10,"pump_nm":405,"schema":"lightwave-seeded-vqvae-state-v1"}
```

This is the state-2 example at 405 nm and Q₁₀=100 mJ: the canonical power coordinate is P₁₀=10 mW. General records replace the corresponding integer fields. Exact serialization is:

```python
json.dumps(record, ensure_ascii=True, sort_keys=True,
           separators=(",", ":")).encode("ascii")
```

The result C_s contains lexicographically sorted keys and no optional whitespace. The experiment identifier and acquisition-condition labels are part of the input; their distinction alone can distinguish two records even if their four measurement fields were equal.

### S3.2 Hash derivation and expansion

Using the terminal zero byte in each prefix,

```python
D_seed = b"LIGHTWAVE_SEEDED_VQVAE/STATE_SEED/V1\x00"
D_book = b"LIGHTWAVE_SEEDED_VQVAE/BASE_CODEBOOK/V1\x00"
version = b"shake256-uniform-center-l2-f32le-v1"
```

define the 32-byte seed and descriptor as

\[
h_s=\operatorname{SHA256}(D_{\rm seed}\Vert C_s),
\]
\[
M_s=D_{\rm book}\Vert v\Vert 0x00\Vert
\operatorname{U32BE}(K_B)\Vert\operatorname{U32BE}(D)\Vert h_s.
\]

U32BE is unsigned 32-bit big-endian encoding and v is the ASCII generator-version byte string above. The deterministic generator is \(T_s=\operatorname{SHAKE256}(M_s,2K_BD)\), where the second argument denotes the requested output length in bytes. Read each successive byte pair of \(T_s\) as an unsigned big-endian 16-bit integer u, in C row-major order, and map it to

\[
x=2\frac{u+1/2}{65536}-1.
\]

Reshape to K_B×D. In float64, center and normalize each row:

\[
c_{k,d}=x_{k,d}-D^{-1}\sum_jx_{k,j},\qquad
b_{k,d}=c_{k,d}/\sqrt{\sum_jc_{k,j}^2}.
\]

Reject a row whose denominator does not exceed float64 epsilon. Cast only the final array to contiguous little-endian float32, `<f4`. Stack the ten banks in the S2 order. All rows consequently have approximately zero mean and unit L2 norm, subject to float32 rounding. The hashes follow FIPS 180-4 and FIPS 202; explicit domain prefixes are used, not cSHAKE or TupleHash. [SHA-256 standard](https://csrc.nist.gov/pubs/fips/180-4/upd1/final), [SHAKE-256 standard](https://csrc.nist.gov/pubs/fips/202/final)

### S3.3 Known-answer checks and storage

For state 2, the canonical JSON above is 235 ASCII bytes. Its seed is `dfe9139e8f6f35c1dae663296c4e334f36c2673fe5dbfee4e43c097d0195107e`. At K_B=128, D=64, the descriptor is 116 bytes, expansion is 16,384 bytes, and the first 16 expanded bytes are `dcf33822ff1f8ff0ac8b1935d2bed962`.

The first eight float32 values of codeword zero are:

```text
0.15065829455852509, -0.1390238106250763, 0.21071894466876984,
0.015302125364542007, 0.06557923555374146, -0.19337911903858185,
0.13271836936473846, 0.14438995718955994
```

Its full state-array SHA-256 is `eb4e2b3d7e492735c9b95498baaa8140951f016b8008f42a5d971e37e3c7bd32`. The [reference manifest](../data/optical/lightwave_bank_d64_k128.manifest.json) contains all ten canonical records, seeds, per-state hashes and source provenance.

The canonical entire-bank hash is SHA-256 over contiguous C-order `<f4` bytes:

```text
shape: [10,128,64]
bank_array_sha256: 89f7c290012917955f8a911f6df85fdca60c7b7e354061a78020e4e8e5cad199
```

This array hash, rather than the compressed-container byte hash, is the decisive regeneration check.

Generation requires ten distinct seeds and bank hashes and refuses to overwrite existing bank/manifest files. The bank is loaded as a non-persistent buffer excluded from the optimizer and checkpoint state dictionary. It is independently required when loading a trained model. Its canonical array hash is checked in run/evaluation provenance. Both the four peak values and the acquisition labels/experiment identifier contribute to the hash; the observed separation between banks cannot be attributed uniquely to the optical spectra.

Seed/bank reproducibility is defined for the supplied canonical records; repeated-measurement robustness and causal attribution are outside the demonstrated scope.

## S4. Encoder and grayscale adaptation

The encoder uses torchvision ResNet18 `IMAGENET1K_V1`, retained through `layer3` and excluding `layer4`, global pooling and the classifier. The upstream initialization is `resnet18-f37072fd.pth` (SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`). The original 7×7 RGB stem kernel is center-cropped to 3×3 and rescaled separately per output channel o:

\[
\widetilde W_o=W^{(7)}_{o,:,2:5,2:5},\quad
\alpha_o=\begin{cases}\|W^{(7)}_o\|_F/\|\widetilde W_o\|_F,&\|\widetilde W_o\|_F>10^{-12},\\1,&\text{otherwise},\end{cases}
\quad W^{(3)}_o=\alpha_o\widetilde W_o.
\]

For MNIST only, the rescaled RGB kernels are summed across input channels, \(W^{\rm gray}_{o,0}=\sum_{c=0}^{2}W^{(3)}_{o,c}\). They are not averaged. The stem uses stride 1, padding 1 and no bias; max-pooling is removed. `layer1` preserves spatial size, while `layer2` and `layer3` each halve it. Pretrained convolutional and BatchNorm affine parameters are retained, while BatchNorm running means, variances and counters are reset before training. A biased 1×1 convolution projects 256 channels to D=64, using Xavier-uniform weights and zero bias. [ResNet paper](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html), [torchvision implementation](https://docs.pytorch.org/vision/0.16/models/generated/torchvision.models.resnet18.html)

RGB input normalization occurs inside the encoder using mean (0.485,0.456,0.406) and standard deviation (0.229,0.224,0.225). Grayscale normalization uses mean 0.5 and standard deviation 0.5. Decoder targets remain the original [0,1] arrays. Encoder normalization therefore requires no inverse transform of decoder outputs when computing image metrics.

## S5. Quantization and receiver mismatch

Permute encoder output to one D-vector per spatial position and normalize each raw token:

\[
z_t=z^{\rm raw}_t/\max(\|z^{\rm raw}_t\|_2,10^{-8}).
\]

For sender record s,

\[
i_{B,t}=\arg\min_k\|z_t-B_s[k]\|_2^2,\quad
e_t=z_t-B_s[i_{B,t}],
\]
\[
i_{R,t}=\arg\min_j\|e_t-R[j]\|_2^2,\quad
q_t=B_s[i_{B,t}]+R[i_{R,t}].
\]

Squared distances use \(\|a\|^2+\|b\|^2-2a^Tb\), clamped below at zero for floating-point roundoff. `torch.argmin` selects the first index in an exact numerical tie. R is initialized componentwise from \(\mathcal N(0,(0.05/\sqrt D)^2)\). It is neither a per-state table nor a directly added correction to every possible base entry.

The decoder receives the straight-through value

\[
q_{\rm ST}=z+\operatorname{sg}(q-z),
\]

where sg denotes stop-gradient. The forward value equals q; the surrogate derivative to z is identity. The selected R vectors receive their fitting gradients through the codebook loss, while B remains frozen. This uses the VQ-VAE straight-through construction with a two-stage residual approximation. Unlike standard fully learned residual quantization, the first bank is frozen and selected by an optical record. No autoregressive prior, GAN discriminator or perceptual loss is part of the final model. [VQ-VAE paper](https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html), [RQ-VAE paper](https://openaccess.thecvf.com/content/CVPR2022/html/Lee_Autoregressive_Image_Generation_Using_Residual_Quantization_CVPR_2022_paper.html), [official RQ-VAE implementation](https://github.com/kakaobrain/rq-vae-transformer)

For receiver s', decode the transmitted sender indices as

\[
\hat x_{s\to s'}=G\left(B_{s'}[i_B^{(s)}]+R[i_R^{(s)}]\right).
\]

All ten entries in one sender row of the mismatch matrix use the same two index maps. Receiver mismatch does not re-encode the image, search another nearest neighbor, or alter R.

## S6. Decoder and initialization

The shared decoder maps D channels to 256 with a biased 1×1 convolution. It applies three residual blocks at the latent resolution, nearest-neighbor upsampling by two followed by a biased 3×3 256→128 convolution and two blocks, then another upsampling with a biased 3×3 128→64 convolution and two blocks. A final biased 3×3 convolution and sigmoid produce one or three [0,1] channels. All 3×3 convolutions use padding 1.

Each residual block is GroupNorm–SiLU–3×3 convolution–GroupNorm–SiLU–3×3 convolution plus identity. Block convolutions have no bias. GroupNorm uses the largest divisor of the channel count not exceeding 32; thus each reported block uses 32 groups. No encoder-to-decoder skip connection exists.

MNIST has the same final [3,2,2] architecture but an explicit initialization adaptation: instantiate the [1,1,1] main path first, append the additional blocks, and initialize only the last convolution of the added blocks to zero. These blocks initially implement identity and remain trainable. STL-10 uses the ordinary initialization. This distinction must accompany any “same architecture” comparison.

## S7. Objective, optimization, and selection

Let N be batch size, C×H×W image size, T total latent tokens, and D_ref=64. The chip model objective is

\[
L_{\rm rec}=\frac{1}{NCHW}\sum_{n,c,h,w}(\hat x_{nchw}-x_{nchw})^2,
\]
\[
L_{\rm commit}=\frac{1}{TD_{\rm ref}}\sum_t\|z_t-\operatorname{sg}(q_t)\|_2^2,
\]
\[
L_{\rm code}=\frac{1}{TD_{\rm ref}}\sum_t\|R[i_{R,t}]-\operatorname{sg}(e_t)\|_2^2,
\qquad
L_{R\rm norm}=\frac{1}{K_RD_{\rm ref}}\sum_j\|R[j]\|_2^2,
\]
\[
L=L_{\rm rec}+0.25L_{\rm commit}+L_{\rm code}+10^{-4}L_{R\rm norm}.
\]

D=D_ref in the final configurations, so the vector losses equal elementwise MSE. No mismatched-state, residual-only, GAN, or perceptual reconstruction term is active.

AdamW uses base learning rate 3×10^-4, beta=(0.9,0.999), epsilon=10^-8 and weight decay 10^-4 for 80 epochs. The pretrained backbone receives 0.1× the base learning rate; the new latent projection, decoder and quantizer use the base rate. There is no warm-up or scheduler. CUDA automatic mixed precision and gradient scaling are enabled during training. Random generators are seeded with 20260907; a fixed seed does not imply bitwise-identical GPU training across implementations. [AdamW paper](https://openreview.net/forum?id=Bkg6RiCqY7)

MNIST uses batch 256 on one GPU. STL-10 uses NCCL DDP with four GPUs and batch 64 per rank (global 256); eight loader workers per rank; validation and artifact writing run on rank zero. BatchNorm statistics in DDP are local to each GPU, not synchronized. Training uses shuffled samples and drops the last incomplete batch. For each rank, a persistent offset o assigns local sample n to \(s_n=(n+o)\bmod10\) and updates \(o\leftarrow(o+N_b)\bmod10\). A fixed image is not permanently tied to one record.

The 80-epoch run records report the model size and elapsed time below. Parameter counts exclude the frozen $B_s$ buffer. Time is measured wall time for each original run, not a hardware-normalized speed comparison.

| Dataset | Model | Trainable parameters | Training GPUs | Recorded elapsed time |
| --- | --- | ---: | ---: | ---: |
| MNIST | $B_s+R$ | 7,465,473 | 1 | 0.309 h |
| MNIST | Software VQ | 7,473,665 | 1 | 0.225 h |
| STL-10 | $B_s+R$ | 7,467,779 | 4 | 1.771 h |
| STL-10 | Software VQ | 7,475,971 | 4 | 0.901 h |

The training histories, runtime settings, validation values and selected epochs are in [`results/training_provenance/`](../results/training_provenance/).

For the primary MNIST and STL-10 experiments, `checkpoint_selection: correct_validation_mse` explicitly selects the minimum pooled matched-state validation MSE across all ten records; the software control has one state. The selected epochs are MNIST chip 78/software 80 and STL-10 chip 76/software 80. The epoch rule uses validation only. Architecture and presentation choices used exploratory test diagnostics; final epoch selection used validation MSE.

Only one independent training realization is reported per final configuration. Image-level bootstrap does not measure training-run variance.

## S8. Matched software reference and component diagnostics

The software VQ reference has the same per-dataset encoder, decoder, normalized latent, D, K_B and K_R, with state-independent fully trainable B and R. B is initialized by L2-normalizing standard-normal rows. Its fitting loss adds

\[
\frac{1}{TD_{\rm ref}}\sum_t\|B[i_{B,t}]-\operatorname{sg}(z_t)\|_2^2
\]

to the residual fitting loss. This reference matches the two-stage index budget but is not a mathematical upper bound on non-convex finite training.

For the trained chip model, diagnostics are G(B_s[i_B]) (B-only), G(R[i_R]) using the transmitted residual indices, and G(R[j]) with \(j=\arg\min_j\|z-R[j]\|^2\) for a separate re-quantized R-only diagnostic. All reuse the trained decoder. They are component-removal tests, not separately trained standalone autoencoders or adversarial recovery models.

For actually selected vectors, record norms of B, R, and B+R; the fraction with norm(R)>norm(B); and quantization errors

\[
E_B=\frac{1}{TD}\sum_t\|z_t-B_s[i_{B,t}]\|^2,\quad
E_{BR}=\frac{1}{TD}\sum_t\|z_t-B_s[i_{B,t}]-R[i_{R,t}]\|^2.
\]

Residual improvement is \(1-E_{BR}/E_B\). For assignment probabilities p_k, coverage is the fraction of entries visited and perplexity is \(\exp(-\sum_{p_k>0}p_k\log p_k)\). Norms do not quantify an information fraction. Low utilization alone does not prove that increasing K could never help.

Corresponding-index cross-state distances \(\|B_s[k]-B_{s'}[k]\|_2\) characterize digital bank distinction. Hash expansion does not preserve a continuous geometry of wavelength, power or the four spectral values.

## S9. Image metrics

Each test image is evaluated at all ten sender/receiver matches and 90 ordered mismatches. Inference uses model evaluation mode, float32, no autocast, batch 128 and an unshuffled official test stream. Original figure reproduction used the same batch context because small floating-point changes can switch nearest-neighbor assignments. The new evaluation manifest records its actual hardware/software and result hashes.

For pixels flattened over channels and space, MSE and RMSE are

\[
\operatorname{MSE}(x,y)=\frac{1}{CHW}\sum_{c,h,w}(x-y)^2,\qquad
\operatorname{RMSE}=\sqrt{\operatorname{MSE}}.
\]

All images of a dataset have equal shape. Pooled MSE is the mean of their per-image/state MSE values; pooled RMSE is the square root of that mean, not the mean of image RMSEs. Scalar per-image metrics are computed in float32 and aggregate reductions use float64. PSNR, if shown, is \(-10\log_{10}(\operatorname{MSE})\) for data range one and is supplementary.

SSIM uses the normalized separable 11×11 Gaussian kernel with sigma 1.5 and valid convolution. Define weighted population means μ, variances σ² and covariance σ_xy. Negative variance values caused by roundoff are clamped to zero. For each window and channel,

\[
\operatorname{SSIM}(x,y)=
\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}
{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)},
\quad C_1=0.01^2,\ C_2=0.03^2.
\]

Average over valid windows and channels to obtain one score per image. The implementation `torch-gaussian-valid-v2` uses `torch.finfo(dtype).tiny` as an underflow-only denominator floor. Tests cover identical black/dark images and agreement with the Gaussian population-moment reference. [Original SSIM paper and reference implementation](https://ece.uwaterloo.ca/~z70wang/research/ssim/)

LPIPS uses package 0.1.4, AlexNet, version 0.1, `spatial=False`, evaluation mode and input transformation 2x−1. For MNIST only, repeat the grayscale image into three channels and bilinearly interpolate to 32×32 with `align_corners=False` in the metric branch. MSE/SSIM and saved image arrays stay 28×28. LPIPS is used for evaluation, not training. Its absolute MNIST value is not directly comparable with native RGB STL-10. [Original LPIPS paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html), [official implementation](https://github.com/richzhang/PerceptualSimilarity)

## S10. Statistical aggregation and figures

Let m_nss' be an image metric. Define within-image correct and wrong MSE by

\[
c_n=\frac{1}{10}\sum_s m_{nss},\qquad
w_n=\frac{1}{90}\sum_{s\ne s'}m_{nss'}.
\]

The headline MSE ratio is \(\bar w/\bar c\), not the average of w_n/c_n. LPIPS separation is mean(wrong LPIPS)−mean(correct LPIPS); SSIM separation has the opposite sign. For two-sided 95% intervals, draw 2,000 bootstrap samples of N image indices with replacement using NumPy PCG64. The same resampled indices are applied to every paired branch and state. Take the 2.5th and 97.5th percentiles. The canonical evaluation invocation uses bootstrap seed 20260907, with correct-versus-wrong resampling internally using the next seed. The evaluation manifest is authoritative if a rerun uses a different recorded seed.

The ten-state mismatch matrix places senders on rows and receivers on columns. Each cell averages over all test images. A diagonal cell corresponds to matched records; an off-diagonal cell changes only the receiver bank. Wrong-wavelength/same-energy, same-wavelength/wrong-energy, and both-label-mismatch groups contain 10, 40 and 40 ordered pairs, respectively. At zero energy, wavelength refers to the source-series label as described in S2. Stored fields named `power` use the corresponding P₁₀ coordinate, so they select exactly the same pairs.

The publication-ready `*_02_state_mismatch_10x10_metrics` files show four panels (MSE, RMSE, SSIM and LPIPS). Their RMSE cell is the square root of that cell's pooled MSE. The `*_03_correct_wrong_metric_distributions` files give two violin/box summaries per metric: one observation per test image, first averaging that image's ten diagonal scores or 90 off-diagonal scores. For the RMSE distribution, the square root is applied after the within-image MSE average. Thus the plotted violin width describes variation across images, not 80,000 or 720,000 independent state-pair samples. The plotted boxes mark median and interquartile range; the numerical result and plotting provenance are stored in each `metric_figures/figure_specification.json`.

The original fixed examples were selected by ranking each class by its mean correct-state MSE across ten states in the original evaluation. The test ID nearest NumPy's linear empirical 25%, 50% or 75% quantile was chosen, with exact distance ties broken by the lower official test ID. This rule was applied after the original evaluation, then frozen before rendering. It was deterministic post-evaluation selection, not preregistration.

Publication galleries retain the supplied registry IDs and receiver choices. Their Q25/Q50/Q75 labels refer to the reference-run selection, rather than guaranteed quantile ranks after retraining. Use `--preserve-registered-samples` to render these identities with a new evaluation; alternatively, select and label quantiles from the new run. Error values shown with any gallery must come from the same evaluation as its reconstructed arrays.

The sender for controlled panels is 405 nm/100 mJ. The fixed wrong-wavelength receiver is 640 nm/100 mJ, and the fixed wrong-energy receiver is 405 nm/200 mJ. The representative wrong receiver is determined from the original evaluation's image-averaged MSE matrix: in the sender row, calculate the median of all nine wrong cells, exclude the two already displayed fixed receivers, and take the remaining cell closest to that median (lower receiver index breaks ties). Original-run Q50 examples include MNIST digits 1 and 8 (IDs 9180 and 436) and STL-10 car and dog (IDs 2359 and 2734); digit 5 is ID 4263 and airplane is ID 5671. All class-wise Q25/Q50/Q75 sample IDs are supplied, not only these illustrative choices.

The full gallery columns are original, software VQ, matched chip reconstruction, its absolute-error map, fixed wavelength mismatch, its error map, fixed energy mismatch, its error map, representative median mismatch, and its error map. The median-wrong record differs between datasets (MNIST 405 nm/50 mJ; STL-10 405 nm/150 mJ), so it is not a shared physical condition for a cross-dataset comparison. Error maps show channel-mean absolute pixel error on a common [0,1] scale. Native arrays are enlarged only on the display canvas; there is no sharpening, denoising, per-image rescaling or super-resolution.

## S11. Index budget and cross-dataset interpretation

With K_B=K_R=128, each token transmits 7+7=14 index bits. Downsampling by four gives \(14/4^2=0.875\) index bit per spatial pixel. MNIST has 49 tokens, 686 bits/image and nominal ratio \((28\times28\times8)/686=9.143:1\). STL-10 has 576 tokens, 8,064 bits/image and ratio \((96\times96\times3\times8)/8064=27.429:1\).

These are index-only rates; state/schedule information, headers, entropy coding, channel coding, models, codebooks and authentication/encryption overhead are excluded. In the separately validated AES-256-GCM protocol, MNIST uses 86 plaintext bytes after two alignment bits and STL-10 uses 1,008 bytes without alignment padding. A 64-byte header, 12-byte nonce and 16-byte tag give 178 and 1,100 application-packet bytes, respectively. This excludes lower-layer framing and channel coding. The completed 600 matched-case software-loopback experiment is described in [AES-GCM validation](aes_gcm.md). MNIST and STL-10 match D, K and spatial index density, not total latent length, channel-normalized bitrate, image complexity, or trained weights. MSE ratios across datasets are not security rankings.

## S12. Full-test results and component diagnostics

The publication metrics below use `torch-gaussian-valid-v2` on the selected final checkpoints. Each software row averages one reconstruction per official test image; each chip-correct row averages ten matched states per image; and each chip-wrong row averages all 90 ordered mismatches per image. All image metrics are computed in native [0,1] space. MSE and RMSE are pooled as defined in S9, while SSIM and LPIPS are means of per-observation scores. Lower MSE, RMSE and LPIPS and higher SSIM indicate closer reconstruction. The unrounded values and input/output hashes are in the [verified summary](../results/audit/verified_metrics_summary.json) and the [STL-10](../results/recomputed/stl10/evaluation_manifest.json) and [MNIST](../results/recomputed/mnist/evaluation_manifest.json) evaluation manifests.

| Dataset (test images) | Reconstruction | Pooled MSE | Pooled RMSE | Mean SSIM | Mean LPIPS |
| --- | --- | ---: | ---: | ---: | ---: |
| STL-10 (8,000) | Software VQ | 0.00185606 | 0.043082 | 0.889938 | 0.063753 |
| STL-10 (8,000) | Chip, matched $s=s'$ | 0.00311294 | 0.055794 | 0.812340 | 0.141596 |
| STL-10 (8,000) | Chip, 90 ordered mismatches | 0.03371626 | 0.183620 | 0.370839 | 0.431635 |
| MNIST (10,000) | Software VQ | 0.00088030 | 0.029670 | 0.990069 | 0.001017 |
| MNIST (10,000) | Chip, matched $s=s'$ | 0.00175440 | 0.041886 | 0.979402 | 0.002637 |
| MNIST (10,000) | Chip, 90 ordered mismatches | 0.03261143 | 0.180586 | 0.687721 | 0.041427 |

For STL-10 and MNIST, respectively, the pooled wrong/correct MSE ratios are 10.8310 (paired image-bootstrap 95% interval 10.7029–10.9630) and 18.5884 (18.3980–18.7783). The corresponding wrong-minus-correct LPIPS differences are 0.290040 and 0.038790. Each dataset uses its own trained model and image statistics; these are within-dataset reconstruction diagnostics.

Component-removal evaluation used the same trained chip decoder and the official full test images. The R-only columns distinguish the residual indices transmitted by the full model from a separate re-quantization of encoder features directly against R. The latter changes the diagnostic selection rule and is not a valid replacement for the transmitted pair of indices.

| Dataset | Full $B_s+R$ MSE | B-only MSE | R-only, transmitted indices MSE | R-only, reselected indices MSE | $1-E_{BR}/E_B$ | Selected mean R/B norm ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| STL-10 | 0.00311294 | 0.01624953 | 0.02302449 | 0.00776438 | 33.57% | 0.616 |
| MNIST | 0.00175440 | 0.02538717 | 0.00762124 | 0.03379112 | 52.44% | 0.771 |

The fraction of selected tokens for which $\|R[i_R]\|_2>\|B_s[i_B]\|_2$ is 2.50% on STL-10 and 22.02% on MNIST. This, the residual-error reduction, occupancy and component reconstructions provide complementary diagnostics; none measures a fraction of image information or excludes a stronger attack. Per-state metrics, all ten-by-ten cells, pair categories, usage/perplexity tables and bootstrap intervals are in [`results/recomputed/`](../results/recomputed/).

Numerical kernels can change near-boundary nearest-codeword decisions. Independent reproduction should report its actual environment and resulting metrics.

## S13. Reproduction and reference checkpoints

Use the four final configurations under `configs/` and the commands in the [reproduction guide](reproducibility.md). The model, bank and evaluation manifest must correspond to the same run. Task weights are not distributed; these fingerprints identify the reference checkpoints and do not reconstruct their parameters.

| Dataset / model | Reference checkpoint SHA-256 |
| --- | --- |
| MNIST chip | `50e82b017e8dc2ec8e2cc59bb8474b18354b6e9031fb057b42dbae237c3ce4ae` |
| MNIST software | `caddd9e4eecafbcb558778d6b6375414d7ef53d4b0926614671b78804ae23c08` |
| STL-10 chip | `7c3fdb0159e69c3fce53a280a7dca8026d61f091852100369cbbd76508d4f411` |
| STL-10 software | `db1cc1d40f6ea34747ff17a651bf93e6a57da5702286c43c11dd9dec17f2055c` |

## S14. Interpretation limits

The experiment demonstrates that a shared reconstruction model can depend on which bank is selected from a reproducible, measured-record-seeded library. Ten records, one chip and one training realization do not establish inter-chip uniqueness, optical entropy, environmental stability, confidentiality or resistance to attacks. Some mismatched images retain coarse silhouettes or digit identity. No random-seed control or repeated optical acquisition study was performed. The separate AES-GCM software-loopback validation supplies a conventional authenticated transport layer; its cryptographic mechanism must not be inferred from the mismatch pictures. Authentication failure produces rejection, not a mismatched reconstruction.

The software VQ gap, residual norms and occupancy are descriptive diagnostics. They do not establish causal superiority of a particular optical waveform over ordinary deterministic digital labels. The retained neural method is a chip-application demonstration with explicit scope; stronger physical or security claims would require additional experiments beyond this release.
