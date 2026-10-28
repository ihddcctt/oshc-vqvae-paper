# Data and code availability

The algorithm code, final MNIST/STL-10 configurations, measured optical-feature tables, fixed sample registry, full-test numerical evaluation source data and K128 AES-GCM validation records are available at https://github.com/ihddcctt/oshc-vqvae-paper, version **v1.0.0**. The repository provides scripts for deterministic seed/codebook generation, model training, evaluation and metric-plot reproduction.

Trained task-specific weights are not included. Independent retraining is supported; reproducing the authors' checkpoint-specific inference images requires the corresponding task weights. The [reproduction guide](reproducibility.md) defines the verification and retraining workflows.

MNIST and STL-10 images are obtained from their original public distributions and are not redistributed here:

- [MNIST](https://yann.lecun.com/exdb/mnist/)
- [STL-10](https://cs.stanford.edu/~acoates/stl10/)

The optical workbooks contain extracted two-peak feature measurements. The bank manifest and numerical result files are listed in `release_manifest.json`.

## Rights

No code licence has yet been assigned. Public visibility does not itself grant an open-source reuse licence; see [GitHub's licensing explanation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository). Dataset and third-party software/model-weight terms apply separately.

ImageNet initialization and LPIPS reference weights are obtained from the [torchvision ResNet18 distribution](https://docs.pytorch.org/vision/0.16/models/generated/torchvision.models.resnet18.html) and [official LPIPS implementation](https://github.com/richzhang/PerceptualSimilarity). They are not bundled in this release. Primary method, dataset and metric references are listed in [references](references.md).
