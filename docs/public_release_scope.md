# Distributed material

This release contains the final MNIST/STL-10 implementation, configurations, extracted optical peak tables, seed/bank manifest, fixed sample IDs, full-test numerical source data and K128 transport-validation records. `release_manifest.json` records payload sizes and SHA-256 checksums; verify it with `python scripts/verify_release.py --require-manifest`.

Task weights, optimizer states, complete image datasets, private machine records, credentials and AES secret keys are not distributed. Numerical metric plots are included; example-image galleries can be generated locally with the corresponding datasets and a trained model.

The exact seed/bank input convention and public workbook fingerprints are specified in the [technical protocol](supporting_information.md). See [availability and rights](data_and_code_availability.md) for access terms and [reproduction instructions](reproducibility.md) for the supported workflows.
