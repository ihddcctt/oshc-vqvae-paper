# Method overview

Ten measured optical records generate deterministic seeds and frozen state-specific codebooks B_s through canonical serialization, SHA-256 and SHAKE256. The records cover two pump scan series and five energy conditions over 10 s; P₁₀=Q₁₀/(10 s) supplies the numerical `power_mw` coordinate.

A shared ImageNet-initialized ResNet18 encoder produces normalized D=64 latent vectors. Each is quantized first against B_s and then against a shared trainable residual codebook R, with K_B=K_R=128. The decoder receives the sum of the two selected codewords. Receiver mismatch retains both transmitted index maps and changes only B_s. MNIST and STL-10 use separately trained networks with the same frozen optical bank.

The final training objective combines pixel MSE, commitment, residual-codebook fitting and residual-norm regularization. Evaluation includes all ten matched states, 90 ordered mismatches, a software VQ reference and component-removal diagnostics. AES-GCM provides a separately evaluated authenticated transport layer.

The complete reproducible specification is maintained in the [technical protocol](supporting_information.md), with [training and evaluation commands](reproducibility.md), [final numerical results](final_results.md), [transport details](aes_gcm.md) and [primary references](references.md).
