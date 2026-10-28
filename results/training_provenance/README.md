# Final training provenance

The MNIST/STL-10 chip and software directories contain the recorded configuration, training history, selected-checkpoint record and runtime settings. Paths are portable placeholders; use `configs/` and `scripts/reproduce.py` for executable local paths.

All four models select epochs by matched-state validation MSE. One independent training run is reported per configuration. Checkpoint SHA-256 values identify the reference artifacts; model and optimizer tensors are not distributed.
