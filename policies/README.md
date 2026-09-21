# Policies

This Python package contains deterministic policies, the custom small VLA, dataset adapters,
training, and ROS 2 inference. PyTorch supplies tensors, autograd, and CUDA kernels; model-level
linear, normalization, attention, feed-forward, embedding, and transformer components are
implemented in this repository.

## Layout

- `scripted/`: joint tests and pose-based baseline policies.
- `data/`: LeRobot loading, adaptation, statistics, and normalization.
- `models/`: neural primitives, transformer, vision, language, action, and VLA modules.
- `training/`: configuration, loss, checkpoints, and training entry point.
- `inference/`: checkpoint loading, ROS image conversion, and the policy node.
- `tests/`: model and training-component smoke tests.

## Commands

```bash
python3 -m pip install --user -e ./policies
python3 policies/tests/test_small_vla.py
python3 policies/tests/test_training_components.py
so101-train-vla --dataset-root datasets/raw/so101_data --epochs 1
```

The verified one-epoch run produced training loss `0.196694` and validation loss `0.092230`.
These values validate the pipeline but do not establish closed-loop policy quality.

Checkpoints belong under `models/checkpoints/`; exported deployment artifacts belong under
`models/exported/`.