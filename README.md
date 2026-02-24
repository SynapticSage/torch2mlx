<p align="center">
  <img src=".assets/logo.png" alt="torch2mlx" width="520">
</p>

<p align="center">
  Translate PyTorch neural network models to Apple's MLX framework.
</p>

## Why

PyTorch models don't run natively on Apple Silicon's GPU/Neural Engine. [MLX](https://github.com/ml-explore/mlx) does — but porting a model means manually transposing weight layouts, renaming state dict keys, rewriting `forward()` calls, and debugging silent numerical mismatches.

**torch2mlx** automates the mechanical parts:

- **Weight conversion** — dispatches the correct transposition per layer type (Conv2d needs `[O,I,H,W]` → `[O,H,W,I]`, Linear is identity, etc.)
- **State dict surgery** — converts PyTorch's flat dot-separated keys to MLX's nested dicts, through safetensors as the interchange format
- **Portability analysis** — tells you _before_ you start porting what percentage of the model converts automatically and what needs manual work
- **MLX templates** — hand-written reference implementations for common patterns (transformer blocks, conv stacks, MLPs)

## Quickstart

```bash
pip install torch2mlx          # core (numpy + safetensors only)
pip install torch2mlx[all]     # with torch + mlx + dev tools
```

```python
from torch2mlx.converter import convert, load_converted

# Convert a PyTorch model → safetensors
# Automatically analyzes coverage and warns about unsupported layers
convert(model, "weights.safetensors")
# ⚠ UserWarning: Model coverage 85.7% — unmapped layers: AdaptiveAvgPool2d

# Load into MLX
params = load_converted("weights.safetensors")
mlx_model.load_weights(list(params.items()))
```

You can also pass a pre-extracted state dict (numpy arrays with dot-separated keys) instead of a live `torch.nn.Module` — no torch installation required for the conversion step itself.

## How it works

torch2mlx walks the PyTorch module tree, looks up each layer in a **registry** to find its MLX equivalent and weight transposition rule, applies the transpositions using **numpy only** (no framework imports during conversion), and saves the result as safetensors. A separate **analyzer** inspects the model's `forward()` source for non-convertible patterns (in-place mutation, custom autograd, etc.) and reports blockers before you invest time porting.

```
src/torch2mlx/
├── registry.py          # torch.nn.X → mlx.nn.X dispatch table
├── op_mapping.py        # torch.cat → mx.concatenate etc.
├── weight_converter.py  # Per-layer transposition rules (numpy only)
├── state_dict.py        # Flat keys ↔ nested dict + safetensors I/O
├── analyzer.py          # Portability report: % convertible, blockers
├── converter.py         # End-to-end orchestration
└── templates/           # Hand-written MLX module implementations
```

## What's supported

**24** layer types, **28** ops, **6** weight transposition rules — covering Linear, Conv1d/2d, ConvTranspose1d/2d, BatchNorm, LayerNorm, Embedding, MultiheadAttention, GroupNorm, InstanceNorm, common activations (GELU, ReLU, SiLU, Tanh, Sigmoid, LeakyReLU, Softmax), and tensor ops (einsum, matmul, reshape, squeeze/unsqueeze, reductions, etc.).

See [docs/support-matrix.md](docs/support-matrix.md) for the full table with test coverage per layer.

**Not supported** (and won't be): pooling layers (MLX lacks primitives), RNNs/LSTMs (stateful, out of scope), Conv3d (MLX lacks it), in-place mutation patterns (`+=`, `.copy_()` — MLX arrays are immutable).

## Templates

Hand-written MLX implementations for common architecture patterns:

| Template | Description |
|---|---|
| `MLP` | Linear stacks with configurable activation, dropout, residual connections |
| `TransformerBlock` | Self-attention + FFN + LayerNorm (pre-norm and post-norm) |
| `ConvBlock` | Conv + normalization + activation |
| `ConvStack` | Stacked ConvBlocks with channel progression |

These are reference implementations, not auto-generated. Use them directly or as a starting point for hand-porting custom architectures.

## Development

```bash
pip install -e ".[all]"     # Install with torch + mlx + dev deps
pytest                       # Run tests (182 tests)
ruff check src/              # Lint
ruff format src/ tests/      # Format
```

## License

Apache 2.0
