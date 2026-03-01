# torch2mlx Support Matrix

> Reference of supported torch operations and their MLX equivalents.

## Layer Mappings

37 `torch.nn` module classes with automatic conversion support.

| Torch Layer | MLX Equivalent | Weight Rule | Template |
|---|---|---|---|
| `nn.Linear` | `nn.Linear` | identity | MLP |
| `nn.Embedding` | `nn.Embedding` | identity | — |
| `nn.LayerNorm` | `nn.LayerNorm` | identity | TransformerBlock |
| `nn.RMSNorm` | `nn.RMSNorm` | identity | — |
| `nn.Conv1d` | `nn.Conv1d` | conv1d | ConvBlock, ConvStack |
| `nn.Conv2d` | `nn.Conv2d` | conv2d | ConvBlock, ConvStack |
| `nn.ConvTranspose1d` | `nn.ConvTranspose1d` | conv_transpose1d | — |
| `nn.ConvTranspose2d` | `nn.ConvTranspose2d` | conv_transpose2d | — |
| `nn.BatchNorm1d` | `nn.BatchNorm` | batch_norm | — |
| `nn.BatchNorm2d` | `nn.BatchNorm` | batch_norm | — |
| `nn.MultiheadAttention` | `nn.MultiHeadAttention` | identity | TransformerBlock |
| `nn.GELU` | `nn.GELU` | identity | MLP, TransformerBlock |
| `nn.ReLU` | `nn.ReLU` | identity | MLP |
| `nn.SiLU` | `nn.SiLU` | identity | MLP |
| `nn.Tanh` | `nn.Tanh` | identity | — |
| `nn.Sigmoid` | `nn.Sigmoid` | identity | — |
| `nn.LeakyReLU` | `nn.LeakyReLU` | identity | — |
| `nn.Softmax` | `nn.Softmax` | identity | — |
| `nn.Dropout` | `nn.Dropout` | identity | MLP, TransformerBlock |
| `nn.GroupNorm` | `nn.GroupNorm` | identity | — |
| `nn.InstanceNorm1d` | `nn.InstanceNorm` | identity | — |
| `nn.InstanceNorm2d` | `nn.InstanceNorm` | identity | — |
| `nn.MaxPool1d` | `nn.MaxPool1d` | identity | — |
| `nn.MaxPool2d` | `nn.MaxPool2d` | identity | — |
| `nn.MaxPool3d` | `nn.MaxPool3d` | identity | — |
| `nn.AvgPool1d` | `nn.AvgPool1d` | identity | — |
| `nn.AvgPool2d` | `nn.AvgPool2d` | identity | — |
| `nn.AvgPool3d` | `nn.AvgPool3d` | identity | — |
| `nn.AdaptiveAvgPool2d` | Custom template | identity | AdaptiveAvgPool2d |
| `nn.TransformerEncoder` | Decomposed into children | identity | — |
| `nn.TransformerDecoder` | Decomposed into children | identity | — |
| `nn.TransformerEncoderLayer` | Decomposed into children | identity | — |
| `nn.TransformerDecoderLayer` | Decomposed into children | identity | — |
| `nn.Flatten` | Stateless (no weights) | identity | — |
| `nn.ModuleList` | Container (no weights) | identity | — |
| `nn.Sequential` | Container (no weights) | identity | — |
| `NonDynamicallyQuantizableLinear` | `nn.Linear` | identity | — |

### Unsupported Layers

| Category | Torch Layers | Notes |
|---|---|---|
| Recurrent | `LSTM`, `GRU`, `RNN` | Stateful + sequential execution. MLX has no built-in RNN. Out of scope. |
| 3D convolution | `Conv3d`, `ConvTranspose3d` | MLX lacks `Conv3d` entirely. |

## Op Mappings

30 functional/tensor operations with automatic mapping.

| Torch Op | MLX Equivalent | Param Renames |
|---|---|---|
| `torch.cat` | `mx.concatenate` | `dim` → `axis` |
| `torch.stack` | `mx.stack` | `dim` → `axis` |
| `torch.split` | `mx.split` | `dim` → `axis` |
| `torch.chunk` | `mx.split` | `dim` → `axis` |
| `x.chunk` | `mx.split` | `dim` → `axis` |
| `torch.einsum` | `mx.einsum` | — |
| `torch.matmul` | `mx.matmul` | — |
| `F.softmax` | `mx.softmax` | `dim` → `axis` |
| `F.relu` | `nn.relu` | — |
| `F.gelu` | `nn.gelu` | — |
| `F.silu` | `nn.silu` | — |
| `F.cross_entropy` | `nn.losses.cross_entropy` | — |
| `F.mse_loss` | `nn.losses.mse_loss` | — |
| `x.view` | `mx.reshape` | — |
| `x.reshape` | `mx.reshape` | — |
| `x.permute` | `mx.transpose` | — |
| `x.transpose` | `mx.swapaxes` | — |
| `x.unsqueeze` | `mx.expand_dims` | `dim` → `axis` |
| `x.squeeze` | `mx.squeeze` | `dim` → `axis` |
| `x.flatten` | `mx.flatten` | — |
| `x.sum` | `mx.sum` | `dim` → `axis` |
| `x.mean` | `mx.mean` | `dim` → `axis` |
| `x.max` | `mx.max` | `dim` → `axis` |
| `x.min` | `mx.min` | `dim` → `axis` |
| `x.to` | no-op | — |
| `x.contiguous` | no-op | — |
| `torch.no_grad` | no-op | — |
| `torch.zeros` | `mx.zeros` | `dtype` → `dtype` |
| `torch.ones` | `mx.ones` | `dtype` → `dtype` |
| `torch.randn` | `mx.random.normal` | — |

### Unsupported Ops

| Category | Torch Ops | Notes |
|---|---|---|
| In-place mutation | `.add_`, `.mul_`, `.zero_`, `x[i] = v` | MLX arrays are immutable. Must refactor to functional style. The analyzer flags these as blockers. |
| Autograd | `torch.autograd.Function`, `.backward()` | MLX uses `mx.grad()` with a different API. Out of scope. |

## Weight Transposition Rules

6 rules mapping PyTorch weight layouts to MLX conventions. All operate on numpy arrays (backend-agnostic).

| Rule Key | Transform | Shape Example |
|---|---|---|
| `identity` | passthrough | `[O, I]` → `[O, I]` |
| `conv1d` | `np.swapaxes(1, 2)` | `[O, I, K]` → `[O, K, I]` |
| `conv2d` | `np.moveaxis(1, -1)` | `[O, I, H, W]` → `[O, H, W, I]` |
| `conv_transpose1d` | `np.transpose(1, 2, 0)` | `[I, O, K]` → `[O, K, I]` |
| `conv_transpose2d` | `np.transpose(1, 2, 3, 0)` | `[I, O, H, W]` → `[O, H, W, I]` |
| `batch_norm` | passthrough (alias for identity) | `[C]` → `[C]` |

## Blocker Detection

Patterns scanned in `forward()` source code to flag non-convertible constructs.

| Pattern | Detection Method |
|---|---|
| `.copy_(` | String match in `inspect.getsource(forward)` |
| `torch.autograd.Function` | String match |
| `.item()` | String match |
| `register_forward_hook` | String match |
| `register_backward_hook` | String match |
| `+=` (in-place add) | Line-by-line scan, excludes counters |

## Summary

| Metric | Count |
|---|---|
| Supported layer types | 37 |
| Supported op mappings | 30 |
| Weight transposition rules | 6 |
| Blocker patterns detected | 6 |
| Templates | 5 (MLP, TransformerBlock, ConvBlock, ConvStack, AdaptiveAvgPool2d) |
| Total tests | 217 |
