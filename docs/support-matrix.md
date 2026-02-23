# torch2mlx Support Matrix

> Living reference of every torch operation's support status.
> Generated from source: `registry.py`, `op_mapping.py`, `weight_converter.py`, `analyzer.py`.

## Layer Mappings (`LAYER_REGISTRY`)

24 torch.nn module classes with automatic conversion support.

| Torch Layer | MLX Equivalent | Weight Rule | Registry Test | Weight Test | E2E Test | Template |
|---|---|---|---|---|---|---|
| `nn.Linear` | `nn.Linear` | identity | test_registry.py | test_weights.py | test_converter.py (`test_end_to_end_linear`) | MLP |
| `nn.Embedding` | `nn.Embedding` | identity | test_registry.py | test_weights.py | — | — |
| `nn.LayerNorm` | `nn.LayerNorm` | identity | test_registry.py | test_weights.py | — | TransformerBlock |
| `nn.RMSNorm` | `nn.RMSNorm` | identity | test_registry.py | test_weights.py | — | — |
| `nn.Conv1d` | `nn.Conv1d` | conv1d | test_registry.py | test_weights.py | test_converter.py (`test_end_to_end_conv1d`) | ConvBlock, ConvStack |
| `nn.Conv2d` | `nn.Conv2d` | conv2d | test_registry.py | test_weights.py | test_converter.py (shape test) | ConvBlock, ConvStack |
| `nn.ConvTranspose1d` | `nn.ConvTranspose1d` | conv_transpose1d | test_registry.py | test_weights.py | — | — |
| `nn.ConvTranspose2d` | `nn.ConvTranspose2d` | conv_transpose2d | test_registry.py | test_weights.py | — | — |
| `nn.BatchNorm1d` | `nn.BatchNorm` | batch_norm | test_registry.py | test_weights.py | — | — |
| `nn.BatchNorm2d` | `nn.BatchNorm` | batch_norm | test_registry.py | test_weights.py | — | — |
| `nn.MultiheadAttention` | `nn.MultiHeadAttention` | identity | test_registry.py | test_weights.py | — | TransformerBlock |
| `nn.GELU` | `nn.GELU` | identity | test_registry.py | — | — | MLP, TransformerBlock |
| `nn.ReLU` | `nn.ReLU` | identity | test_registry.py | — | — | MLP |
| `nn.SiLU` | `nn.SiLU` | identity | test_registry.py | — | — | MLP |
| `nn.Dropout` | `nn.Dropout` | identity | test_registry.py | — | — | MLP, TransformerBlock |
| `nn.ModuleList` | None (wrapper) | identity | test_registry.py | — | — | — |
| `nn.Sequential` | None (wrapper) | identity | test_registry.py | — | — | — |
| `nn.Tanh` | `nn.Tanh` | identity | test_registry.py | — | — | — |
| `nn.Sigmoid` | `nn.Sigmoid` | identity | test_registry.py | — | — | — |
| `nn.LeakyReLU` | `nn.LeakyReLU` | identity | test_registry.py | — | — | — |
| `nn.Softmax` | `nn.Softmax` | identity | test_registry.py | — | — | — |
| `nn.GroupNorm` | `nn.GroupNorm` | identity | test_registry.py | — | — | — |
| `nn.InstanceNorm1d` | `nn.InstanceNorm` | identity | test_registry.py | — | — | — |
| `nn.InstanceNorm2d` | `nn.InstanceNorm` | identity | test_registry.py | — | — | — |

### Notable Unsupported Layers

| Category | Torch Layers | Difficulty | Blockers / Notes |
|---|---|---|---|
| Pooling | `MaxPool1d/2d/3d`, `AvgPool1d/2d/3d`, `AdaptiveAvgPool2d` | Hard | MLX has no pooling primitives. Requires manual reduce or strided conv workaround. `AdaptiveAvgPool2d` needs dynamic output-size logic. |
| Recurrent | `LSTM`, `GRU`, `RNN` | Out of scope | Stateful + sequential execution. MLX has no built-in RNN. Would need hand-written scan loop + hidden state management. |
| Conv variants | `Conv3d`, `ConvTranspose3d` | Medium | MLX lacks `Conv3d` entirely. `ConvTranspose2d` now supported — see layer table above. |
| ~~Normalization~~ | ~~`GroupNorm`, `InstanceNorm1d/2d`~~ | ~~Easy~~ | Now supported — see layer table above |
| ~~Activations~~ | ~~`Tanh`, `Sigmoid`, `LeakyReLU`, `Softmax`~~ | ~~Easy~~ | Now supported — see layer table above |
| Attention | `TransformerEncoder`, `TransformerDecoder` | Medium | Compound modules — no 1:1 MLX equivalent. Must decompose into per-layer mappings or use template approach. Cross-attention in decoder adds complexity. |

## Op Mappings (`OP_REGISTRY`)

28 functional/tensor operations with automatic mapping.

| Torch Op | MLX Equivalent | Param Renames | Registry Test | Used in Templates |
|---|---|---|---|---|
| `torch.cat` | `mx.concatenate` | `dim` → `axis` | test_registry.py | — |
| `torch.stack` | `mx.stack` | `dim` → `axis` | test_registry.py | — |
| `F.softmax` | `mx.softmax` | `dim` → `axis` | test_registry.py | TransformerBlock |
| `x.view` | `mx.reshape` | — | test_registry.py | — |
| `x.permute` | `mx.transpose` | — | test_registry.py | — |
| `x.transpose` | `mx.swapaxes` | — | test_registry.py | TransformerBlock |
| `x.reshape` | `mx.reshape` | — | test_registry.py | — |
| `x.to` | no_op | — | test_registry.py | — |
| `x.contiguous` | no_op | — | test_registry.py | — |
| `torch.no_grad` | no_op | — | test_registry.py | — |
| `F.relu` | `nn.relu` | — | test_registry.py | — |
| `F.gelu` | `nn.gelu` | — | test_registry.py | — |
| `F.silu` | `nn.silu` | — | test_registry.py | — |
| `torch.einsum` | `mx.einsum` | — | test_registry.py | — |
| `torch.matmul` | `mx.matmul` | — | test_registry.py | — |
| `x.unsqueeze` | `mx.expand_dims` | `dim` → `axis` | test_registry.py | — |
| `x.squeeze` | `mx.squeeze` | `dim` → `axis` | test_registry.py | — |
| `x.flatten` | `mx.flatten` | — | test_registry.py | — |
| `torch.split` | `mx.split` | `dim` → `axis` | test_registry.py | — |
| `x.sum` | `mx.sum` | `dim` → `axis` | test_registry.py | — |
| `x.mean` | `mx.mean` | `dim` → `axis` | test_registry.py | — |
| `x.max` | `mx.max` | `dim` → `axis` | test_registry.py | — |
| `x.min` | `mx.min` | `dim` → `axis` | test_registry.py | — |
| `F.cross_entropy` | `nn.losses.cross_entropy` | — | test_registry.py | — |
| `F.mse_loss` | `nn.losses.mse_loss` | — | test_registry.py | — |
| `torch.zeros` | `mx.zeros` | `dtype` → `dtype` | test_registry.py | — |
| `torch.ones` | `mx.ones` | `dtype` → `dtype` | test_registry.py | — |
| `torch.randn` | `mx.random.normal` | — | test_registry.py | — |

### Notable Unsupported Ops

| Category | Torch Ops | Difficulty | Blockers / Notes |
|---|---|---|---|
| ~~Einsum~~ | ~~`torch.einsum`~~ | ~~Easy~~ | Now supported — see op table above |
| ~~Matmul~~ | ~~`torch.matmul`, `@` operator~~ | ~~Easy~~ | Now supported — see op table above |
| Tensor methods | `.chunk` | Easy | MLX lacks direct `.chunk`; can be expressed via `mx.split`. |
| ~~Reduction~~ | ~~`.sum`, `.mean`, `.max`, `.min`~~ | ~~Easy~~ | Now supported — see op table above |
| ~~Loss functions~~ | ~~`F.cross_entropy`, `F.mse_loss`~~ | ~~Medium~~ | Now supported — see op table above. Note: no `reduction` param in MLX, different label format. |
| ~~Creation~~ | ~~`torch.zeros`, `torch.ones`, `torch.randn`~~ | ~~Medium~~ | Now supported — see op table above. Note: dtype mapping needed, `torch.randn` has different seeding. |
| In-place ops | `.add_`, `.mul_`, `.zero_`, `x[i] = v` | Hard | MLX arrays are immutable. No direct equivalent. Must refactor to functional style (`x = x + y`). Analyzer flags these as blockers. |
| Autograd | `torch.autograd.Function`, `.backward()` | Out of scope | MLX uses `mx.grad()` with a fundamentally different API. Custom autograd functions need full rewrite. |

## Weight Transposition Rules (`TRANSPOSITION_RULES`)

6 rules mapping PyTorch weight layouts to MLX conventions. All operate on numpy arrays.

| Rule Key | Transform | Shape Example | Shape Test | Value Test | E2E Test |
|---|---|---|---|---|---|
| `identity` | passthrough | `[O, I]` → `[O, I]` | `TestTranspositionShapes.test_identity` | `TestTranspositionValues.test_identity_values` | `test_end_to_end_linear` |
| `conv1d` | `np.swapaxes(1, 2)` | `[O, I, K]` → `[O, K, I]` | `TestTranspositionShapes.test_conv1d` | `TestTranspositionValues.test_conv1d_values` | `test_end_to_end_conv1d` |
| `conv2d` | `np.moveaxis(1, -1)` | `[O, I, H, W]` → `[O, H, W, I]` | `TestTranspositionShapes.test_conv2d` | `TestTranspositionValues.test_conv2d_values` | `test_convert_state_dict_applies_conv2d_transposition` |
| `conv_transpose1d` | `np.transpose(1, 2, 0)` | `[I, O, K]` → `[O, K, I]` | `TestTranspositionShapes.test_conv_transpose1d` | `TestTranspositionValues.test_conv_transpose1d_values` | — |
| `conv_transpose2d` | `np.transpose(1, 2, 3, 0)` | `[I, O, H, W]` → `[O, H, W, I]` | `TestTranspositionShapes.test_conv_transpose2d` | `TestTranspositionValues.test_conv_transpose2d_values` | — |
| `batch_norm` | alias for identity | `[C]` → `[C]` | `TestTranspositionShapes.test_batch_norm` | `TestTranspositionValues.test_batch_norm_is_identity` | — |

## Blocker Detection (`analyzer.py`)

Patterns scanned in `forward()` source code to flag non-convertible constructs.

| Pattern | Detection Method | Tested |
|---|---|---|
| `.copy_(` | String match in `inspect.getsource(forward)` | `TestBlockerDetection.test_copy_inplace_blocker` |
| `torch.autograd.Function` | String match | — (no dedicated test) |
| `.item()` | String match | `TestBlockerDetection.test_item_blocker` |
| `register_forward_hook` | String match | — (no dedicated test) |
| `register_backward_hook` | String match | — (no dedicated test) |
| `+=` (in-place add heuristic) | Line-by-line scan, excludes counters | — (no dedicated test) |

## Summary

| Metric | Count |
|---|---|
| Supported layer types | 24 |
| Supported op mappings | 28 |
| Weight transposition rules | 6 |
| Blocker patterns detected | 6 |
| Total tests | 182 |
| Templates | 4 (MLP, TransformerBlock, ConvBlock, ConvStack) |

### Test Coverage by Module

| Module | Test File | Tests |
|---|---|---|
| `registry.py` | test_registry.py | 24 parametrized layer + 20 MLX existence + 4 unit |
| `op_mapping.py` | test_registry.py | 28 parametrized op + 24 MLX existence + 3 unit |
| `weight_converter.py` | test_weights.py | 6 shape + 6 value + 2 dispatch |
| `state_dict.py` | test_weights.py | 3 safetensors + 5 flatten/unflatten |
| `analyzer.py` | test_analyzer.py | 5 report unit + 5 analyze + 4 blocker |
| `converter.py` | test_converter.py | 6 state_dict + 4 module_map + 2 roundtrip + 4 e2e |
| `templates/` | test_templates.py | 5 MLP + 4 transformer + 3 conv_block + 3 conv_stack |
