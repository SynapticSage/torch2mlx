<p align="center">
  <img src=".assets/logo.png" alt="torch2mlx" width="520">
</p>

<p align="center">
  Translate PyTorch neural network models to Apple's MLX framework.
</p>

> **Scope**: torch2mlx converts models for **inference** on Apple Silicon. Training support (including a Lightning-compatible MLX Trainer) is on the roadmap.

## Why

PyTorch models don't run natively on Apple Silicon's GPU/Neural Engine. [MLX](https://github.com/ml-explore/mlx) does — but porting a model means manually transposing weight layouts, renaming state dict keys, rewriting `forward()` calls, and debugging silent numerical mismatches.

**torch2mlx** automates the mechanical parts:

- **Weight conversion** — dispatches the correct transposition per layer type (Conv2d needs `[O,I,H,W]` → `[O,H,W,I]`, Linear is identity, etc.)
- **State dict surgery** — converts PyTorch's flat dot-separated keys to MLX's nested dicts, through safetensors as the interchange format
- **Portability analysis** — tells you _before_ you start porting what percentage of the model converts automatically and what needs manual work
- **Code generation** — emits a complete `mlx.nn.Module` `.py` file with `__init__` wired from the module tree and `__call__` translated via `torch.fx` (falls back to a TODO stub for models with dynamic control flow)
- **MLX templates** — hand-written reference implementations for common patterns (transformer blocks, conv stacks, MLPs)

## Quickstart

```bash
pip install torch2mlx          # core (numpy + safetensors only)
pip install torch2mlx[all]     # with torch + mlx + dev tools
```

### Python API

```python
import torch2mlx

# Analyze portability before converting
report = torch2mlx.analyze(model)
print(f"Coverage: {report.coverage:.0%}")

# Convert a PyTorch model → safetensors
torch2mlx.convert(model, "weights.safetensors")

# Load into MLX
params = torch2mlx.load_converted("weights.safetensors")
mlx_model.load_weights(list(params.items()))

# Generate MLX module source code
result = torch2mlx.generate(model)
print(result.source)     # complete .py file
print(result.coverage)   # fraction of children with codegen support
print(result.traced)     # True if torch.fx succeeded for __call__
```

### CLI

```bash
# Convert with portability report
python -m torch2mlx model.pt output/

# Analyze only (no conversion)
python -m torch2mlx model.pt --analyze-only

# Convert + generate MLX module source
python -m torch2mlx model.pt output/ --codegen
```

You can also pass a pre-extracted state dict (numpy arrays with dot-separated keys) instead of a live `torch.nn.Module` — no torch installation required for the conversion step itself.

## How it works

torch2mlx walks the PyTorch module tree, looks up each layer in a **registry** to find its MLX equivalent and weight transposition rule, applies the transpositions using **numpy only** (no framework imports during conversion), and saves the result as safetensors. A separate **analyzer** inspects the model's `forward()` source for non-convertible patterns (in-place mutation, custom autograd, etc.) and reports blockers before you invest time porting.

```
src/torch2mlx/
├── registry.py          # torch.nn.X → mlx.nn.X dispatch table
├── op_mapping.py        # torch.cat → mx.concatenate etc. + dtype mappings
├── weight_converter.py  # Per-layer transposition rules (numpy only)
├── state_dict.py        # Flat keys ↔ nested dict + safetensors I/O
├── analyzer.py          # Portability report: % convertible, blockers
├── codegen.py           # Emit MLX nn.Module .py from torch module tree + fx graph
├── converter.py         # End-to-end orchestration
└── templates/           # Hand-written MLX module implementations
```

## What's supported

**72** layer types, **53** ops, **17** dtype mappings, **7** weight transposition rules — covering Linear, Conv1d/2d, ConvTranspose1d/2d, BatchNorm, LayerNorm, RMSNorm, Embedding, MultiheadAttention, GroupNorm, InstanceNorm, pooling (MaxPool/AvgPool 1d/2d/3d, AdaptiveAvgPool2d), Transformer encoder/decoder, common activations (GELU, ReLU, SiLU, Tanh, Sigmoid, LeakyReLU, Softmax), and tensor ops (einsum, matmul, reshape, squeeze/unsqueeze, reductions, etc.).

Works with `torch.compile()` — compiled models convert identically to uncompiled ones.

See [docs/support-matrix.md](docs/support-matrix.md) for the full table.

### Tested HuggingFace models

The analyzer reports **100% coverage** on all 36 tested architectures:

| Category | Models |
|---|---|
| Encoder | BERT, RoBERTa, DistilBERT, ALBERT, DeBERTa, DeBERTa-v3, Electra, MPNet, Longformer, Funnel, CamemBERT, Data2Vec-Text |
| Decoder / Causal LM | GPT-2, GPT-Neo, OPT, BLOOM, Qwen2, Pythia, CodeGen, Falcon |
| Encoder-Decoder | T5, BART, Pegasus |
| Vision | ViT, CLIP, Swin Transformer, ConvNeXt, DINOv2, BEiT, SegFormer, MobileNetV2, ResNet |
| Speech | Whisper, Wav2Vec2, HuBERT |
| Other | XLNet |

**Not supported** (architectural blockers): RNNs/LSTMs (stateful, out of scope), Conv3d (MLX lacks it), in-place mutation patterns (`+=`, `.copy_()` — MLX arrays are immutable).

## Code generation

`torch2mlx.generate()` emits a complete MLX `nn.Module` `.py` file from a live PyTorch model. The `__init__` is generated recursively from the module tree — composite wrappers (BertEncoder, ViTEmbeddings, etc.) become helper classes with their children wired up. The `__call__` is translated via a 3-tier cascade: `torch.fx` graph (for simple models) → AST rewrite of `forward()` source (for HF models) → TODO stub (last resort).

**Simple model — full translation:**

<table>
<tr><th>PyTorch</th><th>Generated MLX</th></tr>
<tr>
<td>

```python
class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))
```

</td>
<td>

```python
class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def __call__(self, x):
        fc1 = self.fc1(x)
        relu = nn.relu(fc1)
        fc2 = self.fc2(relu)
        return fc2
```

</td>
</tr>
</table>

**HuggingFace DistilBERT — recursive init + AST-rewritten `__call__`:**

```python
class Embeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(30522, 768)
        self.position_embeddings = nn.Embedding(512, 768)
        self.LayerNorm = nn.LayerNorm((768,))
        self.dropout = nn.Dropout(0.1)

    # --- torch2mlx: MECHANICAL (AST rewrite) ---
    def __call__(self, input_ids: mx.array, ...) -> mx.array:
        input_embeds = self.word_embeddings(input_ids)
        position_ids = mx.arange(seq_length, dtype=mx.int64)
        embeddings = input_embeds + self.position_embeddings(position_ids)
        return self.dropout(self.LayerNorm(embeddings))

class DistilBertModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = Embeddings()
        self.transformer = Transformer()
    ...
```

### Codegen coverage

All **36/36** HF architectures get **100% init coverage** via recursive descent into composite wrappers. The AST rewriter mechanically translates `forward()` → `__call__()` for all 36 models, rewriting `torch.*` → `mx.*`, `F.*` → `nn.*`, stripping device/CUDA guards, and converting type annotations.

## Numerical equivalence

Three end-to-end validation examples in `examples/` prove that converted models produce identical outputs:

| Example | Architecture | Max logit diff | MLX speedup |
|---|---|---|---|
| [`validate_mnist.py`](examples/validate_mnist.py) | CNN (Conv2d, MaxPool2d, Linear) | < 1e-5 | ~3x |
| [`validate_transformer.py`](examples/validate_transformer.py) | Transformer (Attention, FFN, LayerNorm) | < 1e-5 | ~2x |
| [`validate_resnet.py`](examples/validate_resnet.py) | ResNet (Conv2d, BatchNorm, skip connections) | < 1e-5 | ~6x |

Each script trains a small model in PyTorch, converts via torch2mlx, loads into an equivalent MLX model, and compares predictions — 100% agreement across all three.

## Templates

Hand-written MLX implementations for common architecture patterns:

| Template | Description |
|---|---|
| `MLP` | Linear stacks with configurable activation, dropout, residual connections |
| `TransformerBlock` | Self-attention + FFN + LayerNorm (pre-norm and post-norm) |
| `ConvBlock` | Conv + normalization + activation |
| `ConvStack` | Stacked ConvBlocks with channel progression |
| `AdaptiveAvgPool2d` | Dynamic kernel/stride computation for adaptive average pooling |

These are reference implementations, not auto-generated. Use them directly or as a starting point for hand-porting custom architectures.

## Progress

| Phase | Status | Highlights |
|---|---|---|
| **P0 — Layer & op coverage** | Done | 72 layer mappings, 53 op mappings, 7 transposition rules, 17 dtype mappings |
| **P1 — CLI & API** | Done | `python -m torch2mlx`, public API (`convert`, `analyze`, `export`), e2e tests, 3 numerical equivalence examples |
| **P2 — Polish** | Done | PyPI metadata, support-matrix cleanup, dtype registry, `torch.compile` interop |
| **P3 — HuggingFace validation** | Done | 36/36 models at 100% analyzer coverage, weight round-trip (MLX→PyTorch) |
| **P4 — Code generation** | Done | `generate()` with recursive init (36/36 at 100%), 3-tier `__call__` cascade (fx → AST → TODO), 53 op mappings |
| **Training support** | Planned | Lightning-compatible MLX Trainer — [see roadmap](next-steps.md) |

### Current numbers

| Metric | |
|---|---|
| Layer types | 72 |
| Op mappings | 53 |
| Dtype mappings | 17 |
| Transposition rules | 7 (+ reverse for round-trip) |
| Constructor specs | 72 (codegen) |
| Templates | 5 (MLP, Transformer, ConvBlock, ConvStack, AdaptiveAvgPool2d) |
| Tests | 464 |
| HuggingFace models tested | 36/36 at 100% analyzer coverage |
| HF codegen init coverage | 36/36 at 100% (recursive) |
| HF codegen `__call__` | 36/36 AST-rewritten at MECHANICAL confidence |

## Roadmap

torch2mlx currently targets **inference-only** conversion of feed-forward architectures.

Planned next:
- **End-to-end codegen validation** — load converted weights into generated models and verify numerical equivalence
- **Decorator API** — `@torch2mlx.export` for compile-style annotation
- **Weight streaming** — convert large models without loading full state dict into memory
- **Training support** — Lightning-compatible MLX Trainer where users provide an MLX-native `forward()` while weights, optimizers, schedulers, and the training loop are automated

See [next-steps.md](next-steps.md) for detailed plans including the three-level Lightning integration strategy.

## Development

```bash
pip install -e ".[all]"          # Install with torch + mlx + dev deps
python -m pytest                 # Run tests (464 tests)
ruff check src/                  # Lint
ruff format src/ tests/          # Format
```

## License

Apache 2.0
