<p align="center">
  <img src=".assets/logo.png" alt="torch2mlx" width="520">
</p>

<p align="center">
  Translate PyTorch neural network models to Apple's MLX framework.
</p>

> **Scope**: torch2mlx converts models for **inference** on Apple Silicon. Training support is on the roadmap.

## What this is

A PyTorch-to-MLX lowering pipeline with three products:

1. **Weight conversion** (solid) — dispatches the correct transposition per layer type (Conv2d: `[O,I,H,W]` → `[O,H,W,I]`, Linear: identity, etc.), restructures state dict keys, and saves as safetensors. Uses **numpy only** — no framework imports during conversion.

2. **Portability analysis** (heuristic) — walks the module tree and reports what percentage of layers have registry mappings. Scans `forward()` source for common blocker _patterns_ (`.copy_()`, `+=`, custom autograd) via string/regex matching. This catches many issues but is not a semantic checker — it won't find runtime-only problems like shape mismatches or dynamic dispatch.

3. **Code generation** (experimental) — emits an `mlx.nn.Module` `.py` file with `__init__` wired from the module tree and `__call__` translated via a 3-tier cascade: `torch.fx` tracing → syntactic AST rewrite → TODO stub. The AST rewriter mechanically renames `torch.*` → `mx.*` using an op registry; operations not in the registry are flagged but may require manual fixes. Some op lowerings are approximate (e.g., attention, activation variants) — treat generated code as a starting point for manual review, not a finished product.

**Weight conversion is the stable product.** The analyzer is a triage tool, and the code generator is an assisted porting layer — not yet a general PyTorch→MLX compiler.

## Quickstart

```bash
pip install torch2mlx          # core (numpy + safetensors only)
pip install torch2mlx[all]     # with torch + mlx + dev tools
```

### Python API

```python
import torch2mlx

# Analyze portability (layer coverage + blocker patterns)
report = torch2mlx.analyze(model)
print(f"Layer coverage: {report.coverage:.0%}")  # % of layers with registry mappings
if report.blockers:
    print(f"Blockers: {report.blockers}")

# Convert weights → safetensors (numpy only, no MLX needed)
torch2mlx.convert(model, "weights.safetensors")

# Load into MLX (flat=True for load_weights format)
params = torch2mlx.load_converted("weights.safetensors", flat=True)
mlx_model.load_weights(list(params.items()))

# Generate MLX module source code (experimental — review output before use)
result = torch2mlx.generate(model)
print(result.source)          # .py file (may include TODO stubs for unmapped ops)
print(result.coverage)        # init_coverage: fraction with real constructor code
print(result.coverage_metrics.registry_coverage)  # includes stateless/skipped leaves
print(result.call_confidence) # "mechanical", "needs_review", or "todo"
```

### CLI

```bash
# Convert with portability report
python -m torch2mlx model.pt output/

# Analyze only (no conversion)
python -m torch2mlx model.pt --analyze-only

# Convert + generate MLX module source (experimental — review before use)
python -m torch2mlx model.pt output/ --codegen
```

You can also pass a pre-extracted state dict (numpy arrays with dot-separated keys) instead of a live `torch.nn.Module` — no torch installation required for the conversion step itself.

## How it works

torch2mlx walks the PyTorch module tree and uses a **registry** to map each layer type to its MLX equivalent and weight transposition rule. Weights are converted using **numpy only** (no framework imports) and saved as safetensors. A separate **analyzer** scans each module's `forward()` source for common blocker patterns using regex matching — this catches many non-convertible patterns but is not exhaustive. The **code generator** emits MLX module source via recursive init + a 3-tier `__call__` cascade.

```
src/torch2mlx/
├── registry.py          # torch.nn.X → mlx.nn.X dispatch table (72 entries)
├── op_mapping.py        # torch.cat → mx.concatenate etc. + dtype mappings
├── weight_converter.py  # Per-layer transposition rules (numpy only)
├── state_dict.py        # Flat keys ↔ nested dict + safetensors I/O
├── analyzer.py          # Layer coverage report + pattern-based blocker detection
├── codegen.py           # Emit MLX nn.Module .py (init + fx/AST __call__)
├── hf_compat.py         # HuggingFace-specific post-processing (pluggable)
├── converter.py         # End-to-end orchestration
└── templates/           # Hand-written MLX module implementations
```

## What's supported

**69** layer types with registry mappings, **79** operator translations, **12** dtype mappings, **7** weight transposition rules.

Includes Conv1d/2d, ConvTranspose1d/2d, BatchNorm, LayerNorm, RMSNorm, Embedding, MultiheadAttention, GroupNorm, InstanceNorm, pooling (MaxPool/AvgPool 1d/2d/3d, AdaptiveAvgPool2d), common activations (GELU, ReLU, SiLU, Tanh, Sigmoid, Softmax, etc.), and tensor ops (matmul, einsum, reshape, squeeze, reductions, etc.). These are **syntactically mapped** — see the validation section below for which architectures have full end-to-end numerical verification.

**Not supported** (architectural blockers): RNNs/LSTMs (stateful, out of scope), Conv3d (MLX lacks it), in-place mutation patterns (`+=`, `.copy_()` — MLX arrays are immutable).

Works with `torch.compile()` — compiled models convert identically to uncompiled ones.

See [docs/support-matrix.md](docs/support-matrix.md) for the full table.

### HuggingFace architectures

The following 36 architectures all achieve **100% layer-level analyzer coverage** and **100% codegen init coverage** (recursive class generation for all submodules):

| Category | Models |
|---|---|
| Encoder | BERT, RoBERTa, DistilBERT, ALBERT, DeBERTa, DeBERTa-v3, Electra, MPNet, Longformer, Funnel, CamemBERT, Data2Vec-Text |
| Decoder / Causal LM | GPT-2, GPT-Neo, OPT, BLOOM, Qwen2, Pythia, CodeGen, Falcon |
| Encoder-Decoder | T5, BART, Pegasus |
| Vision | ViT, CLIP, Swin Transformer, ConvNeXt, DINOv2, BEiT, SegFormer, MobileNetV2, ResNet |
| Speech | Whisper, Wav2Vec2, HuBERT |
| Other | XLNet |

**What "100% coverage" means:** Every child module's class name is in the registry (or is fully composed of registered children). This is *registry coverage* — a necessary but not sufficient condition for a working conversion. Use `result.coverage_metrics` for a breakdown: `init_coverage` (constructor code emitted), `registry_coverage` (leaf recognized), `call_coverage` (forward ops rewritten), and `skipped_leaves` (stateless modules like Identity/DropPath that are recognized but emit no code).

## Validation

### Weight conversion (validated)

Three reference examples in `examples/` validate that weight conversion produces numerically identical outputs when loaded into **hand-written** MLX models:

| Example | Architecture | Max diff |
|---|---|---|
| [`validate_mnist.py`](examples/validate_mnist.py) | CNN (Conv2d, MaxPool2d, Linear) | < 1e-5 |
| [`validate_transformer.py`](examples/validate_transformer.py) | Transformer (Attention, FFN, LayerNorm) | < 1e-5 |
| [`validate_resnet.py`](examples/validate_resnet.py) | ResNet (Conv2d, BatchNorm, skip connections) | < 1e-5 |

These prove the weight conversion pipeline (transposition + safetensors round-trip) is correct. The MLX models in these examples are hand-ported, not generated.

### Code generation (partially validated)

End-to-end codegen validation — generate code, load converted weights, compare forward-pass output — is verified on **10 HuggingFace models**:

| Model | Checkpoint | Max diff |
|---|---|---|
| DistilBERT | `distilbert-base-uncased` | < 2e-3 |
| BERT | `bert-base-uncased` | < 5e-3 |
| RoBERTa | `roberta-base` | < 4e-5 |
| ELECTRA | `google/electra-small-discriminator` | < 3e-2 |
| ViT | `google/vit-base-patch16-224` | < 5e-5 |
| GPT-2 | `gpt2` | < 3e-4 |
| CamemBERT | `camembert-base` | < 4e-5 |
| Data2Vec-Text | `facebook/data2vec-text-base` | < 5e-5 |
| MPNet | `microsoft/mpnet-base` | < 1e-3 |
| DINOv2 | `facebook/dinov2-small` | < 5e-5 |

The remaining 30 HF models have codegen structural coverage (init + AST-rewritten `__call__`) but their forward passes have **not** been numerically validated. Generated code for untested models may contain unmapped operations requiring manual fixes.

Simple architectures (MLP, transformer block, embedding net, multi-LayerNorm, deep MLP) are also validated end-to-end via `torch.fx` tracing with < 1e-5 tolerance.

## Code generation

`torch2mlx.generate()` emits a complete MLX `nn.Module` `.py` file. The `__init__` is generated recursively — composite wrappers (BertEncoder, ViTEmbeddings, etc.) become helper classes. The `__call__` uses a 3-tier cascade:

1. **torch.fx tracing** — captures the compute graph for simple, fully-traceable models (e.g., MLP, basic CNNs). Produces the most reliable output.
2. **AST rewrite** — parses `forward()` source as a Python AST and mechanically renames `torch.*` → `mx.*`, `F.*` → `nn.*`, strips device/CUDA guards, and maps known operations via OP_REGISTRY. Unmapped torch APIs are preserved with annotations. This handles most HuggingFace models.
3. **TODO stub** — fallback when both tracing and rewriting fail (e.g., C extensions, obfuscated source).

**Simple model — full translation (via fx):**

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

### Codegen customization

The `generate()` function accepts a `post_processors` parameter for controlling HF-specific post-processing:

```python
# Default: applies HF compat patches (private attrs, method stubs, output classes)
result = generate(model)

# Disable HF post-processing for non-HF models
result = generate(model, post_processors=[])

# Custom post-processor
result = generate(model, post_processors=[my_custom_processor])
```

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
| **P0 — Layer & op coverage** | Done | 72 layer mappings, 59 op mappings, 7 transposition rules, 12 dtype mappings |
| **P1 — CLI & API** | Done | `python -m torch2mlx`, public API (`convert`, `analyze`, `export`, `generate`), e2e tests |
| **P2 — Polish** | Done | PyPI metadata, support-matrix, dtype registry, `torch.compile` interop |
| **P3 — HuggingFace validation** | Done | 36/36 models at 100% analyzer coverage, weight round-trip (MLX→PyTorch) |
| **P4 — Code generation** | Done | Recursive init (36/36 at 100%), 3-tier `__call__` cascade, HF compat layer |
| **P5 — Forward-pass validation** | In progress | 10/36 HF models numerically validated end-to-end |
| **Training support** | Planned | Lightning-compatible MLX Trainer — [see roadmap](next-steps.md) |

### Current numbers

| Metric | |
|---|---|
| Layer types | 69 |
| Op mappings | 79 |
| Dtype mappings | 12 |
| Transposition rules | 7 (+ reverse for round-trip) |
| Constructor specs | 69 (codegen) |
| Templates | 5 (MLP, Transformer, ConvBlock, ConvStack, AdaptiveAvgPool2d) |
| Tests | 545+ (non-HF) + 367 (HF codegen) |
| HF analyzer coverage | 36/36 at 100% (layer-level) |
| HF codegen init coverage | 36/36 at 100% (recursive) |
| HF codegen `__call__` | 36/36 AST-rewritten at MECHANICAL confidence |
| HF forward-pass validated | 10 models (BERT, RoBERTa, DistilBERT, ELECTRA, ViT, GPT-2, CamemBERT, Data2Vec, MPNet, DINOv2) |

## Roadmap

torch2mlx currently targets **inference-only** conversion of feed-forward architectures.

Planned next:
- **Expand forward-pass validation** — validate remaining HF architectures end-to-end (currently 4/36)
- **Decorator API** — `@torch2mlx.export` for compile-style annotation
- **Weight streaming** — convert large models without loading full state dict into memory
- **Training support** — Lightning-compatible MLX Trainer where users provide an MLX-native `forward()` while weights, optimizers, schedulers, and the training loop are automated

See [next-steps.md](next-steps.md) for detailed plans including the three-level Lightning integration strategy.

## Development

```bash
pip install -e ".[all]"          # Install with torch + mlx + dev deps
python -m pytest                 # Run tests
ruff check src/                  # Lint
ruff format src/ tests/          # Format
```

## License

Apache 2.0
