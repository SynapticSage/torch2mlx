# Next Steps

## Vision

Two ways to use torch2mlx:

1. **CLI tool** — point at a model, get converted weights + portability report
2. **Drop-in decorator** — `torch.compile`-style annotation in existing PyTorch scripts

```python
# Future API: annotate a model for MLX export
import torch2mlx

@torch2mlx.export("weights.safetensors")
class MyModel(nn.Module):
    ...

# or inline
model = MyModel()
torch2mlx.export(model, "weights.safetensors")
```

The decorator would hook into the model lifecycle — convert on first `forward()` call, on `save()`, or on explicit `torch2mlx.export()`. Think `torch.compile` but targeting MLX instead of Triton.

---

## ~~P0 — complete layer & op coverage~~ DONE

Completed. 37 layers, 30 ops, 6 transposition rules, 5 templates. See `docs/support-matrix.md`.

Not planned (architectural blockers):
- `Conv3d`, `ConvTranspose3d` — MLX lacks `Conv3d`
- `LSTM`, `GRU`, `RNN` — stateful + sequential, out of scope
- In-place ops — MLX immutability, analyzer flags these
- `torch.autograd.Function` — fundamentally different paradigm

---

## ~~P1 — make it usable~~ DONE

Completed. CLI (`python -m torch2mlx`), public API (`export`, `convert`, `analyze`), e2e tests (mini ResNet, TransformerEncoder).

- CLI: `__main__.py` with `--analyze-only` and `--no-analyze` flags
- API: `torch2mlx.export(model, path)` as convenience wrapper
- E2E: mini ResNet (100% coverage), TransformerEncoder (100% coverage), CLI smoke tests
- Discovered and added 4 missing registry entries: Flatten, TransformerEncoderLayer, TransformerDecoderLayer, NonDynamicallyQuantizableLinear

### Validation examples

Three end-to-end validation scripts in `examples/`, each proving numerical equivalence between PyTorch and MLX with speed comparisons:

| Script | Architecture | Exercises |
|---|---|---|
| `validate_mnist.py` | Small CNN | Conv2d, ReLU, MaxPool2d, Linear |
| `validate_transformer.py` | TransformerEncoder classifier | Attention, FFN, LayerNorm |
| `validate_resnet.py` | ResNet-style CNN | Conv2d, BatchNorm2d, skip connections, AdaptiveAvgPool2d |

### Analyzer improvements

- Recursive composition analysis (`_is_fully_composed`): unregistered container modules (e.g. custom `ResidualBlock`) are now recognized as convertible when all their children map to known MLX layers. This fixed false negatives where the analyzer would report 0% coverage on models using composition.

---

## ~~P2 — polish~~ DONE

Completed. PyPI metadata, support-matrix cleanup, dtype registry (12 dtypes), torch.compile interop (4 tests).

| Item | Status |
|---|---|
| PyPI packaging | Done — authors, keywords, classifiers, project URLs |
| Clean up support-matrix.md | Done — removed dev-era markers, 161→136 lines |
| Dtype mapping registry | Done — `DTYPE_REGISTRY` in `op_mapping.py`, 12 entries |
| `torch.compile` interop | Done — 4 tests, `_orig_mod.` prefix documented |
| Logo / branding | Logo in `.assets/logo.png` |

---

## P3 — future directions

### Inference tooling

| Direction | Description |
|---|---|
| Auto template generation | Generate MLX module stubs from torch module trees, reducing manual template work |
| Decorator API | `@torch2mlx.export` — compile-style annotation that triggers conversion at save/forward |
| ~~HuggingFace model testing~~ | ~~Done — 11/11 models at 100%. Added 7 HF leaf types (3 activations, 2 norms, 1 embedding, 1 linear variant)~~ |
| Weight streaming | Stream large model weights to safetensors without loading full state dict into memory |

### Training support

Current scope is inference-only. Training is hard because PyTorch uses imperative autograd (`loss.backward()` + in-place mutation) while MLX uses functional gradients (`mx.grad()` + immutable arrays). Full training loop translation is a compiler problem — but Lightning's structured API provides an escape hatch.

#### Foundation (no Lightning required)

| Direction | Difficulty | Description |
|---|---|---|
| ~~Weight round-trip (MLX→PyTorch)~~ | ~~Easy~~ | ~~Done — `convert_weight_reverse()` + `convert_state_dict_to_pytorch()` with roundtrip tests~~ |
| Optimizer state conversion | Medium | Convert Adam `exp_avg`/`exp_avg_sq` between frameworks (same transposition rules as weights) |
| Training recipe templates | Medium | Hand-written MLX training loops for common patterns (classifier fine-tune, LoRA) that consume torch2mlx-converted weights |
| LR scheduler mapping | Medium | 1:1 mappings for common schedulers (cosine, step, linear warmup) |

#### Lightning integration (three levels)

The key obstacle to converting a LightningModule is `forward()` — it contains framework-specific ops (`torch.relu`, `x.mean(dim=1)`, `F.cross_entropy`). Weights, optimizers, schedulers, and the training loop itself are all automatable. The three levels below represent increasing ambition in handling `forward()`:

**Level 1 — Paired definition (Medium for us, ~30 min for user)**

User provides an MLX-native `forward()`. Everything else is automated: weights from checkpoint, optimizer mapped from `configure_optimizers`, training loop via `mx.value_and_grad(training_step)`.

```python
from torch2mlx.lightning import MLXModule, MLXTrainer

class MyModelMLX(MLXModule):
    def forward(self, x):
        # User writes MLX-native forward — guided by op_mapping docs
        x = self.embed(x)
        x = self.encoder(x, mask=None)
        return self.head(mx.mean(x, axis=1))

    # training_step auto-mapped: self(x) dispatches to MLX forward()
    # configure_optimizers auto-mapped: torch.optim.Adam → mlx.optimizers.Adam

model = MyModelMLX.from_checkpoint("model.ckpt", source_class=OriginalPyTorchModel)
trainer = MLXTrainer(max_epochs=5)
trainer.fit(model, train_dataloader)
```

What's automated: weight conversion, optimizer mapping, LR schedulers, training loop (`mx.value_and_grad`), validation loop, callbacks.
What the user writes: `forward()` in MLX ops. User effort scales with model complexity — 5-line forward ≈ 5 minutes.

**Level 2 — Draft generation via torch.fx (Medium-Hard for us, minimal for user)**

Use `torch.fx` to trace `forward()`, walk the IR graph, apply `op_mapping.py` substitutions, emit MLX Python code as a reviewable draft. User reviews, fixes edge cases, and runs.

```python
# Auto-generate MLX forward() from PyTorch model
draft = torch2mlx.lightning.generate_forward(OriginalPyTorchModel)
print(draft)
# def forward(self, x):
#     x = self.embed(x)
#     x = self.encoder(x, mask=None)       # ← auto-mapped
#     return self.head(mx.mean(x, axis=1)) # ← torch.mean(dim=1) → mx.mean(axis=1)
#     # TODO: verify self.encoder call (custom module)
```

Works for: standard feed-forward architectures (our declared scope) — Linear, Conv, Attention, standard activations/losses. We already have 30 op mappings to drive the translation.
Breaks on: dynamic control flow (`if x.shape[0] > 10`), in-place mutation, custom autograd Functions — but the analyzer already flags these.

**Level 3 — Transparent conversion (Very Hard, likely a separate project)**

User passes their existing LightningModule unchanged. Runtime interception replaces every torch op with its MLX equivalent during execution.

```python
# Dream API — user changes nothing
mlx_model = torch2mlx.lightning.convert(MyLightningModule.load_from_checkpoint("model.ckpt"))
trainer = MLXTrainer(max_epochs=5)
trainer.fit(mlx_model, train_dataloader)
```

This requires either: (a) a torch.fx full-graph capture + complete op_mapping coverage, or (b) a runtime proxy layer that intercepts tensor operations — essentially `torch.compile` targeting MLX instead of Triton. The long tail of edge cases (custom ops, dynamic shapes, Python control flow) makes this a compiler project. Superseded by Level 1+2 for most practical use cases.

#### Recommended rollout

1. Ship Level 1 (MLXModule + MLXTrainer) — immediately useful, validates the API
2. Build Level 2 (fx draft generation) — reduces user effort to reviewing generated code
3. Evaluate Level 3 based on demand — may not be needed if Level 2 coverage is high enough

---

## Current state

| Metric | Count |
|---|---|
| Supported layers | 62 |
| Supported ops | 30 |
| Weight transposition rules | 7 |
| Dtype mappings | 12 |
| Blocker patterns detected | 6 |
| Tests | 292 |
| Templates | 5 |
| Validation examples | 3 |
| HuggingFace coverage | 22/22 models at 100% |
