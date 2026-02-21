# torch2mlx

Translate PyTorch neural network models to Apple's MLX framework.

**Approach**: Module-Tree Walk + Weight Convert (Approach C from plan.md)
- Mechanically convert weights via registry dispatch
- Hand-port forward() logic using MLX templates
- Analyzer reports what's auto-convertible vs needs manual work

**Scope**: Inference-only conversion of feed-forward architectures (MLP, CNN, standard transformers). Training loops, custom CUDA kernels, and in-place mutation patterns are out of scope.

## Commands

```bash
python3.10 -m pip install -e ".[dev]"   # Install with dev deps
python3.10 -m pip install -e ".[all]"   # Install with torch + mlx + dev
python3.10 -m pytest                     # Run tests
ruff check src/                          # Lint
ruff format src/ tests/                  # Format
```

## Architecture

```
src/torch2mlx/
├── registry.py          # torch.nn.X → mlx.nn.X dispatch table
├── weight_converter.py  # Per-layer transposition rules (numpy only)
├── state_dict.py        # Flat keys ↔ nested dict conversion
├── op_mapping.py        # torch.cat → mx.concatenate etc.
├── analyzer.py          # Portability report: % convertible, blockers
├── converter.py         # End-to-end orchestration
└── templates/           # Hand-written MLX module implementations
    ├── transformer.py   # Attention + FFN + LayerNorm
    ├── cnn.py           # Conv stacks (channels-last)
    └── mlp.py           # Linear + activation stacks
```

### Module dependency graph

```
registry.py, op_mapping.py          ← foundation, no internal deps
    ↓
weight_converter.py, state_dict.py  ← depend on registry (layer types)
analyzer.py                         ← depends on registry + op_mapping
templates/*                         ← depend on registry
    ↓
converter.py                        ← wires everything together
```

## Domain knowledge

### Weight transposition rules

Dispatch is keyed on **module type**, not tensor shape. Identical shapes can need different transpositions:

| Layer | Torch layout | MLX layout | Rule |
|-------|-------------|------------|------|
| Conv1d | `[O, I, K]` | `[O, K, I]` | swap axes 1,2 |
| Conv2d | `[O, I, H, W]` | `[O, H, W, I]` | move I to last |
| ConvTranspose1d | `[I, O, K]` | `[O, K, I]` | transpose (1,2,0) |
| Linear | `[O, I]` | `[O, I]` | identity |
| Embedding | `[V, D]` | `[V, D]` | identity |

### State dict surgery

- PyTorch: flat keys like `encoder.layers.0.self_attn.q_proj.weight`
- MLX: nested dicts `{"encoder": {"layers": {"0": {"self_attn": ...}}}}`
- Safetensors is the interchange format (backend-agnostic, numpy-only I/O)

### Immutability constraint

MLX arrays are immutable — no `.copy_()`, no `+=`, no `x[i] = v`. This is the biggest architectural mismatch with PyTorch. The analyzer should flag in-place patterns as blockers.

## Conventions

- **Registry-driven**: All layer/op mappings go through registries, never ad-hoc
- **Backend-agnostic weight conversion**: `weight_converter.py` and `state_dict.py` use only numpy, never import torch or mlx
- **Test tolerances**: `atol=1e-5` for float32, `atol=1e-2` for float16
- **src layout**: Package lives under `src/torch2mlx/`, requires install to import
- **Type hints**: Use `from __future__ import annotations` in all modules

## Team structure (for parallel implementation)

### Phase 1 — Foundation (no deps)

**registry-agent** owns `registry.py` + `op_mapping.py`
- Build dispatch tables mapping torch→MLX layers and ops
- Pure data work: dataclasses + dicts

### Phase 2 — Parallel (depends on registry)

**weights-agent** owns `weight_converter.py` + `state_dict.py`
- Transposition rules per layer type, flat↔nested key conversion
- Safetensors I/O, numpy only

**analyzer-agent** owns `analyzer.py`
- Walk torch Module tree, produce portability report
- Coverage %, unmapped layers, blocker patterns

**templates-agent** owns `templates/*.py`
- MLX implementations of transformer, CNN, MLP patterns
- Hand-written forward() equivalents

### Phase 3 — Sequential (depends on all above)

**converter-agent** owns `converter.py`
- End-to-end orchestration: walk → build → convert → load → verify

### Phase 4 — Validation

**test-agent** owns `tests/`
- Numerical equivalence, roundtrip verification, shape validation
