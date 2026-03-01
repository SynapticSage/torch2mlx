"""Numerical equivalence validation: PyTorch vs MLX Transformer.

Trains a small TransformerEncoder classifier on synthetic sequence data,
converts weights via torch2mlx, loads into an equivalent MLX model,
and compares outputs.

The task: classify sequences by whether their mean token ID exceeds
a threshold. Simple enough to learn in a few epochs, complex enough
to exercise the full attention + FFN + LayerNorm pipeline.

Usage:
    python3.10 examples/validate_transformer.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import time

import numpy as np


# ── Synthetic data ────────────────────────────────────────────────────────────

def _make_sequence_data(
    n_samples: int, seq_len: int, vocab_size: int, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate token sequences with binary labels.

    Label 1 if the mean token ID > vocab_size/2, else 0.
    """
    rng = np.random.RandomState(seed)
    tokens = rng.randint(0, vocab_size, size=(n_samples, seq_len))
    labels = (tokens.mean(axis=1) > vocab_size / 2).astype(np.int64)
    return tokens, labels


# ── PyTorch model ─────────────────────────────────────────────────────────────

def _build_and_train_pytorch(
    train_x: np.ndarray, train_y: np.ndarray,
    vocab_size: int, d_model: int, n_heads: int, n_layers: int,
) -> object:
    """Train a small Transformer classifier, return model in eval mode.

    Uses a custom encoder layer with bias=False on attention (matching MLX's
    MultiHeadAttention which has no bias) while keeping biases on FFN and norms.
    """
    import torch
    import torch.nn as nn

    class EncoderLayer(nn.Module):
        """TransformerEncoderLayer with bias-free attention (MLX-compatible)."""
        def __init__(self):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(
                d_model, n_heads, bias=False, batch_first=True,
            )
            self.linear1 = nn.Linear(d_model, d_model * 4)
            self.linear2 = nn.Linear(d_model * 4, d_model)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, x):
            # Pre-norm style matches MLX's TransformerEncoder default
            residual = x
            x = self.norm1(x)
            x, _ = self.self_attn(x, x, x)
            x = residual + x
            residual = x
            x = self.norm2(x)
            x = self.linear2(torch.relu(self.linear1(x)))
            return residual + x

    class TransformerClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
            self.classifier = nn.Linear(d_model, 2)

        def forward(self, x):
            x = self.embed(x)                   # (B, seq, d_model)
            for layer in self.layers:
                x = layer(x)                    # (B, seq, d_model)
            x = x.mean(dim=1)                   # (B, d_model) — mean pooling
            return self.classifier(x)           # (B, 2)

    model = TransformerClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x_train = torch.from_numpy(train_x).long()
    y_train = torch.from_numpy(train_y).long()

    model.train()
    batch_size = 256
    for epoch in range(10):
        perm = torch.randperm(len(x_train))
        total_loss = 0.0
        for i in range(0, len(x_train), batch_size):
            idx = perm[i : i + batch_size]
            logits = model(x_train[idx])
            loss = loss_fn(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        n_batches = (len(x_train) + batch_size - 1) // batch_size
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/10  loss={total_loss / n_batches:.4f}")

    model.eval()
    return model


def _eval_pytorch(model: object, test_x: np.ndarray, test_y: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Run PyTorch inference, return (accuracy, logits, elapsed_seconds)."""
    import torch

    x = torch.from_numpy(test_x).long()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x).numpy()
    elapsed = time.perf_counter() - t0
    preds = logits.argmax(axis=1)
    acc = (preds == test_y).mean()
    return float(acc), logits, elapsed


# ── MLX model ─────────────────────────────────────────────────────────────────

def _build_mlx_model(vocab_size: int, d_model: int, n_heads: int, n_layers: int) -> object:
    """Build equivalent MLX TransformerClassifier.

    Uses manual encoder layers (not nn.TransformerEncoder) so the parameter
    structure matches the PyTorch model exactly — no extra final LayerNorm.
    """
    import mlx.core as mx
    import mlx.nn as nn

    class EncoderLayer(nn.Module):
        """Pre-norm encoder layer matching PyTorch's custom EncoderLayer."""
        def __init__(self):
            super().__init__()
            self.self_attn = nn.MultiHeadAttention(d_model, n_heads)
            self.linear1 = nn.Linear(d_model, d_model * 4)
            self.linear2 = nn.Linear(d_model * 4, d_model)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        def __call__(self, x):
            residual = x
            x = self.norm1(x)
            x = self.self_attn(x, x, x)
            x = residual + x
            residual = x
            x = self.norm2(x)
            x = self.linear2(nn.relu(self.linear1(x)))
            return residual + x

    class TransformerClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.layers = [EncoderLayer() for _ in range(n_layers)]
            self.classifier = nn.Linear(d_model, 2)

        def __call__(self, x):
            x = self.embed(x)                   # (B, seq, d_model)
            for layer in self.layers:
                x = layer(x)                    # (B, seq, d_model)
            x = mx.mean(x, axis=1)             # (B, d_model) — mean pooling
            return self.classifier(x)           # (B, 2)

    return TransformerClassifier()


def _eval_mlx(model: object, test_x: np.ndarray, test_y: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Run MLX inference, return (accuracy, logits, elapsed_seconds)."""
    import mlx.core as mx

    x = mx.array(test_x.astype(np.int32))
    t0 = time.perf_counter()
    logits = model(x)
    mx.eval(logits)
    elapsed = time.perf_counter() - t0
    logits_np = np.array(logits)
    preds = logits_np.argmax(axis=1)
    acc = (preds == test_y).mean()
    return float(acc), logits_np, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import torch2mlx
    import mlx.core as mx

    # Hyperparams
    vocab_size, d_model, n_heads, n_layers = 100, 64, 4, 2
    seq_len = 20

    print("Generating synthetic sequence data...")
    train_x, train_y = _make_sequence_data(5000, seq_len, vocab_size, seed=42)
    test_x, test_y = _make_sequence_data(1000, seq_len, vocab_size, seed=99)
    print(f"  Train: {train_x.shape}, Test: {test_x.shape}")
    print(f"  Label balance: {train_y.mean():.2%} positive")

    # 1. Train PyTorch
    print("\nTraining PyTorch Transformer (10 epochs)...")
    pt_model = _build_and_train_pytorch(
        train_x, train_y, vocab_size, d_model, n_heads, n_layers,
    )

    # 2. Evaluate PyTorch
    print("\nPyTorch inference...")
    pt_acc, pt_logits, pt_time = _eval_pytorch(pt_model, test_x, test_y)
    print(f"  PyTorch accuracy: {pt_acc:.2%}  ({pt_time:.3f}s)")

    # 3. Analyze and convert
    report = torch2mlx.analyze(pt_model)
    print(f"\n  Analyzer coverage: {report.coverage:.0%} ({report.mapped_layers}/{report.total_layers})")

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "transformer.safetensors"
        torch2mlx.convert(pt_model, out_path, analyze_first=False)
        print(f"  Converted to {out_path.stat().st_size / 1024:.1f} KB")

        # 4. Load into MLX
        # PyTorch TransformerEncoder and MLX TransformerEncoder have different
        # internal structures — we load weights manually with key remapping
        print("\nLoading into MLX model...")
        flat = torch2mlx.state_dict.load_safetensors(out_path)

        # Key mapping: PyTorch → MLX
        # Both models now use the same custom structure, so the only
        # remapping needed is splitting PyTorch's fused in_proj_weight
        # into separate query/key/value projections for MLX.
        #
        # PyTorch: layers.{i}.self_attn.in_proj_weight  [3*d, d]
        #          layers.{i}.self_attn.out_proj.weight  [d, d]
        #          layers.{i}.{linear1,linear2,norm1,norm2}.{weight,bias}
        # MLX:     layers.{i}.self_attn.{query,key,value}_proj.weight [d, d]
        #          layers.{i}.self_attn.out_proj.weight  [d, d]
        #          layers.{i}.{linear1,linear2,norm1,norm2}.{weight,bias}

        mlx_weights = {}
        for k, v in flat.items():
            # Split fused in_proj_weight into separate Q/K/V
            if "self_attn.in_proj_weight" in k:
                prefix = k.replace("self_attn.in_proj_weight", "self_attn.")
                q, k_w, v_w = np.split(v, 3, axis=0)
                mlx_weights[f"{prefix}query_proj.weight"] = q
                mlx_weights[f"{prefix}key_proj.weight"] = k_w
                mlx_weights[f"{prefix}value_proj.weight"] = v_w
                continue
            mlx_weights[k] = v

        mlx_model = _build_mlx_model(vocab_size, d_model, n_heads, n_layers)
        weight_pairs = [(k, mx.array(v)) for k, v in mlx_weights.items()]
        mlx_model.load_weights(weight_pairs)

    # 5. Evaluate MLX
    print("MLX inference...")
    mlx_acc, mlx_logits, mlx_time = _eval_mlx(mlx_model, test_x, test_y)
    print(f"  MLX accuracy: {mlx_acc:.2%}  ({mlx_time:.3f}s)")

    # 6. Compare
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  PyTorch accuracy:    {pt_acc:.2%}")
    print(f"  MLX accuracy:        {mlx_acc:.2%}")
    print(f"  Accuracy delta:      {abs(pt_acc - mlx_acc):.4%}")

    max_diff = np.abs(pt_logits - mlx_logits).max()
    mean_diff = np.abs(pt_logits - mlx_logits).mean()
    print(f"\n  Max logit diff:      {max_diff:.6f}")
    print(f"  Mean logit diff:     {mean_diff:.6f}")

    pt_preds = pt_logits.argmax(axis=1)
    mlx_preds = mlx_logits.argmax(axis=1)
    agreement = (pt_preds == mlx_preds).mean()
    print(f"  Prediction agreement:{agreement:.2%}")

    # Speed comparison
    speedup = pt_time / mlx_time if mlx_time > 0 else float("inf")
    faster = "MLX" if mlx_time < pt_time else "PyTorch"
    print(f"\n  PyTorch inference:   {pt_time:.3f}s")
    print(f"  MLX inference:       {mlx_time:.3f}s")
    print(f"  Speedup:             {faster} is {speedup:.1f}x {'faster' if faster == 'MLX' else 'slower'}")

    print("\n" + "-" * 60)
    if max_diff < 1e-3 and agreement == 1.0:
        print("  PASS: Numerically equivalent")
    elif agreement > 0.99:
        print(f"  PASS: Functionally equivalent ({agreement:.2%} agreement)")
    else:
        print(f"  FAIL: Significant divergence (agreement={agreement:.2%})")
    print("-" * 60)


if __name__ == "__main__":
    main()
