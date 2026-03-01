"""Numerical equivalence validation: PyTorch vs MLX on MNIST.

Trains a small CNN in PyTorch, converts weights via torch2mlx,
loads into an equivalent MLX model, and compares:
  1. Per-sample logit agreement (max abs difference)
  2. Classification accuracy on test set
  3. Prediction agreement (% of samples where both models agree)

Usage:
    python3.10 examples/validate_mnist.py
"""

from __future__ import annotations

import gzip
import struct
import tempfile
import urllib.request
from pathlib import Path

import time

import numpy as np

# ── MNIST loading (no torchvision dependency) ─────────────────────────────────

MNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}


def _fetch_mnist(cache_dir: Path) -> dict[str, np.ndarray]:
    """Download and parse MNIST IDX files, caching locally."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for key, url in MNIST_URLS.items():
        cached = cache_dir / f"{key}.npy"
        if cached.exists():
            data[key] = np.load(cached)
            continue

        print(f"  Downloading {key}...")
        resp = urllib.request.urlopen(url, timeout=30)
        raw = gzip.decompress(resp.read())

        if "images" in key:
            _, n, rows, cols = struct.unpack(">IIII", raw[:16])
            arr = np.frombuffer(raw[16:], dtype=np.uint8).reshape(n, rows, cols)
        else:
            _, n = struct.unpack(">II", raw[:8])
            arr = np.frombuffer(raw[8:], dtype=np.uint8)

        np.save(cached, arr)
        data[key] = arr
    return data


# ── PyTorch model ─────────────────────────────────────────────────────────────

def _build_and_train_pytorch(train_x: np.ndarray, train_y: np.ndarray) -> object:
    """Train a small CNN on MNIST, return the model in eval mode."""
    import torch
    import torch.nn as nn

    class MnistCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(32 * 7 * 7, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))  # (B,16,14,14)
            x = self.pool(torch.relu(self.conv2(x)))  # (B,32,7,7)
            x = x.flatten(1)
            return self.fc(x)

    model = MnistCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Normalize to [0, 1], add channel dim: (N, 1, 28, 28)
    x_train = torch.from_numpy(train_x[:10000].astype(np.float32) / 255.0).unsqueeze(1)
    y_train = torch.from_numpy(train_y[:10000].astype(np.int64))

    model.train()
    batch_size = 256
    for epoch in range(5):
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
        print(f"  Epoch {epoch + 1}/5  loss={total_loss / n_batches:.4f}")

    model.eval()
    return model


def _eval_pytorch(model: object, test_x: np.ndarray, test_y: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Run PyTorch inference, return (accuracy, logits, elapsed_seconds)."""
    import torch

    x = torch.from_numpy(test_x.astype(np.float32) / 255.0).unsqueeze(1)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x).numpy()
    elapsed = time.perf_counter() - t0
    preds = logits.argmax(axis=1)
    acc = (preds == test_y).mean()
    return float(acc), logits, elapsed


# ── MLX model (equivalent architecture, channels-last) ───────────────────────

def _build_mlx_model() -> object:
    """Build the MLX equivalent of MnistCNN."""
    import mlx.core as mx
    import mlx.nn as nn

    class MnistCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(32 * 7 * 7, 10)

        def __call__(self, x):
            x = self.pool(nn.relu(self.conv1(x)))  # (B,14,14,16) channels-last
            x = self.pool(nn.relu(self.conv2(x)))  # (B,7,7,32)
            # Transpose NHWC→NCHW before flatten so element ordering
            # matches the PyTorch-trained Linear weight matrix
            x = mx.transpose(x, (0, 3, 1, 2))     # (B,32,7,7)
            x = mx.flatten(x, 1)
            return self.fc(x)

    return MnistCNN()


def _eval_mlx(model: object, test_x: np.ndarray, test_y: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Run MLX inference, return (accuracy, logits, elapsed_seconds)."""
    import mlx.core as mx

    # channels-last: (N, H, W, C)
    x = mx.array(test_x.astype(np.float32) / 255.0)[..., None]
    t0 = time.perf_counter()
    logits = model(x)
    mx.eval(logits)
    elapsed = time.perf_counter() - t0
    logits_np = np.array(logits)
    preds = logits_np.argmax(axis=1)
    acc = (preds == test_y).mean()
    return float(acc), logits_np, elapsed


# ── Main validation ──────────────────────────────────────────────────────────

def main() -> None:
    import torch2mlx

    cache_dir = Path.home() / ".cache" / "torch2mlx" / "mnist"
    print("Loading MNIST...")
    data = _fetch_mnist(cache_dir)
    train_x, train_y = data["train_images"], data["train_labels"]
    test_x, test_y = data["test_images"], data["test_labels"]
    print(f"  Train: {train_x.shape}, Test: {test_x.shape}")

    # 1. Train PyTorch model
    print("\nTraining PyTorch CNN (5 epochs, 10k samples)...")
    pt_model = _build_and_train_pytorch(train_x, train_y)

    # 2. Evaluate PyTorch
    print("\nPyTorch inference on test set...")
    pt_acc, pt_logits, pt_time = _eval_pytorch(pt_model, test_x, test_y)
    print(f"  PyTorch accuracy: {pt_acc:.2%}  ({pt_time:.3f}s)")

    # 3. Analyze and convert
    print("\nAnalyzing model...")
    report = torch2mlx.analyze(pt_model)
    print(f"  Coverage: {report.coverage:.0%} ({report.mapped_layers}/{report.total_layers})")

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "mnist_cnn.safetensors"
        print("\nConverting weights...")
        torch2mlx.convert(pt_model, out_path, analyze_first=False)
        print(f"  Saved {out_path.stat().st_size / 1024:.1f} KB")

        # 4. Load into MLX model
        print("\nLoading into MLX model...")
        import mlx.core as mx
        mlx_model = _build_mlx_model()
        # load_weights expects flat (dotted-key, mx.array) pairs
        flat = torch2mlx.state_dict.load_safetensors(out_path)
        weights = [(k, mx.array(v)) for k, v in flat.items()]
        mlx_model.load_weights(weights)

    # 5. Evaluate MLX
    print("MLX inference on test set...")
    mlx_acc, mlx_logits, mlx_time = _eval_mlx(mlx_model, test_x, test_y)
    print(f"  MLX accuracy: {mlx_acc:.2%}  ({mlx_time:.3f}s)")

    # 6. Compare
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  PyTorch accuracy:    {pt_acc:.2%}")
    print(f"  MLX accuracy:        {mlx_acc:.2%}")
    print(f"  Accuracy delta:      {abs(pt_acc - mlx_acc):.4%}")

    # Numerical comparison
    max_diff = np.abs(pt_logits - mlx_logits).max()
    mean_diff = np.abs(pt_logits - mlx_logits).mean()
    print(f"\n  Max logit diff:      {max_diff:.6f}")
    print(f"  Mean logit diff:     {mean_diff:.6f}")

    # Prediction agreement
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

    # Pass/fail
    print("\n" + "-" * 60)
    if max_diff < 1e-4 and agreement == 1.0:
        print("  PASS: Numerically equivalent (max diff < 1e-4)")
    elif agreement > 0.99:
        print(f"  PASS: Functionally equivalent ({agreement:.2%} agreement)")
    else:
        print(f"  FAIL: Significant divergence (agreement={agreement:.2%})")
    print("-" * 60)


if __name__ == "__main__":
    main()
