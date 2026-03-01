"""Numerical equivalence validation: PyTorch vs MLX ResNet.

Trains a small ResNet-style CNN on Fashion-MNIST, converts weights via
torch2mlx, loads into an equivalent MLX model, and compares outputs.

Exercises: Conv2d, BatchNorm2d, ReLU, skip connections, AdaptiveAvgPool2d,
Linear — the core building blocks of residual networks.

Usage:
    python3.10 examples/validate_resnet.py
"""

from __future__ import annotations

import gzip
import struct
import tempfile
import urllib.request
from pathlib import Path

import time

import numpy as np


# ── Fashion-MNIST loading ────────────────────────────────────────────────────

FMNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}


def _fetch_data(cache_dir: Path) -> dict[str, np.ndarray]:
    """Download and parse MNIST IDX files, caching locally."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for key, url in FMNIST_URLS.items():
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

def _build_and_train_pytorch(
    train_x: np.ndarray, train_y: np.ndarray,
) -> object:
    """Train a small ResNet on MNIST, return model in eval mode."""
    import torch
    import torch.nn as nn

    class ResBlock(nn.Module):
        """Basic residual block: two 3x3 convs with skip connection."""
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            residual = x
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return torch.relu(out + residual)

    class MiniResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )
            self.block1 = ResBlock(16)
            self.pool = nn.MaxPool2d(2)        # 28→14
            self.expand = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
            self.block2 = ResBlock(32)
            self.pool2 = nn.MaxPool2d(2)       # 14→7
            self.avgpool = nn.AdaptiveAvgPool2d(1)  # 7→1
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            x = self.stem(x)
            x = self.block1(x)
            x = self.pool(x)
            x = self.expand(x)
            x = self.block2(x)
            x = self.pool2(x)
            x = self.avgpool(x)                # (B, 32, 1, 1)
            x = x.flatten(1)                   # (B, 32)
            return self.fc(x)

    model = MiniResNet()
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
    """Build the MLX equivalent of MiniResNet."""
    import mlx.core as mx
    import mlx.nn as nn

    class ResBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm(channels)

        def __call__(self, x):
            residual = x
            out = nn.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return nn.relu(out + residual)

    class MiniResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem_conv = nn.Conv2d(1, 16, 3, padding=1, bias=False)
            self.stem_bn = nn.BatchNorm(16)
            self.block1 = ResBlock(16)
            self.pool = nn.MaxPool2d(2)
            self.expand_conv = nn.Conv2d(16, 32, 3, padding=1, bias=False)
            self.expand_bn = nn.BatchNorm(32)
            self.block2 = ResBlock(32)
            self.pool2 = nn.MaxPool2d(2)
            self.fc = nn.Linear(32, 10)

        def __call__(self, x):
            # Stem
            x = nn.relu(self.stem_bn(self.stem_conv(x)))
            x = self.block1(x)
            x = self.pool(x)
            # Expand
            x = nn.relu(self.expand_bn(self.expand_conv(x)))
            x = self.block2(x)
            x = self.pool2(x)                          # (B, 7, 7, 32) NHWC
            # Global average pool: mean over spatial dims
            x = mx.mean(x, axis=(1, 2))               # (B, 32)
            return self.fc(x)

    return MiniResNet()


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


# ── Key remapping ─────────────────────────────────────────────────────────────

def _remap_resnet_keys(flat: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Remap PyTorch nn.Sequential keys to MLX flat-attribute keys.

    PyTorch uses nn.Sequential for stem/expand, producing keys like
    stem.0.weight, stem.1.weight. MLX uses flat attributes (stem_conv,
    stem_bn), so we remap accordingly.
    """
    out = {}
    remap = {
        "stem.0.": "stem_conv.",
        "stem.1.": "stem_bn.",
        "expand.0.": "expand_conv.",
        "expand.1.": "expand_bn.",
    }
    for k, v in flat.items():
        # PyTorch BatchNorm tracks batch count — MLX doesn't need it
        if "num_batches_tracked" in k:
            continue
        new_key = k
        for old, new in remap.items():
            new_key = new_key.replace(old, new)
        out[new_key] = v
    return out


# ── Main validation ──────────────────────────────────────────────────────────

def main() -> None:
    import torch2mlx
    import mlx.core as mx

    cache_dir = Path.home() / ".cache" / "torch2mlx" / "mnist"
    print("Loading MNIST...")
    data = _fetch_data(cache_dir)
    train_x, train_y = data["train_images"], data["train_labels"]
    test_x, test_y = data["test_images"], data["test_labels"]
    print(f"  Train: {train_x.shape}, Test: {test_x.shape}")

    # 1. Train PyTorch model
    print("\nTraining PyTorch MiniResNet (5 epochs, 10k samples)...")
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
        out_path = Path(tmp) / "resnet.safetensors"
        print("\nConverting weights...")
        torch2mlx.convert(pt_model, out_path, analyze_first=False)
        print(f"  Saved {out_path.stat().st_size / 1024:.1f} KB")

        # 4. Load into MLX model
        print("\nLoading into MLX model...")
        flat = torch2mlx.state_dict.load_safetensors(out_path)

        # Remap Sequential keys to flat-attribute keys
        remapped = _remap_resnet_keys(flat)
        mlx_model = _build_mlx_model()
        weights = [(k, mx.array(v)) for k, v in remapped.items()]
        mlx_model.load_weights(weights)
        mlx_model.eval()  # Critical: use running stats, not batch stats

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
    if max_diff < 1e-4 and agreement == 1.0:
        print("  PASS: Numerically equivalent (max diff < 1e-4)")
    elif agreement > 0.99:
        print(f"  PASS: Functionally equivalent ({agreement:.2%} agreement)")
    else:
        print(f"  FAIL: Significant divergence (agreement={agreement:.2%})")
    print("-" * 60)


if __name__ == "__main__":
    main()
