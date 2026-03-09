"""End-to-end codegen validation: generate → convert → load → compare outputs.

These tests verify that generated MLX module source code can:
1. Parse without errors
2. Instantiate an mlx.nn.Module
3. Load converted weights
4. Produce numerically equivalent outputs to the PyTorch original
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mx = pytest.importorskip("mlx.core")
mlx_nn = pytest.importorskip("mlx.nn")

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from torch2mlx import convert, load_converted  # noqa: E402
from torch2mlx.codegen import generate  # noqa: E402
from torch2mlx.state_dict import flatten  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_e2e(
    model: nn.Module,
    input_shape: tuple[int, ...],
    *,
    atol: float = 1e-5,
    class_name: str | None = None,
) -> None:
    """Generate MLX code, convert weights, load, and compare outputs."""
    model.eval()
    result = generate(model, class_name=class_name)

    # Source must parse
    ast.parse(result.source)

    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "weights")
        convert(model, out)
        params = load_converted(out)

        flat = flatten(params)
        weights = [(k, mx.array(v)) for k, v in flat.items()]

        # Exec generated source
        ns: dict = {}
        exec(result.source, ns)
        cls_name = class_name or type(model).__name__
        mlx_model = ns[cls_name]()
        mlx_model.load_weights(weights)

        # Run both
        x_torch = torch.randn(*input_shape)
        with torch.no_grad():
            y_torch = model(x_torch).numpy()

        x_mlx = mx.array(x_torch.numpy())
        y_mlx = np.array(mlx_model(x_mlx))

        diff = np.abs(y_torch - y_mlx).max()
        assert diff < atol, (
            f"Max diff {diff:.2e} exceeds atol={atol:.0e}\n"
            f"PyTorch: {y_torch.ravel()[:5]}\n"
            f"MLX:     {y_mlx.ravel()[:5]}"
        )


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(32)
        self.proj = nn.Linear(32, 32)
        self.norm2 = nn.LayerNorm(32)
        self.ff1 = nn.Linear(32, 64)
        self.ff2 = nn.Linear(64, 32)

    def forward(self, x):
        h = self.proj(self.norm1(x))
        x = x + h
        h = self.ff2(F.gelu(self.ff1(self.norm2(x))))
        return x + h


class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 16)
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc(self.embed(x))


class MultiLayerNorm(nn.Module):
    """Model with multiple differently-sized LayerNorms."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.norm1 = nn.LayerNorm(16)
        self.fc2 = nn.Linear(16, 8)
        self.norm2 = nn.LayerNorm(8)

    def forward(self, x):
        x = self.norm1(F.silu(self.fc1(x)))
        return self.norm2(self.fc2(x))


class DeepMLP(nn.Module):
    """Deeper model to test longer fx graphs."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class BareLinear(nn.Module):
    """Root-as-leaf: a bare Linear layer (no children)."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 8))
        self.bias = nn.Parameter(torch.randn(4))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------


class TestCodegenE2E:
    """End-to-end: generate → convert → load → compare."""

    def test_tiny_mlp(self):
        _run_e2e(TinyMLP(), (2, 4))

    def test_transformer_block(self):
        _run_e2e(TransformerBlock(), (2, 8, 32))

    def test_embedding(self):
        model = EmbeddingNet()
        model.eval()
        result = generate(model)
        ast.parse(result.source)

        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "weights")
            convert(model, out)
            params = load_converted(out)

            flat = flatten(params)
            weights = [(k, mx.array(v)) for k, v in flat.items()]

            ns: dict = {}
            exec(result.source, ns)
            mlx_model = ns["EmbeddingNet"]()
            mlx_model.load_weights(weights)

            # Embedding takes int input
            x_torch = torch.randint(0, 100, (2, 5))
            with torch.no_grad():
                y_torch = model(x_torch).numpy()

            x_mlx = mx.array(x_torch.numpy())
            y_mlx = np.array(mlx_model(x_mlx))

            diff = np.abs(y_torch - y_mlx).max()
            assert diff < 1e-5

    def test_multi_layernorm(self):
        _run_e2e(MultiLayerNorm(), (2, 8))

    def test_deep_mlp(self):
        _run_e2e(DeepMLP(), (3, 8))

    def test_custom_class_name(self):
        _run_e2e(TinyMLP(), (2, 4), class_name="MyCustomMLP")


class TestCodegenE2ECoverage:
    """Verify coverage and metadata from e2e runs."""

    def test_mlp_coverage(self):
        result = generate(TinyMLP())
        assert result.coverage == 1.0
        assert result.traced is True
        assert len(result.unmapped) == 0

    def test_transformer_coverage(self):
        result = generate(TransformerBlock())
        assert result.coverage == 1.0
        assert result.traced is True

    def test_source_parses(self):
        for model in [TinyMLP(), TransformerBlock(), MultiLayerNorm(), DeepMLP()]:
            result = generate(model)
            ast.parse(result.source)  # Must not raise
