"""Tests for MLX template modules (MLP, TransformerBlock, ConvBlock, ConvStack).

All tests are skipped if mlx is not installed.
"""

from __future__ import annotations

import pytest

mlx = pytest.importorskip("mlx")

import mlx.core as mx
from torch2mlx.templates.mlp import MLP
from torch2mlx.templates.transformer import TransformerBlock
from torch2mlx.templates.cnn import ConvBlock, ConvStack


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


def test_mlp_forward_shape():
    model = MLP([64, 128, 64])
    x = mx.random.normal((2, 64))
    out = model(x)
    mx.eval(out)
    assert out.shape == (2, 64)


@pytest.mark.parametrize("activation", ["gelu", "relu", "silu"])
def test_mlp_activations(activation: str):
    model = MLP([32, 64, 32], activation=activation)
    x = mx.random.normal((4, 32))
    out = model(x)
    mx.eval(out)
    assert out.shape == (4, 32)


def test_mlp_with_dropout():
    model = MLP([64, 128, 64], dropout=0.1)
    x = mx.random.normal((2, 64))
    out = model(x)
    mx.eval(out)
    assert out.shape == (2, 64)


def test_mlp_no_activation_after_final_layer():
    # Single linear layer: dims=[64, 32] — no activation at all
    model = MLP([64, 32])
    x = mx.random.normal((3, 64))
    out = model(x)
    mx.eval(out)
    assert out.shape == (3, 32)


def test_mlp_invalid_activation():
    with pytest.raises(ValueError, match="activation must be one of"):
        MLP([64, 32], activation="tanh")


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------


def test_transformer_block_forward_shape():
    model = TransformerBlock(d_model=64, n_heads=4)
    x = mx.random.normal((2, 10, 64))
    out = model(x)
    mx.eval(out)
    assert out.shape == (2, 10, 64)


def test_transformer_block_prenorm():
    model = TransformerBlock(64, 4, norm_first=True)
    x = mx.random.normal((2, 10, 64))
    out = model(x)
    mx.eval(out)
    assert out.shape == (2, 10, 64)


def test_transformer_block_postnorm():
    model = TransformerBlock(64, 4, norm_first=False)
    x = mx.random.normal((2, 10, 64))
    out = model(x)
    mx.eval(out)
    assert out.shape == (2, 10, 64)


def test_transformer_block_custom_dff():
    model = TransformerBlock(64, 4, d_ff=512)
    x = mx.random.normal((1, 8, 64))
    out = model(x)
    mx.eval(out)
    assert out.shape == (1, 8, 64)


# ---------------------------------------------------------------------------
# ConvBlock
# ---------------------------------------------------------------------------


def test_conv_block_1d_shape():
    # MLX expects channels-last: (batch, length, channels)
    model = ConvBlock(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    x = mx.random.normal((2, 100, 3))
    out = model(x)
    mx.eval(out)
    assert out.shape == (2, 100, 16)


def test_conv_block_2d_shape():
    # MLX expects channels-last: (batch, H, W, channels)
    model = ConvBlock(in_channels=3, out_channels=16, kernel_size=3, padding=1, conv_type="2d")
    x = mx.random.normal((2, 32, 32, 3))
    out = model(x)
    mx.eval(out)
    assert out.shape == (2, 32, 32, 16)


def test_conv_block_invalid_conv_type():
    with pytest.raises(ValueError, match="conv_type must be one of"):
        ConvBlock(3, 16, conv_type="3d")


# ---------------------------------------------------------------------------
# ConvStack
# ---------------------------------------------------------------------------


def test_conv_stack_1d_shape():
    # channels-last: (batch, length, channels)
    # Two k=3 convs, no padding: 64 -> 62 -> 60
    model = ConvStack(channels=[3, 16, 32], kernel_sizes=3)
    x = mx.random.normal((2, 64, 3))
    out = model(x)
    mx.eval(out)
    assert out.shape == (2, 60, 32)


def test_conv_stack_2d_shape():
    # Two k=3 convs, no padding: 32 -> 30 -> 28
    model = ConvStack(channels=[3, 8, 16], kernel_sizes=3, conv_type="2d")
    x = mx.random.normal((2, 32, 32, 3))
    out = model(x)
    mx.eval(out)
    assert out.shape == (2, 28, 28, 16)


def test_conv_stack_per_layer_kernel_sizes():
    model = ConvStack(channels=[3, 16, 32], kernel_sizes=[3, 5])
    x = mx.random.normal((1, 64, 3))
    out = model(x)
    mx.eval(out)
    # With no padding: 64 ->(k=3) 62 ->(k=5) 58
    assert out.shape[0] == 1
    assert out.shape[2] == 32
