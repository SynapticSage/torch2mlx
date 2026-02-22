"""Tests for converter.py — end-to-end conversion pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from torch2mlx.converter import build_module_map, convert, convert_state_dict, load_converted
from torch2mlx.state_dict import save_safetensors

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# ---------------------------------------------------------------------------
# convert_state_dict
# ---------------------------------------------------------------------------

def test_convert_state_dict_applies_conv2d_transposition():
    """conv2d rule: [O, I, H, W] -> [O, H, W, I]."""
    arr = np.zeros((8, 3, 5, 5), dtype=np.float32)
    flat = {"conv.weight": arr}
    module_map = {"conv": "conv2d"}
    result = convert_state_dict(flat, module_map)
    assert result["conv.weight"].shape == (8, 5, 5, 3)


def test_convert_state_dict_bias_passthrough():
    """Bias keys must NOT be transposed even when module prefix matches."""
    bias = np.zeros((16,), dtype=np.float32)
    flat = {"conv.bias": bias}
    module_map = {"conv": "conv2d"}
    result = convert_state_dict(flat, module_map)
    assert result["conv.bias"].shape == (16,)
    np.testing.assert_array_equal(result["conv.bias"], bias)


def test_convert_state_dict_no_prefix_match():
    """Keys with no matching prefix pass through unchanged."""
    arr = np.ones((4, 2), dtype=np.float32)
    flat = {"unknown.weight": arr}
    module_map = {"fc": "identity"}
    result = convert_state_dict(flat, module_map)
    np.testing.assert_array_equal(result["unknown.weight"], arr)


def test_convert_state_dict_identity_preserves_shape():
    """identity rule leaves weight shape unchanged."""
    arr = np.zeros((20, 10), dtype=np.float32)
    flat = {"fc.weight": arr}
    module_map = {"fc": "identity"}
    result = convert_state_dict(flat, module_map)
    assert result["fc.weight"].shape == (20, 10)


def test_convert_state_dict_longest_prefix_wins():
    """Longer prefix beats shorter when both match."""
    # key "enc.conv.weight" — module_map has both "enc" and "enc.conv"
    arr = np.zeros((8, 3, 5, 5), dtype=np.float32)
    flat = {"enc.conv.weight": arr}
    module_map = {"enc": "identity", "enc.conv": "conv2d"}
    result = convert_state_dict(flat, module_map)
    # enc.conv -> conv2d: [O,I,H,W] -> [O,H,W,I]
    assert result["enc.conv.weight"].shape == (8, 5, 5, 3)


def test_convert_state_dict_non_weight_keys_untouched():
    """running_mean, running_var etc. should pass through unchanged."""
    running_mean = np.zeros((16,), dtype=np.float32)
    flat = {"bn.running_mean": running_mean, "bn.weight": np.ones((16,))}
    module_map = {"bn": "batch_norm"}
    result = convert_state_dict(flat, module_map)
    np.testing.assert_array_equal(result["bn.running_mean"], running_mean)


# ---------------------------------------------------------------------------
# build_module_map
# ---------------------------------------------------------------------------

def test_build_module_map_known_classes():
    """Known torch class names are mapped to their transposition rules."""
    named = [("fc", "Linear"), ("conv", "Conv2d"), ("embed", "Embedding")]
    result = build_module_map(named)
    assert result["fc"] == "identity"
    assert result["conv"] == "conv2d"
    assert result["embed"] == "identity"


def test_build_module_map_unknown_class_omitted():
    """Unknown class names are silently omitted from the map."""
    named = [("custom", "MySpecialLayer"), ("fc", "Linear")]
    result = build_module_map(named)
    assert "custom" not in result
    assert "fc" in result


def test_build_module_map_empty():
    """Empty input yields empty map."""
    assert build_module_map([]) == {}


def test_build_module_map_conv1d_rule():
    """Conv1d maps to conv1d transposition rule."""
    result = build_module_map([("conv1", "Conv1d")])
    assert result["conv1"] == "conv1d"


# ---------------------------------------------------------------------------
# load_converted roundtrip
# ---------------------------------------------------------------------------

def test_load_converted_roundtrip(tmp_path):
    """save_safetensors -> load_converted produces correct nested structure."""
    flat = {
        "encoder.fc.weight": np.zeros((10, 5), dtype=np.float32),
        "encoder.fc.bias": np.zeros((10,), dtype=np.float32),
    }
    out = tmp_path / "model.safetensors"
    save_safetensors(flat, out)

    nested = load_converted(out)
    assert "encoder" in nested
    assert "fc" in nested["encoder"]
    assert "weight" in nested["encoder"]["fc"]
    assert nested["encoder"]["fc"]["weight"].shape == (10, 5)


def test_load_converted_flat_key(tmp_path):
    """Single-level key is preserved correctly."""
    flat = {"weight": np.eye(3, dtype=np.float32)}
    out = tmp_path / "model.safetensors"
    save_safetensors(flat, out)
    nested = load_converted(out)
    np.testing.assert_array_equal(nested["weight"], np.eye(3))


# ---------------------------------------------------------------------------
# Torch-dependent end-to-end tests
# ---------------------------------------------------------------------------

@requires_torch
def test_end_to_end_linear(tmp_path):
    """nn.Linear converts and loads back with correct shapes."""
    model = nn.Linear(10, 20)
    out = tmp_path / "linear.safetensors"
    convert(model, out)

    nested = load_converted(out)
    # Linear weight: [O, I] -> identity, shape unchanged
    assert nested["weight"].shape == (20, 10)
    assert nested["bias"].shape == (20,)


@requires_torch
def test_end_to_end_conv1d(tmp_path):
    """nn.Conv1d weight shape is transposed from [O,I,K] to [O,K,I]."""
    model = nn.Conv1d(3, 16, 5)  # weight: [16, 3, 5]
    out = tmp_path / "conv1d.safetensors"
    convert(model, out)

    nested = load_converted(out)
    # conv1d rule: [O, I, K] -> [O, K, I] = [16, 5, 3]
    assert nested["weight"].shape == (16, 5, 3)


@requires_torch
def test_end_to_end_dict_passthrough(tmp_path):
    """Passing a plain dict skips transpositions and saves unchanged."""
    flat = {"fc.weight": np.zeros((4, 2), dtype=np.float32)}
    out = tmp_path / "dict.safetensors"
    convert(flat, out)
    nested = load_converted(out)
    assert nested["fc"]["weight"].shape == (4, 2)


@requires_torch
def test_convert_returns_path(tmp_path):
    """convert() returns a Path to the saved file."""
    model = nn.Linear(4, 8)
    out = tmp_path / "out.safetensors"
    result = convert(model, out)
    assert isinstance(result, Path)
    assert result.exists()
