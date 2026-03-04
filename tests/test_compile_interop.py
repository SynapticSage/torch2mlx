"""Tests for torch.compile interop.

Validates that torch2mlx.convert() handles torch.compile-wrapped models
correctly. Key behavior: torch.compile wraps models in OptimizedModule,
which prefixes all state_dict keys with ``_orig_mod.``. The converter
should still produce valid safetensors — weights are numerically identical
to the unwrapped model, just nested under an extra ``_orig_mod`` level.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

# torch.compile requires Python 3.8+ and torch >= 2.0
HAS_COMPILE = HAS_TORCH and hasattr(torch, "compile")

requires_compile = pytest.mark.skipif(not HAS_COMPILE, reason="torch.compile not available")


def _make_simple_model():
    """Small model exercising Linear + ReLU (common compile target)."""
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


# ---------------------------------------------------------------------------
# Core interop: compiled model converts without errors
# ---------------------------------------------------------------------------


@requires_compile
def test_compile_model_converts(tmp_path):
    """torch.compile-wrapped model converts to safetensors without error."""
    from torch2mlx.converter import convert

    model = _make_simple_model()
    compiled = torch.compile(model)

    out = tmp_path / "compiled.safetensors"
    result = convert(compiled, out, analyze_first=False)
    assert result.exists()


@requires_compile
def test_compile_weights_match_original(tmp_path):
    """Compiled and original models produce identical weight values.

    torch.compile wraps the module in OptimizedModule, which nests keys
    under ``_orig_mod.``. The raw weight values should be numerically
    identical — only the key prefix differs.
    """
    from torch2mlx.converter import convert, load_converted

    model = _make_simple_model()
    compiled = torch.compile(model)

    orig_out = tmp_path / "original.safetensors"
    compiled_out = tmp_path / "compiled.safetensors"

    convert(model, orig_out, analyze_first=False)
    convert(compiled, compiled_out, analyze_first=False)

    orig = load_converted(orig_out)
    comp = load_converted(compiled_out)

    # Compiled output nests under _orig_mod; strip that for comparison
    assert "_orig_mod" in comp, (
        f"Expected '_orig_mod' wrapper in compiled output, got keys: {list(comp.keys())}"
    )
    inner = comp["_orig_mod"]

    # Same structure once unwrapped
    assert set(_leaf_keys(orig)) == set(_leaf_keys(inner)), (
        f"Key mismatch: orig={set(_leaf_keys(orig))}, compiled={set(_leaf_keys(inner))}"
    )

    # Same weight values
    for key_path, orig_arr in _leaf_items(orig):
        comp_arr = _nested_get(inner, key_path)
        np.testing.assert_array_equal(
            orig_arr,
            comp_arr,
            err_msg=f"Weight mismatch at {'.'.join(key_path)}",
        )


@requires_compile
def test_compile_named_modules_class_names(tmp_path):
    """Registry recognizes class names through the OptimizedModule wrapper.

    The converter walks named_modules() to build the module map. With
    torch.compile, modules are prefixed ``_orig_mod.`` but class names
    (Linear, ReLU) stay the same — registry lookup should still work.
    """
    from torch2mlx.converter import build_module_map

    model = _make_simple_model()
    compiled = torch.compile(model)

    named = [(name, type(m).__name__) for name, m in compiled.named_modules()]

    module_map = build_module_map(named)

    # Linear modules should be found (with _orig_mod prefix)
    assert "_orig_mod.0" in module_map  # first Linear
    assert "_orig_mod.2" in module_map  # second Linear
    # Both should map to identity (Linear rule)
    assert module_map["_orig_mod.0"] == "identity"
    assert module_map["_orig_mod.2"] == "identity"


@requires_compile
def test_compile_with_analysis(tmp_path):
    """Analyzer works on compiled models (walks through OptimizedModule)."""
    from torch2mlx.converter import convert

    model = _make_simple_model()
    compiled = torch.compile(model)

    out = tmp_path / "analyzed_compiled.safetensors"
    # analyze_first=True (default) should not crash
    result = convert(compiled, out)
    assert result.exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _leaf_keys(d, prefix=()):
    """Yield dot-joined paths to all leaf (non-dict) values."""
    for k, v in d.items():
        path = (*prefix, k)
        if isinstance(v, dict):
            yield from _leaf_keys(v, path)
        else:
            yield ".".join(path)


def _leaf_items(d, prefix=()):
    """Yield (path_tuple, value) for all leaf values."""
    for k, v in d.items():
        path = (*prefix, k)
        if isinstance(v, dict):
            yield from _leaf_items(v, path)
        else:
            yield path, v


def _nested_get(d, keys):
    """Traverse nested dict by key tuple."""
    for k in keys:
        d = d[k]
    return d
