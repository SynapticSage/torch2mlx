"""Tests for LAYER_REGISTRY and OP_REGISTRY dispatch tables."""

from __future__ import annotations

import pytest

from torch2mlx import op_mapping, registry
from torch2mlx.op_mapping import OP_REGISTRY, OpMapping
from torch2mlx.registry import LAYER_REGISTRY, LayerMapping

# ── Expected contents ────────────────────────────────────────────────────────

EXPECTED_LAYERS = [
    ("Linear",            "nn.Linear",            "identity"),
    ("Embedding",         "nn.Embedding",          "identity"),
    ("LayerNorm",         "nn.LayerNorm",          "identity"),
    ("RMSNorm",           "nn.RMSNorm",            "identity"),
    ("Conv1d",            "nn.Conv1d",             "conv1d"),
    ("Conv2d",            "nn.Conv2d",             "conv2d"),
    ("ConvTranspose1d",   "nn.ConvTranspose1d",    "conv_transpose1d"),
    ("BatchNorm1d",       "nn.BatchNorm",          "batch_norm"),
    ("BatchNorm2d",       "nn.BatchNorm",          "batch_norm"),
    ("MultiheadAttention","nn.MultiHeadAttention", "identity"),
    ("GELU",              "nn.GELU",               "identity"),
    ("ReLU",              "nn.ReLU",               "identity"),
    ("SiLU",              "nn.SiLU",               "identity"),
    ("Dropout",           "nn.Dropout",            "identity"),
    ("ModuleList",        "None",                  "identity"),
    ("Sequential",        "None",                  "identity"),
]

EXPECTED_OPS = [
    ("torch.cat",     "mx.concatenate", {"dim": "axis"}),
    ("torch.stack",   "mx.stack",       {"dim": "axis"}),
    ("F.softmax",     "mx.softmax",     {"dim": "axis"}),
    ("x.view",        "mx.reshape",     {}),
    ("x.permute",     "mx.transpose",   {}),
    ("x.transpose",   "mx.swapaxes",    {}),
    ("x.reshape",     "mx.reshape",     {}),
    ("x.to",          "no_op",          {}),
    ("x.contiguous",  "no_op",          {}),
    ("torch.no_grad", "no_op",          {}),
    ("F.relu",        "nn.relu",        {}),
    ("F.gelu",        "nn.gelu",        {}),
    ("F.silu",        "nn.silu",        {}),
]

# ── LAYER_REGISTRY tests ─────────────────────────────────────────────────────

@pytest.mark.parametrize("torch_name,mlx_name,transposition", EXPECTED_LAYERS)
def test_layer_mapping_present(torch_name, mlx_name, transposition):
    """Every expected layer is in the registry with correct fields."""
    m = LAYER_REGISTRY.get(torch_name)
    assert m is not None, f"{torch_name!r} missing from LAYER_REGISTRY"
    assert isinstance(m, LayerMapping)
    assert m.torch_name == torch_name
    assert m.mlx_name == mlx_name
    assert m.weight_transposition == transposition


def test_lookup_returns_correct_mapping():
    m = registry.lookup("Conv2d")
    assert m is not None
    assert m.mlx_name == "nn.Conv2d"
    assert m.weight_transposition == "conv2d"


def test_lookup_returns_none_for_unknown():
    assert registry.lookup("NonExistentLayer") is None
    assert registry.lookup("") is None


def test_registered_names_contains_all():
    names = registry.registered_names()
    expected = {t for t, _, _ in EXPECTED_LAYERS}
    assert expected.issubset(set(names)), f"Missing: {expected - set(names)}"


def test_registry_size():
    assert len(LAYER_REGISTRY) == len(EXPECTED_LAYERS)


# ── OP_REGISTRY tests ────────────────────────────────────────────────────────

@pytest.mark.parametrize("torch_op,mlx_op,param_renames", EXPECTED_OPS)
def test_op_mapping_present(torch_op, mlx_op, param_renames):
    """Every expected op is in the registry with correct fields."""
    m = OP_REGISTRY.get(torch_op)
    assert m is not None, f"{torch_op!r} missing from OP_REGISTRY"
    assert isinstance(m, OpMapping)
    assert m.torch_op == torch_op
    assert m.mlx_op == mlx_op
    assert m.param_renames == param_renames


def test_lookup_op_returns_correct_mapping():
    m = op_mapping.lookup_op("torch.cat")
    assert m is not None
    assert m.mlx_op == "mx.concatenate"
    assert m.param_renames == {"dim": "axis"}


def test_lookup_op_returns_none_for_unknown():
    assert op_mapping.lookup_op("torch.unknown_op") is None
    assert op_mapping.lookup_op("") is None


def test_op_registry_size():
    assert len(OP_REGISTRY) == len(EXPECTED_OPS)


def test_no_op_mappings_present():
    """Ops that map to no_op should all be present."""
    no_ops = [op for op, mlx, _ in EXPECTED_OPS if mlx == "no_op"]
    for torch_op in no_ops:
        assert op_mapping.lookup_op(torch_op) is not None
