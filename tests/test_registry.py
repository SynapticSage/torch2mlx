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
    ("ConvTranspose2d",   "nn.ConvTranspose2d",    "conv_transpose2d"),
    ("BatchNorm1d",       "nn.BatchNorm",          "batch_norm"),
    ("BatchNorm2d",       "nn.BatchNorm",          "batch_norm"),
    ("MultiheadAttention","nn.MultiHeadAttention", "identity"),
    ("GELU",              "nn.GELU",               "identity"),
    ("ReLU",              "nn.ReLU",               "identity"),
    ("SiLU",              "nn.SiLU",               "identity"),
    ("Dropout",           "nn.Dropout",            "identity"),
    ("ModuleList",        "None",                  "identity"),
    ("Sequential",        "None",                  "identity"),
    ("Tanh",              "nn.Tanh",               "identity"),
    ("Sigmoid",           "nn.Sigmoid",            "identity"),
    ("LeakyReLU",         "nn.LeakyReLU",          "identity"),
    ("Softmax",           "nn.Softmax",            "identity"),
    ("GroupNorm",         "nn.GroupNorm",           "identity"),
    ("InstanceNorm1d",    "nn.InstanceNorm",        "identity"),
    ("InstanceNorm2d",    "nn.InstanceNorm",        "identity"),
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
    ("torch.einsum",  "mx.einsum",     {}),
    ("torch.matmul",  "mx.matmul",     {}),
    ("x.unsqueeze",   "mx.expand_dims", {"dim": "axis"}),
    ("x.squeeze",     "mx.squeeze",    {"dim": "axis"}),
    ("x.flatten",     "mx.flatten",    {}),
    ("torch.split",   "mx.split",      {"dim": "axis"}),
    ("x.sum",         "mx.sum",        {"dim": "axis"}),
    ("x.mean",        "mx.mean",       {"dim": "axis"}),
    ("x.max",         "mx.max",        {"dim": "axis"}),
    ("x.min",         "mx.min",        {"dim": "axis"}),
    ("F.cross_entropy", "nn.losses.cross_entropy", {}),
    ("F.mse_loss",      "nn.losses.mse_loss",      {}),
    ("torch.zeros",   "mx.zeros",         {"dtype": "dtype"}),
    ("torch.ones",    "mx.ones",          {"dtype": "dtype"}),
    ("torch.randn",   "mx.random.normal", {}),
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


# ── MLX existence validation ────────────────────────────────────────────────
# These tests verify that every MLX symbol referenced in the registries
# actually exists in the installed MLX package.

mlx_core = pytest.importorskip("mlx.core", reason="mlx not installed")
mlx_nn = pytest.importorskip("mlx.nn", reason="mlx not installed")

SKIP_LAYER_NAMES = {"None"}
SKIP_OP_NAMES = {"no_op"}


def _resolve_mlx_attr(dotted_name: str):
    """Resolve 'nn.Linear' -> mlx.nn.Linear, 'mx.sum' -> mlx.core.sum, etc."""
    parts = dotted_name.split(".")
    if parts[0] == "nn":
        obj = mlx_nn
        parts = parts[1:]
    elif parts[0] == "mx":
        obj = mlx_core
        parts = parts[1:]
    else:
        raise ValueError(f"Unknown prefix in {dotted_name!r}")
    for part in parts:
        obj = getattr(obj, part)
    return obj


_LAYER_NAMES_TO_CHECK = [
    m.mlx_name
    for m in LAYER_REGISTRY.values()
    if m.mlx_name not in SKIP_LAYER_NAMES
]

_OP_NAMES_TO_CHECK = [
    m.mlx_op
    for m in OP_REGISTRY.values()
    if m.mlx_op not in SKIP_OP_NAMES
]


@pytest.mark.parametrize("mlx_name", sorted(set(_LAYER_NAMES_TO_CHECK)))
def test_mlx_layer_exists(mlx_name):
    """Every non-None mlx_name in LAYER_REGISTRY resolves to a real MLX symbol."""
    obj = _resolve_mlx_attr(mlx_name)
    assert callable(obj), f"{mlx_name} exists but is not callable"


@pytest.mark.parametrize("mlx_op", sorted(set(_OP_NAMES_TO_CHECK)))
def test_mlx_op_exists(mlx_op):
    """Every non-no_op mlx_op in OP_REGISTRY resolves to a real MLX symbol."""
    obj = _resolve_mlx_attr(mlx_op)
    assert callable(obj), f"{mlx_op} exists but is not callable"
