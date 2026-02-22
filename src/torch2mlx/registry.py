"""Layer registry: torch.nn.X -> mlx.nn.X mapping table.

Maps each PyTorch module class to its MLX equivalent constructor
and any API adaptation needed (e.g., class vs function activations).
Layers not in the registry are reported as manual-port items.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerMapping:
    """Single torch -> MLX layer mapping entry."""

    torch_name: str
    mlx_name: str
    weight_transposition: str  # key into weight_converter dispatch
    notes: str = ""


# Populated by registry-agent
LAYER_REGISTRY: dict[str, LayerMapping] = {}


def lookup(torch_class_name: str) -> LayerMapping | None:
    """Find the MLX equivalent for a torch module class name."""
    return LAYER_REGISTRY.get(torch_class_name)


def register(mapping: LayerMapping) -> None:
    """Add a layer mapping to the registry."""
    LAYER_REGISTRY[mapping.torch_name] = mapping


def registered_names() -> list[str]:
    """Return all registered torch class names."""
    return list(LAYER_REGISTRY.keys())


def _populate() -> None:
    _ENTRIES = [
        LayerMapping("Linear",           "nn.Linear",           "identity",        "Identical API"),
        LayerMapping("Embedding",        "nn.Embedding",        "identity",        "Identical"),
        LayerMapping("LayerNorm",        "nn.LayerNorm",        "identity",        "Identical"),
        LayerMapping("RMSNorm",          "nn.RMSNorm",          "identity",        "MLX has this natively"),
        LayerMapping("Conv1d",           "nn.Conv1d",           "conv1d",          "Weight layout differs"),
        LayerMapping("Conv2d",           "nn.Conv2d",           "conv2d",          "Weight layout differs"),
        LayerMapping("ConvTranspose1d",  "nn.ConvTranspose1d",  "conv_transpose1d","Weight layout differs"),
        LayerMapping("BatchNorm1d",      "nn.BatchNorm",        "batch_norm",      "Per-param identity"),
        LayerMapping("BatchNorm2d",      "nn.BatchNorm",        "batch_norm",      "Per-param identity"),
        LayerMapping("MultiheadAttention","nn.MultiHeadAttention","identity",      "Different API surface"),
        LayerMapping("GELU",             "nn.GELU",             "identity",        "Class vs function"),
        LayerMapping("ReLU",             "nn.ReLU",             "identity",        "Class vs function"),
        LayerMapping("SiLU",             "nn.SiLU",             "identity",        "Class vs function"),
        LayerMapping("Dropout",          "nn.Dropout",          "identity",        ""),
        LayerMapping("ModuleList",       "None",                "identity",        "No MLX equivalent — needs wrapper"),
        LayerMapping("Sequential",       "None",                "identity",        "No MLX equivalent — needs wrapper"),
    ]
    for entry in _ENTRIES:
        register(entry)


_populate()
