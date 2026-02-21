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
