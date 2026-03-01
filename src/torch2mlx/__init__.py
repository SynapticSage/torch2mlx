"""torch2mlx — Translate PyTorch models to Apple's MLX framework.

Approach: Module-Tree Walk + Weight Convert
  1. Walk torch.nn.Module tree, map each layer via registry
  2. Convert state dict (transpositions, key restructuring)
  3. Load weights into equivalent MLX modules
  4. Verify numerical equivalence

Public API::

    from torch2mlx import convert, load_converted, export, analyze
"""

from __future__ import annotations

from typing import Any

from pathlib import Path

__version__ = "0.1.0"

from torch2mlx.analyzer import PortabilityReport, analyze
from torch2mlx.converter import convert, load_converted


def export(
    model: Any,
    path: str | Path,
    *,
    analyze_first: bool = True,
) -> Path:
    """Convert a PyTorch model to MLX-compatible safetensors.

    Convenience alias for :func:`torch2mlx.converter.convert`.

    Args:
        model: a torch.nn.Module or flat state dict (numpy arrays)
        path: output safetensors file path
        analyze_first: run portability analysis before converting

    Returns:
        Path to the saved safetensors file
    """
    return convert(model, path, analyze_first=analyze_first)


__all__ = [
    "convert",
    "load_converted",
    "export",
    "analyze",
    "PortabilityReport",
]
