"""torch2mlx — Translate PyTorch models to Apple's MLX framework.

Stable:
  - Weight conversion (convert, load_converted, export)
  - Portability analysis (analyze)

Experimental:
  - Code generation (generate) — assisted porting, output may need manual review

Public API::

    from torch2mlx import convert, load_converted, export, analyze, generate
"""

from __future__ import annotations

from typing import Any

from pathlib import Path

__version__ = "0.1.0"

from torch2mlx.analyzer import PortabilityReport, analyze
from torch2mlx.codegen import Confidence, CoverageMetrics, GeneratedCode, RewriteResult, generate
from torch2mlx.converter import convert, load_converted


def export(
    model: Any,
    path: str | Path,
    *,
    analyze_first: bool = True,
    module_map: dict[str, str] | None = None,
) -> Path:
    """Convert a PyTorch model to MLX-compatible safetensors.

    Convenience alias for :func:`torch2mlx.converter.convert`.

    Args:
        model: a torch.nn.Module or flat state dict (numpy arrays)
        path: output safetensors file path
        analyze_first: run portability analysis before converting
        module_map: explicit prefix-to-rule mapping for weight transpositions

    Returns:
        Path to the saved safetensors file
    """
    return convert(model, path, analyze_first=analyze_first, module_map=module_map)


__all__ = [
    "convert",
    "load_converted",
    "export",
    "analyze",
    "generate",
    "PortabilityReport",
    "GeneratedCode",
    "CoverageMetrics",
    "Confidence",
    "RewriteResult",
]
