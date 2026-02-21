"""Weight converter: per-layer-type transposition rules.

Converts PyTorch weight tensors to MLX layout using numpy.
Dispatch is keyed on the originating module type (not tensor shape),
because identical shapes can require different transpositions
(e.g., Conv1d vs ConvTranspose1d).

Backend-agnostic: operates on numpy arrays, never imports torch or mlx.
"""

from __future__ import annotations

from numpy.typing import NDArray


# Transposition dispatch table: rule_name -> callable(ndarray) -> ndarray
TRANSPOSITION_RULES: dict[str, callable] = {}


def convert_weight(array: NDArray, rule_name: str) -> NDArray:
    """Apply the named transposition rule to a weight array."""
    rule = TRANSPOSITION_RULES.get(rule_name)
    if rule is None:
        raise KeyError(f"Unknown transposition rule: {rule_name!r}")
    return rule(array)
