"""Weight converter: per-layer-type transposition rules.

Converts PyTorch weight tensors to MLX layout using numpy.
Dispatch is keyed on the originating module type (not tensor shape),
because identical shapes can require different transpositions
(e.g., Conv1d vs ConvTranspose1d).

Backend-agnostic: operates on numpy arrays, never imports torch or mlx.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# Transposition dispatch table: rule_name -> callable(ndarray) -> ndarray
TRANSPOSITION_RULES: dict[str, callable] = {}


def _identity(arr: NDArray) -> NDArray:
    return arr


def _conv1d(arr: NDArray) -> NDArray:
    # [O, I, K] -> [O, K, I]
    return np.swapaxes(arr, 1, 2)


def _conv2d(arr: NDArray) -> NDArray:
    # [O, I, H, W] -> [O, H, W, I]
    return np.moveaxis(arr, 1, -1)


def _conv_transpose1d(arr: NDArray) -> NDArray:
    # [I, O, K] -> [O, K, I]
    return np.transpose(arr, (1, 2, 0))


def _populate() -> None:
    TRANSPOSITION_RULES["identity"] = _identity
    TRANSPOSITION_RULES["conv1d"] = _conv1d
    TRANSPOSITION_RULES["conv2d"] = _conv2d
    TRANSPOSITION_RULES["conv_transpose1d"] = _conv_transpose1d
    TRANSPOSITION_RULES["batch_norm"] = _identity  # alias


_populate()


def convert_weight(array: NDArray, rule_name: str) -> NDArray:
    """Apply the named transposition rule to a weight array."""
    rule = TRANSPOSITION_RULES.get(rule_name)
    if rule is None:
        raise KeyError(f"Unknown transposition rule: {rule_name!r}")
    return rule(array)


def available_rules() -> list[str]:
    """Return all registered transposition rule names."""
    return list(TRANSPOSITION_RULES.keys())
