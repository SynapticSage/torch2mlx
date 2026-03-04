"""Weight converter: per-layer-type transposition rules.

Converts PyTorch weight tensors to MLX layout (and back) using numpy.
Dispatch is keyed on the originating module type (not tensor shape),
because identical shapes can require different transpositions
(e.g., Conv1d vs ConvTranspose1d).

Backend-agnostic: operates on numpy arrays, never imports torch or mlx.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# Forward dispatch: PyTorch → MLX
TRANSPOSITION_RULES: dict[str, callable] = {}

# Reverse dispatch: MLX → PyTorch
REVERSE_TRANSPOSITION_RULES: dict[str, callable] = {}


# -- Forward rules (PyTorch → MLX) -------------------------------------------

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


def _conv_transpose2d(arr: NDArray) -> NDArray:
    # [I, O, H, W] -> [O, H, W, I]
    return np.transpose(arr, (1, 2, 3, 0))


def _linear_transposed(arr: NDArray) -> NDArray:
    # HF Conv1D: [in, out] -> [out, in] (standard Linear layout)
    return arr.T


# -- Reverse rules (MLX → PyTorch) -------------------------------------------

def _rev_conv1d(arr: NDArray) -> NDArray:
    # [O, K, I] -> [O, I, K]  (swapaxes is its own inverse)
    return np.swapaxes(arr, 1, 2)


def _rev_conv2d(arr: NDArray) -> NDArray:
    # [O, H, W, I] -> [O, I, H, W]
    return np.moveaxis(arr, -1, 1)


def _rev_conv_transpose1d(arr: NDArray) -> NDArray:
    # [O, K, I] -> [I, O, K]
    return np.transpose(arr, (2, 0, 1))


def _rev_conv_transpose2d(arr: NDArray) -> NDArray:
    # [O, H, W, I] -> [I, O, H, W]
    return np.transpose(arr, (3, 0, 1, 2))


def _rev_linear_transposed(arr: NDArray) -> NDArray:
    # [out, in] -> [in, out] (back to HF Conv1D layout)
    return arr.T


# -- Populate both tables ----------------------------------------------------

def _populate() -> None:
    TRANSPOSITION_RULES["identity"] = _identity
    TRANSPOSITION_RULES["conv1d"] = _conv1d
    TRANSPOSITION_RULES["conv2d"] = _conv2d
    TRANSPOSITION_RULES["conv_transpose1d"] = _conv_transpose1d
    TRANSPOSITION_RULES["conv_transpose2d"] = _conv_transpose2d
    TRANSPOSITION_RULES["batch_norm"] = _identity  # alias
    TRANSPOSITION_RULES["linear_transposed"] = _linear_transposed

    REVERSE_TRANSPOSITION_RULES["identity"] = _identity
    REVERSE_TRANSPOSITION_RULES["conv1d"] = _rev_conv1d
    REVERSE_TRANSPOSITION_RULES["conv2d"] = _rev_conv2d
    REVERSE_TRANSPOSITION_RULES["conv_transpose1d"] = _rev_conv_transpose1d
    REVERSE_TRANSPOSITION_RULES["conv_transpose2d"] = _rev_conv_transpose2d
    REVERSE_TRANSPOSITION_RULES["batch_norm"] = _identity  # alias
    REVERSE_TRANSPOSITION_RULES["linear_transposed"] = _rev_linear_transposed


_populate()


def convert_weight(array: NDArray, rule_name: str) -> NDArray:
    """Apply the named transposition rule (PyTorch -> MLX)."""
    rule = TRANSPOSITION_RULES.get(rule_name)
    if rule is None:
        raise KeyError(f"Unknown transposition rule: {rule_name!r}")
    return rule(array)


def convert_weight_reverse(array: NDArray, rule_name: str) -> NDArray:
    """Apply the reverse transposition rule (MLX -> PyTorch)."""
    rule = REVERSE_TRANSPOSITION_RULES.get(rule_name)
    if rule is None:
        raise KeyError(f"Unknown transposition rule: {rule_name!r}")
    return rule(array)


def available_rules() -> list[str]:
    """Return all registered transposition rule names."""
    return list(TRANSPOSITION_RULES.keys())
