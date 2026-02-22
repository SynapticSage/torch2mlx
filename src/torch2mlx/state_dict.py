"""State dict surgery: flat key <-> nested dict conversion.

Handles the structural mismatch between PyTorch's flat state_dict keys
(e.g., "encoder.layers.0.self_attn.q_proj.weight") and MLX's nested
parameter tree (dicts of dicts).

Also handles safetensors I/O as the interchange format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from numpy.typing import NDArray
from safetensors.numpy import load_file, save_file


def flatten(nested: dict[str, Any], prefix: str = "") -> dict[str, NDArray]:
    """Convert nested parameter dict to flat dot-separated keys."""
    flat: dict[str, NDArray] = {}
    for key, value in nested.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten(value, full_key))
        else:
            flat[full_key] = value
    return flat


def unflatten(flat: dict[str, NDArray]) -> dict[str, Any]:
    """Convert flat dot-separated keys to nested parameter dict."""
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(".")
        current = nested
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return nested


def load_safetensors(path: str | Path) -> dict[str, NDArray]:
    """Load tensors from a safetensors file."""
    return load_file(str(path))


def save_safetensors(tensors: dict[str, NDArray], path: str | Path) -> None:
    """Save tensors to a safetensors file."""
    save_file(tensors, str(path))
