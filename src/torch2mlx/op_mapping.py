"""Operation mapping: torch functional ops -> MLX equivalents.

Maps PyTorch tensor operations and functional API calls to their
MLX counterparts. Used by the analyzer to assess forward() portability
and by templates as a reference for hand-porting.

Example: torch.cat(tensors, dim=d) -> mx.concatenate(tensors, axis=d)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpMapping:
    """Single torch op -> MLX op mapping."""

    torch_op: str
    mlx_op: str
    param_renames: dict[str, str]  # e.g., {"dim": "axis"}
    notes: str = ""


# Populated by registry-agent
OP_REGISTRY: dict[str, OpMapping] = {}


def lookup_op(torch_op: str) -> OpMapping | None:
    """Find the MLX equivalent for a torch operation."""
    return OP_REGISTRY.get(torch_op)


def register_op(mapping: OpMapping) -> None:
    """Add an operation mapping to the registry."""
    OP_REGISTRY[mapping.torch_op] = mapping
