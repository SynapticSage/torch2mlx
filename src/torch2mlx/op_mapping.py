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


def _populate() -> None:
    _ENTRIES = [
        OpMapping("torch.cat",       "mx.concatenate", {"dim": "axis"}, ""),
        OpMapping("torch.stack",     "mx.stack",       {"dim": "axis"}, ""),
        OpMapping("F.softmax",       "mx.softmax",     {"dim": "axis"}, ""),
        OpMapping("x.view",          "mx.reshape",     {},              "method -> function"),
        OpMapping("x.permute",       "mx.transpose",   {},              "method -> function"),
        OpMapping("x.transpose",     "mx.swapaxes",    {},              "2-arg transpose"),
        OpMapping("x.reshape",       "mx.reshape",     {},              ""),
        OpMapping("x.to",            "no_op",          {},              "Unified memory"),
        OpMapping("x.contiguous",    "no_op",          {},              "No-op"),
        OpMapping("torch.no_grad",   "no_op",          {},              "MLX doesn't track by default"),
        OpMapping("F.relu",          "nn.relu",        {},              ""),
        OpMapping("F.gelu",          "nn.gelu",        {},              ""),
        OpMapping("F.silu",          "nn.silu",        {},              ""),
    ]
    for entry in _ENTRIES:
        register_op(entry)


_populate()
