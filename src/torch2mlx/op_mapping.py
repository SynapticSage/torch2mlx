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
        OpMapping("torch.einsum",    "mx.einsum",      {},              ""),
        OpMapping("torch.matmul",    "mx.matmul",      {},              ""),
        OpMapping("x.unsqueeze",     "mx.expand_dims", {"dim": "axis"}, "method -> function"),
        OpMapping("x.squeeze",       "mx.squeeze",     {"dim": "axis"}, ""),
        OpMapping("x.flatten",       "mx.flatten",     {},              "start_dim/end_dim differ"),
        OpMapping("torch.split",     "mx.split",       {"dim": "axis"}, ""),
        OpMapping("x.sum",           "mx.sum",          {"dim": "axis"}, "method -> function"),
        OpMapping("x.mean",          "mx.mean",         {"dim": "axis"}, "method -> function"),
        OpMapping("x.max",           "mx.max",          {"dim": "axis"}, "method -> function"),
        OpMapping("x.min",           "mx.min",          {"dim": "axis"}, "method -> function"),
        OpMapping("F.cross_entropy", "nn.losses.cross_entropy", {}, "No reduction param in MLX"),
        OpMapping("F.mse_loss",      "nn.losses.mse_loss",      {}, "No reduction param in MLX"),
        OpMapping("torch.zeros",     "mx.zeros",          {"dtype": "dtype"}, "dtype values differ across frameworks"),
        OpMapping("torch.ones",      "mx.ones",           {"dtype": "dtype"}, "dtype values differ across frameworks"),
        OpMapping("torch.randn",     "mx.random.normal",  {},                 "Different seeding semantics"),
    ]
    for entry in _ENTRIES:
        register_op(entry)


_populate()
