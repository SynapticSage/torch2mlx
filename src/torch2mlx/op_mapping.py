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


@dataclass(frozen=True)
class DtypeMapping:
    """Single torch dtype -> MLX dtype mapping."""

    torch_dtype: str
    mlx_dtype: str
    notes: str = ""


# Populated by registry-agent
OP_REGISTRY: dict[str, OpMapping] = {}

# Dtype registry: torch dtype string -> MLX dtype string
DTYPE_REGISTRY: dict[str, DtypeMapping] = {}


def lookup_op(torch_op: str) -> OpMapping | None:
    """Find the MLX equivalent for a torch operation."""
    return OP_REGISTRY.get(torch_op)


def get_dtype_mapping(torch_dtype: str) -> str | None:
    """Return the MLX dtype string for a torch dtype, or None if unmapped."""
    mapping = DTYPE_REGISTRY.get(torch_dtype)
    return mapping.mlx_dtype if mapping is not None else None


def register_op(mapping: OpMapping) -> None:
    """Add an operation mapping to the registry."""
    OP_REGISTRY[mapping.torch_op] = mapping


def register_dtype(mapping: DtypeMapping) -> None:
    """Add a dtype mapping to the registry."""
    DTYPE_REGISTRY[mapping.torch_dtype] = mapping


def _populate() -> None:
    _ENTRIES = [
        OpMapping("torch.cat", "mx.concatenate", {"dim": "axis"}, ""),
        OpMapping("torch.stack", "mx.stack", {"dim": "axis"}, ""),
        OpMapping("F.softmax", "mx.softmax", {"dim": "axis"}, ""),
        OpMapping("x.view", "mx.reshape", {}, "method -> function"),
        OpMapping("x.permute", "mx.transpose", {}, "method -> function"),
        OpMapping("x.transpose", "mx.swapaxes", {}, "2-arg transpose"),
        OpMapping("x.reshape", "mx.reshape", {}, ""),
        OpMapping("x.to", "no_op", {}, "Unified memory"),
        OpMapping("x.contiguous", "no_op", {}, "No-op"),
        OpMapping("torch.no_grad", "no_op", {}, "MLX doesn't track by default"),
        OpMapping("F.relu", "nn.relu", {}, ""),
        OpMapping("F.gelu", "nn.gelu", {}, ""),
        OpMapping("F.silu", "nn.silu", {}, ""),
        OpMapping("torch.einsum", "mx.einsum", {}, ""),
        OpMapping("torch.matmul", "mx.matmul", {}, ""),
        OpMapping("x.unsqueeze", "mx.expand_dims", {"dim": "axis"}, "method -> function"),
        OpMapping("x.squeeze", "mx.squeeze", {"dim": "axis"}, ""),
        OpMapping("x.flatten", "mx.flatten", {}, "start_dim/end_dim differ"),
        OpMapping("torch.split", "mx.split", {"dim": "axis"}, ""),
        OpMapping("x.sum", "mx.sum", {"dim": "axis"}, "method -> function"),
        OpMapping("x.mean", "mx.mean", {"dim": "axis"}, "method -> function"),
        OpMapping("x.max", "mx.max", {"dim": "axis"}, "method -> function"),
        OpMapping("x.min", "mx.min", {"dim": "axis"}, "method -> function"),
        OpMapping("F.cross_entropy", "nn.losses.cross_entropy", {}, "No reduction param in MLX"),
        OpMapping("F.mse_loss", "nn.losses.mse_loss", {}, "No reduction param in MLX"),
        OpMapping(
            "torch.zeros", "mx.zeros", {"dtype": "dtype"}, "dtype values differ across frameworks"
        ),
        OpMapping(
            "torch.ones", "mx.ones", {"dtype": "dtype"}, "dtype values differ across frameworks"
        ),
        OpMapping("torch.randn", "mx.random.normal", {}, "Different seeding semantics"),
        OpMapping("x.chunk", "mx.split", {"dim": "axis"}, "Method form of chunk"),
        OpMapping("torch.chunk", "mx.split", {"dim": "axis"}, "Functional form of chunk"),
        # Tensor creation (additional)
        OpMapping("torch.arange", "mx.arange", {}, ""),
        OpMapping("torch.full", "mx.full", {}, ""),
        OpMapping("torch.zeros_like", "mx.zeros_like", {}, ""),
        OpMapping("torch.ones_like", "mx.ones_like", {}, ""),
        OpMapping("torch.tensor", "mx.array", {}, "torch.tensor → mx.array"),
        OpMapping("torch.finfo", "mx.finfo", {}, "Float type info"),
        OpMapping("torch.iinfo", "mx.iinfo", {}, "Integer type info"),
        # Math functions
        OpMapping("torch.where", "mx.where", {}, ""),
        OpMapping("torch.clamp", "mx.clip", {"min": "a_min", "max": "a_max"}, ""),
        OpMapping("torch.abs", "mx.abs", {}, ""),
        OpMapping("torch.sqrt", "mx.sqrt", {}, ""),
        OpMapping("torch.pow", "mx.power", {}, ""),
        OpMapping("torch.log", "mx.log", {}, ""),
        OpMapping("torch.exp", "mx.exp", {}, ""),
        OpMapping("torch.tanh", "mx.tanh", {}, ""),
        # Tensor methods (additional)
        OpMapping("x.expand", "mx.broadcast_to", {}, "method -> function"),
        OpMapping("x.clamp", "mx.clip", {"min": "a_min", "max": "a_max"}, "method -> function"),
        OpMapping("x.abs", "mx.abs", {}, "method -> function"),
        OpMapping("x.sqrt", "mx.sqrt", {}, "method -> function"),
        OpMapping("x.repeat", "mx.tile", {}, "method -> function"),
        OpMapping("x.split", "mx.split", {"dim": "axis"}, "method -> function"),
        OpMapping("x.matmul", "mx.matmul", {}, "method -> function"),
        # F.* functions (additional)
        OpMapping("F.dropout", "no_op", {}, "No-op at eval time"),
        # Python operators (emitted by torch.fx for arithmetic expressions)
        OpMapping("operator.add", "mx.add", {}, "Python + operator"),
        OpMapping("operator.mul", "mx.multiply", {}, "Python * operator"),
        OpMapping("operator.sub", "mx.subtract", {}, "Python - operator"),
        OpMapping("operator.truediv", "mx.divide", {}, "Python / operator"),
        OpMapping("operator.floordiv", "mx.floor_divide", {}, "Python // operator"),
        OpMapping("operator.getitem", "operator.getitem", {}, "Indexing — passthrough"),
    ]
    for entry in _ENTRIES:
        register_op(entry)


_populate()


def _populate_dtypes() -> None:
    _ENTRIES = [
        DtypeMapping("torch.float16", "mx.float16", ""),
        DtypeMapping("torch.float32", "mx.float32", ""),
        DtypeMapping("torch.bfloat16", "mx.bfloat16", ""),
        DtypeMapping("torch.int8", "mx.int8", ""),
        DtypeMapping("torch.int16", "mx.int16", ""),
        DtypeMapping("torch.int32", "mx.int32", ""),
        DtypeMapping("torch.int64", "mx.int64", ""),
        DtypeMapping("torch.uint8", "mx.uint8", ""),
        DtypeMapping("torch.bool", "mx.bool_", "MLX uses trailing underscore"),
        DtypeMapping("torch.float64", "mx.float32", "MLX lacks float64; downcast"),
        DtypeMapping("torch.complex64", "unsupported", "MLX lacks complex dtypes"),
        DtypeMapping("torch.complex128", "unsupported", "MLX lacks complex dtypes"),
    ]
    for entry in _ENTRIES:
        register_dtype(entry)


_populate_dtypes()
