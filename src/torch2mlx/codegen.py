"""Code generation: emit MLX nn.Module source from a torch.nn.Module.

Generates a `.py` file containing:
  - __init__ with constructor calls derived from the module tree (always)
  - __call__ translated from torch.fx graph (when tracing succeeds)

Uses registry.py for layer mapping and op_mapping.py for operator translation.
"""

from __future__ import annotations

import ast as _ast
import enum
import inspect
import operator
import re as _re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch2mlx.hf_compat import (
    HF_METHOD_STUBS,
    SCALAR_OVERRIDES,
    hf_post_process,
)
from torch2mlx.op_mapping import DTYPE_REGISTRY, OP_REGISTRY

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArgSpec:
    """Describes how to extract one constructor arg from a torch module."""

    attr: str  # attribute name on the torch module
    mlx_name: str | None = None  # MLX param name if different
    transform: str = "identity"  # "identity" | "bias_check" | "tuple_to_scalar" | "last_element"
    default: Any = None  # omit kwarg if value equals this


@dataclass(frozen=True)
class ConstructorSpec:
    """Recipe for generating one MLX constructor call."""

    mlx_call: str  # e.g. "nn.Linear"
    args: tuple[ArgSpec, ...]


class Confidence(enum.Enum):
    """Confidence level for AST-rewritten code."""

    MECHANICAL = "mechanical"  # Pure syntactic rename, high confidence
    NEEDS_REVIEW = "needs_review"  # Ambiguous or approximate translation
    BLOCKER = "blocker"  # Known incompatibility, manual fix needed


_CONFIDENCE_ORDER = {
    Confidence.MECHANICAL: 0,
    Confidence.NEEDS_REVIEW: 1,
    Confidence.BLOCKER: 2,
}


@dataclass
class RewriteResult:
    """Result of AST-rewriting a forward() method."""

    source: str  # Rewritten __call__ body
    confidence: Confidence  # Overall confidence
    annotations: list[tuple[int, Confidence, str]]  # (line, level, note)
    unmapped_calls: list[str]  # Torch APIs not in OP_REGISTRY
    total_ops: int = 0  # Total torch/F/method calls encountered
    mapped_ops: int = 0  # Successfully rewritten ops


@dataclass(frozen=True)
class CoverageMetrics:
    """Disaggregated code generation coverage.

    Separates "we recognize this module" (registry_coverage) from
    "we actually emitted constructor code" (init_coverage).
    """

    total_leaves: int
    mapped_leaves: int  # Non-None spec — real constructor emitted
    skipped_leaves: int  # None spec — Identity, DropPath, RoPE, containers, etc.
    unmapped_leaves: int  # Not in CONSTRUCTOR_SPECS at all
    total_ops: int = 0  # Torch/F/method calls encountered in forward()
    mapped_ops: int = 0  # Successfully rewritten ops

    @property
    def init_coverage(self) -> float:
        """Fraction of code-bearing leaves that have a real constructor."""
        effective = self.total_leaves - self.skipped_leaves
        return self.mapped_leaves / effective if effective > 0 else 1.0

    @property
    def registry_coverage(self) -> float:
        """Fraction of leaves recognized by CONSTRUCTOR_SPECS (including None)."""
        if self.total_leaves == 0:
            return 1.0
        return (self.mapped_leaves + self.skipped_leaves) / self.total_leaves

    @property
    def call_coverage(self) -> float:
        """Fraction of forward() ops successfully rewritten."""
        return self.mapped_ops / self.total_ops if self.total_ops > 0 else 1.0


@dataclass
class GeneratedCode:
    """Result of code generation."""

    source: str  # complete .py source
    class_name: str
    coverage_metrics: CoverageMetrics
    todos: list[str] = field(default_factory=list)
    unmapped: list[str] = field(default_factory=list)
    traced: bool = False  # True if fx trace succeeded for __call__
    ast_rewritten: bool = False  # True if AST rewrite succeeded for __call__
    call_confidence: str = "todo"  # "mechanical" | "needs_review" | "todo"

    @property
    def coverage(self) -> float:
        """Init coverage — honest metric excluding skipped (None-spec) leaves."""
        return self.coverage_metrics.init_coverage


@dataclass
class _ClassDef:
    """Helper class definition to emit before the main class."""

    name: str  # e.g. "BertEmbeddings"
    init_body: str  # indented init lines (joined with newlines)
    forward_sig: str  # original forward() signature for TODO stub
    call_body: str | None = None  # AST-rewritten __call__, None → TODO stub
    call_confidence: str = "todo"  # "mechanical" | "needs_review" | "todo"
    extra_methods: str = ""  # Additional non-forward methods (e.g., ff_chunk)


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------


def _apply_transform(value: Any, transform: str, module: Any = None) -> Any:
    """Apply a named transform to extract a constructor arg value."""
    if transform == "identity":
        return value
    if transform == "bias_check":
        # Module has a .bias attribute; check if it's not None
        return value is not None
    if transform == "tuple_to_scalar":
        # (3, 3) -> 3 when all elements are equal, else keep tuple
        if isinstance(value, (tuple, list)) and len(value) > 0 and len(set(value)) == 1:
            return value[0]
        return value
    if transform == "last_element":
        # Return last element of a sequence (e.g., Conv1D nf from [in, out] -> out)
        if isinstance(value, (tuple, list)):
            return value[-1]
        return value
    if transform.startswith("fixed:"):
        # Return a fixed string value, ignoring the module attr
        return transform[len("fixed:") :]
    raise ValueError(f"Unknown transform: {transform!r}")


def _format_value(value: Any) -> str:
    """Format a Python value for code generation."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, tuple):
        if len(value) == 1:
            return f"({_format_value(value[0])},)"
        return "(" + ", ".join(_format_value(v) for v in value) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(v) for v in value) + "]"
    if value is None:
        return "None"
    return repr(value)


def _infer_buffer_initializer(buf: Any) -> str:
    """Infer a computed initializer for a non-persistent buffer.

    Detects common patterns: arange, zeros, ones, and falls back to
    mx.array(literal) for small buffers or mx.zeros(shape) for large ones.
    """
    import numpy as _np

    arr = buf.detach().cpu().numpy()
    shape = tuple(arr.shape)
    flat = arr.ravel()

    # All zeros
    if _np.all(flat == 0):
        return f"mx.zeros({_format_value(shape)}, dtype=mx.int64)"

    # All ones
    if _np.all(flat == 1):
        return f"mx.ones({_format_value(shape)}, dtype=mx.int64)"

    # arange pattern: values are 0, 1, 2, ..., N-1 (possibly reshaped)
    n = flat.size
    if _np.array_equal(flat, _np.arange(n)):
        if len(shape) == 1:
            return f"mx.arange({n})"
        return f"mx.reshape(mx.arange({n}), {_format_value(shape)})"

    # Lower-triangular pattern (causal mask): tril(ones(...))
    if arr.ndim >= 2:
        # Squeeze leading singleton dims for the tril check
        squeezed = arr.squeeze()
        if squeezed.ndim == 2:
            rows, cols = squeezed.shape
            expected_tril = _np.tril(_np.ones((rows, cols), dtype=arr.dtype))
            if _np.array_equal(squeezed, expected_tril):
                return f"mx.reshape(mx.tril(mx.ones(({rows}, {cols}))), {_format_value(shape)})"

    # Small buffer: emit literal array
    if flat.size <= 64:
        return f"mx.array({arr.tolist()})"

    # Fallback: zeros with shape (will be overwritten if loaded from weights)
    return f"mx.zeros({_format_value(shape)})"


# ---------------------------------------------------------------------------
# Shared constructor specs
# ---------------------------------------------------------------------------

_LINEAR_SPEC = ConstructorSpec(
    "nn.Linear",
    (
        ArgSpec("in_features", "input_dims"),
        ArgSpec("out_features", "output_dims"),
        ArgSpec("bias", "bias", "bias_check", default=True),
    ),
)

_EMBEDDING_SPEC = ConstructorSpec(
    "nn.Embedding",
    (
        ArgSpec("num_embeddings"),
        ArgSpec("embedding_dim"),
    ),
)

_LAYERNORM_SPEC = ConstructorSpec(
    "nn.LayerNorm",
    (
        ArgSpec("normalized_shape", "dims", "tuple_to_scalar"),
        ArgSpec("eps", default=1e-5),
    ),
)

_RMSNORM_SPEC = ConstructorSpec(
    "nn.RMSNorm",
    (
        ArgSpec("normalized_shape", "dims"),
        ArgSpec("eps", default=1e-5),
    ),
)

_CONV1D_SPEC = ConstructorSpec(
    "nn.Conv1d",
    (
        ArgSpec("in_channels"),
        ArgSpec("out_channels"),
        ArgSpec("kernel_size", transform="tuple_to_scalar"),
        ArgSpec("stride", transform="tuple_to_scalar", default=1),
        ArgSpec("padding", transform="tuple_to_scalar", default=0),
        ArgSpec("bias", "bias", "bias_check", default=True),
    ),
)

_CONV2D_SPEC = ConstructorSpec(
    "nn.Conv2d",
    (
        ArgSpec("in_channels"),
        ArgSpec("out_channels"),
        ArgSpec("kernel_size", transform="tuple_to_scalar"),
        ArgSpec("stride", transform="tuple_to_scalar", default=1),
        ArgSpec("padding", transform="tuple_to_scalar", default=0),
        ArgSpec("bias", "bias", "bias_check", default=True),
    ),
)

_CONV_T1D_SPEC = ConstructorSpec(
    "nn.ConvTranspose1d",
    (
        ArgSpec("in_channels"),
        ArgSpec("out_channels"),
        ArgSpec("kernel_size", transform="tuple_to_scalar"),
        ArgSpec("stride", transform="tuple_to_scalar", default=1),
        ArgSpec("padding", transform="tuple_to_scalar", default=0),
        ArgSpec("bias", "bias", "bias_check", default=True),
    ),
)

_CONV_T2D_SPEC = ConstructorSpec(
    "nn.ConvTranspose2d",
    (
        ArgSpec("in_channels"),
        ArgSpec("out_channels"),
        ArgSpec("kernel_size", transform="tuple_to_scalar"),
        ArgSpec("stride", transform="tuple_to_scalar", default=1),
        ArgSpec("padding", transform="tuple_to_scalar", default=0),
        ArgSpec("bias", "bias", "bias_check", default=True),
    ),
)

_BATCHNORM_SPEC = ConstructorSpec(
    "nn.BatchNorm",
    (
        ArgSpec("num_features"),
        ArgSpec("eps", default=1e-5),
        ArgSpec("momentum", default=0.1),
        ArgSpec("affine", default=True),
    ),
)

_GROUPNORM_SPEC = ConstructorSpec(
    "nn.GroupNorm",
    (
        ArgSpec("num_groups"),
        ArgSpec("num_channels", "dims"),
        ArgSpec("eps", default=1e-5),
        ArgSpec("affine", default=True),
    ),
)

_INSTANCENORM_SPEC = ConstructorSpec(
    "nn.InstanceNorm",
    (
        ArgSpec("num_features", "dims"),
        ArgSpec("eps", default=1e-5),
        ArgSpec("affine", default=False),
    ),
)

_DROPOUT_SPEC = ConstructorSpec(
    "nn.Dropout",
    (ArgSpec("p", default=0.5),),
)

_MHA_SPEC = ConstructorSpec(
    "nn.MultiHeadAttention",
    (
        ArgSpec("embed_dim"),
        ArgSpec("num_heads"),
    ),
)

_LEAKY_RELU_SPEC = ConstructorSpec(
    "nn.LeakyReLU",
    (ArgSpec("negative_slope", default=0.01),),
)

_MAXPOOL1D_SPEC = ConstructorSpec(
    "nn.MaxPool1d",
    (
        ArgSpec("kernel_size", transform="tuple_to_scalar"),
        ArgSpec("stride", transform="tuple_to_scalar"),
        ArgSpec("padding", transform="tuple_to_scalar", default=0),
    ),
)

_MAXPOOL2D_SPEC = ConstructorSpec(
    "nn.MaxPool2d",
    (
        ArgSpec("kernel_size", transform="tuple_to_scalar"),
        ArgSpec("stride", transform="tuple_to_scalar"),
        ArgSpec("padding", transform="tuple_to_scalar", default=0),
    ),
)

_AVGPOOL1D_SPEC = ConstructorSpec(
    "nn.AvgPool1d",
    (
        ArgSpec("kernel_size", transform="tuple_to_scalar"),
        ArgSpec("stride", transform="tuple_to_scalar"),
        ArgSpec("padding", transform="tuple_to_scalar", default=0),
    ),
)

_AVGPOOL2D_SPEC = ConstructorSpec(
    "nn.AvgPool2d",
    (
        ArgSpec("kernel_size", transform="tuple_to_scalar"),
        ArgSpec("stride", transform="tuple_to_scalar"),
        ArgSpec("padding", transform="tuple_to_scalar", default=0),
    ),
)


def _noarg(mlx_call: str) -> ConstructorSpec:
    """Spec for a stateless module with no constructor args."""
    return ConstructorSpec(mlx_call, ())


# HF Conv1D: linear with [in, out] weight layout, extract nf from weight shape
_HF_CONV1D_SPEC = ConstructorSpec(
    "nn.Linear",
    (
        ArgSpec("nx", "input_dims"),
        ArgSpec("nf", "output_dims"),
        ArgSpec("bias", "bias", "bias_check", default=True),
    ),
)

# ---------------------------------------------------------------------------
# Constructor spec registry — must cover every LAYER_REGISTRY entry
# None means "skip this type" (containers, computed embeddings, identity, etc.)
# ---------------------------------------------------------------------------

CONSTRUCTOR_SPECS: dict[str, ConstructorSpec | None] = {
    # Core layers
    "Linear": _LINEAR_SPEC,
    "Embedding": _EMBEDDING_SPEC,
    "LayerNorm": _LAYERNORM_SPEC,
    "RMSNorm": _RMSNORM_SPEC,
    "Conv1d": _CONV1D_SPEC,
    "Conv2d": _CONV2D_SPEC,
    "ConvTranspose1d": _CONV_T1D_SPEC,
    "ConvTranspose2d": _CONV_T2D_SPEC,
    "BatchNorm1d": _BATCHNORM_SPEC,
    "BatchNorm2d": _BATCHNORM_SPEC,
    "MultiheadAttention": _MHA_SPEC,
    "GroupNorm": _GROUPNORM_SPEC,
    "InstanceNorm1d": _INSTANCENORM_SPEC,
    "InstanceNorm2d": _INSTANCENORM_SPEC,
    "Dropout": _DROPOUT_SPEC,
    "LeakyReLU": _LEAKY_RELU_SPEC,
    # Stateless activations
    "GELU": _noarg("nn.GELU"),
    "ReLU": _noarg("nn.ReLU"),
    "SiLU": _noarg("nn.SiLU"),
    "Tanh": _noarg("nn.Tanh"),
    "Sigmoid": _noarg("nn.Sigmoid"),
    "Softmax": _noarg("nn.Softmax"),
    # Pooling
    "MaxPool1d": _MAXPOOL1D_SPEC,
    "MaxPool2d": _MAXPOOL2D_SPEC,
    "MaxPool3d": None,  # no MLX equivalent
    "AvgPool1d": _AVGPOOL1D_SPEC,
    "AvgPool2d": _AVGPOOL2D_SPEC,
    "AvgPool3d": None,  # no MLX equivalent
    "AdaptiveAvgPool2d": None,  # custom template
    "AdaptiveAvgPool1d": None,
    "Flatten": None,  # stateless, mx.flatten
    # Containers — skip, children are emitted individually
    "ModuleList": None,
    "Sequential": None,
    "ModuleDict": None,
    "TransformerEncoder": None,
    "TransformerDecoder": None,
    "TransformerEncoderLayer": None,
    "TransformerDecoderLayer": None,
    # Identity / passthrough
    "Identity": None,
    # PyTorch internals
    "NonDynamicallyQuantizableLinear": _LINEAR_SPEC,
    "ParametrizedConv1d": _CONV1D_SPEC,
    "ParametrizationList": None,
    "_WeightNorm": None,
    # HuggingFace — activations (stateless)
    "GELUActivation": _noarg("nn.GELU"),
    "NewGELUActivation": ConstructorSpec(
        "nn.GELU",
        (ArgSpec("_unused", "approx", transform="fixed:precise", default=""),),
    ),
    "QuickGELUActivation": _noarg("QuickGELU"),
    "BloomGelu": _noarg("nn.GELU"),
    "SiLUActivation": _noarg("nn.SiLU"),
    "ReLU6": None,
    # HuggingFace — norms
    "T5LayerNorm": _RMSNORM_SPEC,
    "DebertaLayerNorm": _LAYERNORM_SPEC,
    "Qwen2RMSNorm": _RMSNORM_SPEC,
    "ConvNextLayerNorm": _LAYERNORM_SPEC,
    # HuggingFace — embeddings
    "WhisperPositionalEmbedding": _EMBEDDING_SPEC,
    "OPTLearnedPositionalEmbedding": _EMBEDDING_SPEC,
    "BartLearnedPositionalEmbedding": _EMBEDDING_SPEC,
    "BartScaledWordEmbedding": _EMBEDDING_SPEC,
    "PegasusSinusoidalPositionalEmbedding": _EMBEDDING_SPEC,
    # HuggingFace — linear subclasses
    "Conv1D": _HF_CONV1D_SPEC,  # HF GPT-2 Conv1D (NOT torch.nn.Conv1d)
    "FalconLinear": _LINEAR_SPEC,
    # HuggingFace — computed / stateless (skip)
    "Qwen2RotaryEmbedding": None,
    "GPTNeoXRotaryEmbedding": None,
    "FalconRotaryEmbedding": None,
    "SwinDropPath": None,
    "BeitDropPath": None,
    "SegformerDropPath": None,
    # Dinov2LayerScale: emitted as helper class (x * self.lambda1)
    "Wav2Vec2SamePadLayer": None,
    "HubertSamePadLayer": None,
    "BeitRelativePositionBias": None,
}

# Synthetic helper classes — emitted when a specific torch type is encountered,
# providing a correct MLX implementation rather than an approximate alias.
_SYNTHETIC_HELPERS: dict[str, _ClassDef] = {
    "QuickGELU": _ClassDef(
        name="QuickGELU",
        init_body="",
        forward_sig="forward(self, x)",
        call_body=("    def __call__(self, x):\n        return x * mx.sigmoid(1.702 * x)\n"),
        call_confidence="mechanical",
    ),
}

# Runtime adapter for F.pad — emitted into generated source when F.pad is used.
# PyTorch pad format: flat tuple (left, right, top, bottom, ...) for last N dims
# in reverse order.  MLX pad format: [(before_0, after_0), ...] per axis in
# forward order.  This helper bridges the two at runtime.
_TORCH_PAD_HELPER = (
    "def _torch_pad(tensor, pad, mode='constant', value=None):\n"
    '    """F.pad adapter: convert PyTorch flat pad format to MLX nested format."""\n'
    "    ndim = len(tensor.shape)\n"
    "    n_pad_dims = len(pad) // 2\n"
    "    pad_width = [(0, 0)] * ndim\n"
    "    for i in range(n_pad_dims):\n"
    "        axis = ndim - 1 - i\n"
    "        pad_width[axis] = (pad[2 * i], pad[2 * i + 1])\n"
    "    if mode == 'constant':\n"
    "        return mx.pad(tensor, pad_width, constant_values=value if value is not None else 0)\n"
    "    return mx.pad(tensor, pad_width)\n"
)

# Types whose children should be emitted as list items, not named attributes.
_CONTAINER_TYPES = frozenset(
    {
        "ModuleList",
        "Sequential",
        "ModuleDict",
    }
)


# ---------------------------------------------------------------------------
# __init__ generation
# ---------------------------------------------------------------------------


def _format_constructor(module: Any, spec: ConstructorSpec) -> str:
    """Format a single MLX constructor call from a torch module and its spec."""
    positional: list[str] = []
    keyword: list[str] = []
    use_keyword = False  # Once we skip a positional arg, rest must be keyword

    for arg in spec.args:
        raw = getattr(module, arg.attr, None)
        value = _apply_transform(raw, arg.transform, module)

        # Omit kwargs that match their default
        if arg.default is not None and value == arg.default:
            use_keyword = True  # Gap in positional args → rest are keyword
            continue

        formatted = _format_value(value)
        # Use keyword when: we've already skipped a positional arg, OR
        # this arg has a default (optional kwarg that differs from default)
        if use_keyword or arg.default is not None:
            kw_name = arg.mlx_name or arg.attr
            keyword.append(f"{kw_name}={formatted}")
            use_keyword = True
        else:
            positional.append(formatted)

    all_parts = positional + keyword
    return f"{spec.mlx_call}({', '.join(all_parts)})"


def _sanitize_name(name: str) -> str:
    """Convert a dotted module path to a valid Python identifier for __init__."""
    name = name.replace(".", "_")
    if name and name[0].isdigit():
        name = f"layer_{name}"
    return name


def _dotted_to_access(target: str) -> str:
    """Convert a dotted fx target to a Python attribute/index access chain.

    Maps numeric path segments to index access so generated __call__ code
    matches the list-based __init__ structure from container codegen.

    Examples:
        "layers.0.fc"  → "layers[0].fc"
        "fc"           → "fc"
        "0.linear"     → "layer_0.linear"  (numeric root gets sanitized)
    """
    parts = target.split(".")
    result: list[str] = []
    for i, part in enumerate(parts):
        if part.isdigit():
            if not result:
                # Numeric root (from Sequential direct children) — sanitize
                result.append(_sanitize_name(part))
            else:
                # Numeric after a named container → index access
                result.append(f"[{part}]")
        else:
            if not result:
                result.append(part)
            else:
                result.append(f".{part}")
    return "".join(result)


def _class_name_from_module(module: Any) -> str:
    """Derive a class name from a torch module."""
    name = type(module).__name__
    if name in ("Module", "Sequential"):
        return "ConvertedModel"
    return name


# ---------------------------------------------------------------------------
# fx graph translation
# ---------------------------------------------------------------------------

# Map torch function objects to their string keys in OP_REGISTRY
_FX_FUNCTION_MAP: dict[Any, str] = {}

# PyTorch-only kwargs to strip from fx call_function nodes
_FX_STRIP_KWARGS: frozenset[str] = frozenset(
    {
        "inplace",
        "device",
        "pin_memory",
        "requires_grad",
        "memory_format",
    }
)

# Map torch method names to their string keys in OP_REGISTRY
_FX_METHOD_MAP: dict[str, str] = {
    "view": "x.view",
    "permute": "x.permute",
    "transpose": "x.transpose",
    "reshape": "x.reshape",
    "to": "x.to",
    "contiguous": "x.contiguous",
    "unsqueeze": "x.unsqueeze",
    "squeeze": "x.squeeze",
    "flatten": "x.flatten",
    "sum": "x.sum",
    "mean": "x.mean",
    "max": "x.max",
    "min": "x.min",
    "chunk": "x.chunk",
    "expand": "x.expand",
    "clamp": "x.clamp",
    "abs": "x.abs",
    "sqrt": "x.sqrt",
    "repeat": "x.repeat",
    "split": "x.split",
    "matmul": "x.matmul",
}


def _build_fx_function_map() -> None:
    """Populate _FX_FUNCTION_MAP from actual torch/operator callables."""
    if not HAS_TORCH:
        return

    import torch.nn.functional as F  # noqa: N812

    # torch.* functions
    _torch_funcs = {
        torch.cat: "torch.cat",
        torch.stack: "torch.stack",
        torch.einsum: "torch.einsum",
        torch.matmul: "torch.matmul",
        torch.split: "torch.split",
        torch.chunk: "torch.chunk",
        torch.zeros: "torch.zeros",
        torch.ones: "torch.ones",
        torch.randn: "torch.randn",
        torch.arange: "torch.arange",
        torch.full: "torch.full",
        torch.zeros_like: "torch.zeros_like",
        torch.ones_like: "torch.ones_like",
        torch.where: "torch.where",
        torch.clamp: "torch.clamp",
        torch.abs: "torch.abs",
        torch.sqrt: "torch.sqrt",
        torch.pow: "torch.pow",
        torch.log: "torch.log",
        torch.exp: "torch.exp",
        torch.tanh: "torch.tanh",
    }
    # F.* functions
    _f_funcs = {
        F.relu: "F.relu",
        F.gelu: "F.gelu",
        F.silu: "F.silu",
        F.softmax: "F.softmax",
        F.cross_entropy: "F.cross_entropy",
        F.mse_loss: "F.mse_loss",
        F.dropout: "F.dropout",
        F.pad: "F.pad",
    }
    # Python operators
    _op_funcs = {
        operator.add: "operator.add",
        operator.mul: "operator.mul",
        operator.sub: "operator.sub",
        operator.truediv: "operator.truediv",
        operator.floordiv: "operator.floordiv",
        operator.getitem: "operator.getitem",
    }

    _FX_FUNCTION_MAP.update(_torch_funcs)
    _FX_FUNCTION_MAP.update(_f_funcs)
    _FX_FUNCTION_MAP.update(_op_funcs)


def _try_trace(model: Any) -> Any | None:
    """Attempt torch.fx symbolic trace; return GraphModule or None on failure."""
    if not HAS_TORCH:
        return None
    try:
        import torch.fx

        return torch.fx.symbolic_trace(model)
    except Exception:
        return None


def _node_arg_repr(arg: Any) -> str:
    """Convert an fx node argument to a source code string."""
    if HAS_TORCH:
        import torch.fx

        if isinstance(arg, torch.fx.Node):
            return str(arg.name)
    if isinstance(arg, (list, tuple)):
        inner = ", ".join(_node_arg_repr(a) for a in arg)
        if isinstance(arg, tuple):
            return f"({inner},)" if len(arg) == 1 else f"({inner})"
        return f"[{inner}]"
    return _format_value(arg)


def _translate_node(node: Any) -> str | None:
    """Translate a single fx graph node to MLX source. Returns None for placeholders."""
    op = node.op

    if op == "placeholder":
        return None  # handled as function args

    if op == "output":
        args = node.args[0]
        if isinstance(args, (tuple, list)):
            return f"return {_node_arg_repr(args)}"
        return f"return {_node_arg_repr(args)}"

    if op == "get_attr":
        return f"{node.name} = self.{_dotted_to_access(node.target)}"

    if op == "call_module":
        args_str = ", ".join(_node_arg_repr(a) for a in node.args)
        if node.kwargs:
            kw = ", ".join(f"{k}={_node_arg_repr(v)}" for k, v in node.kwargs.items())
            args_str = f"{args_str}, {kw}" if args_str else kw
        target = _dotted_to_access(node.target)
        return f"{node.name} = self.{target}({args_str})"

    if op == "call_function":
        reg_key = _FX_FUNCTION_MAP.get(node.target)
        if reg_key is not None:
            mapping = OP_REGISTRY.get(reg_key)
            if mapping is not None:
                mlx_op = mapping.mlx_op

                # No-op operations
                if mlx_op == "no_op":
                    if node.args:
                        return f"{node.name} = {_node_arg_repr(node.args[0])}"
                    return None

                # Special case: operator.getitem passes through
                if reg_key == "operator.getitem":
                    args_strs = [_node_arg_repr(a) for a in node.args]
                    if len(args_strs) == 2:
                        return f"{node.name} = {args_strs[0]}[{args_strs[1]}]"

                # F.pad: route through _torch_pad adapter (format conversion)
                if reg_key == "F.pad":
                    args_strs = [_node_arg_repr(a) for a in node.args]
                    kw_parts = []
                    for k, v in node.kwargs.items():
                        kw_parts.append(f"{k}={_node_arg_repr(v)}")
                    all_args = ", ".join(args_strs + kw_parts)
                    return f"{node.name} = _torch_pad({all_args})"

                # Build call with param renames, stripping PyTorch-only kwargs
                args_strs = [_node_arg_repr(a) for a in node.args]
                kw_parts: list[str] = []
                for k, v in node.kwargs.items():
                    if k in _FX_STRIP_KWARGS:
                        continue
                    mlx_k = mapping.param_renames.get(k, k)
                    kw_parts.append(f"{mlx_k}={_node_arg_repr(v)}")
                all_args = ", ".join(args_strs + kw_parts)
                return f"{node.name} = {mlx_op}({all_args})"

        # Unmapped function — emit with qualified name as comment
        fname = getattr(
            node.target, "__qualname__", getattr(node.target, "__name__", str(node.target))
        )
        args_str = ", ".join(_node_arg_repr(a) for a in node.args)
        return f"{node.name} = {fname}({args_str})  # TODO: unmapped function"

    if op == "call_method":
        method_name = node.target
        reg_key = _FX_METHOD_MAP.get(method_name)
        if reg_key is not None:
            mapping = OP_REGISTRY.get(reg_key)
            if mapping is not None:
                mlx_op = mapping.mlx_op

                if mlx_op == "no_op":
                    return f"{node.name} = {_node_arg_repr(node.args[0])}"

                # Method → function: first arg is self
                self_arg = _node_arg_repr(node.args[0])
                rest_args = [_node_arg_repr(a) for a in node.args[1:]]
                kw_parts = []
                for k, v in node.kwargs.items():
                    mlx_k = mapping.param_renames.get(k, k)
                    kw_parts.append(f"{mlx_k}={_node_arg_repr(v)}")
                all_args = ", ".join([self_arg] + rest_args + kw_parts)
                return f"{node.name} = {mlx_op}({all_args})"

        # Unmapped method — keep as method call
        self_arg = _node_arg_repr(node.args[0])
        rest_args = ", ".join(_node_arg_repr(a) for a in node.args[1:])
        return f"{node.name} = {self_arg}.{method_name}({rest_args})  # TODO: unmapped method"

    return f"# TODO: unknown node op {op!r}"


def _translate_graph(graph_module: Any) -> tuple[str, list[str]]:
    """Translate an fx GraphModule into a __call__ method body.

    Returns:
        (source_lines, placeholder_names)
    """
    lines: list[str] = []
    placeholders: list[str] = []

    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            placeholders.append(node.name)
            continue
        line = _translate_node(node)
        if line is not None:
            lines.append(line)

    return "\n".join(f"        {line}" for line in lines), placeholders


# ---------------------------------------------------------------------------
# AST-based forward() rewriting
# ---------------------------------------------------------------------------

# Methods that are no-ops in MLX (unified memory, no contiguity concept)
_NOOP_METHODS = frozenset({"contiguous", "to", "cuda", "cpu", "detach", "requires_grad_"})

# Tensor cast methods → MLX dtype attribute names
_CAST_DTYPES: dict[str, str] = {
    "float": "float32",
    "half": "float16",
    "double": "float32",  # MLX lacks float64; downcast
    "int": "int32",
    "long": "int64",
    "bool": "bool_",
    "bfloat16": "bfloat16",
}


def _is_self_access(node: _ast.expr) -> bool:
    """Check if an AST node is a `self.xxx` attribute chain."""
    while isinstance(node, _ast.Attribute):
        node = node.value
    return isinstance(node, _ast.Name) and node.id == "self"


def _make_mx_attr(parts: str) -> _ast.expr:
    """Build an AST node for a dotted name like 'mx.reshape' or 'nn.relu'."""
    segs = parts.split(".")
    node: _ast.expr = _ast.Name(id=segs[0], ctx=_ast.Load())
    for seg in segs[1:]:
        node = _ast.Attribute(value=node, attr=seg, ctx=_ast.Load())
    return node


# Standard library / non-torch module names — never rewrite their method calls
_STDLIB_MODULES = frozenset(
    {
        "math",
        "os",
        "sys",
        "re",
        "json",
        "copy",
        "functools",
        "itertools",
        "collections",
        "logging",
        "warnings",
        "operator",
        "typing",
        "np",
        "numpy",
        "logger",
    }
)


class _TorchToMLXRewriter(_ast.NodeTransformer):
    """Rewrite a torch forward() AST into an MLX __call__() AST."""

    def __init__(self) -> None:
        self.annotations: list[tuple[int, Confidence, str]] = []
        self.unmapped_calls: list[str] = []
        self.total_ops: int = 0
        self.mapped_ops: int = 0
        self._confidence = Confidence.MECHANICAL
        self._needs_simplenamespace = False

    def _lower_confidence(self, level: Confidence) -> None:
        if _CONFIDENCE_ORDER[level] > _CONFIDENCE_ORDER[self._confidence]:
            self._confidence = level

    def _annotate(self, lineno: int, level: Confidence, note: str) -> None:
        self.annotations.append((lineno, level, note))
        self._lower_confidence(level)

    # --- FunctionDef: forward → __call__ ---

    def visit_FunctionDef(self, node: _ast.FunctionDef) -> _ast.FunctionDef:
        if node.name == "forward":
            node.name = "__call__"
            node.decorator_list = []  # Strip decorators
            # Strip docstrings (HF forward() docstrings reference torch types)
            if (
                node.body
                and isinstance(node.body[0], _ast.Expr)
                and isinstance(node.body[0].value, _ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
        # Convert type annotations
        if node.returns:
            node.returns = self._convert_annotation(node.returns)
        for arg in node.args.args:
            if arg.annotation:
                arg.annotation = self._convert_annotation(arg.annotation)
        self.generic_visit(node)
        return node

    def _convert_annotation(self, node: _ast.expr) -> _ast.expr:
        """Recursively convert torch type annotations to MLX equivalents."""
        if isinstance(node, _ast.Attribute):
            if isinstance(node.value, _ast.Name) and node.value.id == "torch":
                if "Tensor" in node.attr:
                    return _ast.Attribute(
                        value=_ast.Name(id="mx", ctx=_ast.Load()),
                        attr="array",
                        ctx=_ast.Load(),
                    )
        # Subscript: Optional[X], Tuple[X, Y], Union[X, Y], etc.
        if isinstance(node, _ast.Subscript):
            node.slice = self._convert_annotation(node.slice)
            return node
        # Tuple node: (X, Y, Z) inside Subscript slice
        if isinstance(node, _ast.Tuple):
            node.elts = [self._convert_annotation(e) for e in node.elts]
            return node
        # String annotations containing "Tensor"
        if isinstance(node, _ast.Constant) and isinstance(node.value, str):
            if "Tensor" in node.value:
                return _ast.Constant(value="mx.array")
        return node

    # --- Attribute: torch.float32 → mx.float32, no-op removal ---

    def visit_Attribute(self, node: _ast.Attribute) -> _ast.expr:
        self.generic_visit(node)
        # torch.float32 → mx.float32 (dtype constants)
        if isinstance(node.value, _ast.Name) and node.value.id == "torch":
            dtype_key = f"torch.{node.attr}"
            if dtype_key in DTYPE_REGISTRY:
                mapping = DTYPE_REGISTRY[dtype_key]
                if mapping.mlx_dtype != "unsupported":
                    mlx_attr = mapping.mlx_dtype.replace("mx.", "")
                    return _ast.Attribute(
                        value=_ast.Name(id="mx", ctx=_ast.Load()),
                        attr=mlx_attr,
                        ctx=node.ctx,
                    )
        # self.dtype → mx.float32  (default parameter dtype for inference)
        if node.attr == "dtype" and _is_self_access(node.value):
            self._annotate(
                getattr(node, "lineno", 0),
                Confidence.NEEDS_REVIEW,
                "self.dtype assumed float32; verify model default dtype",
            )
            return _ast.Attribute(
                value=_ast.Name(id="mx", ctx=_ast.Load()),
                attr="float32",
                ctx=node.ctx,
            )
        # x.device → SimpleNamespace(type="cpu")  (MLX unified memory)
        # Handles both `x.device` and `x.device.type == 'cuda'` patterns
        if node.attr == "device":
            self._needs_simplenamespace = True
            self._annotate(
                getattr(node, "lineno", 0),
                Confidence.NEEDS_REVIEW,
                ".device replaced with SimpleNamespace; MLX uses unified memory",
            )
            return _ast.Call(
                func=_ast.Name(id="SimpleNamespace", ctx=_ast.Load()),
                args=[],
                keywords=[
                    _ast.keyword(
                        arg="type",
                        value=_ast.Constant(value="cpu"),
                    ),
                ],
            )
        return node

    # --- Call: the main rewriting engine ---

    def visit_Call(self, node: _ast.Call) -> _ast.expr:
        # Visit children first so nested transforms resolve
        self.generic_visit(node)

        # Strip device=/pin_memory=/requires_grad= from ALL calls
        node.keywords = [kw for kw in node.keywords if kw.arg not in self._STRIP_KWARGS]

        func = node.func
        if not isinstance(func, _ast.Attribute):
            return node

        attr_name = func.attr

        # self.xxx(args) → leave alone (submodule calls)
        # But self.xxx.method(args) → rewrite method (tensor ops on params/buffers)
        if _is_self_access(func.value):
            if isinstance(func.value, _ast.Name) and func.value.id == "self":
                # Direct self.xxx(args) — submodule call, leave alone
                return node
            # self.xxx.forward(args) → self.xxx(args)
            if attr_name == "forward":
                return _ast.Call(func=func.value, args=node.args, keywords=node.keywords)
            # self.xxx.method(args) — fall through to method rewriting below

        # super().forward(args) → super().__call__(args)
        if attr_name == "forward" and self._is_super_call(func.value):
            func.attr = "__call__"
            return node

        # torch.jit.is_tracing() → False (always false in MLX)
        if (
            attr_name == "is_tracing"
            and isinstance(func.value, _ast.Attribute)
            and func.value.attr == "jit"
            and isinstance(func.value.value, _ast.Name)
            and func.value.value.id == "torch"
        ):
            return _ast.Constant(value=False)

        # torch.func(args)
        if isinstance(func.value, _ast.Name) and func.value.id == "torch":
            return self._rewrite_torch_func(node, attr_name)

        # torch.nn.functional.xxx(args) — multi-level attribute
        if self._is_torch_nn_functional(func.value):
            return self._rewrite_f_func(node, attr_name)

        # F.func(args)
        if isinstance(func.value, _ast.Name) and func.value.id == "F":
            return self._rewrite_f_func(node, attr_name)

        # Skip known non-torch module calls (math.sqrt, os.path, etc.)
        if isinstance(func.value, _ast.Name) and func.value.id in _STDLIB_MODULES:
            return node

        # x.method(args) — tensor methods
        return self._rewrite_method(node, attr_name)

    def _is_super_call(self, node: _ast.expr) -> bool:
        """Check if node is a super() call."""
        return (
            isinstance(node, _ast.Call)
            and isinstance(node.func, _ast.Name)
            and node.func.id == "super"
        )

    def _is_torch_nn_functional(self, node: _ast.expr) -> bool:
        """Check if node is torch.nn.functional or nn.functional."""
        # torch.nn.functional.xxx
        if (
            isinstance(node, _ast.Attribute)
            and node.attr == "functional"
            and isinstance(node.value, _ast.Attribute)
            and node.value.attr == "nn"
            and isinstance(node.value.value, _ast.Name)
            and node.value.value.id == "torch"
        ):
            return True
        # nn.functional.xxx
        if (
            isinstance(node, _ast.Attribute)
            and node.attr == "functional"
            and isinstance(node.value, _ast.Name)
            and node.value.id == "nn"
        ):
            return True
        return False

    # --- torch.func() rewriting ---

    def _rewrite_torch_func(self, node: _ast.Call, func_name: str) -> _ast.expr:
        self.total_ops += 1
        # torch.split(x, chunk_size, dim) — needs semantic translation
        if func_name == "split" and len(node.args) >= 2:
            tensor = node.args[0]
            self.mapped_ops += 1
            return self._handle_split(tensor, list(node.args[1:]), node.keywords)

        # torch.min(a, b) → mx.minimum(a, b); torch.max(a, b) → mx.maximum(a, b)
        # (2-arg elementwise form vs 1-arg reduction)
        if func_name in ("min", "max") and len(node.args) == 2 and not node.keywords:
            self.mapped_ops += 1
            mlx_fn = "mx.minimum" if func_name == "min" else "mx.maximum"
            return _ast.Call(
                func=_make_mx_attr(mlx_fn),
                args=list(node.args),
                keywords=[],
            )

        # torch.zeros/ones(a, b, c) → mx.zeros((a, b, c))  — varargs to shape tuple
        if func_name in ("zeros", "ones", "empty") and len(node.args) > 1:
            self.mapped_ops += 1
            shape_tuple = _ast.Tuple(elts=list(node.args), ctx=_ast.Load())
            mlx_fn = "mx.zeros" if func_name in ("zeros", "empty") else "mx.ones"
            return _ast.Call(
                func=_make_mx_attr(mlx_fn),
                args=[shape_tuple],
                keywords=self._rename_kwargs(node.keywords, {}),
            )

        # torch.full_like(input, fill_value) → mx.full(input.shape, fill_value, dtype=input.dtype)
        if func_name == "full_like" and len(node.args) >= 2:
            self.mapped_ops += 1
            inp = node.args[0]
            fill_val = node.args[1]
            return _ast.Call(
                func=_make_mx_attr("mx.full"),
                args=[
                    _ast.Attribute(value=inp, attr="shape", ctx=_ast.Load()),
                    fill_val,
                ],
                keywords=[
                    _ast.keyword(
                        arg="dtype",
                        value=_ast.Attribute(value=inp, attr="dtype", ctx=_ast.Load()),
                    ),
                ],
            )

        reg_key = f"torch.{func_name}"
        mapping = OP_REGISTRY.get(reg_key)
        if mapping is None:
            self.unmapped_calls.append(reg_key)
            self._lower_confidence(Confidence.NEEDS_REVIEW)
            return node

        self.mapped_ops += 1
        if mapping.mlx_op == "no_op":
            return node.args[0] if node.args else node

        return _ast.Call(
            func=_make_mx_attr(mapping.mlx_op),
            args=list(node.args),
            keywords=self._rename_kwargs(node.keywords, mapping.param_renames),
        )

    # --- F.func() rewriting ---

    def _rewrite_f_func(self, node: _ast.Call, func_name: str) -> _ast.expr:
        self.total_ops += 1
        reg_key = f"F.{func_name}"
        mapping = OP_REGISTRY.get(reg_key)
        if mapping is None:
            self.unmapped_calls.append(reg_key)
            self._lower_confidence(Confidence.NEEDS_REVIEW)
            return node

        self.mapped_ops += 1
        if mapping.mlx_op == "no_op":
            return node.args[0] if node.args else node

        # F.pad: rewrite to _torch_pad() adapter (handles format conversion at runtime)
        if func_name == "pad":
            # Map keyword names: mode and value pass through unchanged
            return _ast.Call(
                func=_ast.Name(id="_torch_pad", ctx=_ast.Load()),
                args=list(node.args),
                keywords=list(node.keywords),
            )

        # SDPA: MLX takes (q, k, v, *, scale, mask=) — only 3 positional
        if func_name == "scaled_dot_product_attention" and node.args:
            pos_args = list(node.args[:3])
            kws = self._rename_kwargs(node.keywords, mapping.param_renames)
            # args[3] = attn_mask → mask= keyword
            if len(node.args) > 3:
                mask_arg = node.args[3]
                # Skip None masks
                if not (isinstance(mask_arg, _ast.Constant) and mask_arg.value is None):
                    kws.append(_ast.keyword(arg="mask", value=mask_arg))
            # args[4] = dropout_p → drop (no-op at inference)
            # Compute scale from query's last dim if not provided or None
            # Remove scale=None keywords (MLX requires a float)
            kws = [
                kw
                for kw in kws
                if not (
                    kw.arg == "scale"
                    and isinstance(kw.value, _ast.Constant)
                    and kw.value.value is None
                )
            ]
            has_scale = any(kw.arg == "scale" for kw in kws)
            if not has_scale:
                q_arg = node.args[0]
                scale_expr = _ast.BinOp(
                    left=_ast.Subscript(
                        value=_ast.Attribute(value=q_arg, attr="shape", ctx=_ast.Load()),
                        slice=_ast.Constant(value=-1),
                        ctx=_ast.Load(),
                    ),
                    op=_ast.Pow(),
                    right=_ast.UnaryOp(op=_ast.USub(), operand=_ast.Constant(value=0.5)),
                )
                kws.append(_ast.keyword(arg="scale", value=scale_expr))
            return _ast.Call(
                func=_make_mx_attr(mapping.mlx_op),
                args=pos_args,
                keywords=kws,
            )

        rewritten = _ast.Call(
            func=_make_mx_attr(mapping.mlx_op),
            args=list(node.args),
            keywords=self._rename_kwargs(node.keywords, mapping.param_renames),
        )

        return rewritten

    # --- x.method() rewriting ---

    def _rewrite_method(self, node: _ast.Call, method_name: str) -> _ast.expr:
        receiver = node.func.value

        # Special methods (all are known torch tensor methods → counted as mapped)
        if (
            method_name
            in (
                "split",
                "size",
                "dim",
                "numel",
                "type_as",
                "forward",
                "expand_as",
            )
            or method_name in _CAST_DTYPES
            or method_name in _NOOP_METHODS
            or method_name
            in (
                "masked_fill",
                "masked_fill_",
            )
        ):
            self.total_ops += 1
            self.mapped_ops += 1

        if method_name == "split":
            return self._handle_split(receiver, node.args, node.keywords)
        if method_name == "size":
            return self._handle_size(node)
        if method_name == "dim":
            return self._handle_dim(node)
        if method_name == "numel":
            return _ast.Attribute(value=receiver, attr="size", ctx=_ast.Load())
        if method_name == "type_as":
            return self._handle_type_as(node)
        if method_name in _CAST_DTYPES:
            return self._handle_cast(node, method_name)
        if method_name in _NOOP_METHODS:
            return self._handle_noop_method(node)
        if method_name == "forward":
            # obj.forward(args) → obj(args)
            return _ast.Call(func=receiver, args=node.args, keywords=node.keywords)
        if method_name in ("masked_fill", "masked_fill_"):
            return self._handle_masked_fill(node)
        if method_name == "expand_as":
            return self._handle_expand_as(node)

        # Registry-mapped methods
        reg_key = _FX_METHOD_MAP.get(method_name)
        if reg_key is None:
            # Not a known tensor method — leave as-is (could be dict/list method)
            return node

        self.total_ops += 1
        mapping = OP_REGISTRY.get(reg_key)
        if mapping is None:
            self.unmapped_calls.append(method_name)
            self._lower_confidence(Confidence.NEEDS_REVIEW)
            return node

        self.mapped_ops += 1

        if mapping.mlx_op == "no_op":
            return receiver

        # Methods whose varargs become a single tuple arg in MLX:
        # x.view(a, b, c) → mx.reshape(x, (a, b, c))
        # x.reshape(a, b, c) → mx.reshape(x, (a, b, c))
        # x.expand(a, b, c) → mx.broadcast_to(x, (a, b, c))
        # x.permute(0, 2, 1, 3) → mx.transpose(x, (0, 2, 1, 3))
        # x.transpose(0, 1) → mx.swapaxes(x, 0, 1)  [2 args stays as-is]
        _varargs_methods = ("view", "reshape", "expand", "permute")
        if method_name in _varargs_methods:
            # Three calling conventions:
            #   x.view(a, b, c)       → args=[a, b, c]
            #   x.view((a, b, c))     → args=[Tuple(a,b,c)]
            #   x.view(*shape)        → args=[Starred(shape)]
            if len(node.args) > 1:
                elts = list(node.args)
            elif len(node.args) == 1 and isinstance(node.args[0], _ast.Tuple):
                elts = list(node.args[0].elts)
            elif len(node.args) == 1 and isinstance(node.args[0], _ast.Starred):
                # x.view(*shape) → mx.reshape(x, shape) — pass the tuple directly
                shape_arg = node.args[0].value
                return _ast.Call(
                    func=_make_mx_attr(mapping.mlx_op),
                    args=[receiver, shape_arg],
                    keywords=self._rename_kwargs(node.keywords, mapping.param_renames),
                )
            else:
                elts = None  # fall through to default handling

            if elts is not None:
                # expand(-1) means "keep existing dim" — replace with receiver.shape[i]
                if method_name == "expand":
                    for i, elt in enumerate(elts):
                        if self._is_neg_one(elt):
                            elts[i] = _ast.Subscript(
                                value=_ast.Attribute(value=receiver, attr="shape", ctx=_ast.Load()),
                                slice=_ast.Constant(value=i),
                                ctx=_ast.Load(),
                            )
                shape_tuple = _ast.Tuple(elts=elts, ctx=_ast.Load())
                return _ast.Call(
                    func=_make_mx_attr(mapping.mlx_op),
                    args=[receiver, shape_tuple],
                    keywords=self._rename_kwargs(node.keywords, mapping.param_renames),
                )

        # Method → function: prepend receiver as first arg
        return _ast.Call(
            func=_make_mx_attr(mapping.mlx_op),
            args=[receiver] + list(node.args),
            keywords=self._rename_kwargs(node.keywords, mapping.param_renames),
        )

    # --- Special method handlers ---

    @staticmethod
    def _is_neg_one(node: _ast.expr) -> bool:
        """Check if an AST node represents the constant -1."""
        if isinstance(node, _ast.Constant) and node.value == -1:
            return True
        if (
            isinstance(node, _ast.UnaryOp)
            and isinstance(node.op, _ast.USub)
            and isinstance(node.operand, _ast.Constant)
            and node.operand.value == 1
        ):
            return True
        return False

    def _handle_size(self, node: _ast.Call) -> _ast.expr:
        """x.size() → x.shape, x.size(dim) → x.shape[dim]."""
        receiver = node.func.value
        if not node.args:
            return _ast.Attribute(value=receiver, attr="shape", ctx=_ast.Load())
        return _ast.Subscript(
            value=_ast.Attribute(value=receiver, attr="shape", ctx=_ast.Load()),
            slice=node.args[0],
            ctx=_ast.Load(),
        )

    def _handle_dim(self, node: _ast.Call) -> _ast.expr:
        """x.dim() → len(x.shape)."""
        receiver = node.func.value
        return _ast.Call(
            func=_ast.Name(id="len", ctx=_ast.Load()),
            args=[_ast.Attribute(value=receiver, attr="shape", ctx=_ast.Load())],
            keywords=[],
        )

    def _handle_type_as(self, node: _ast.Call) -> _ast.expr:
        """x.type_as(y) → x.astype(y.dtype)."""
        receiver = node.func.value
        if node.args:
            other = node.args[0]
            return _ast.Call(
                func=_ast.Attribute(value=receiver, attr="astype", ctx=_ast.Load()),
                args=[_ast.Attribute(value=other, attr="dtype", ctx=_ast.Load())],
                keywords=[],
            )
        return node

    def _handle_cast(self, node: _ast.Call, method_name: str) -> _ast.expr:
        """x.float() → x.astype(mx.float32), x.half() → x.astype(mx.float16)."""
        receiver = node.func.value
        mlx_dtype = _CAST_DTYPES[method_name]
        return _ast.Call(
            func=_ast.Attribute(value=receiver, attr="astype", ctx=_ast.Load()),
            args=[_make_mx_attr(f"mx.{mlx_dtype}")],
            keywords=[],
        )

    def _handle_noop_method(self, node: _ast.Call) -> _ast.expr:
        """x.contiguous() → x, x.to(device) → x, x.to(dtype) → x.astype(dtype)."""
        method = node.func.attr
        receiver = node.func.value

        if method == "to":
            # Check if first positional arg or dtype= kwarg is a dtype constant
            dtype_node = None
            if node.args:
                arg = node.args[0]
                # torch.float16, torch.float32, etc.
                if (
                    isinstance(arg, _ast.Attribute)
                    and isinstance(arg.value, _ast.Name)
                    and arg.value.id in ("torch", "mx")
                ):
                    dtype_key = f"torch.{arg.attr}"
                    if dtype_key in DTYPE_REGISTRY:
                        dtype_node = arg
                # Already-converted mx.float16 etc.
                elif (
                    isinstance(arg, _ast.Attribute)
                    and isinstance(arg.value, _ast.Name)
                    and arg.value.id == "mx"
                ):
                    dtype_node = arg
            # Check dtype= keyword
            for kw in node.keywords:
                if kw.arg == "dtype" and isinstance(kw.value, _ast.Attribute):
                    dtype_node = kw.value
                    break

            if dtype_node is not None:
                # Convert dtype to MLX equivalent
                if isinstance(dtype_node.value, _ast.Name) and dtype_node.value.id == "torch":
                    dtype_key = f"torch.{dtype_node.attr}"
                    mapping = DTYPE_REGISTRY.get(dtype_key)
                    if mapping and mapping.mlx_dtype != "unsupported":
                        mlx_attr = mapping.mlx_dtype.replace("mx.", "")
                        target = _ast.Attribute(
                            value=_ast.Name(id="mx", ctx=_ast.Load()),
                            attr=mlx_attr,
                            ctx=_ast.Load(),
                        )
                    else:
                        target = dtype_node
                else:
                    target = dtype_node
                return _ast.Call(
                    func=_ast.Attribute(value=receiver, attr="astype", ctx=_ast.Load()),
                    args=[target],
                    keywords=[],
                )

        return receiver

    def _handle_masked_fill(self, node: _ast.Call) -> _ast.expr:
        """x.masked_fill(mask, value) → mx.where(mask, value, x)."""
        receiver = node.func.value
        if len(node.args) >= 2:
            self._annotate(
                getattr(node, "lineno", 0),
                Confidence.NEEDS_REVIEW,
                "masked_fill arg order differs from mx.where",
            )
            return _ast.Call(
                func=_make_mx_attr("mx.where"),
                args=[node.args[0], node.args[1], receiver],
                keywords=[],
            )
        return node

    def _handle_split(
        self,
        receiver: _ast.expr,
        args: list[_ast.expr],
        keywords: list[_ast.keyword],
    ) -> _ast.expr:
        """Translate torch split semantics to MLX.

        torch.split(x, chunk_size, dim) splits by SIZE (last chunk may be smaller).
        mx.split(x, indices, axis) splits at the given indices.
        Convert: mx.split(x, list(range(chunk_size, x.shape[axis], chunk_size)), axis=axis)
        """
        if not args:
            return _ast.Call(func=_make_mx_attr("mx.split"), args=[receiver], keywords=keywords)
        chunk_arg = args[0]
        # Determine axis from args[1] or keyword 'dim'
        axis_node: _ast.expr = _ast.Constant(value=0)
        if len(args) > 1:
            axis_node = args[1]
        else:
            for kw in keywords:
                if kw.arg == "dim":
                    axis_node = kw.value
                    break
        # chunk_size → indices: list(range(chunk_size, x.shape[axis], chunk_size))
        # This preserves remainder semantics (torch.split returns a smaller last chunk).
        shape_subscript = _ast.Subscript(
            value=_ast.Attribute(value=receiver, attr="shape", ctx=_ast.Load()),
            slice=axis_node,
            ctx=_ast.Load(),
        )
        indices_expr = _ast.Call(
            func=_ast.Name(id="list", ctx=_ast.Load()),
            args=[
                _ast.Call(
                    func=_ast.Name(id="range", ctx=_ast.Load()),
                    args=[chunk_arg, shape_subscript, chunk_arg],
                    keywords=[],
                )
            ],
            keywords=[],
        )
        return _ast.Call(
            func=_make_mx_attr("mx.split"),
            args=[receiver, indices_expr],
            keywords=[_ast.keyword(arg="axis", value=axis_node)],
        )

    def _handle_expand_as(self, node: _ast.Call) -> _ast.expr:
        """x.expand_as(y) → mx.broadcast_to(x, y.shape)."""
        receiver = node.func.value
        if node.args:
            target = node.args[0]
            return _ast.Call(
                func=_make_mx_attr("mx.broadcast_to"),
                args=[
                    receiver,
                    _ast.Attribute(value=target, attr="shape", ctx=_ast.Load()),
                ],
                keywords=[],
            )
        return node

    # --- Helpers ---

    # Keywords to strip from all rewritten calls (MLX unified memory)
    _STRIP_KWARGS = frozenset(
        {
            "device",
            "pin_memory",
            "requires_grad",
            "non_blocking",
            "dropout_p",
            "is_causal",  # PyTorch SDPA kwargs not in MLX
        }
    )

    def _rename_kwargs(
        self,
        keywords: list[_ast.keyword],
        renames: dict[str, str],
    ) -> list[_ast.keyword]:
        """Apply parameter renames and strip device= etc."""
        result = []
        for kw in keywords:
            if kw.arg in self._STRIP_KWARGS:
                continue
            new_arg = renames.get(kw.arg, kw.arg) if kw.arg else kw.arg
            result.append(_ast.keyword(arg=new_arg, value=kw.value))
        return result


def _rewrite_forward_ast(module: Any) -> RewriteResult | None:
    """AST-rewrite a module's forward() to MLX __call__().

    Returns None if source is unavailable or unparseable.
    """
    forward = getattr(module, "forward", None)
    if forward is None:
        return None

    try:
        source = inspect.getsource(forward)
    except (OSError, TypeError):
        return None

    source = textwrap.dedent(source)
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return None

    rewriter = _TorchToMLXRewriter()
    tree = rewriter.visit(tree)
    _ast.fix_missing_locations(tree)

    try:
        result_source = _ast.unparse(tree)
    except Exception:
        return None

    confidence = rewriter._confidence
    if rewriter.unmapped_calls:
        if _CONFIDENCE_ORDER.get(Confidence.NEEDS_REVIEW, 1) > _CONFIDENCE_ORDER.get(confidence, 0):
            confidence = Confidence.NEEDS_REVIEW

    return RewriteResult(
        source=result_source,
        confidence=confidence,
        annotations=rewriter.annotations,
        unmapped_calls=rewriter.unmapped_calls,
        total_ops=rewriter.total_ops,
        mapped_ops=rewriter.mapped_ops,
    )


def _format_ast_call(source: str, confidence: Confidence) -> str:
    """Indent and annotate an AST-rewritten __call__ for class body."""
    header = f"    # --- torch2mlx: {confidence.value.upper()} (AST rewrite) ---"
    indented = "\n".join(f"    {line}" for line in source.split("\n"))
    return f"{header}\n{indented}"


def _rewrite_method_ast(method: Any) -> str | None:
    """AST-rewrite a single method (non-forward) for MLX compatibility.

    Returns rewritten source or None if unavailable.
    """
    try:
        source = inspect.getsource(method)
    except (OSError, TypeError):
        return None

    source = textwrap.dedent(source)
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return None

    rewriter = _TorchToMLXRewriter()
    # Don't rename the function (it's not forward→__call__)
    # Override visit_FunctionDef to only strip annotations, not rename
    original_visit = rewriter.visit_FunctionDef

    def _keep_name(node: _ast.FunctionDef) -> _ast.FunctionDef:
        orig_name = node.name
        result = original_visit(node)
        result.name = orig_name  # Keep original method name
        return result

    rewriter.visit_FunctionDef = _keep_name
    tree = rewriter.visit(tree)
    _ast.fix_missing_locations(tree)

    try:
        return _ast.unparse(tree)
    except Exception:
        return None


def _get_extra_methods(module: Any, call_source: str | None) -> str:
    """Find and rewrite non-forward methods referenced in the call body.

    Catches both self.method() calls AND self.method references (callbacks).
    Recurses: if a captured method itself references further self.xxx methods,
    those are captured too (transitive discovery).
    """
    if call_source is None:
        return ""

    child_names = {name for name, _ in module.named_children()}
    param_names = {name for name, _ in module.named_parameters(recurse=False)}
    buffer_names = {name for name, _ in module.named_buffers(recurse=False)}
    # Exclude submodules, params, buffers, known names, and HF stubs
    exclude = (
        child_names
        | param_names
        | buffer_names
        | {"__init__", "__call__", "forward", "config", "training"}
        | HF_METHOD_STUBS.keys()
    )

    # Worklist: discover methods transitively from call_source
    discovered: list[str] = []  # ordered for deterministic output
    seen: set[str] = set()
    pending_sources = [call_source]

    while pending_sources:
        src = pending_sources.pop()
        refs = set(_re.findall(r"self\.(\w+)", src)) - exclude - seen
        for method_name in sorted(refs):
            seen.add(method_name)
            method = getattr(type(module), method_name, None)
            if method is None or not callable(method):
                continue
            rewritten = _rewrite_method_ast(method)
            if rewritten is not None:
                discovered.append(rewritten)
                # Scan the rewritten method for further self.xxx references
                pending_sources.append(rewritten)

    extra_source = ""
    for rewritten in discovered:
        indented = "\n".join(f"    {line}" for line in rewritten.split("\n"))
        extra_source += f"\n{indented}\n"
    return extra_source


def _try_ast_for_classdef(child: Any) -> tuple[str | None, str, str]:
    """Try AST rewrite for a helper class __call__ and extra methods.

    Returns:
        (call_body, confidence, extra_methods_source)
    """
    rewrite = _rewrite_forward_ast(child)
    if rewrite is not None and rewrite.confidence != Confidence.BLOCKER:
        extra = _get_extra_methods(child, rewrite.source)
        return rewrite.source, rewrite.confidence.value, extra
    return None, "todo", ""


# ---------------------------------------------------------------------------
# Top-level generation
# ---------------------------------------------------------------------------


def _get_forward_signature(module: Any) -> str:
    """Extract the original forward() signature for the TODO stub."""
    forward = getattr(module, "forward", None)
    if forward is None:
        return "forward(self, x)"
    try:
        sig = inspect.signature(forward)
        params = list(sig.parameters.keys())
        return f"forward({', '.join(['self'] + params)})"
    except (ValueError, TypeError):
        return "forward(self, x)"


# ---------------------------------------------------------------------------
# Recursive module tree walk
# ---------------------------------------------------------------------------


def _walk_module(
    module: Any,
    seen_classes: dict[str, _ClassDef],
) -> tuple[list[str], int, int, int, list[str], list[str]]:
    """Recursively walk module tree for __init__ generation.

    Returns:
        (init_lines, total_leaves, mapped_leaves, skipped_leaves, todos, unmapped)
    """
    init_lines: list[str] = []
    total = 0
    mapped = 0
    skipped = 0
    todos: list[str] = []
    unmapped: list[str] = []

    for name, child in module.named_children():
        safe_name = _sanitize_name(name)
        child_type = type(child).__name__
        spec = CONSTRUCTOR_SPECS.get(child_type)

        if spec is not None:
            # CASE 1: Known leaf with constructor spec
            total += 1
            try:
                cstr = _format_constructor(child, spec)
                init_lines.append(f"        self.{safe_name} = {cstr}")
                mapped += 1
                # Register synthetic helper class if needed (e.g. QuickGELU)
                helper_name = spec.mlx_call
                if helper_name in _SYNTHETIC_HELPERS and helper_name not in seen_classes:
                    seen_classes[helper_name] = _SYNTHETIC_HELPERS[helper_name]
            except (AttributeError, TypeError) as exc:
                todos.append(f"self.{safe_name}: {child_type} — {exc}")
                init_lines.append(
                    f"        # TODO: self.{safe_name} = {spec.mlx_call}(...)  # {exc}"
                )

        elif child_type in CONSTRUCTOR_SPECS:
            # CASE 2: In CONSTRUCTOR_SPECS with None value
            has_children = bool(list(child.children()))
            if child_type in _CONTAINER_TYPES:
                if has_children:
                    # 2a: List-like container with children → emit list syntax
                    c_lines, c_total, c_mapped, c_skipped, c_todos, c_unmapped = _handle_container(
                        safe_name, child, seen_classes
                    )
                    init_lines.extend(c_lines)
                    total += c_total
                    mapped += c_mapped
                    skipped += c_skipped
                    todos.extend(c_todos)
                    unmapped.extend(c_unmapped)
                # else: empty container → 0 leaves, skip silently
            elif has_children:
                # 2b: Composite in CONSTRUCTOR_SPECS (e.g. TransformerEncoderLayer)
                sub_lines, sub_total, sub_mapped, sub_skipped, sub_todos, sub_unmapped = (
                    _walk_module(child, seen_classes)
                )
                total += sub_total
                mapped += sub_mapped
                skipped += sub_skipped
                todos.extend(sub_todos)
                unmapped.extend(sub_unmapped)
                if child_type not in seen_classes:
                    cb, cc, em = _try_ast_for_classdef(child)
                    seen_classes[child_type] = _ClassDef(
                        name=child_type,
                        init_body="\n".join(sub_lines),
                        forward_sig=_get_forward_signature(child),
                        call_body=cb,
                        call_confidence=cc,
                        extra_methods=em,
                    )
                init_lines.append(f"        self.{safe_name} = {child_type}()")
            else:
                # 2c: Stateless skip (Identity, DropPath, RoPE, etc.)
                # Emit as identity lambda so self.X(y) works in __call__
                init_lines.append(f"        self.{safe_name} = lambda x, *a, **kw: x")
                total += 1
                skipped += 1

        else:
            # CASE 3: Not in CONSTRUCTOR_SPECS at all
            if list(child.named_children()):
                # 3a: Composite — recurse, register helper class
                sub_lines, sub_total, sub_mapped, sub_skipped, sub_todos, sub_unmapped = (
                    _walk_module(child, seen_classes)
                )
                total += sub_total
                mapped += sub_mapped
                skipped += sub_skipped
                todos.extend(sub_todos)
                unmapped.extend(sub_unmapped)
                if child_type not in seen_classes:
                    cb, cc, em = _try_ast_for_classdef(child)
                    seen_classes[child_type] = _ClassDef(
                        name=child_type,
                        init_body="\n".join(sub_lines),
                        forward_sig=_get_forward_signature(child),
                        call_body=cb,
                        call_confidence=cc,
                        extra_methods=em,
                    )
                init_lines.append(f"        self.{safe_name} = {child_type}()")
            elif list(child.named_parameters(recurse=False)):
                # 3b: No children but has parameters — emit helper class
                # (e.g. Dinov2LayerScale: just self.lambda1 * x)
                sub_lines, sub_total, sub_mapped, sub_skipped, sub_todos, sub_unmapped = (
                    _walk_module(child, seen_classes)
                )
                total += sub_total
                mapped += sub_mapped
                skipped += sub_skipped
                todos.extend(sub_todos)
                unmapped.extend(sub_unmapped)
                if child_type not in seen_classes:
                    cb, cc, em = _try_ast_for_classdef(child)
                    seen_classes[child_type] = _ClassDef(
                        name=child_type,
                        init_body="\n".join(sub_lines),
                        forward_sig=_get_forward_signature(child),
                        call_body=cb,
                        call_confidence=cc,
                        extra_methods=em,
                    )
                init_lines.append(f"        self.{safe_name} = {child_type}()")
            else:
                # 3c: Truly unmapped leaf (no children, no parameters)
                total += 1
                unmapped.append(child_type)
                todos.append(f"self.{safe_name}: {child_type} has no constructor spec")
                init_lines.append(
                    f"        # TODO: self.{safe_name} = ...  # {child_type} — no constructor spec"
                )

    # Emit orphan nn.Parameters (not inside any child submodule)
    child_names = {name for name, _ in module.named_children()}
    for pname, param in module.named_parameters(recurse=False):
        if pname in child_names:
            continue  # belongs to a child, already handled
        safe_pname = _sanitize_name(pname)
        shape = tuple(param.shape)
        init_lines.append(f"        self.{safe_pname} = mx.zeros({_format_value(shape)})")
        total += 1
        mapped += 1

    # Emit orphan buffers (position_ids, causal masks, index tables, etc.)
    # Persistent buffers: appear in state_dict, emit as mx.zeros (load_weights fills them)
    # Non-persistent buffers: NOT in state_dict, emit with computed initializers
    non_persistent = getattr(module, "_non_persistent_buffers_set", set())
    for bname, buf in module.named_buffers(recurse=False):
        if bname in child_names:
            continue
        safe_bname = _sanitize_name(bname)
        shape = tuple(buf.shape)
        if bname in non_persistent:
            # Emit computed initializer for non-persistent buffers
            init_lines.append(f"        self.{safe_bname} = {_infer_buffer_initializer(buf)}")
        else:
            init_lines.append(f"        self.{safe_bname} = mx.zeros({_format_value(shape)})")

    # Emit plain scalar attributes (n_heads, dim, hidden_size, etc.)
    # These are architectural constants set in __init__ that forward() references.
    param_names = {n for n, _ in module.named_parameters(recurse=False)}
    buffer_names = {n for n, _ in module.named_buffers(recurse=False)}
    # "training" is a read-only property in mlx.nn.Module — skip it
    skip_names = (
        child_names
        | param_names
        | buffer_names
        | {
            "training",
            "T_destination",
            "_parameters",
            "_buffers",
            "_modules",
            "_backward_hooks",
            "_forward_hooks",
            "_forward_pre_hooks",
            "_state_dict_hooks",
            "_load_state_dict_pre_hooks",
            "_non_persistent_buffers_set",
            "_is_full_backward_hook",
        }
    )
    for attr_name, attr_val in vars(module).items():
        if attr_name in skip_names or attr_name.startswith("_"):
            continue
        # Override known HF attributes that need different values in MLX
        if attr_name in SCALAR_OVERRIDES:
            init_lines.append(
                f"        self.{_sanitize_name(attr_name)} = {SCALAR_OVERRIDES[attr_name]}"
            )
        elif isinstance(attr_val, (int, float, bool)):
            init_lines.append(f"        self.{_sanitize_name(attr_name)} = {attr_val!r}")
        elif isinstance(attr_val, str):
            init_lines.append(f"        self.{_sanitize_name(attr_name)} = {attr_val!r}")
        elif attr_val is None:
            init_lines.append(f"        self.{_sanitize_name(attr_name)} = None")
        elif isinstance(attr_val, (tuple, list)):
            init_lines.append(
                f"        self.{_sanitize_name(attr_name)} = {_format_value(attr_val)}"
            )

    return init_lines, total, mapped, skipped, todos, unmapped


def _module_fingerprint(module: Any) -> tuple:
    """Hashable structural fingerprint for uniformity detection.

    Compares direct children (names + types), direct parameters (names + shapes),
    direct buffers (names + shapes), and scalar attributes.  Two modules with the
    same class name but different sub-module structures or scalar constants
    (e.g., self.scale=2.0 vs 4.0) produce different fingerprints.
    """
    children = tuple((n, type(c).__name__) for n, c in module.named_children())
    params = tuple((n, tuple(p.shape)) for n, p in module.named_parameters(recurse=False))
    buffers = tuple((n, tuple(b.shape)) for n, b in module.named_buffers(recurse=False))

    # Scalar attributes — same filter as _walk_module's scalar emission
    child_names = {n for n, _ in module.named_children()}
    param_names = {n for n, _ in module.named_parameters(recurse=False)}
    buffer_names = {n for n, _ in module.named_buffers(recurse=False)}
    skip = child_names | param_names | buffer_names
    scalars = tuple(
        sorted(
            (k, v)
            for k, v in vars(module).items()
            if k not in skip
            and not k.startswith("_")
            and isinstance(v, (int, float, bool, str, type(None), tuple, list))
        )
    )
    return (children, params, buffers, scalars)


def _handle_nonuniform_container(
    safe_name: str,
    children: list[tuple[str, Any]],
    seen_classes: dict[str, _ClassDef],
) -> tuple[list[str], int, int, int, list[str], list[str]]:
    """Handle containers where items share a type name but differ structurally.

    Groups items by structural fingerprint, creates variant classes for each
    unique structure, and emits a mixed list.  Weight keys use numeric indices
    so MLX load_weights maps correctly regardless of variant class names.
    """
    base_type = type(children[0][1]).__name__
    spec = CONSTRUCTOR_SPECS.get(base_type)

    # --- Leaf items with constructor spec: emit individual constructors ---
    if spec is not None:
        items: list[str] = []
        total = len(children)
        mapped_count = 0
        todos: list[str] = []
        for child_name, child in children:
            try:
                items.append(_format_constructor(child, spec))
                mapped_count += 1
            except (AttributeError, TypeError) as exc:
                items.append(f"None  # TODO: {base_type} — {exc}")
                todos.append(f"{safe_name}[{child_name}]: {base_type} — {exc}")

        init_lines = [f"        self.{safe_name} = ["]
        for item in items:
            init_lines.append(f"            {item},")
        init_lines.append("        ]")
        return init_lines, total, mapped_count, 0, todos, []

    # --- Composite items: group by fingerprint, create variant classes ---

    # Collect union of all child names across all items — needed to emit
    # self.attr = None for attributes absent in some variants but referenced
    # in the shared forward() body (e.g., `if self.downsample is not None:`)
    all_child_names: set[str] = set()
    for _, child in children:
        all_child_names.update(n for n, _ in child.named_children())
        # Also include None-valued _modules entries
        all_child_names.update(n for n, m in child._modules.items() if m is None)
    all_child_names_frozen = frozenset(all_child_names)

    fp_to_variant: dict[tuple, str] = {}
    variant_leaves: dict[str, tuple[int, int, int]] = {}  # variant → (total, mapped, skipped)
    todos_out: list[str] = []
    unmapped_out: list[str] = []
    variant_counter = 0

    for _, child in children:
        fp = _module_fingerprint(child)
        if fp in fp_to_variant:
            continue  # already registered this variant

        # First encounter of this fingerprint — pick a name and walk
        variant_name = base_type if variant_counter == 0 else f"{base_type}_v{variant_counter + 1}"
        variant_counter += 1
        fp_to_variant[fp] = variant_name

        if list(child.named_children()):
            sub_lines, sub_total, sub_mapped, sub_skipped, sub_todos, sub_unmapped = _walk_module(
                child, seen_classes
            )
            # Emit None for any absent sub-modules referenced by forward()
            none_lines = _emit_none_modules(child, all_child_names_frozen)
            all_lines = sub_lines + none_lines
            if variant_name not in seen_classes:
                cb, cc, em = _try_ast_for_classdef(child)
                seen_classes[variant_name] = _ClassDef(
                    name=variant_name,
                    init_body="\n".join(all_lines),
                    forward_sig=_get_forward_signature(child),
                    call_body=cb,
                    call_confidence=cc,
                    extra_methods=em,
                )
            variant_leaves[variant_name] = (sub_total, sub_mapped, sub_skipped)
            todos_out.extend(sub_todos)
            unmapped_out.extend(sub_unmapped)
        else:
            variant_leaves[variant_name] = (0, 0, 0)

    # Build the list with per-item variant types
    items_out: list[str] = []
    total = 0
    mapped_count = 0
    skipped_count = 0
    for _, child in children:
        fp = _module_fingerprint(child)
        variant_name = fp_to_variant[fp]
        items_out.append(f"{variant_name}()")
        vt, vm, vs = variant_leaves[variant_name]
        total += vt
        mapped_count += vm
        skipped_count += vs

    init_lines = [f"        self.{safe_name} = ["]
    for item in items_out:
        init_lines.append(f"            {item},")
    init_lines.append("        ]")

    return init_lines, total, mapped_count, skipped_count, todos_out, unmapped_out


def _emit_none_modules(module: Any, all_child_names: frozenset[str] | None = None) -> list[str]:
    """Emit `self.attr = None` for absent module slots.

    Two sources of None attributes:
    1. PyTorch _modules dict entries set to None (e.g., via register_module)
    2. Child names present in other variants but absent in this one
       (passed via all_child_names from non-uniform container detection)

    The AST-rewritten __call__ may reference these attributes
    (e.g., `if self.downsample is not None:`), so they must exist.
    """
    lines: list[str] = []
    # Source 1: explicit None in _modules
    for name, child in module._modules.items():
        if child is None:
            safe = _sanitize_name(name)
            lines.append(f"        self.{safe} = None")

    # Source 2: children present in other variants but absent here
    if all_child_names is not None:
        present = {n for n, _ in module.named_children()}
        present.update(n for n, _ in module._modules.items() if _ is None)
        for name in sorted(all_child_names - present):
            safe = _sanitize_name(name)
            lines.append(f"        self.{safe} = None")

    return lines


def _handle_container(
    safe_name: str,
    container: Any,
    seen_classes: dict[str, _ClassDef],
) -> tuple[list[str], int, int, int, list[str], list[str]]:
    """Handle ModuleList/Sequential/ModuleDict containers.

    Uniform type → list comprehension.  Mixed types → individual items.
    Same-type but structurally different → variant classes.

    Returns:
        (init_lines, total_leaves, mapped_leaves, skipped_leaves, todos, unmapped)
    """
    children = list(container.named_children())
    if not children:
        return [], 0, 0, 0, [], []

    child_types = [type(c).__name__ for _, c in children]
    count = len(children)

    # --- Uniform type → list comprehension ---
    if len(set(child_types)) == 1:
        item_type = child_types[0]
        rep = children[0][1]

        # Check structural uniformity — same class name ≠ same structure
        fingerprints = [_module_fingerprint(c) for _, c in children]
        if len(set(fingerprints)) > 1:
            return _handle_nonuniform_container(safe_name, children, seen_classes)

        # Uniform leaf with constructor
        spec = CONSTRUCTOR_SPECS.get(item_type)
        if spec is not None:
            try:
                cstr = _format_constructor(rep, spec)
                line = f"        self.{safe_name} = [{cstr} for _ in range({count})]"
                return [line], count, count, 0, [], []
            except (AttributeError, TypeError):
                pass  # fall through to mixed path

        # Uniform composite/container with children → recurse representative
        if list(rep.named_children()):
            sub_lines, sub_total, sub_mapped, sub_skipped, sub_todos, sub_unmapped = _walk_module(
                rep, seen_classes
            )
            if item_type not in seen_classes:
                cb, cc, em = _try_ast_for_classdef(rep)
                seen_classes[item_type] = _ClassDef(
                    name=item_type,
                    init_body="\n".join(sub_lines),
                    forward_sig=_get_forward_signature(rep),
                    call_body=cb,
                    call_confidence=cc,
                    extra_methods=em,
                )
            line = f"        self.{safe_name} = [{item_type}() for _ in range({count})]"
            return (
                [line],
                sub_total * count,
                sub_mapped * count,
                sub_skipped * count,
                sub_todos,
                sub_unmapped,
            )

        # Uniform stateless skip / unmapped without children
        if item_type in CONSTRUCTOR_SPECS:
            return [], count, 0, count, [], []  # skipped, not mapped
        return (
            [f"        # TODO: self.{safe_name} = [...]  # {count}x {item_type}"],
            count,
            0,
            0,
            [f"{safe_name}: {count}x {item_type} has no constructor spec"],
            [item_type],
        )

    # --- Mixed types → emit items individually ---
    items: list[str] = []
    total = 0
    mapped_count = 0
    skipped_count = 0
    todos: list[str] = []
    unmapped_list: list[str] = []

    for child_name, child in children:
        child_type = type(child).__name__
        spec = CONSTRUCTOR_SPECS.get(child_type)

        if spec is not None:
            # Leaf with constructor
            total += 1
            try:
                items.append(_format_constructor(child, spec))
                mapped_count += 1
            except (AttributeError, TypeError) as exc:
                items.append(f"None  # TODO: {child_type} — {exc}")
                todos.append(f"{safe_name}[{child_name}]: {child_type} — {exc}")
        elif list(child.named_children()):
            # Composite with children — recurse
            sub_lines, sub_total, sub_mapped, sub_skipped, sub_todos, sub_unmapped = _walk_module(
                child, seen_classes
            )
            total += sub_total
            mapped_count += sub_mapped
            skipped_count += sub_skipped
            todos.extend(sub_todos)
            unmapped_list.extend(sub_unmapped)
            if child_type not in seen_classes:
                cb, cc, em = _try_ast_for_classdef(child)
                seen_classes[child_type] = _ClassDef(
                    name=child_type,
                    init_body="\n".join(sub_lines),
                    forward_sig=_get_forward_signature(child),
                    call_body=cb,
                    call_confidence=cc,
                    extra_methods=em,
                )
            items.append(f"{child_type}()")
        elif child_type in CONSTRUCTOR_SPECS:
            # Stateless skip (None spec, no children)
            total += 1
            skipped_count += 1
        else:
            # Unmapped leaf
            total += 1
            unmapped_list.append(child_type)
            items.append(f"None  # TODO: {child_type}")
            todos.append(f"{safe_name}[{child_name}]: {child_type} has no constructor spec")

    if not items:
        return [], total, mapped_count, skipped_count, todos, unmapped_list

    init_lines = [f"        self.{safe_name} = ["]
    for item in items:
        init_lines.append(f"            {item},")
    init_lines.append("        ]")

    return init_lines, total, mapped_count, skipped_count, todos, unmapped_list


def _make_todo_call_helper(type_name: str, forward_sig: str) -> str:
    """Generate a TODO __call__ stub for a helper class."""
    return (
        "    def __call__(self, x: mx.array) -> mx.array:\n"
        f"        # TODO: Translate {type_name}.{forward_sig}\n"
        f'        raise NotImplementedError("{type_name}.forward() requires manual translation")'
    )


_CONFIG_ATTR_RE = _re.compile(r"self\.config\.(\w+)")


def _extract_config_refs(source: str) -> set[str]:
    """Extract all `self.config.X` attribute names from generated source."""
    return set(_CONFIG_ATTR_RE.findall(source))


# Config attribute overrides: force specific values in emitted SimpleNamespace
_CONFIG_OVERRIDES: dict[str, str] = {
    "_attn_implementation": "'eager'",
    "attn_implementation": "'eager'",
}


def _emit_config_line(model: Any, config_attrs: set[str]) -> str | None:
    """Build a `self.config = SimpleNamespace(...)` line from model.config.

    Returns None if no config attributes are referenced or model has no config.
    """
    config = getattr(model, "config", None)
    if config is None or not config_attrs:
        return None

    parts: list[str] = []
    for attr in sorted(config_attrs):
        if attr in _CONFIG_OVERRIDES:
            parts.append(f"{attr}={_CONFIG_OVERRIDES[attr]}")
            continue
        val = getattr(config, attr, None)
        if val is None:
            parts.append(f"{attr}=None")
        elif isinstance(val, bool):
            parts.append(f"{attr}={val}")
        elif isinstance(val, int):
            parts.append(f"{attr}={val}")
        elif isinstance(val, float):
            parts.append(f"{attr}={val}")
        elif isinstance(val, str):
            parts.append(f"{attr}={val!r}")
        else:
            parts.append(f"{attr}={val!r}")

    if not parts:
        return None

    joined = ", ".join(parts)
    return f"        self.config = SimpleNamespace({joined})"


def _inject_config_into_source(source: str, model: Any) -> str:
    """Post-process generated source to inject config constants where needed.

    Scans each class body for self.config.X references and injects a
    SimpleNamespace with the relevant values.  Adds the import if needed.
    """
    config = getattr(model, "config", None)
    if config is None:
        return source

    all_refs = _extract_config_refs(source)
    if not all_refs:
        return source

    # Build per-class config lines by scanning class bodies
    lines = source.split("\n")
    result_lines: list[str] = []
    needs_import = False
    i = 0

    while i < len(lines):
        line = lines[i]
        result_lines.append(line)

        # Detect class definitions and find their super().__init__() call
        if line.strip().startswith("class ") and "(nn.Module):" in line:
            # Collect this class's body to check for self.config refs
            class_start = i
            j = i + 1
            # Find the end of __init__ (next def or class or end of indent)
            init_end = None
            while j < len(lines):
                if lines[j].strip() == "super().__init__()":
                    init_end = j
                    break
                j += 1

            if init_end is not None:
                # Scan the rest of the class for self.config refs
                class_refs: set[str] = set()
                k = class_start
                while k < len(lines):
                    if (
                        k > class_start
                        and lines[k].strip().startswith("class ")
                        and "(nn.Module):" in lines[k]
                    ):
                        break
                    class_refs.update(_CONFIG_ATTR_RE.findall(lines[k]))
                    k += 1

                if class_refs:
                    config_line = _emit_config_line(model, class_refs)
                    if config_line is not None:
                        # Copy lines up to and including super().__init__()
                        for m in range(i + 1, init_end + 1):
                            result_lines.append(lines[m])
                        result_lines.append(config_line)
                        needs_import = True
                        i = init_end + 1
                        continue

        i += 1

    if needs_import:
        # Add SimpleNamespace import after the existing imports
        import_line = "from types import SimpleNamespace\n"
        # Insert after "import mlx.nn as nn"
        final = "\n".join(result_lines)
        final = final.replace(
            "import mlx.nn as nn\n",
            "import mlx.nn as nn\n" + import_line,
        )
        return final

    return source


def generate(
    model: Any,
    class_name: str | None = None,
    *,
    post_processors: list[Any] | None = None,
) -> GeneratedCode:
    """Generate MLX module source code from a torch.nn.Module (experimental).

    Output is an assisted port — treat as a starting point for manual review,
    not a finished product.  Some op lowerings are approximate; check
    ``result.coverage_metrics`` and ``result.call_confidence`` for diagnostics.

    Args:
        model: a torch.nn.Module instance
        class_name: name for the generated class (default: derived from model)
        post_processors: list of callables ``(source, model) -> source`` applied
            after assembly.  Defaults to ``[hf_post_process]``.  Pass ``[]``
            to disable HF-specific post-processing.

    Returns:
        GeneratedCode with the complete .py source
    """
    if not HAS_TORCH:
        raise ImportError("torch is required for code generation")

    if class_name is None:
        class_name = _class_name_from_module(model)

    # Ensure fx function map is populated
    if not _FX_FUNCTION_MAP:
        _build_fx_function_map()

    # Walk module tree for __init__
    seen_classes: dict[str, _ClassDef] = {}
    children = list(model.named_children())

    # Root-as-leaf: bare modules like nn.Linear(10, 20) have no children
    if not children:
        root_type = type(model).__name__
        root_spec = CONSTRUCTOR_SPECS.get(root_type)
        init_lines: list[str] = []
        total_leaves = 0
        mapped_leaves = 0
        skipped_leaves = 0
        todos: list[str] = []
        unmapped: list[str] = []
        if root_spec is not None:
            total_leaves = 1
            try:
                constructor_str = _format_constructor(model, root_spec)
                init_lines.append(f"        self.module = {constructor_str}")
                mapped_leaves = 1
            except (AttributeError, TypeError) as exc:
                todos.append(f"self.module: {root_type} — {exc}")
                init_lines.append(
                    f"        # TODO: self.module = {root_spec.mlx_call}(...)  # {exc}"
                )
    else:
        init_lines, total_leaves, mapped_leaves, skipped_leaves, todos, unmapped = _walk_module(
            model, seen_classes
        )

    # metrics is constructed after __call__ cascade so we can include op counts
    metrics = None  # set after __call__ generation

    # __call__ generation cascade: fx trace → AST rewrite → TODO stub
    traced = False
    ast_rewritten = False
    call_confidence = "todo"
    forward_total_ops = 0
    forward_mapped_ops = 0

    # 1. Try fx trace (works for simple traceable models)
    graph_module = _try_trace(model)
    if graph_module is not None:
        try:
            body, placeholders = _translate_graph(graph_module)
            params = ", ".join(placeholders)
            call_method = f"    def __call__(self, {params}):\n{body}"
            traced = True
        except Exception:
            pass

    # 2. AST rewrite (handles dynamic control flow that fx cannot)
    root_extra_methods = ""
    if not traced:
        rewrite = _rewrite_forward_ast(model)
        if rewrite is not None and rewrite.confidence != Confidence.BLOCKER:
            call_method = _format_ast_call(rewrite.source, rewrite.confidence)
            ast_rewritten = True
            call_confidence = rewrite.confidence.value
            root_extra_methods = _get_extra_methods(model, rewrite.source)
            forward_total_ops = rewrite.total_ops
            forward_mapped_ops = rewrite.mapped_ops
            # Merge unmapped calls into todos
            for call in rewrite.unmapped_calls:
                todos.append(f"__call__: unmapped call {call}")
        else:
            call_method = _make_todo_call(model)

    metrics = CoverageMetrics(
        total_leaves=total_leaves,
        mapped_leaves=mapped_leaves,
        skipped_leaves=skipped_leaves,
        unmapped_leaves=total_leaves - mapped_leaves - skipped_leaves,
        total_ops=forward_total_ops,
        mapped_ops=forward_mapped_ops,
    )

    # Assemble helper class source (post-order: deepest first)
    helper_source = ""
    for cls_def in seen_classes.values():
        cls_init = cls_def.init_body if cls_def.init_body.strip() else "        pass"
        if cls_def.call_body is not None:
            call_str = _format_ast_call(cls_def.call_body, Confidence(cls_def.call_confidence))
        else:
            call_str = _make_todo_call_helper(cls_def.name, cls_def.forward_sig)
        extra = cls_def.extra_methods if cls_def.extra_methods else ""
        helper_source += (
            f"class {cls_def.name}(nn.Module):\n"
            f"    def __init__(self) -> None:\n"
            f"        super().__init__()\n"
            f"{cls_init}\n\n"
            f"{extra}"
            f"{call_str}\n\n\n"
        )

    # Assemble main class source
    init_body = "\n".join(init_lines) if init_lines else "        pass"
    source = (
        f'"""MLX module generated by torch2mlx from {class_name}."""\n'
        f"from __future__ import annotations\n\n"
        f"import mlx.core as mx\n"
        f"import mlx.nn as nn\n\n\n"
        f"{helper_source}"
        f"class {class_name}(nn.Module):\n"
        f"    def __init__(self) -> None:\n"
        f"        super().__init__()\n"
        f"{init_body}\n\n"
        f"{root_extra_methods}"
        f"{call_method}\n"
    )

    # Post-process: inject self.config = SimpleNamespace(...) where referenced
    source = _inject_config_into_source(source, model)

    # Apply configurable post-processors (default: HF compat)
    if post_processors is None:
        post_processors = [hf_post_process]
    for pp in post_processors:
        source = pp(source, model)

    # Ensure SimpleNamespace import if used (by config injection or .device rewrite)
    if "SimpleNamespace" in source and "from types import SimpleNamespace" not in source:
        source = source.replace(
            "import mlx.nn as nn\n",
            "import mlx.nn as nn\nfrom types import SimpleNamespace\n",
        )

    # Add typing imports if type annotations reference them
    _typing_names = {"Optional", "Union", "Tuple", "List", "Dict", "Callable"}
    used_typing = [t for t in _typing_names if t in source]
    if used_typing and "from typing import" not in source:
        typing_import = f"from typing import {', '.join(sorted(used_typing))}\n"
        source = source.replace(
            "import mlx.core as mx\n",
            f"import mlx.core as mx\n{typing_import}",
        )

    # Add stdlib imports when referenced in generated code
    if "math." in source and "import math" not in source:
        source = source.replace(
            "import mlx.core as mx\n",
            "import math\nimport mlx.core as mx\n",
        )

    # Inject _torch_pad adapter when F.pad was rewritten
    if "_torch_pad(" in source and "def _torch_pad(" not in source:
        idx = source.find("\nclass ")
        if idx >= 0:
            source = source[:idx] + "\n" + _TORCH_PAD_HELPER + "\n" + source[idx:]

    return GeneratedCode(
        source=source,
        class_name=class_name,
        coverage_metrics=metrics,
        todos=todos,
        unmapped=unmapped,
        traced=traced,
        ast_rewritten=ast_rewritten,
        call_confidence=call_confidence,
    )


def _make_todo_call(model: Any) -> str:
    """Generate a TODO stub __call__ when fx tracing fails."""
    sig = _get_forward_signature(model)
    return (
        "    def __call__(self, x: mx.array) -> mx.array:\n"
        f"        # TODO: torch.fx tracing failed for this model.\n"
        f"        # Original forward signature: {sig}\n"
        f"        # See torch2mlx.templates for common patterns.\n"
        f'        raise NotImplementedError("Forward method requires manual translation")'
    )


def generate_to_file(
    model: Any,
    path: str | Path,
    class_name: str | None = None,
) -> Path:
    """Generate MLX module source and write to a file.

    Args:
        model: a torch.nn.Module instance
        path: output .py file path
        class_name: name for the generated class

    Returns:
        Path to the written file
    """
    path = Path(path)
    result = generate(model, class_name=class_name)
    path.write_text(result.source)
    return path
