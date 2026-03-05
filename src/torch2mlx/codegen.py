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
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


@dataclass
class GeneratedCode:
    """Result of code generation."""

    source: str  # complete .py source
    class_name: str
    coverage: float  # fraction of leaves with specs
    todos: list[str] = field(default_factory=list)
    unmapped: list[str] = field(default_factory=list)
    traced: bool = False  # True if fx trace succeeded for __call__
    ast_rewritten: bool = False  # True if AST rewrite succeeded for __call__
    call_confidence: str = "todo"  # "mechanical" | "needs_review" | "todo"


@dataclass
class _ClassDef:
    """Helper class definition to emit before the main class."""

    name: str  # e.g. "BertEmbeddings"
    init_body: str  # indented init lines (joined with newlines)
    forward_sig: str  # original forward() signature for TODO stub
    call_body: str | None = None  # AST-rewritten __call__, None → TODO stub
    call_confidence: str = "todo"  # "mechanical" | "needs_review" | "todo"


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
    (ArgSpec("normalized_shape", "dims"),),
)

_RMSNORM_SPEC = ConstructorSpec(
    "nn.RMSNorm",
    (ArgSpec("normalized_shape", "dims"),),
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
        ArgSpec("nf", "output_dims"),
        ArgSpec("nx", "input_dims"),
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
    "NewGELUActivation": _noarg("nn.GELU"),
    "QuickGELUActivation": _noarg("nn.GELU"),
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
    "Dinov2LayerScale": None,
    "Wav2Vec2SamePadLayer": None,
    "HubertSamePadLayer": None,
    "BeitRelativePositionBias": None,
}

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
    parts: list[str] = []
    for arg in spec.args:
        raw = getattr(module, arg.attr, None)
        value = _apply_transform(raw, arg.transform, module)

        # Omit kwargs that match their default
        if arg.default is not None and value == arg.default:
            continue

        parts.append(_format_value(value))

    return f"{spec.mlx_call}({', '.join(parts)})"


def _sanitize_name(name: str) -> str:
    """Convert a dotted module path to a valid Python identifier for __init__."""
    name = name.replace(".", "_")
    if name and name[0].isdigit():
        name = f"layer_{name}"
    return name


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
        return f"{node.name} = self.{node.target}"

    if op == "call_module":
        args_str = ", ".join(_node_arg_repr(a) for a in node.args)
        if node.kwargs:
            kw = ", ".join(f"{k}={_node_arg_repr(v)}" for k, v in node.kwargs.items())
            args_str = f"{args_str}, {kw}" if args_str else kw
        target = _sanitize_name(node.target)
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

                # Build call with param renames
                args_strs = [_node_arg_repr(a) for a in node.args]
                kw_parts: list[str] = []
                for k, v in node.kwargs.items():
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


class _TorchToMLXRewriter(_ast.NodeTransformer):
    """Rewrite a torch forward() AST into an MLX __call__() AST."""

    def __init__(self) -> None:
        self.annotations: list[tuple[int, Confidence, str]] = []
        self.unmapped_calls: list[str] = []
        self._confidence = Confidence.MECHANICAL

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
        # Convert type annotations
        if node.returns:
            node.returns = self._convert_annotation(node.returns)
        for arg in node.args.args:
            if arg.annotation:
                arg.annotation = self._convert_annotation(arg.annotation)
        self.generic_visit(node)
        return node

    def _convert_annotation(self, node: _ast.expr) -> _ast.expr:
        """torch.Tensor / torch.*Tensor → mx.array."""
        if isinstance(node, _ast.Attribute):
            if isinstance(node.value, _ast.Name) and node.value.id == "torch":
                if "Tensor" in node.attr:
                    return _ast.Attribute(
                        value=_ast.Name(id="mx", ctx=_ast.Load()),
                        attr="array",
                        ctx=_ast.Load(),
                    )
        # Optional[torch.Tensor] → Optional[mx.array]
        if isinstance(node, _ast.Subscript):
            node.slice = self._convert_annotation(node.slice)
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
        return node

    # --- Call: the main rewriting engine ---

    def visit_Call(self, node: _ast.Call) -> _ast.expr:
        # Visit children first so nested transforms resolve
        self.generic_visit(node)

        func = node.func
        if not isinstance(func, _ast.Attribute):
            return node

        attr_name = func.attr

        # self.xxx(args) → leave alone (submodule calls)
        if _is_self_access(func.value):
            # Special case: self.xxx.forward(args) → self.xxx(args)
            if attr_name == "forward" and isinstance(func.value, _ast.Attribute):
                return _ast.Call(func=func.value, args=node.args, keywords=node.keywords)
            return node

        # super().forward(args) → super().__call__(args)
        if attr_name == "forward" and self._is_super_call(func.value):
            func.attr = "__call__"
            return node

        # torch.func(args)
        if isinstance(func.value, _ast.Name) and func.value.id == "torch":
            return self._rewrite_torch_func(node, attr_name)

        # F.func(args) or torch.nn.functional.func(args)
        if isinstance(func.value, _ast.Name) and func.value.id == "F":
            return self._rewrite_f_func(node, attr_name)

        # x.method(args) — tensor methods
        return self._rewrite_method(node, attr_name)

    def _is_super_call(self, node: _ast.expr) -> bool:
        """Check if node is a super() call."""
        return (
            isinstance(node, _ast.Call)
            and isinstance(node.func, _ast.Name)
            and node.func.id == "super"
        )

    # --- torch.func() rewriting ---

    def _rewrite_torch_func(self, node: _ast.Call, func_name: str) -> _ast.expr:
        reg_key = f"torch.{func_name}"
        mapping = OP_REGISTRY.get(reg_key)
        if mapping is None:
            self.unmapped_calls.append(reg_key)
            self._lower_confidence(Confidence.NEEDS_REVIEW)
            return node

        if mapping.mlx_op == "no_op":
            return node.args[0] if node.args else node

        return _ast.Call(
            func=_make_mx_attr(mapping.mlx_op),
            args=list(node.args),
            keywords=self._rename_kwargs(node.keywords, mapping.param_renames),
        )

    # --- F.func() rewriting ---

    def _rewrite_f_func(self, node: _ast.Call, func_name: str) -> _ast.expr:
        reg_key = f"F.{func_name}"
        mapping = OP_REGISTRY.get(reg_key)
        if mapping is None:
            self.unmapped_calls.append(reg_key)
            self._lower_confidence(Confidence.NEEDS_REVIEW)
            return node

        if mapping.mlx_op == "no_op":
            return node.args[0] if node.args else node

        return _ast.Call(
            func=_make_mx_attr(mapping.mlx_op),
            args=list(node.args),
            keywords=self._rename_kwargs(node.keywords, mapping.param_renames),
        )

    # --- x.method() rewriting ---

    def _rewrite_method(self, node: _ast.Call, method_name: str) -> _ast.expr:
        receiver = node.func.value

        # Special methods
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

        # Registry-mapped methods
        reg_key = _FX_METHOD_MAP.get(method_name)
        if reg_key is None:
            # Not a known tensor method — leave as-is (could be dict/list method)
            return node

        mapping = OP_REGISTRY.get(reg_key)
        if mapping is None:
            self.unmapped_calls.append(method_name)
            self._lower_confidence(Confidence.NEEDS_REVIEW)
            return node

        if mapping.mlx_op == "no_op":
            return receiver

        # Method → function: prepend receiver as first arg
        return _ast.Call(
            func=_make_mx_attr(mapping.mlx_op),
            args=[receiver] + list(node.args),
            keywords=self._rename_kwargs(node.keywords, mapping.param_renames),
        )

    # --- Special method handlers ---

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
        """x.contiguous() → x, x.to(...) → x, etc."""
        method = node.func.attr
        if method == "to" and node.args:
            self._annotate(
                getattr(node, "lineno", 0),
                Confidence.NEEDS_REVIEW,
                ".to() may be a dtype cast, not just device move",
            )
        return node.func.value

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

    # --- Helpers ---

    def _rename_kwargs(
        self,
        keywords: list[_ast.keyword],
        renames: dict[str, str],
    ) -> list[_ast.keyword]:
        """Apply parameter renames (e.g. dim → axis)."""
        result = []
        for kw in keywords:
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
    )


def _format_ast_call(source: str, confidence: Confidence) -> str:
    """Indent and annotate an AST-rewritten __call__ for class body."""
    header = f"    # --- torch2mlx: {confidence.value.upper()} (AST rewrite) ---"
    indented = "\n".join(f"    {line}" for line in source.split("\n"))
    return f"{header}\n{indented}"


def _try_ast_for_classdef(child: Any) -> tuple[str | None, str]:
    """Try AST rewrite for a helper class __call__."""
    rewrite = _rewrite_forward_ast(child)
    if rewrite is not None and rewrite.confidence != Confidence.BLOCKER:
        return rewrite.source, rewrite.confidence.value
    return None, "todo"


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
) -> tuple[list[str], int, int, list[str], list[str]]:
    """Recursively walk module tree for __init__ generation.

    Returns:
        (init_lines, total_leaves, mapped_leaves, todos, unmapped)
    """
    init_lines: list[str] = []
    total = 0
    mapped = 0
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
                    c_lines, c_total, c_mapped, c_todos, c_unmapped = _handle_container(
                        safe_name, child, seen_classes
                    )
                    init_lines.extend(c_lines)
                    total += c_total
                    mapped += c_mapped
                    todos.extend(c_todos)
                    unmapped.extend(c_unmapped)
                # else: empty container → 0 leaves, skip silently
            elif has_children:
                # 2b: Composite in CONSTRUCTOR_SPECS (e.g. TransformerEncoderLayer)
                sub_lines, sub_total, sub_mapped, sub_todos, sub_unmapped = _walk_module(
                    child, seen_classes
                )
                total += sub_total
                mapped += sub_mapped
                todos.extend(sub_todos)
                unmapped.extend(sub_unmapped)
                if child_type not in seen_classes:
                    cb, cc = _try_ast_for_classdef(child)
                    seen_classes[child_type] = _ClassDef(
                        name=child_type,
                        init_body="\n".join(sub_lines),
                        forward_sig=_get_forward_signature(child),
                        call_body=cb,
                        call_confidence=cc,
                    )
                init_lines.append(f"        self.{safe_name} = {child_type}()")
            else:
                # 2c: Stateless skip (Identity, DropPath, RoPE, etc.)
                total += 1
                mapped += 1

        else:
            # CASE 3: Not in CONSTRUCTOR_SPECS at all
            if list(child.named_children()):
                # 3a: Composite — recurse, register helper class
                sub_lines, sub_total, sub_mapped, sub_todos, sub_unmapped = _walk_module(
                    child, seen_classes
                )
                total += sub_total
                mapped += sub_mapped
                todos.extend(sub_todos)
                unmapped.extend(sub_unmapped)
                if child_type not in seen_classes:
                    cb, cc = _try_ast_for_classdef(child)
                    seen_classes[child_type] = _ClassDef(
                        name=child_type,
                        init_body="\n".join(sub_lines),
                        forward_sig=_get_forward_signature(child),
                        call_body=cb,
                        call_confidence=cc,
                    )
                init_lines.append(f"        self.{safe_name} = {child_type}()")
            else:
                # 3b: Truly unmapped leaf
                total += 1
                unmapped.append(child_type)
                todos.append(f"self.{safe_name}: {child_type} has no constructor spec")
                init_lines.append(
                    f"        # TODO: self.{safe_name} = ...  # {child_type} — no constructor spec"
                )

    return init_lines, total, mapped, todos, unmapped


def _handle_container(
    safe_name: str,
    container: Any,
    seen_classes: dict[str, _ClassDef],
) -> tuple[list[str], int, int, list[str], list[str]]:
    """Handle ModuleList/Sequential/ModuleDict containers.

    Uniform type → list comprehension.  Mixed types → individual items.

    Returns:
        (init_lines, total_leaves, mapped_leaves, todos, unmapped)
    """
    children = list(container.named_children())
    if not children:
        return [], 0, 0, [], []

    child_types = [type(c).__name__ for _, c in children]
    count = len(children)

    # --- Uniform type → list comprehension ---
    if len(set(child_types)) == 1:
        item_type = child_types[0]
        rep = children[0][1]

        # Uniform leaf with constructor
        spec = CONSTRUCTOR_SPECS.get(item_type)
        if spec is not None:
            try:
                cstr = _format_constructor(rep, spec)
                line = f"        self.{safe_name} = [{cstr} for _ in range({count})]"
                return [line], count, count, [], []
            except (AttributeError, TypeError):
                pass  # fall through to mixed path

        # Uniform composite/container with children → recurse representative
        if list(rep.named_children()):
            sub_lines, sub_total, sub_mapped, sub_todos, sub_unmapped = _walk_module(
                rep, seen_classes
            )
            if item_type not in seen_classes:
                cb, cc = _try_ast_for_classdef(rep)
                seen_classes[item_type] = _ClassDef(
                    name=item_type,
                    init_body="\n".join(sub_lines),
                    forward_sig=_get_forward_signature(rep),
                    call_body=cb,
                    call_confidence=cc,
                )
            line = f"        self.{safe_name} = [{item_type}() for _ in range({count})]"
            return [line], sub_total * count, sub_mapped * count, sub_todos, sub_unmapped

        # Uniform stateless skip / unmapped without children
        if item_type in CONSTRUCTOR_SPECS:
            return [], count, count, [], []
        return (
            [f"        # TODO: self.{safe_name} = [...]  # {count}x {item_type}"],
            count,
            0,
            [f"{safe_name}: {count}x {item_type} has no constructor spec"],
            [item_type],
        )

    # --- Mixed types → emit items individually ---
    items: list[str] = []
    total = 0
    mapped_count = 0
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
            sub_lines, sub_total, sub_mapped, sub_todos, sub_unmapped = _walk_module(
                child, seen_classes
            )
            total += sub_total
            mapped_count += sub_mapped
            todos.extend(sub_todos)
            unmapped_list.extend(sub_unmapped)
            if child_type not in seen_classes:
                cb, cc = _try_ast_for_classdef(child)
                seen_classes[child_type] = _ClassDef(
                    name=child_type,
                    init_body="\n".join(sub_lines),
                    forward_sig=_get_forward_signature(child),
                    call_body=cb,
                    call_confidence=cc,
                )
            items.append(f"{child_type}()")
        elif child_type in CONSTRUCTOR_SPECS:
            # Stateless skip (None spec, no children)
            total += 1
            mapped_count += 1
        else:
            # Unmapped leaf
            total += 1
            unmapped_list.append(child_type)
            items.append(f"None  # TODO: {child_type}")
            todos.append(f"{safe_name}[{child_name}]: {child_type} has no constructor spec")

    if not items:
        return [], total, mapped_count, todos, unmapped_list

    init_lines = [f"        self.{safe_name} = ["]
    for item in items:
        init_lines.append(f"            {item},")
    init_lines.append("        ]")

    return init_lines, total, mapped_count, todos, unmapped_list


def _make_todo_call_helper(type_name: str, forward_sig: str) -> str:
    """Generate a TODO __call__ stub for a helper class."""
    return (
        "    def __call__(self, x: mx.array) -> mx.array:\n"
        f"        # TODO: Translate {type_name}.{forward_sig}\n"
        f'        raise NotImplementedError("{type_name}.forward() requires manual translation")'
    )


def generate(model: Any, class_name: str | None = None) -> GeneratedCode:
    """Generate MLX module source code from a torch.nn.Module.

    Args:
        model: a torch.nn.Module instance
        class_name: name for the generated class (default: derived from model)

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
        init_lines, total_leaves, mapped_leaves, todos, unmapped = _walk_module(model, seen_classes)

    coverage = mapped_leaves / total_leaves if total_leaves > 0 else 1.0

    # __call__ generation cascade: fx trace → AST rewrite → TODO stub
    traced = False
    ast_rewritten = False
    call_confidence = "todo"

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
    if not traced:
        rewrite = _rewrite_forward_ast(model)
        if rewrite is not None and rewrite.confidence != Confidence.BLOCKER:
            call_method = _format_ast_call(rewrite.source, rewrite.confidence)
            ast_rewritten = True
            call_confidence = rewrite.confidence.value
            # Merge unmapped calls into todos
            for call in rewrite.unmapped_calls:
                todos.append(f"__call__: unmapped call {call}")
        else:
            call_method = _make_todo_call(model)

    # Assemble helper class source (post-order: deepest first)
    helper_source = ""
    for cls_def in seen_classes.values():
        cls_init = cls_def.init_body if cls_def.init_body.strip() else "        pass"
        if cls_def.call_body is not None:
            call_str = _format_ast_call(cls_def.call_body, Confidence(cls_def.call_confidence))
        else:
            call_str = _make_todo_call_helper(cls_def.name, cls_def.forward_sig)
        helper_source += (
            f"class {cls_def.name}(nn.Module):\n"
            f"    def __init__(self) -> None:\n"
            f"        super().__init__()\n"
            f"{cls_init}\n\n"
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
        f"{call_method}\n"
    )

    return GeneratedCode(
        source=source,
        class_name=class_name,
        coverage=coverage,
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
