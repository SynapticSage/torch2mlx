"""Code generation: emit MLX nn.Module source from a torch.nn.Module.

Generates a `.py` file containing:
  - __init__ with constructor calls derived from the module tree (always)
  - __call__ translated from torch.fx graph (when tracing succeeds)

Uses registry.py for layer mapping and op_mapping.py for operator translation.
"""

from __future__ import annotations

import inspect
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch2mlx.op_mapping import OP_REGISTRY

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


@dataclass
class GeneratedCode:
    """Result of code generation."""

    source: str  # complete .py source
    class_name: str
    coverage: float  # fraction of children with specs
    todos: list[str] = field(default_factory=list)
    unmapped: list[str] = field(default_factory=list)
    traced: bool = False  # True if fx trace succeeded for __call__


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
    }
    # F.* functions
    _f_funcs = {
        F.relu: "F.relu",
        F.gelu: "F.gelu",
        F.silu: "F.silu",
        F.softmax: "F.softmax",
        F.cross_entropy: "F.cross_entropy",
        F.mse_loss: "F.mse_loss",
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
    init_lines: list[str] = []
    total_children = 0
    mapped_children = 0
    todos: list[str] = []
    unmapped: list[str] = []

    children = list(model.named_children())

    # Root-as-leaf: bare modules like nn.Linear(10, 20) have no children
    if not children:
        root_type = type(model).__name__
        root_spec = CONSTRUCTOR_SPECS.get(root_type)
        if root_spec is not None:
            total_children = 1
            try:
                constructor_str = _format_constructor(model, root_spec)
                init_lines.append(f"        self.module = {constructor_str}")
                mapped_children = 1
            except (AttributeError, TypeError) as exc:
                todos.append(f"self.module: {root_type} — {exc}")
                init_lines.append(
                    f"        # TODO: self.module = {root_spec.mlx_call}(...)  # {exc}"
                )
    else:
        for name, child in children:
            safe_name = _sanitize_name(name)
            child_type = type(child).__name__
            total_children += 1

            spec = CONSTRUCTOR_SPECS.get(child_type)
            if spec is None and child_type in CONSTRUCTOR_SPECS:
                # Explicitly skipped type (containers, etc.)
                mapped_children += 1
                continue

            if spec is None:
                # Not in CONSTRUCTOR_SPECS at all
                unmapped.append(child_type)
                todos.append(f"self.{safe_name}: {child_type} has no constructor spec")
                init_lines.append(
                    f"        # TODO: self.{safe_name} = ...  # {child_type} — no constructor spec"
                )
                continue

            try:
                constructor_str = _format_constructor(child, spec)
                init_lines.append(f"        self.{safe_name} = {constructor_str}")
                mapped_children += 1
            except (AttributeError, TypeError) as exc:
                todos.append(f"self.{safe_name}: {child_type} — {exc}")
                init_lines.append(
                    f"        # TODO: self.{safe_name} = {spec.mlx_call}(...)  # {exc}"
                )

    coverage = mapped_children / total_children if total_children > 0 else 1.0

    # Try fx trace for __call__
    traced = False
    graph_module = _try_trace(model)
    if graph_module is not None:
        try:
            body, placeholders = _translate_graph(graph_module)
            params = ", ".join(placeholders)
            call_method = f"    def __call__(self, {params}):\n{body}"
            traced = True
        except Exception:
            call_method = _make_todo_call(model)
    else:
        call_method = _make_todo_call(model)

    # Assemble source
    init_body = "\n".join(init_lines) if init_lines else "        pass"
    source = (
        f'"""MLX module generated by torch2mlx from {class_name}."""\n'
        f"from __future__ import annotations\n\n"
        f"import mlx.core as mx\n"
        f"import mlx.nn as nn\n\n\n"
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
