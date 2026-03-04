"""Layer registry: torch.nn.X -> mlx.nn.X mapping table.

Maps each PyTorch module class to its MLX equivalent constructor
and any API adaptation needed (e.g., class vs function activations).
Layers not in the registry are reported as manual-port items.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerMapping:
    """Single torch -> MLX layer mapping entry."""

    torch_name: str
    mlx_name: str
    weight_transposition: str  # key into weight_converter dispatch
    notes: str = ""


# Populated by registry-agent
LAYER_REGISTRY: dict[str, LayerMapping] = {}


def lookup(torch_class_name: str) -> LayerMapping | None:
    """Find the MLX equivalent for a torch module class name."""
    return LAYER_REGISTRY.get(torch_class_name)


def register(mapping: LayerMapping) -> None:
    """Add a layer mapping to the registry."""
    LAYER_REGISTRY[mapping.torch_name] = mapping


def registered_names() -> list[str]:
    """Return all registered torch class names."""
    return list(LAYER_REGISTRY.keys())


def _populate() -> None:
    _ENTRIES = [
        LayerMapping("Linear", "nn.Linear", "identity", "Identical API"),
        LayerMapping("Embedding", "nn.Embedding", "identity", "Identical"),
        LayerMapping("LayerNorm", "nn.LayerNorm", "identity", "Identical"),
        LayerMapping("RMSNorm", "nn.RMSNorm", "identity", "MLX has this natively"),
        LayerMapping("Conv1d", "nn.Conv1d", "conv1d", "Weight layout differs"),
        LayerMapping("Conv2d", "nn.Conv2d", "conv2d", "Weight layout differs"),
        LayerMapping(
            "ConvTranspose1d", "nn.ConvTranspose1d", "conv_transpose1d", "Weight layout differs"
        ),
        LayerMapping(
            "ConvTranspose2d", "nn.ConvTranspose2d", "conv_transpose2d", "Weight layout differs"
        ),
        LayerMapping("BatchNorm1d", "nn.BatchNorm", "batch_norm", "Per-param identity"),
        LayerMapping("BatchNorm2d", "nn.BatchNorm", "batch_norm", "Per-param identity"),
        LayerMapping(
            "MultiheadAttention", "nn.MultiHeadAttention", "identity", "Different API surface"
        ),
        LayerMapping("GELU", "nn.GELU", "identity", "Class vs function"),
        LayerMapping("ReLU", "nn.ReLU", "identity", "Class vs function"),
        LayerMapping("SiLU", "nn.SiLU", "identity", "Class vs function"),
        LayerMapping("Dropout", "nn.Dropout", "identity", ""),
        LayerMapping("ModuleList", "None", "identity", "No MLX equivalent — needs wrapper"),
        LayerMapping("Sequential", "None", "identity", "No MLX equivalent — needs wrapper"),
        LayerMapping("Tanh", "nn.Tanh", "identity", "Stateless activation"),
        LayerMapping("Sigmoid", "nn.Sigmoid", "identity", "Stateless activation"),
        LayerMapping("LeakyReLU", "nn.LeakyReLU", "identity", "negative_slope param identical"),
        LayerMapping("Softmax", "nn.Softmax", "identity", "Stateless"),
        LayerMapping("GroupNorm", "nn.GroupNorm", "identity", "MLX uses dims not num_channels"),
        LayerMapping("InstanceNorm1d", "nn.InstanceNorm", "identity", "MLX default affine=False"),
        LayerMapping("InstanceNorm2d", "nn.InstanceNorm", "identity", "MLX default affine=False"),
        # Pooling — native MLX, no learnable parameters
        LayerMapping("MaxPool1d", "nn.MaxPool1d", "identity", "Native MLX, no params"),
        LayerMapping("MaxPool2d", "nn.MaxPool2d", "identity", "Native MLX, no params"),
        LayerMapping("MaxPool3d", "nn.MaxPool3d", "identity", "Native MLX, no params"),
        LayerMapping("AvgPool1d", "nn.AvgPool1d", "identity", "Native MLX, no params"),
        LayerMapping("AvgPool2d", "nn.AvgPool2d", "identity", "Native MLX, no params"),
        LayerMapping("AvgPool3d", "nn.AvgPool3d", "identity", "Native MLX, no params"),
        LayerMapping(
            "AdaptiveAvgPool2d", "None", "identity", "Custom template, dynamic kernel/stride"
        ),
        LayerMapping("Flatten", "None", "identity", "Stateless, use mx.flatten"),
        # Compound modules — decompose into registered children
        LayerMapping(
            "TransformerEncoder", "None", "identity", "Decomposes into registered children"
        ),
        LayerMapping(
            "TransformerDecoder", "None", "identity", "Decomposes into registered children"
        ),
        LayerMapping(
            "TransformerEncoderLayer", "None", "identity", "Decomposes into registered children"
        ),
        LayerMapping(
            "TransformerDecoderLayer", "None", "identity", "Decomposes into registered children"
        ),
        # PyTorch internal Linear subclass (used in MultiheadAttention)
        LayerMapping(
            "NonDynamicallyQuantizableLinear",
            "nn.Linear",
            "identity",
            "Internal torch Linear subclass",
        ),
        # HuggingFace custom types
        LayerMapping("GELUActivation", "nn.GELU", "identity", "HF custom GELU wrapper"),
        LayerMapping("NewGELUActivation", "nn.GELU", "identity", "HF GPT-2 GELU variant"),
        LayerMapping(
            "QuickGELUActivation", "nn.GELU", "identity", "HF CLIP x*sigmoid(1.702x) approx"
        ),
        LayerMapping(
            "Conv1D", "nn.Linear", "linear_transposed", "HF GPT-2 Linear with [in,out] weights"
        ),
        LayerMapping("T5LayerNorm", "nn.RMSNorm", "identity", "HF T5 RMSNorm (no bias, no mean)"),
        LayerMapping(
            "WhisperPositionalEmbedding",
            "nn.Embedding",
            "identity",
            "HF Whisper nn.Embedding subclass",
        ),
        LayerMapping("DebertaLayerNorm", "nn.LayerNorm", "identity", "HF DeBERTa custom LayerNorm"),
        LayerMapping("BloomGelu", "nn.GELU", "identity", "HF BLOOM custom GELU"),
        LayerMapping("SiLUActivation", "nn.SiLU", "identity", "HF SiLU wrapper (Qwen2, etc.)"),
        LayerMapping("Qwen2RMSNorm", "nn.RMSNorm", "identity", "HF Qwen2 RMSNorm"),
        LayerMapping(
            "ConvNextLayerNorm", "nn.LayerNorm", "identity", "HF ConvNeXt LayerNorm subclass"
        ),
        LayerMapping(
            "OPTLearnedPositionalEmbedding",
            "nn.Embedding",
            "identity",
            "HF OPT nn.Embedding subclass",
        ),
        LayerMapping(
            "BartLearnedPositionalEmbedding",
            "nn.Embedding",
            "identity",
            "HF BART nn.Embedding subclass",
        ),
        LayerMapping(
            "BartScaledWordEmbedding", "nn.Embedding", "identity", "HF BART scaled nn.Embedding"
        ),
        # PyTorch built-ins not yet covered
        LayerMapping("Identity", "None", "identity", "torch.nn.Identity — passthrough"),
        LayerMapping("AdaptiveAvgPool1d", "None", "identity", "No MLX equivalent, no params"),
        LayerMapping("ModuleDict", "None", "identity", "Container, no MLX equivalent"),
        # HuggingFace architecture-specific (no learnable weights)
        LayerMapping("SwinDropPath", "None", "identity", "Stochastic depth — identity at eval"),
        LayerMapping("Dinov2LayerScale", "None", "identity", "Learnable scalar multiply"),
        LayerMapping(
            "Qwen2RotaryEmbedding",
            "None",
            "identity",
            "RoPE — computed sin/cos, no learned weights",
        ),
        LayerMapping("Wav2Vec2SamePadLayer", "None", "identity", "Padding removal, no params"),
        # PyTorch parametrization internals (e.g., weight_norm on Conv1d)
        LayerMapping(
            "ParametrizedConv1d", "nn.Conv1d", "conv1d", "Conv1d with parametrization (weight_norm)"
        ),
        LayerMapping(
            "ParametrizationList", "None", "identity", "PyTorch parametrization container"
        ),
        LayerMapping("_WeightNorm", "None", "identity", "PyTorch weight_norm internal, no params"),
        # More HuggingFace activations & norms
        LayerMapping("FalconLinear", "nn.Linear", "identity", "HF Falcon nn.Linear subclass"),
        LayerMapping("ReLU6", "None", "identity", "min(max(0,x),6) — stateless, no MLX equiv"),
        # Rotary embedding variants (computed sin/cos, no learned weights)
        LayerMapping("GPTNeoXRotaryEmbedding", "None", "identity", "Pythia/GPT-NeoX RoPE"),
        LayerMapping("FalconRotaryEmbedding", "None", "identity", "Falcon RoPE"),
        # Embedding subclasses
        LayerMapping(
            "PegasusSinusoidalPositionalEmbedding",
            "nn.Embedding",
            "identity",
            "HF Pegasus sinusoidal pos embed",
        ),
        # DropPath variants (stochastic depth — identity at eval)
        LayerMapping("BeitDropPath", "None", "identity", "BEiT stochastic depth"),
        LayerMapping("SegformerDropPath", "None", "identity", "SegFormer stochastic depth"),
        # Architecture-specific leaves
        LayerMapping(
            "BeitRelativePositionBias", "None", "identity", "BEiT learned relative pos bias table"
        ),
        LayerMapping("HubertSamePadLayer", "None", "identity", "HuBERT padding removal, no params"),
    ]
    for entry in _ENTRIES:
        register(entry)


_populate()
