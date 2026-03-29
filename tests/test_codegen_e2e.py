"""End-to-end codegen validation: generate → convert → load → compare outputs.

These tests verify that generated MLX module source code can:
1. Parse without errors
2. Instantiate an mlx.nn.Module
3. Load converted weights
4. Produce numerically equivalent outputs to the PyTorch original
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mx = pytest.importorskip("mlx.core")
mlx_nn = pytest.importorskip("mlx.nn")

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from torch2mlx import convert, load_converted  # noqa: E402
from torch2mlx.codegen import generate  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_e2e(
    model: nn.Module,
    input_shape: tuple[int, ...],
    *,
    atol: float = 1e-5,
    class_name: str | None = None,
) -> None:
    """Generate MLX code, convert weights, load, and compare outputs."""
    model.eval()
    result = generate(model, class_name=class_name)

    # Source must parse
    ast.parse(result.source)

    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "weights")
        convert(model, out)
        flat = load_converted(out, flat=True)
        weights = [(k, mx.array(v)) for k, v in flat.items()]

        # Exec generated source
        ns: dict = {}
        exec(result.source, ns)
        cls_name = class_name or type(model).__name__
        mlx_model = ns[cls_name]()
        mlx_model.load_weights(weights)

        # Run both
        x_torch = torch.randn(*input_shape)
        with torch.no_grad():
            y_torch = model(x_torch).numpy()

        x_mlx = mx.array(x_torch.numpy())
        y_mlx = np.array(mlx_model(x_mlx))

        diff = np.abs(y_torch - y_mlx).max()
        assert diff < atol, (
            f"Max diff {diff:.2e} exceeds atol={atol:.0e}\n"
            f"PyTorch: {y_torch.ravel()[:5]}\n"
            f"MLX:     {y_mlx.ravel()[:5]}"
        )


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(32)
        self.proj = nn.Linear(32, 32)
        self.norm2 = nn.LayerNorm(32)
        self.ff1 = nn.Linear(32, 64)
        self.ff2 = nn.Linear(64, 32)

    def forward(self, x):
        h = self.proj(self.norm1(x))
        x = x + h
        h = self.ff2(F.gelu(self.ff1(self.norm2(x))))
        return x + h


class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 16)
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc(self.embed(x))


class MultiLayerNorm(nn.Module):
    """Model with multiple differently-sized LayerNorms."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.norm1 = nn.LayerNorm(16)
        self.fc2 = nn.Linear(16, 8)
        self.norm2 = nn.LayerNorm(8)

    def forward(self, x):
        x = self.norm1(F.silu(self.fc1(x)))
        return self.norm2(self.fc2(x))


class DeepMLP(nn.Module):
    """Deeper model to test longer fx graphs."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class PaddedLinear(nn.Module):
    """Model using F.pad before linear — tests pad format adapter."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 4)

    def forward(self, x):
        # Pad last dim by (1, 1): shape (B, 4) → (B, 6)
        x = F.pad(x, (1, 1))
        return self.fc(x)


class BareLinear(nn.Module):
    """Root-as-leaf: a bare Linear layer (no children)."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 8))
        self.bias = nn.Parameter(torch.randn(4))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class ModuleListModel(nn.Module):
    """fx-traceable model with ModuleList — tests container access in fx codegen."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------


class TestCodegenE2E:
    """End-to-end: generate → convert → load → compare."""

    def test_tiny_mlp(self):
        _run_e2e(TinyMLP(), (2, 4))

    def test_transformer_block(self):
        _run_e2e(TransformerBlock(), (2, 8, 32))

    def test_embedding(self):
        model = EmbeddingNet()
        model.eval()
        result = generate(model)
        ast.parse(result.source)

        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "weights")
            convert(model, out)
            flat = load_converted(out, flat=True)
            weights = [(k, mx.array(v)) for k, v in flat.items()]

            ns: dict = {}
            exec(result.source, ns)
            mlx_model = ns["EmbeddingNet"]()
            mlx_model.load_weights(weights)

            # Embedding takes int input
            x_torch = torch.randint(0, 100, (2, 5))
            with torch.no_grad():
                y_torch = model(x_torch).numpy()

            x_mlx = mx.array(x_torch.numpy())
            y_mlx = np.array(mlx_model(x_mlx))

            diff = np.abs(y_torch - y_mlx).max()
            assert diff < 1e-5

    def test_multi_layernorm(self):
        _run_e2e(MultiLayerNorm(), (2, 8))

    def test_deep_mlp(self):
        _run_e2e(DeepMLP(), (3, 8))

    def test_custom_class_name(self):
        _run_e2e(TinyMLP(), (2, 4), class_name="MyCustomMLP")

    def test_modulelist_fx_container_access(self):
        """fx-traced ModuleList: forward refs use layers[i] not layers_i."""
        model = ModuleListModel()
        result = generate(model)
        # Should generate indexing access, not flat names
        if result.traced:
            assert "self.layers[" in result.source or "self.layers" in result.source
            assert "self.layers_0" not in result.source
        _run_e2e(model, (2, 4))

    def test_padded_linear(self):
        """F.pad adapter: PyTorch flat pad format → MLX nested format."""
        model = PaddedLinear()
        result = generate(model)
        # Should emit _torch_pad helper, not raw mx.pad
        assert "def _torch_pad(" in result.source
        assert "_torch_pad(" in result.source
        # Input (B, 4) → pad last dim by (1,1) → (B, 6) → Linear(6, 4) → (B, 4)
        _run_e2e(model, (2, 4))

    def test_torch_pad_helper_2d(self):
        """_torch_pad handles multi-axis padding (last 2 dims)."""
        # Directly test the helper at runtime
        result = generate(PaddedLinear())  # just to get the source with helper
        ns: dict = {}
        exec(result.source, ns)
        pad_fn = ns["_torch_pad"]

        x = mx.zeros((2, 3, 4))
        # Pad last dim by (1,1), second-to-last by (2,2) — PyTorch reverse order
        out = pad_fn(x, (1, 1, 2, 2))
        assert out.shape == (2, 7, 6)  # (2, 3+4, 4+2)


class TestCodegenE2ECoverage:
    """Verify coverage and metadata from e2e runs."""

    def test_mlp_coverage(self):
        result = generate(TinyMLP())
        assert result.coverage == 1.0
        assert result.traced is True
        assert len(result.unmapped) == 0

    def test_transformer_coverage(self):
        result = generate(TransformerBlock())
        assert result.coverage == 1.0
        assert result.traced is True

    def test_source_parses(self):
        for model in [TinyMLP(), TransformerBlock(), MultiLayerNorm(), DeepMLP()]:
            result = generate(model)
            ast.parse(result.source)  # Must not raise


# ---------------------------------------------------------------------------
# HuggingFace e2e tests: generate → parse → instantiate → load weights
# ---------------------------------------------------------------------------

transformers = pytest.importorskip("transformers")

# Models that successfully load weights (no ParameterList / custom buffer issues)
_HF_MODELS = [
    ("BertModel", "bert-base-uncased"),
    ("RobertaModel", "roberta-base"),
    ("ElectraModel", "google/electra-small-discriminator"),
    ("DistilBertModel", "distilbert-base-uncased"),
    ("ViTModel", "google/vit-base-patch16-224"),
    ("XLNetModel", "xlnet-base-cased"),
    ("GPT2Model", "gpt2"),
]


@pytest.fixture(scope="module")
def _hf_cache():
    """Cache generated code + weights across tests in this module."""
    return {}


def _get_hf_data(cls_name, checkpoint, cache):
    """Load or retrieve cached HF model data."""
    if cls_name in cache:
        return cache[cls_name]

    cls = getattr(transformers, cls_name)
    model = cls.from_pretrained(checkpoint)
    model.eval()

    result = generate(model)

    tmp = tempfile.mkdtemp()
    out = str(Path(tmp) / "weights")
    convert(model, out)
    flat = load_converted(out, flat=True)
    weights = [(k, mx.array(v)) for k, v in flat.items()]

    data = {
        "result": result,
        "weights": weights,
        "n_weights": len(weights),
    }
    cache[cls_name] = data
    return data


class TestHFCodegenE2E:
    """HuggingFace models: generate → parse → instantiate → load weights."""

    @pytest.mark.parametrize("cls_name,checkpoint", _HF_MODELS)
    def test_parses(self, cls_name, checkpoint, _hf_cache):
        data = _get_hf_data(cls_name, checkpoint, _hf_cache)
        ast.parse(data["result"].source)

    @pytest.mark.parametrize("cls_name,checkpoint", _HF_MODELS)
    def test_coverage_100(self, cls_name, checkpoint, _hf_cache):
        data = _get_hf_data(cls_name, checkpoint, _hf_cache)
        assert data["result"].coverage_metrics.registry_coverage == 1.0

    @pytest.mark.parametrize("cls_name,checkpoint", _HF_MODELS)
    def test_instantiates(self, cls_name, checkpoint, _hf_cache):
        data = _get_hf_data(cls_name, checkpoint, _hf_cache)
        ns: dict = {}
        exec(data["result"].source, ns)
        mlx_model = ns[cls_name]()
        assert mlx_model is not None

    @pytest.mark.parametrize("cls_name,checkpoint", _HF_MODELS)
    def test_loads_weights(self, cls_name, checkpoint, _hf_cache):
        data = _get_hf_data(cls_name, checkpoint, _hf_cache)
        ns: dict = {}
        exec(data["result"].source, ns)
        mlx_model = ns[cls_name]()
        mlx_model.load_weights(data["weights"], strict=False)
        # Verify converted weight count is close to the model's parameter count.
        # Gap allowed for codegen-emitted zero-initialized buffers (e.g.
        # position_ids, causal mask buffers in decoder models like GPT-2).
        from mlx.utils import tree_flatten

        n_model_params = len(tree_flatten(mlx_model.parameters()))
        gap = n_model_params - data["n_weights"]
        max_gap = max(5, n_model_params // 6)
        assert gap <= max_gap, (
            f"{cls_name}: converted {data['n_weights']} weights but model has "
            f"{n_model_params} parameters (gap: {gap}, max: {max_gap})"
        )

    @pytest.mark.parametrize("cls_name,checkpoint", _HF_MODELS)
    def test_has_helper_classes(self, cls_name, checkpoint, _hf_cache):
        """Nested models should emit helper class definitions."""
        data = _get_hf_data(cls_name, checkpoint, _hf_cache)
        source = data["result"].source
        n_classes = source.count("class ")
        # All these models are nested — should have multiple helper classes
        assert n_classes > 1, f"{cls_name} should emit helper classes, got {n_classes}"

    @pytest.mark.parametrize("cls_name,checkpoint", _HF_MODELS)
    def test_ast_rewritten_calls(self, cls_name, checkpoint, _hf_cache):
        """Generated __call__ methods should be AST-rewritten, not TODO stubs."""
        data = _get_hf_data(cls_name, checkpoint, _hf_cache)
        source = data["result"].source
        assert "MECHANICAL" in source, f"{cls_name} should have MECHANICAL __call__"
        # Should NOT have NotImplementedError inside class __call__ methods (all get AST rewrite).
        # Global function stubs (e.g. _prepare_4d_*) may use NotImplementedError — that's fine.
        in_class = False
        class_nies = 0
        for line in source.split("\n"):
            if line.startswith("class "):
                in_class = True
            elif not line.startswith(" ") and not line.startswith("\t") and line.strip():
                if not line.startswith("class "):
                    in_class = False
            if in_class and "NotImplementedError" in line:
                class_nies += 1
        assert class_nies == 0, f"{cls_name} has {class_nies} NotImplementedError in class bodies"


# ---------------------------------------------------------------------------
# HF forward pass e2e: generate → load → run → compare numerical output
# ---------------------------------------------------------------------------

# Models verified to produce numerically equivalent output
# Format: (cls_name, checkpoint, input_kind)
#   input_kind: "text" → random input_ids, "vision" → random pixel_values
#
# Not yet validated (codegen gaps):
#   XLNet — relative_positional_encoding + torch.eye/F.one_hot needed
_HF_FORWARD_MODELS = [
    ("DistilBertModel", "distilbert-base-uncased", "text"),
    ("BertModel", "bert-base-uncased", "text"),
    ("RobertaModel", "roberta-base", "text"),
    ("ElectraModel", "google/electra-small-discriminator", "text"),
    ("ViTModel", "google/vit-base-patch16-224", "vision"),
    ("GPT2Model", "gpt2", "text"),
    # Phase 2: near-clones of validated models
    ("CamembertModel", "camembert-base", "text"),
    ("Data2VecTextModel", "facebook/data2vec-text-base", "text"),
    ("MPNetModel", "microsoft/mpnet-base", "text"),
    ("Dinov2Model", "facebook/dinov2-small", "vision"),
    # Phase 3: decoder + encoder variants
    ("GPTNeoModel", "EleutherAI/gpt-neo-125m", "text"),
    ("AlbertModel", "albert-base-v2", "text"),
    ("OPTModel", "facebook/opt-125m", "text"),
    # BART — deferred: present_key_value None+None when use_cache=False in decoder layer
    # Longformer — deferred: torch.div with rounding_mode, sliding window attention
    ("XLNetModel", "xlnet-base-cased", "text"),
]


def _make_hf_inputs(cls_name, checkpoint, input_kind):
    """Create dummy inputs appropriate for the model modality."""
    pt_model = getattr(transformers, cls_name).from_pretrained(checkpoint)
    pt_model.eval()
    config = pt_model.config

    if input_kind == "vision":
        # Vision models expect pixel_values: (B, C, H, W)
        image_size = getattr(config, "image_size", 224)
        num_channels = getattr(config, "num_channels", 3)
        x_torch = torch.randn(1, num_channels, image_size, image_size)
        kwargs = {"pixel_values": x_torch}
    elif input_kind == "encoder_decoder":
        # Encoder-decoder models need both input_ids and decoder_input_ids
        vocab_size = getattr(config, "vocab_size", 30522)
        x_torch = torch.randint(0, vocab_size, (1, 16))
        decoder_ids = torch.randint(0, vocab_size, (1, 8))
        kwargs = {"input_ids": x_torch, "decoder_input_ids": decoder_ids}
    else:
        # Text models expect input_ids: (B, seq_len)
        vocab_size = getattr(config, "vocab_size", 30522)
        x_torch = torch.randint(0, vocab_size, (1, 16))
        kwargs = {"input_ids": x_torch}

    return pt_model, kwargs


def _extract_output(out):
    """Extract the main tensor from an HF model output."""
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if isinstance(out, tuple):
        return out[0]
    return out


class TestHFForwardPass:
    """End-to-end: generate → convert → load → forward → compare outputs."""

    @pytest.mark.parametrize("cls_name,checkpoint,input_kind", _HF_FORWARD_MODELS)
    def test_forward_numerical_equivalence(self, cls_name, checkpoint, input_kind, _hf_cache):
        """Generated MLX model produces numerically equivalent output to PyTorch."""
        data = _get_hf_data(cls_name, checkpoint, _hf_cache)
        result = data["result"]

        ns: dict = {}
        exec(result.source, ns)
        mlx_model = ns[cls_name]()
        mlx_model.load_weights(data["weights"], strict=False)
        mlx_model.eval()

        pt_model, kwargs = _make_hf_inputs(cls_name, checkpoint, input_kind)

        # PyTorch forward
        with torch.no_grad():
            y_torch = _extract_output(pt_model(**kwargs)).numpy()

        # MLX forward — pass all inputs as keyword arguments
        mlx_kwargs = {k: mx.array(v.numpy()) for k, v in kwargs.items()}
        out_mlx = mlx_model(**mlx_kwargs)
        y_mlx = np.array(_extract_output(out_mlx))

        assert y_torch.shape == y_mlx.shape, (
            f"Shape mismatch: torch={y_torch.shape}, mlx={y_mlx.shape}"
        )
        diff = np.abs(y_torch - y_mlx).max()
        tol = 1e-3
        assert diff < tol, (
            f"{cls_name}: max diff {diff:.2e} exceeds {tol:.0e}\n"
            f"PyTorch: {y_torch.ravel()[:5]}\n"
            f"MLX:     {y_mlx.ravel()[:5]}"
        )
