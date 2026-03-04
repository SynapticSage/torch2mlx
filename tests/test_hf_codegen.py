"""HuggingFace codegen tests.

Verifies that codegen produces valid, parseable MLX module source for all
36 HuggingFace architectures. Tests validate:

1. Generated source always parses (ast.parse) — hard requirement
2. No constructor spec failures (AttributeError etc.) — codegen must not crash
3. Flat-architecture models (GPT-2 etc.) get 100% init coverage
4. Nested-architecture models record current coverage as baseline
5. All unmapped types are composite wrappers (not leaf types)

These tests guide codegen development: failing tests indicate regressions,
and coverage baselines track improvement as we add recursive init generation.

Requires: torch, transformers (skipped if absent).
"""

from __future__ import annotations

import ast

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
transformers = pytest.importorskip("transformers", reason="transformers not installed")

from torch2mlx.codegen import generate  # noqa: E402
from torch2mlx.registry import LAYER_REGISTRY  # noqa: E402


def _codegen_hf(cls_name: str, checkpoint: str) -> dict:
    """Load a HuggingFace model and run codegen."""
    try:
        cls = getattr(transformers, cls_name)
        model = cls.from_pretrained(checkpoint)
    except RuntimeError as exc:
        pytest.skip(f"Model import failed: {exc}")

    result = generate(model)
    parses = True
    try:
        ast.parse(result.source)
    except SyntaxError:
        parses = False

    return {
        "result": result,
        "parses": parses,
        "model": model,
    }


# ── Model list (same as test_hf_coverage.py) ─────────────────────────────────

# (test_id, HF class name, checkpoint)
_HF_MODELS = [
    # Encoder models
    ("bert", "BertModel", "bert-base-uncased"),
    ("roberta", "RobertaModel", "roberta-base"),
    ("distilbert", "DistilBertModel", "distilbert-base-uncased"),
    ("albert", "AlbertModel", "albert-base-v2"),
    ("deberta", "DebertaModel", "microsoft/deberta-base"),
    ("electra", "ElectraModel", "google/electra-small-discriminator"),
    ("mpnet", "MPNetModel", "microsoft/mpnet-base"),
    # Decoder / causal LMs
    ("gpt2", "GPT2Model", "gpt2"),
    ("gpt_neo", "GPTNeoModel", "EleutherAI/gpt-neo-125m"),
    ("opt", "OPTModel", "facebook/opt-125m"),
    ("bloom", "BloomModel", "bigscience/bloom-560m"),
    ("qwen2", "Qwen2Model", "Qwen/Qwen2-0.5B"),
    # Encoder-decoder
    ("t5", "T5Model", "t5-small"),
    ("bart", "BartModel", "facebook/bart-base"),
    # Vision
    ("vit", "ViTModel", "google/vit-base-patch16-224"),
    ("clip", "CLIPModel", "openai/clip-vit-base-patch32"),
    ("swin", "SwinModel", "microsoft/swin-tiny-patch4-window7-224"),
    ("convnext", "ConvNextModel", "facebook/convnext-tiny-224"),
    ("dinov2", "Dinov2Model", "facebook/dinov2-small"),
    # Speech
    ("whisper", "WhisperModel", "openai/whisper-tiny"),
    ("wav2vec2", "Wav2Vec2Model", "facebook/wav2vec2-base"),
    # More decoders / causal LMs
    ("pythia", "GPTNeoXModel", "EleutherAI/pythia-70m"),
    ("codegen", "CodeGenModel", "Salesforce/codegen-350M-mono"),
    ("falcon", "FalconModel", "tiiuae/falcon-rw-1b"),
    # More encoders
    ("longformer", "LongformerModel", "allenai/longformer-base-4096"),
    ("deberta_v3", "DebertaV2Model", "microsoft/deberta-v3-small"),
    ("funnel", "FunnelModel", "funnel-transformer/small"),
    ("camembert", "CamembertModel", "camembert-base"),
    ("data2vec_text", "Data2VecTextModel", "facebook/data2vec-text-base"),
    # More encoder-decoder
    ("pegasus", "PegasusModel", "google/pegasus-xsum"),
    # More vision
    ("resnet_hf", "ResNetModel", "microsoft/resnet-18"),
    ("beit", "BeitModel", "microsoft/beit-base-patch16-224"),
    ("segformer", "SegformerModel", "nvidia/mit-b0"),
    ("mobilenet", "MobileNetV2Model", "google/mobilenet_v2_1.0_224"),
    # More speech
    ("hubert", "HubertModel", "facebook/hubert-base-ls960"),
    # Misc
    ("xlnet", "XLNetModel", "xlnet-base-cased"),
]

# ── Models whose direct children are all registry/skip types ──────────────────
# These get 100% init coverage because named_children() yields only leaf types.
_FLAT_MODELS = {"gpt2", "gpt_neo", "bloom", "qwen2", "pythia", "codegen", "falcon", "xlnet"}


# ── 1. Source always parses as valid Python ───────────────────────────────────


@pytest.mark.parametrize(
    "cls_name,checkpoint",
    [(c, ck) for _, c, ck in _HF_MODELS],
    ids=[tid for tid, _, _ in _HF_MODELS],
)
def test_hf_codegen_parses(cls_name: str, checkpoint: str) -> None:
    """Generated source is valid Python for every HF model."""
    data = _codegen_hf(cls_name, checkpoint)
    assert data["parses"], f"{cls_name}: generated source has syntax errors"


# ── 2. No codegen crashes ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cls_name,checkpoint",
    [(c, ck) for _, c, ck in _HF_MODELS],
    ids=[tid for tid, _, _ in _HF_MODELS],
)
def test_hf_codegen_no_crash(cls_name: str, checkpoint: str) -> None:
    """generate() completes without exception for every HF model."""
    # _codegen_hf already calls generate(); if it raises, pytest reports it
    _codegen_hf(cls_name, checkpoint)


# ── 3. Flat models get 100% init coverage ────────────────────────────────────


@pytest.mark.parametrize(
    "cls_name,checkpoint",
    [(c, ck) for tid, c, ck in _HF_MODELS if tid in _FLAT_MODELS],
    ids=[tid for tid, _, _ in _HF_MODELS if tid in _FLAT_MODELS],
)
def test_hf_codegen_flat_model_full_coverage(cls_name: str, checkpoint: str) -> None:
    """Flat-architecture HF models have 100% init coverage."""
    data = _codegen_hf(cls_name, checkpoint)
    result = data["result"]
    assert result.coverage == 1.0, (
        f"{cls_name}: coverage {result.coverage:.0%}, unmapped: {result.unmapped}"
    )
    assert result.unmapped == [], f"{cls_name}: unexpected unmapped types: {result.unmapped}"


# ── 4. Unmapped types are always composites, never leaf types ────────────────


@pytest.mark.parametrize(
    "cls_name,checkpoint",
    [(c, ck) for _, c, ck in _HF_MODELS],
    ids=[tid for tid, _, _ in _HF_MODELS],
)
def test_hf_codegen_unmapped_are_composites(cls_name: str, checkpoint: str) -> None:
    """Any unmapped type in codegen output should NOT be a leaf type from LAYER_REGISTRY.

    If a type is in LAYER_REGISTRY, it should also be in CONSTRUCTOR_SPECS.
    Unmapped types should only be HF composite wrappers (BertEncoder, etc.)
    that decompose into registered children.
    """
    data = _codegen_hf(cls_name, checkpoint)
    result = data["result"]
    for utype in result.unmapped:
        assert utype not in LAYER_REGISTRY, (
            f"{cls_name}: '{utype}' is in LAYER_REGISTRY but missing from CONSTRUCTOR_SPECS"
        )


# ── 5. Generated class name matches model ───────────────────────────────────


@pytest.mark.parametrize(
    "cls_name,checkpoint",
    [(c, ck) for _, c, ck in _HF_MODELS],
    ids=[tid for tid, _, _ in _HF_MODELS],
)
def test_hf_codegen_class_name(cls_name: str, checkpoint: str) -> None:
    """Generated class name matches the HF model class."""
    data = _codegen_hf(cls_name, checkpoint)
    assert data["result"].class_name == cls_name


# ── 6. Source structure: imports + class + init + call ────────────────────────


@pytest.mark.parametrize(
    "cls_name,checkpoint",
    [(c, ck) for _, c, ck in _HF_MODELS],
    ids=[tid for tid, _, _ in _HF_MODELS],
)
def test_hf_codegen_source_structure(cls_name: str, checkpoint: str) -> None:
    """Generated source has required structural elements."""
    data = _codegen_hf(cls_name, checkpoint)
    source = data["result"].source
    assert "import mlx.core as mx" in source
    assert "import mlx.nn as nn" in source
    assert f"class {cls_name}(nn.Module):" in source
    assert "def __init__(self)" in source
    assert "super().__init__()" in source
    # Either a real __call__ or a TODO stub
    assert "def __call__" in source


# ── 7. fx tracing: all HF models fail (dynamic control flow) ────────────────


@pytest.mark.parametrize(
    "cls_name,checkpoint",
    [(c, ck) for _, c, ck in _HF_MODELS],
    ids=[tid for tid, _, _ in _HF_MODELS],
)
def test_hf_codegen_fx_trace_fails(cls_name: str, checkpoint: str) -> None:
    """HF models use dynamic control flow — fx tracing should fail gracefully."""
    data = _codegen_hf(cls_name, checkpoint)
    result = data["result"]
    # All current HF models fail fx tracing
    assert result.traced is False, f"{cls_name}: fx tracing unexpectedly succeeded"
    # Fallback TODO stub should be present
    assert "NotImplementedError" in result.source


# ── 8. Coverage baselines — track improvement over time ──────────────────────
# These record the current init coverage for nested models.
# As we add recursive init generation, update these expected values upward.

_COVERAGE_BASELINES: dict[str, float] = {
    # Flat models — already 100%
    "gpt2": 1.0,
    "gpt_neo": 1.0,
    "bloom": 1.0,
    "qwen2": 1.0,
    "pythia": 1.0,
    "codegen": 1.0,
    "falcon": 1.0,
    "xlnet": 1.0,
    # Nested models — current baselines (direct children only)
    "bert": 0.0,
    "roberta": 0.0,
    "distilbert": 0.0,
    "deberta": 0.0,
    "mpnet": 0.0,
    "opt": 0.0,
    "whisper": 0.0,
    "wav2vec2": 0.0,
    "longformer": 0.0,
    "deberta_v3": 0.0,
    "funnel": 0.0,
    "camembert": 0.0,
    "data2vec_text": 0.0,
    "segformer": 0.0,
    "hubert": 0.0,
    "albert": 0.5,
    "electra": 1 / 3,
    "t5": 1 / 3,
    "bart": 1 / 3,
    "vit": 0.25,
    "clip": 0.5,
    "swin": 0.5,
    "convnext": 1 / 3,
    "dinov2": 1 / 3,
    "pegasus": 1 / 3,
    "resnet_hf": 1 / 3,
    "beit": 0.25,
    "mobilenet": 0.5,
}


@pytest.mark.parametrize(
    "tid,cls_name,checkpoint",
    [(tid, c, ck) for tid, c, ck in _HF_MODELS],
    ids=[tid for tid, _, _ in _HF_MODELS],
)
def test_hf_codegen_coverage_baseline(tid: str, cls_name: str, checkpoint: str) -> None:
    """Coverage meets or exceeds recorded baseline.

    If codegen improves (e.g., recursive init), update the baseline upward.
    This test catches regressions — coverage should never decrease.
    """
    data = _codegen_hf(cls_name, checkpoint)
    result = data["result"]
    expected = _COVERAGE_BASELINES[tid]
    assert result.coverage >= expected - 1e-9, (
        f"{cls_name}: coverage {result.coverage:.0%} < baseline {expected:.0%}"
    )


# ── 9. Flat models: init contains expected constructor calls ─────────────────


def test_gpt2_init_contains_embeddings():
    """GPT-2 codegen should emit wte and wpe embeddings with correct dims."""
    data = _codegen_hf("GPT2Model", "gpt2")
    source = data["result"].source
    assert "nn.Embedding(50257, 768)" in source  # wte
    assert "nn.Embedding(1024, 768)" in source  # wpe


def test_gpt2_init_contains_layernorm():
    """GPT-2 codegen should emit ln_f (final LayerNorm)."""
    data = _codegen_hf("GPT2Model", "gpt2")
    source = data["result"].source
    # normalized_shape is (768,) — a 1-tuple
    assert "nn.LayerNorm((768,))" in source


def test_bloom_init_contains_embeddings():
    """BLOOM codegen should emit word_embeddings and word_embeddings_layernorm."""
    data = _codegen_hf("BloomModel", "bigscience/bloom-560m")
    source = data["result"].source
    assert "nn.Embedding(" in source
    assert "nn.LayerNorm(" in source


def test_qwen2_init_contains_rmsnorm():
    """Qwen2 codegen should emit nn.RMSNorm for its norm layer."""
    data = _codegen_hf("Qwen2Model", "Qwen/Qwen2-0.5B")
    source = data["result"].source
    assert "nn.RMSNorm(" in source


def test_falcon_init_contains_embeddings():
    """Falcon codegen should emit word_embeddings."""
    data = _codegen_hf("FalconModel", "tiiuae/falcon-rw-1b")
    source = data["result"].source
    assert "nn.Embedding(" in source
