"""HuggingFace model coverage tests.

Verifies that the analyzer reports 100% coverage on popular
HuggingFace architectures. Each test downloads a small checkpoint,
runs the analyzer, and asserts full coverage.

Requires: torch, transformers (skipped if absent).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
transformers = pytest.importorskip("transformers", reason="transformers not installed")

import torch2mlx  # noqa: E402


def _analyze_hf(cls_name: str, checkpoint: str) -> tuple[float, int, list[str]]:
    """Load a HuggingFace model and return (coverage, total, unmapped)."""
    try:
        cls = getattr(transformers, cls_name)
        model = cls.from_pretrained(checkpoint)
    except RuntimeError as exc:
        # Broken torchvision install can prevent model imports
        pytest.skip(f"Model import failed: {exc}")
    report = torch2mlx.analyze(model)
    return report.coverage, report.total_layers, report.unmapped_layers


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


@pytest.mark.parametrize(
    "cls_name,checkpoint",
    [(c, ck) for _, c, ck in _HF_MODELS],
    ids=[tid for tid, _, _ in _HF_MODELS],
)
def test_hf_model_full_coverage(cls_name: str, checkpoint: str) -> None:
    """Analyzer reports 100% coverage on this HuggingFace architecture."""
    coverage, total, unmapped = _analyze_hf(cls_name, checkpoint)
    assert coverage == 1.0, (
        f"{cls_name} coverage {coverage:.1%} ({total} layers), unmapped: {unmapped}"
    )
