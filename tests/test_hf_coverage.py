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
    ("bert", "BertModel", "bert-base-uncased"),
    ("gpt2", "GPT2Model", "gpt2"),
    ("vit", "ViTModel", "google/vit-base-patch16-224"),
    ("t5", "T5Model", "t5-small"),
    ("roberta", "RobertaModel", "roberta-base"),
    ("distilbert", "DistilBertModel", "distilbert-base-uncased"),
    ("albert", "AlbertModel", "albert-base-v2"),
    ("deberta", "DebertaModel", "microsoft/deberta-base"),
    ("clip", "CLIPModel", "openai/clip-vit-base-patch32"),
    ("whisper", "WhisperModel", "openai/whisper-tiny"),
    ("gpt_neo", "GPTNeoModel", "EleutherAI/gpt-neo-125m"),
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
        f"{cls_name} coverage {coverage:.1%} ({total} layers), "
        f"unmapped: {unmapped}"
    )
