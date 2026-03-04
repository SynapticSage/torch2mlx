"""End-to-end tests with real model architectures and CLI smoke tests.

Validates the full pipeline: analyze -> convert -> load_converted on
architectures that exercise multiple registry entries simultaneously.
"""

from __future__ import annotations

import numpy as np
import pytest

from torch2mlx.converter import convert, load_converted
from torch2mlx.state_dict import save_safetensors

try:
    import torch  # noqa: F401
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# ---------------------------------------------------------------------------
# Mini ResNet (no torchvision dependency)
# Uses: Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear
# ---------------------------------------------------------------------------


def _make_mini_resnet():
    """Build a small ResNet-like model from standard torch modules."""
    return nn.Sequential(
        # stem
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # block
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        # pool + classifier
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 10),
    )


@requires_torch
def test_mini_resnet_analysis(tmp_path):
    """Analyzer reports 100% coverage on mini ResNet (all layers registered)."""
    from torch2mlx.analyzer import analyze

    model = _make_mini_resnet()
    report = analyze(model)
    assert report.coverage == 1.0, (
        f"Expected 100% coverage, got {report.coverage:.1%}. Unmapped: {report.unmapped_layers}"
    )
    assert len(report.blockers) == 0


@requires_torch
def test_mini_resnet_conversion(tmp_path):
    """Mini ResNet converts and loads back with correct structure and shapes."""
    model = _make_mini_resnet()
    out = tmp_path / "resnet.safetensors"
    convert(model, out)

    nested = load_converted(out)

    # Conv2d weight at index "0": [O,I,H,W]=[16,3,3,3] -> [O,H,W,I]=[16,3,3,3]
    assert nested["0"]["weight"].shape == (16, 3, 3, 3)
    # BatchNorm2d at index "1": weight is 1-d [C]
    assert nested["1"]["weight"].shape == (16,)
    # Conv2d at index "4": [32,16,3,3] -> [32,3,3,16]
    assert nested["4"]["weight"].shape == (32, 3, 3, 16)
    # Linear at index "9": [10,32] -> identity [10,32]
    assert nested["9"]["weight"].shape == (10, 32)


# ---------------------------------------------------------------------------
# TransformerEncoder
# Uses: Linear, MultiheadAttention, LayerNorm, Dropout
# ---------------------------------------------------------------------------


@requires_torch
def test_transformer_encoder_analysis(tmp_path):
    """TransformerEncoder analyzer coverage is high (all children registered)."""
    from torch2mlx.analyzer import analyze

    encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
    model = nn.TransformerEncoder(encoder_layer, num_layers=2)
    report = analyze(model)
    # All sub-modules (Linear, MultiheadAttention, LayerNorm, Dropout) are registered
    assert report.coverage > 0.9, (
        f"Expected >90% coverage, got {report.coverage:.1%}. Unmapped: {report.unmapped_layers}"
    )


@requires_torch
def test_transformer_encoder_conversion(tmp_path):
    """TransformerEncoder converts and preserves layer hierarchy."""
    encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
    model = nn.TransformerEncoder(encoder_layer, num_layers=2)
    out = tmp_path / "transformer.safetensors"
    convert(model, out)

    nested = load_converted(out)
    # Verify nested structure: layers.0 and layers.1 exist
    assert "layers" in nested
    assert "0" in nested["layers"]
    assert "1" in nested["layers"]
    # Each layer should have self_attn and linear sub-modules
    layer0 = nested["layers"]["0"]
    assert "self_attn" in layer0


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


def test_cli_help(capsys):
    """CLI --help exits cleanly."""
    from torch2mlx.__main__ import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_cli_convert_safetensors(tmp_path):
    """CLI converts a .safetensors input (no torch required)."""
    from torch2mlx.__main__ import main

    # Create dummy input
    input_file = tmp_path / "input.safetensors"
    save_safetensors(
        {"fc.weight": np.zeros((4, 2), dtype=np.float32)},
        input_file,
    )

    output_dir = tmp_path / "output"
    main([str(input_file), str(output_dir)])

    output_file = output_dir / "input.safetensors"
    assert output_file.exists()
    nested = load_converted(output_file)
    assert nested["fc"]["weight"].shape == (4, 2)


def test_cli_missing_file(tmp_path):
    """CLI exits with error for missing input file."""
    from torch2mlx.__main__ import main

    with pytest.raises(SystemExit) as exc_info:
        main([str(tmp_path / "nonexistent.pt"), str(tmp_path / "out")])
    assert exc_info.value.code == 1


def test_cli_unsupported_format(tmp_path):
    """CLI exits with error for unsupported file format."""
    from torch2mlx.__main__ import main

    bad_file = tmp_path / "model.onnx"
    bad_file.write_text("")

    with pytest.raises(SystemExit) as exc_info:
        main([str(bad_file), str(tmp_path / "out")])
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@requires_torch
def test_export_api(tmp_path):
    """torch2mlx.export() is a working alias for convert()."""
    from torch2mlx import export

    model = nn.Linear(8, 4)
    out = tmp_path / "export.safetensors"
    result = export(model, out)
    assert result.exists()
    nested = load_converted(result)
    assert nested["weight"].shape == (4, 8)
