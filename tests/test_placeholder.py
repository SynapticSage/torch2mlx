"""Placeholder tests to verify scaffolding works."""

import torch2mlx
from torch2mlx import registry, op_mapping, state_dict, weight_converter, analyzer


def test_version():
    assert torch2mlx.__version__ == "0.1.0"


def test_registry_populated():
    assert len(registry.registered_names()) > 0
    assert registry.lookup("Linear") is not None


def test_op_registry_populated():
    assert op_mapping.lookup_op("torch.cat") is not None


def test_state_dict_roundtrip():
    """Flatten -> unflatten should be identity for nested dicts."""
    import numpy as np

    nested = {
        "encoder": {
            "layers": {
                "0": {
                    "weight": np.array([1.0, 2.0]),
                    "bias": np.array([0.1]),
                }
            }
        }
    }
    flat = state_dict.flatten(nested)
    assert "encoder.layers.0.weight" in flat
    assert "encoder.layers.0.bias" in flat

    restored = state_dict.unflatten(flat)
    assert list(restored["encoder"]["layers"]["0"].keys()) == ["weight", "bias"]


def test_portability_report_coverage():
    report = analyzer.PortabilityReport(total_layers=10, mapped_layers=8)
    assert report.coverage == 0.8


def test_portability_report_empty():
    report = analyzer.PortabilityReport()
    assert report.coverage == 0.0
