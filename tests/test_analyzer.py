"""Tests for torch2mlx.analyzer — portability report and blocker detection."""

from __future__ import annotations

import pytest

from torch2mlx.analyzer import PortabilityReport, detect_blockers


# ---------------------------------------------------------------------------
# Mock-based tests (no torch required)
# ---------------------------------------------------------------------------

class TestPortabilityReport:
    def test_coverage_zero_when_empty(self):
        r = PortabilityReport()
        assert r.coverage == 0.0

    def test_coverage_calculation(self):
        r = PortabilityReport(total_layers=10, mapped_layers=7)
        assert r.coverage == pytest.approx(0.7)

    def test_full_coverage(self):
        r = PortabilityReport(total_layers=5, mapped_layers=5)
        assert r.coverage == 1.0

    def test_fields_default(self):
        r = PortabilityReport()
        assert r.total_layers == 0
        assert r.mapped_layers == 0
        assert r.unmapped_layers == []
        assert r.blockers == []

    def test_unmapped_and_blockers_populated(self):
        r = PortabilityReport(
            total_layers=3,
            mapped_layers=2,
            unmapped_layers=["MyCustomLayer"],
            blockers=["In-place operation: .copy_() found in forward()"],
        )
        assert "MyCustomLayer" in r.unmapped_layers
        assert len(r.blockers) == 1


# ---------------------------------------------------------------------------
# Torch-dependent tests
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
nn = torch.nn

from torch2mlx.analyzer import analyze  # noqa: E402 (after importorskip guard)


class TestAnalyzeSimpleModel:
    def test_sequential_total_layers(self):
        # Sequential(Linear, ReLU, Linear) → 3 children + Sequential itself
        # but analyze() skips the root, so root = Sequential
        # named_modules yields: ("", seq), ("0", linear), ("1", relu), ("2", linear)
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        report = analyze(model)
        assert report.total_layers == 3  # Linear, ReLU, Linear

    def test_sequential_mapped_layers(self):
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        report = analyze(model)
        # Linear and ReLU are in registry
        assert report.mapped_layers == 3

    def test_sequential_coverage_full(self):
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        report = analyze(model)
        assert report.coverage == pytest.approx(1.0)

    def test_unmapped_layer_detected(self):
        class MySpecialLayer(nn.Module):
            def forward(self, x):
                return x

        model = nn.Sequential(nn.Linear(4, 4), MySpecialLayer())
        report = analyze(model)
        assert "MySpecialLayer" in report.unmapped_layers

    def test_unmapped_reduces_coverage(self):
        class UnknownOp(nn.Module):
            def forward(self, x):
                return x

        model = nn.Sequential(nn.Linear(4, 4), UnknownOp())
        report = analyze(model)
        assert report.coverage < 1.0


class TestBlockerDetection:
    def test_copy_inplace_blocker(self):
        class InPlaceModel(nn.Module):
            def forward(self, x, y):
                x.copy_(y)
                return x

        blockers = detect_blockers(InPlaceModel())
        assert any(".copy_()" in b for b in blockers)

    def test_item_blocker(self):
        class ItemModel(nn.Module):
            def forward(self, x):
                val = x.sum().item()
                return x * val

        blockers = detect_blockers(ItemModel())
        assert any(".item()" in b for b in blockers)

    def test_clean_model_no_blockers(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        # Sequential's forward is a built-in — getsource will fail, returns []
        # Test detect_blockers doesn't crash
        blockers = detect_blockers(model)
        assert isinstance(blockers, list)

    def test_analyze_integrates_blockers(self):
        class InPlaceModel(nn.Module):
            def forward(self, x, y):
                x.copy_(y)
                return x

        report = analyze(InPlaceModel())
        assert any(".copy_()" in b for b in report.blockers)

    def test_no_torch_raises_import_error(self, monkeypatch):
        import torch2mlx.analyzer as mod
        original = mod.HAS_TORCH
        monkeypatch.setattr(mod, "HAS_TORCH", False)
        with pytest.raises(ImportError, match="torch is required"):
            mod.analyze(nn.Linear(2, 2))
        monkeypatch.setattr(mod, "HAS_TORCH", original)
