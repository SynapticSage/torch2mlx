"""Tests for codegen.py — MLX module source generation."""

from __future__ import annotations

import ast

import pytest

from torch2mlx.codegen import (
    CONSTRUCTOR_SPECS,
    ArgSpec,
    ConstructorSpec,
    GeneratedCode,
    _apply_transform,
    _format_value,
)
from torch2mlx.registry import LAYER_REGISTRY

# ── Always-run tests (no torch required) ─────────────────────────────────────


class TestFormatValue:
    def test_int(self):
        assert _format_value(42) == "42"

    def test_float(self):
        assert _format_value(3.14) == "3.14"

    def test_bool_true(self):
        assert _format_value(True) == "True"

    def test_bool_false(self):
        assert _format_value(False) == "False"

    def test_string(self):
        assert _format_value("hello") == "'hello'"

    def test_tuple(self):
        assert _format_value((3, 3)) == "(3, 3)"

    def test_tuple_single(self):
        assert _format_value((3,)) == "(3,)"

    def test_list(self):
        assert _format_value([1, 2]) == "[1, 2]"

    def test_none(self):
        assert _format_value(None) == "None"

    def test_nested_tuple(self):
        assert _format_value((1, (2, 3))) == "(1, (2, 3))"


class TestApplyTransform:
    def test_identity(self):
        assert _apply_transform(42, "identity") == 42

    def test_bias_check_not_none(self):
        """bias_check returns True when value is not None."""
        import numpy as np

        arr = np.zeros(5)
        assert _apply_transform(arr, "bias_check") is True

    def test_bias_check_none(self):
        assert _apply_transform(None, "bias_check") is False

    def test_tuple_to_scalar_uniform(self):
        assert _apply_transform((3, 3), "tuple_to_scalar") == 3

    def test_tuple_to_scalar_nonuniform(self):
        assert _apply_transform((3, 5), "tuple_to_scalar") == (3, 5)

    def test_tuple_to_scalar_single(self):
        assert _apply_transform((7,), "tuple_to_scalar") == 7

    def test_last_element(self):
        assert _apply_transform((10, 20), "last_element") == 20

    def test_last_element_scalar(self):
        assert _apply_transform(5, "last_element") == 5

    def test_unknown_transform(self):
        with pytest.raises(ValueError, match="Unknown transform"):
            _apply_transform(1, "bogus")


class TestDataclasses:
    def test_generated_code_defaults(self):
        gc = GeneratedCode(source="x", class_name="Foo", coverage=1.0)
        assert gc.todos == []
        assert gc.unmapped == []
        assert gc.traced is False

    def test_argspec_frozen(self):
        a = ArgSpec("x")
        with pytest.raises(AttributeError):
            a.attr = "y"  # type: ignore[misc]

    def test_constructor_spec_frozen(self):
        cs = ConstructorSpec("nn.Linear", ())
        with pytest.raises(AttributeError):
            cs.mlx_call = "nn.ReLU"  # type: ignore[misc]


class TestSpecRegistryConsistency:
    """Every LAYER_REGISTRY entry must have a CONSTRUCTOR_SPECS entry."""

    def test_all_layer_registry_entries_covered(self):
        missing = []
        for name in LAYER_REGISTRY:
            if name not in CONSTRUCTOR_SPECS:
                missing.append(name)
        assert missing == [], f"CONSTRUCTOR_SPECS missing entries for: {missing}"


# ── Torch-dependent tests ────────────────────────────────────────────────────

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional


class TestGenerateLinear:
    def test_linear_source(self):
        from torch2mlx.codegen import generate

        result = generate(nn.Linear(10, 20))
        assert "nn.Linear(10, 20)" in result.source
        assert result.class_name == "Linear"

    def test_linear_bias_false(self):
        from torch2mlx.codegen import generate

        result = generate(nn.Linear(10, 20, bias=False))
        assert "nn.Linear(10, 20, False)" in result.source

    def test_linear_parses(self):
        from torch2mlx.codegen import generate

        result = generate(nn.Linear(10, 20))
        ast.parse(result.source)


class TestGenerateConv:
    def test_conv2d(self):
        from torch2mlx.codegen import generate

        result = generate(nn.Conv2d(3, 16, 3))
        assert "nn.Conv2d(3, 16, 3)" in result.source
        ast.parse(result.source)

    def test_conv1d(self):
        from torch2mlx.codegen import generate

        result = generate(nn.Conv1d(1, 8, 5, stride=2))
        assert "nn.Conv1d(1, 8, 5, 2)" in result.source


class TestGenerateSequential:
    def test_sequential_traced(self):
        from torch2mlx.codegen import generate

        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        result = generate(model)
        assert "nn.Linear(784, 256)" in result.source
        assert "nn.Linear(256, 10)" in result.source
        assert result.traced is True
        ast.parse(result.source)


class TestGenerateCustomModel:
    def test_relu_translated(self):
        from torch2mlx.codegen import generate

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 5)

            def forward(self, x):
                return self.fc2(F.relu(self.fc1(x)))

        result = generate(TinyModel())
        assert result.traced is True
        assert "nn.relu" in result.source
        assert "nn.Linear(10, 20)" in result.source
        ast.parse(result.source)

    def test_residual_add(self):
        from torch2mlx.codegen import generate

        class ResModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                return x + self.fc(x)

        result = generate(ResModel())
        assert result.traced is True
        assert "mx.add" in result.source
        ast.parse(result.source)


class TestGenerateFallback:
    def test_dynamic_control_flow(self):
        from torch2mlx.codegen import generate

        class DynamicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 10)
                self.fc2 = nn.Linear(10, 10)

            def forward(self, x):
                if x.sum() > 0:
                    return self.fc1(x)
                return self.fc2(x)

        result = generate(DynamicModel())
        assert result.traced is False
        assert "TODO" in result.source
        assert "NotImplementedError" in result.source
        ast.parse(result.source)


class TestGenerateToFile:
    def test_write_file(self, tmp_path):
        from torch2mlx.codegen import generate_to_file

        path = tmp_path / "model.py"
        result_path = generate_to_file(nn.Linear(5, 3), path)
        assert result_path == path
        assert path.exists()
        source = path.read_text()
        assert "nn.Linear(5, 3)" in source
        ast.parse(source)


class TestCoverage:
    def test_full_coverage(self):
        from torch2mlx.codegen import generate

        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        result = generate(model)
        assert result.coverage == 1.0
        assert result.unmapped == []

    def test_unmapped_type(self):
        from torch2mlx.codegen import generate

        class WeirdLayer(nn.Module):
            def forward(self, x):
                return x

        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
                self.weird = WeirdLayer()

            def forward(self, x):
                return self.weird(self.fc(x))

        result = generate(MyModel())
        assert result.coverage < 1.0
        assert "WeirdLayer" in result.unmapped
        assert any("WeirdLayer" in t for t in result.todos)


class TestFxMethodTranslation:
    def test_view_becomes_reshape(self):
        from torch2mlx.codegen import generate

        class ViewModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 20)

            def forward(self, x):
                out = self.fc(x)
                return out.view(-1)

        result = generate(ViewModule())
        if result.traced:
            assert "mx.reshape" in result.source

    def test_contiguous_becomes_noop(self):
        from torch2mlx.codegen import generate

        class ContiguousModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                return self.fc(x).contiguous()

        result = generate(ContiguousModule())
        if result.traced:
            # contiguous should not appear as a function call in the output
            assert ".contiguous()" not in result.source


class TestFxOperators:
    def test_multiplication(self):
        from torch2mlx.codegen import generate

        class MulModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                return self.fc(x) * 2.0

        result = generate(MulModel())
        if result.traced:
            assert "mx.multiply" in result.source

    def test_subtraction(self):
        from torch2mlx.codegen import generate

        class SubModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                return self.fc(x) - x

        result = generate(SubModel())
        if result.traced:
            assert "mx.subtract" in result.source


class TestCustomClassName:
    def test_custom_name(self):
        from torch2mlx.codegen import generate

        result = generate(nn.Linear(5, 3), class_name="MyLinear")
        assert "class MyLinear(nn.Module):" in result.source
        assert result.class_name == "MyLinear"


class TestBatchNorm:
    def test_batchnorm2d(self):
        from torch2mlx.codegen import generate

        result = generate(nn.BatchNorm2d(64))
        assert "nn.BatchNorm(64)" in result.source
        ast.parse(result.source)


class TestEmbedding:
    def test_embedding(self):
        from torch2mlx.codegen import generate

        result = generate(nn.Embedding(1000, 128))
        assert "nn.Embedding(1000, 128)" in result.source
        ast.parse(result.source)
