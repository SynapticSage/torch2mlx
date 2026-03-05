"""Tests for codegen.py — MLX module source generation."""

from __future__ import annotations

import ast

import pytest

from torch2mlx.codegen import (
    CONSTRUCTOR_SPECS,
    ArgSpec,
    Confidence,
    ConstructorSpec,
    GeneratedCode,
    _apply_transform,
    _format_value,
    _rewrite_forward_ast,
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
    def test_dynamic_control_flow_ast_rewrite(self):
        """Dynamic control flow: fx fails, AST rewrite succeeds."""
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
        assert result.ast_rewritten is True
        # x.sum() → mx.sum(x), control flow preserved
        assert "mx.sum(x)" in result.source
        assert "if" in result.source
        assert "NotImplementedError" not in result.source
        ast.parse(result.source)

    def test_todo_fallback_when_both_fail(self):
        """TODO stub when both fx and AST rewrite fail."""
        from unittest.mock import patch

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

        # Simulate AST rewrite failure (e.g., C extension, generated code)
        with patch("torch2mlx.codegen._rewrite_forward_ast", return_value=None):
            result = generate(DynamicModel())
        assert result.traced is False
        assert result.ast_rewritten is False
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


# ── Recursive codegen tests ──────────────────────────────────────────────────


class TestNestedModelRecurses:
    """Composite wrapper with leaf children → recurse and emit helper class."""

    def test_helper_class_emitted(self):
        from torch2mlx.codegen import generate

        class InnerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)
                self.act = nn.ReLU()

            def forward(self, x):
                return self.act(self.fc(x))

        class OuterModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = InnerBlock()
                self.head = nn.Linear(10, 2)

            def forward(self, x):
                return self.head(self.block(x))

        result = generate(OuterModel())
        assert "class InnerBlock(nn.Module):" in result.source
        assert "class OuterModel(nn.Module):" in result.source
        assert "self.block = InnerBlock()" in result.source
        assert "nn.Linear(10, 10)" in result.source
        assert "nn.Linear(10, 2)" in result.source
        assert result.coverage == 1.0
        ast.parse(result.source)

    def test_unmapped_still_reported(self):
        """Composite with an unmapped leaf child still reports it."""
        from torch2mlx.codegen import generate

        class Mystery(nn.Module):
            def forward(self, x):
                return x

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = Mystery()
                self.fc = nn.Linear(5, 5)

            def forward(self, x):
                return self.fc(self.m(x))

        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = Wrapper()

            def forward(self, x):
                return self.w(x)

        result = generate(Root())
        assert result.coverage < 1.0
        assert "Mystery" in result.unmapped


class TestModuleListUniform:
    """ModuleList of identical leaf items → list comprehension."""

    def test_uniform_leaf_list(self):
        from torch2mlx.codegen import generate

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        result = generate(Model())
        assert "nn.Linear(8, 8) for _ in range(4)" in result.source
        assert result.coverage == 1.0
        ast.parse(result.source)


class TestModuleListComposite:
    """ModuleList of composite blocks → helper class + list comprehension."""

    def test_composite_list(self):
        from torch2mlx.codegen import generate

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(16, 16)
                self.norm = nn.LayerNorm(16)

            def forward(self, x):
                return self.norm(self.fc(x))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Block() for _ in range(3)])

            def forward(self, x):
                for b in self.blocks:
                    x = b(x)
                return x

        result = generate(Model())
        assert "class Block(nn.Module):" in result.source
        assert "Block() for _ in range(3)" in result.source
        assert "nn.Linear(16, 16)" in result.source
        assert "nn.LayerNorm((16,))" in result.source
        assert result.coverage == 1.0
        ast.parse(result.source)


class TestSequentialMixed:
    """Sequential with mixed leaf types → individual items in list."""

    def test_mixed_sequential(self):
        from torch2mlx.codegen import generate

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU(),
                    nn.Linear(20, 5),
                )

            def forward(self, x):
                return self.net(x)

        result = generate(Model())
        assert "nn.Linear(10, 20)" in result.source
        assert "nn.ReLU()" in result.source
        assert "nn.Linear(20, 5)" in result.source
        assert result.coverage == 1.0
        ast.parse(result.source)


class TestDeduplication:
    """Same composite type referenced twice → only one helper class."""

    def test_dedup(self):
        from torch2mlx.codegen import generate

        class SubBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def forward(self, x):
                return self.fc(x)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = SubBlock()
                self.b = SubBlock()

            def forward(self, x):
                return self.a(x) + self.b(x)

        result = generate(Model())
        # Should appear exactly once as a class definition
        assert result.source.count("class SubBlock(nn.Module):") == 1
        assert "self.a = SubBlock()" in result.source
        assert "self.b = SubBlock()" in result.source
        assert result.coverage == 1.0
        ast.parse(result.source)


class TestDeepNesting:
    """3 levels deep → helper classes emitted in topological order."""

    def test_topological_order(self):
        from torch2mlx.codegen import generate

        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 3)

            def forward(self, x):
                return self.fc(x)

        class Middle(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()
                self.norm = nn.LayerNorm(3)

            def forward(self, x):
                return self.norm(self.inner(x))

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mid = Middle()

            def forward(self, x):
                return self.mid(x)

        result = generate(Outer())
        source = result.source
        # Inner must appear before Middle (dependency order)
        inner_pos = source.index("class Inner(nn.Module):")
        middle_pos = source.index("class Middle(nn.Module):")
        outer_pos = source.index("class Outer(nn.Module):")
        assert inner_pos < middle_pos < outer_pos
        assert result.coverage == 1.0
        ast.parse(source)


class TestCoverageCountsLeaves:
    """Coverage = mapped_leaves / total_leaves across entire tree."""

    def test_nested_all_mapped(self):
        from torch2mlx.codegen import generate

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 10)
                self.fc2 = nn.Linear(10, 10)

            def forward(self, x):
                return self.fc2(self.fc1(x))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = Block()
                self.head = nn.Linear(10, 1)

            def forward(self, x):
                return self.head(self.block(x))

        result = generate(Model())
        # 3 leaves total (fc1, fc2, head), all mapped
        assert result.coverage == 1.0

    def test_nested_partial_mapped(self):
        from torch2mlx.codegen import generate

        class Weird(nn.Module):
            def forward(self, x):
                return x

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(5, 5)
                self.w = Weird()

            def forward(self, x):
                return self.w(self.fc(x))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = Block()

            def forward(self, x):
                return self.block(x)

        result = generate(Model())
        # 2 leaves (fc is mapped, Weird is unmapped) → 50%
        assert result.coverage == pytest.approx(0.5)


# ── Edge case tests (skeptic review) ─────────────────────────────────────────


class TestEmptyContainer:
    """Empty ModuleList should contribute 0 leaves, not 1."""

    def test_empty_modulelist_coverage(self):
        from torch2mlx.codegen import generate

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([])
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        result = generate(Model())
        # Only fc is a real leaf; empty ModuleList has 0 leaves
        assert result.coverage == 1.0
        assert result.unmapped == []
        ast.parse(result.source)


class TestModuleDict:
    """ModuleDict with string keys → valid Python output."""

    def test_uniform_dict(self):
        from torch2mlx.codegen import generate

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = nn.ModuleDict(
                    {
                        "cls": nn.Linear(10, 2),
                        "reg": nn.Linear(10, 4),
                    }
                )

            def forward(self, x):
                return self.heads["cls"](x)

        result = generate(Model())
        assert result.coverage == 1.0
        assert "nn.Linear(" in result.source
        ast.parse(result.source)

    def test_mixed_dict(self):
        from torch2mlx.codegen import generate

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.parts = nn.ModuleDict(
                    {
                        "encoder": nn.Linear(10, 20),
                        "act": nn.ReLU(),
                    }
                )

            def forward(self, x):
                return self.parts["act"](self.parts["encoder"](x))

        result = generate(Model())
        assert result.coverage == 1.0
        ast.parse(result.source)


class TestContainerInContainer:
    """ModuleList of Sequential — container-in-container recursion."""

    def test_nested_containers(self):
        from torch2mlx.codegen import generate

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.towers = nn.ModuleList(
                    [
                        nn.Sequential(nn.Linear(8, 8), nn.ReLU()),
                        nn.Sequential(nn.Linear(8, 8), nn.ReLU()),
                    ]
                )

            def forward(self, x):
                return sum(t(x) for t in self.towers)

        result = generate(Model())
        # 2 Sequentials × (Linear + ReLU) = 4 leaves, all mapped
        assert result.coverage == 1.0
        ast.parse(result.source)


class TestMixedContainerStatelessOnly:
    """Sequential where all children are stateless skips."""

    def test_all_stateless_children(self):
        from torch2mlx.codegen import generate

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.skips = nn.Sequential(nn.Identity(), nn.Dropout(0.1))
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(self.skips(x))

        result = generate(Model())
        assert result.coverage == 1.0
        ast.parse(result.source)


class TestMakeTodoCallHelper:
    """Direct test of _make_todo_call_helper output content."""

    def test_content(self):
        from torch2mlx.codegen import _make_todo_call_helper

        stub = _make_todo_call_helper("MyLayer", "forward(self, hidden, mask)")
        assert "Translate MyLayer.forward(self, hidden, mask)" in stub
        assert "MyLayer.forward() requires manual translation" in stub
        assert "raise NotImplementedError" in stub
        assert "def __call__" in stub


# ── AST rewriter tests ───────────────────────────────────────────────────────


class TestASTRewriter:
    """Unit tests for individual AST transformations."""

    def test_torch_cat_to_mx_concatenate(self):
        """torch.cat([a,b], dim=1) → mx.concatenate([a,b], axis=1)."""
        from torch2mlx.codegen import generate

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(20, 10)

            def forward(self, x, y):
                return self.fc(torch.cat([x, y], dim=1))

        result = generate(Model())
        assert "mx.concatenate" in result.source
        assert "axis=1" in result.source

    def test_method_view_to_reshape(self):
        """x.view(batch, -1) → mx.reshape(x, (batch, -1))."""
        from torch2mlx.codegen import generate

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                out = self.fc(x)
                return out.view(-1, 10)

        result = generate(Model())
        # Either fx or AST should handle this
        assert "mx.reshape" in result.source

    def test_contiguous_removed(self):
        """x.contiguous() → x (no-op removal)."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                return self.fc(x).contiguous()

        result = _rewrite_forward_ast(Model())
        assert result is not None
        assert ".contiguous()" not in result.source

    def test_dim_renamed_to_axis(self):
        """F.softmax(x, dim=-1) → mx.softmax(x, axis=-1)."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return F.softmax(x, dim=-1)

        result = _rewrite_forward_ast(Model())
        assert result is not None
        assert "mx.softmax" in result.source
        assert "axis=" in result.source
        assert "dim=" not in result.source

    def test_dtype_mapping(self):
        """torch.float32 → mx.float32 in non-no-op contexts."""

        class Model(nn.Module):
            def forward(self, x):
                return torch.zeros(10, dtype=torch.float32)

        result = _rewrite_forward_ast(Model())
        assert result is not None
        assert "mx.float32" in result.source
        assert "mx.zeros" in result.source

    def test_forward_renamed_to_call(self):
        """forward → __call__."""

        class Model(nn.Module):
            def forward(self, x):
                return x

        result = _rewrite_forward_ast(Model())
        assert result is not None
        assert "def __call__" in result.source
        assert "def forward" not in result.source

    def test_type_annotations_converted(self):
        """torch.Tensor → mx.array in annotations."""

        class Model(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        result = _rewrite_forward_ast(Model())
        assert result is not None
        assert "mx.array" in result.source

    def test_noop_to_removal(self):
        """.to(device) is a no-op in MLX unified memory."""

        class Model(nn.Module):
            def forward(self, x):
                return x.to("cuda").contiguous()

        result = _rewrite_forward_ast(Model())
        assert result is not None
        assert ".to(" not in result.source
        assert ".contiguous()" not in result.source

    def test_unmapped_call_preserved(self):
        """Unmapped torch calls are preserved with annotation."""

        class Model(nn.Module):
            def forward(self, x):
                return torch.unique(x)

        result = _rewrite_forward_ast(Model())
        assert result is not None
        assert "unique" in result.source
        assert "torch.unique" in result.unmapped_calls

    def test_size_to_shape(self):
        """x.size() → x.shape, x.size(0) → x.shape[0]."""

        class Model(nn.Module):
            def forward(self, x):
                b = x.size(0)
                s = x.size()
                return b, s

        result = _rewrite_forward_ast(Model())
        assert result is not None
        assert "x.shape[0]" in result.source
        assert "x.shape" in result.source
        assert ".size(" not in result.source

    def test_float_cast(self):
        """x.float() → x.astype(mx.float32)."""

        class Model(nn.Module):
            def forward(self, x):
                return x.float()

        result = _rewrite_forward_ast(Model())
        assert result is not None
        assert "astype" in result.source
        assert "mx.float32" in result.source

    def test_dim_method(self):
        """x.dim() → len(x.shape)."""

        class Model(nn.Module):
            def forward(self, x):
                return x.dim()

        result = _rewrite_forward_ast(Model())
        assert result is not None
        assert "len(x.shape)" in result.source

    def test_f_dropout_removed(self):
        """F.dropout(x, ...) → x (no-op at eval)."""

        class Model(nn.Module):
            def forward(self, x):
                return F.dropout(x, p=0.1, training=self.training)

        result = _rewrite_forward_ast(Model())
        assert result is not None
        assert "dropout" not in result.source

    def test_torch_arange(self):
        """torch.arange(n) → mx.arange(n)."""

        class Model(nn.Module):
            def forward(self, x):
                return torch.arange(x.size(0))

        result = _rewrite_forward_ast(Model())
        assert result is not None
        assert "mx.arange" in result.source

    def test_torch_zeros(self):
        """torch.zeros(shape) → mx.zeros(shape)."""

        class Model(nn.Module):
            def forward(self, x):
                return torch.zeros(10, 20)

        result = _rewrite_forward_ast(Model())
        assert result is not None
        assert "mx.zeros" in result.source

    def test_super_forward_renamed(self):
        """super().forward(x) → super().__call__(x)."""

        class Base(nn.Module):
            def forward(self, x):
                return x

        class Model(Base):
            def forward(self, x):
                return super().forward(x) + 1

        result = _rewrite_forward_ast(Model())
        assert result is not None
        assert "super().__call__" in result.source
        assert "super().forward" not in result.source

    def test_self_submodule_calls_preserved(self):
        """self.fc(x) stays as self.fc(x)."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                return self.fc(x)

        result = _rewrite_forward_ast(Model())
        assert result is not None
        assert "self.fc(x)" in result.source

    def test_unsqueeze_to_expand_dims(self):
        """x.unsqueeze(0) → mx.expand_dims(x, axis=0)."""

        class Model(nn.Module):
            def forward(self, x):
                return x.unsqueeze(0)

        result = _rewrite_forward_ast(Model())
        assert result is not None
        assert "mx.expand_dims" in result.source

    def test_permute_to_transpose(self):
        """x.permute(0, 2, 1) → mx.transpose(x, (0, 2, 1))."""

        class Model(nn.Module):
            def forward(self, x):
                return x.permute(0, 2, 1)

        result = _rewrite_forward_ast(Model())
        assert result is not None
        assert "mx.transpose" in result.source

    def test_chained_noop(self):
        """x.contiguous().view(-1) → mx.reshape(x, (-1,))."""

        class Model(nn.Module):
            def forward(self, x):
                return x.contiguous().view(-1)

        result = _rewrite_forward_ast(Model())
        assert result is not None
        assert "mx.reshape" in result.source
        assert ".contiguous()" not in result.source


class TestASTCascade:
    """Tests for the fx → AST → TODO cascade."""

    def test_fx_preferred_over_ast(self):
        """Simple traceable model uses fx, not AST."""
        from torch2mlx.codegen import generate

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        result = generate(SimpleModel())
        assert result.traced is True
        assert result.ast_rewritten is False

    def test_ast_used_when_fx_fails(self):
        """Dynamic model falls through to AST rewrite."""
        from torch2mlx.codegen import generate

        class DynamicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                if x.sum() > 0:
                    return self.fc(x)
                return x

        result = generate(DynamicModel())
        assert result.traced is False
        assert result.ast_rewritten is True

    def test_helper_classes_get_ast_call(self):
        """Helper classes get AST-rewritten __call__ instead of TODO stub."""
        from torch2mlx.codegen import generate

        class InnerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)
                self.act = nn.ReLU()

            def forward(self, x):
                return self.act(self.fc(x))

        class OuterModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = InnerBlock()
                self.head = nn.Linear(10, 2)

            def forward(self, x):
                return self.head(self.block(x))

        result = generate(OuterModel())
        # Helper class should have __call__ from AST, not TODO
        assert "class InnerBlock(nn.Module):" in result.source
        assert "NotImplementedError" not in result.source
        # Should have AST rewrite header
        assert "AST rewrite" in result.source
        ast.parse(result.source)


class TestConfidenceAnnotations:
    """Tests for confidence level tracking."""

    def test_mechanical_ops(self):
        """All-mapped operations yield MECHANICAL confidence."""

        class Model(nn.Module):
            def forward(self, x):
                return F.softmax(x, dim=-1)

        result = _rewrite_forward_ast(Model())
        assert result is not None
        assert result.confidence == Confidence.MECHANICAL

    def test_unmapped_yields_needs_review(self):
        """Unmapped torch calls lower confidence to NEEDS_REVIEW."""

        class Model(nn.Module):
            def forward(self, x):
                return torch.unique(x)

        result = _rewrite_forward_ast(Model())
        assert result is not None
        assert result.confidence == Confidence.NEEDS_REVIEW
        assert len(result.unmapped_calls) > 0

    def test_generated_code_confidence_field(self):
        """GeneratedCode has call_confidence field."""
        from torch2mlx.codegen import generate

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                if x.sum() > 0:
                    return self.fc(x)
                return x

        result = generate(Model())
        assert result.call_confidence in ("mechanical", "needs_review", "todo")
