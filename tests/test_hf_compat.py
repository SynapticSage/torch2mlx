"""Unit tests for hf_compat.py — HF-specific post-processing injectors."""

from __future__ import annotations

from torch2mlx.hf_compat import (
    _inject_hf_global_stubs,
    _inject_hf_method_stubs,
    _inject_hf_output_stubs,
    _inject_hf_private_attrs,
    _inject_logger_stub,
    _inject_nchw_to_nhwc,
    hf_post_process,
)


# ── _inject_hf_private_attrs ─────────────────────────────────────────────────


class TestInjectHFPrivateAttrs:
    def test_injects_defaults_after_super_init(self):
        source = (
            "class Foo(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.fc = nn.Linear(10, 5)\n"
            "\n"
            "    def __call__(self, x):\n"
            "        if self._use_sdpa:\n"
            "            pass\n"
        )
        result = _inject_hf_private_attrs(source)
        assert "self._use_sdpa = False" in result
        # Injected after super().__init__(), before self.fc
        lines = result.split("\n")
        super_idx = next(i for i, ln in enumerate(lines) if "super().__init__()" in ln)
        sdpa_idx = next(i for i, ln in enumerate(lines) if "_use_sdpa = False" in ln)
        fc_idx = next(i for i, ln in enumerate(lines) if "self.fc" in ln)
        assert super_idx < sdpa_idx < fc_idx

    def test_noop_on_clean_source(self):
        source = (
            "class Foo(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.fc = nn.Linear(10, 5)\n"
        )
        assert _inject_hf_private_attrs(source) == source

    def test_gradient_checkpointing_detected(self):
        source = (
            "class Foo(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "    def __call__(self, x):\n"
            "        if self.gradient_checkpointing:\n"
            "            pass\n"
        )
        result = _inject_hf_private_attrs(source)
        assert "self.gradient_checkpointing = False" in result


# ── _inject_hf_method_stubs ──────────────────────────────────────────────────


class TestInjectHFMethodStubs:
    def test_injects_get_head_mask(self):
        source = (
            "class Foo(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "\n"
            "    def __call__(self, x):\n"
            "        mask = self.get_head_mask(None, 12)\n"
            "        return x\n"
        )
        result = _inject_hf_method_stubs(source)
        assert "def get_head_mask" in result

    def test_injects_extended_attention_mask(self):
        source = (
            "class Foo(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "\n"
            "    def __call__(self, x, mask):\n"
            "        ext = self.get_extended_attention_mask(mask, x.shape)\n"
            "        return x\n"
        )
        result = _inject_hf_method_stubs(source)
        assert "def get_extended_attention_mask" in result

    def test_noop_without_calls(self):
        source = "class Foo(nn.Module):\n    def __call__(self, x):\n        return x\n"
        assert _inject_hf_method_stubs(source) == source


# ── _inject_hf_global_stubs ──────────────────────────────────────────────────


class TestInjectHFGlobalStubs:
    def test_injects_before_first_class(self):
        source = (
            "import mlx.core as mx\n"
            "\n"
            "class Foo(nn.Module):\n"
            "    def __call__(self, x):\n"
            "        return apply_chunking_to_forward(self.ff, 0, 0, x)\n"
        )
        result = _inject_hf_global_stubs(source)
        assert "def apply_chunking_to_forward" in result
        # Stub should appear before the class
        stub_idx = result.index("def apply_chunking_to_forward")
        class_idx = result.index("class Foo")
        assert stub_idx < class_idx

    def test_eager_attention_before_all_attention_functions(self):
        source = (
            "import mlx.core as mx\n"
            "\n"
            "class Foo(nn.Module):\n"
            "    def __call__(self, x):\n"
            "        fn = ALL_ATTENTION_FUNCTIONS['eager']\n"
            "        return eager_attention_forward(self, x, x, x)\n"
        )
        result = _inject_hf_global_stubs(source)
        eager_idx = result.index("def eager_attention_forward")
        all_idx = result.index("ALL_ATTENTION_FUNCTIONS = ")
        assert eager_idx < all_idx

    def test_noop_without_references(self):
        source = "class Foo(nn.Module):\n    pass\n"
        assert _inject_hf_global_stubs(source) == source


# ── _inject_hf_output_stubs ──────────────────────────────────────────────────


class TestInjectHFOutputStubs:
    def test_creates_aliases(self):
        source = (
            "import mlx.core as mx\n"
            "\n"
            "class Foo(nn.Module):\n"
            "    def __call__(self, x):\n"
            "        return BaseModelOutput(last_hidden_state=x)\n"
        )
        result = _inject_hf_output_stubs(source)
        assert "class _HFOutput:" in result
        assert "BaseModelOutput = _HFOutput" in result

    def test_noop_without_output_calls(self):
        source = "class Foo(nn.Module):\n    pass\n"
        assert _inject_hf_output_stubs(source) == source


# ── _inject_nchw_to_nhwc ─────────────────────────────────────────────────────


class TestInjectNCHWtoNHWC:
    def test_rewrites_shape_unpack(self):
        source = (
            "    def __call__(self, pixel_values):\n"
            "        (batch_size, num_channels, height, width) = pixel_values.shape\n"
            "        x = self.projection(pixel_values)\n"
        )
        result = _inject_nchw_to_nhwc(source)
        assert "mx.transpose(pixel_values, (0, 2, 3, 1))" in result
        assert "(batch_size, height, width, num_channels) = pixel_values.shape" in result

    def test_noop_without_pattern(self):
        source = "    def __call__(self, x):\n        return self.fc(x)\n"
        assert _inject_nchw_to_nhwc(source) == source


# ── _inject_logger_stub ──────────────────────────────────────────────────────


class TestInjectLoggerStub:
    def test_injects_when_logger_used(self):
        source = (
            "import mlx.core as mx\n"
            "\n"
            "class Foo(nn.Module):\n"
            "    def __call__(self, x):\n"
            "        logger.warning('test')\n"
            "        return x\n"
        )
        result = _inject_logger_stub(source)
        assert "import logging" in result
        assert "logger = logging.getLogger(__name__)" in result

    def test_noop_without_logger(self):
        source = "class Foo(nn.Module):\n    pass\n"
        assert _inject_logger_stub(source) == source

    def test_noop_when_already_imported(self):
        source = (
            "import logging\n"
            "class Foo(nn.Module):\n"
            "    def __call__(self, x):\n"
            "        logger.info('test')\n"
        )
        assert _inject_logger_stub(source) == source


# ── hf_post_process ──────────────────────────────────────────────────────────


class TestHFPostProcess:
    def test_noop_on_non_hf_source(self):
        """Non-HF source should pass through unchanged."""
        source = (
            "import mlx.core as mx\n"
            "import mlx.nn as nn\n"
            "\n"
            "class MyModel(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.fc = nn.Linear(10, 5)\n"
            "\n"
            "    def __call__(self, x):\n"
            "        return self.fc(x)\n"
        )
        assert hf_post_process(source, None) == source
