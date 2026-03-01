"""Tests for DTYPE_REGISTRY in op_mapping."""

from __future__ import annotations

import pytest

from torch2mlx.op_mapping import DTYPE_REGISTRY, DtypeMapping, get_dtype_mapping


# ── Expected contents ────────────────────────────────────────────────────────

EXPECTED_DTYPES = [
    ("torch.float16",    "mx.float16"),
    ("torch.float32",    "mx.float32"),
    ("torch.bfloat16",   "mx.bfloat16"),
    ("torch.int8",       "mx.int8"),
    ("torch.int16",      "mx.int16"),
    ("torch.int32",      "mx.int32"),
    ("torch.int64",      "mx.int64"),
    ("torch.uint8",      "mx.uint8"),
    ("torch.bool",       "mx.bool_"),
]


class TestDtypeRegistry:
    """DTYPE_REGISTRY population and lookup."""

    def test_registry_populated(self) -> None:
        assert len(DTYPE_REGISTRY) >= 9, f"Only {len(DTYPE_REGISTRY)} dtypes registered"

    @pytest.mark.parametrize("torch_dtype,mlx_dtype", EXPECTED_DTYPES)
    def test_dtype_present(self, torch_dtype: str, mlx_dtype: str) -> None:
        mapping = DTYPE_REGISTRY[torch_dtype]
        assert mapping.mlx_dtype == mlx_dtype

    def test_all_values_are_dtype_mappings(self) -> None:
        for key, val in DTYPE_REGISTRY.items():
            assert isinstance(val, DtypeMapping), f"{key} is {type(val)}"

    def test_frozen(self) -> None:
        mapping = DTYPE_REGISTRY["torch.float32"]
        with pytest.raises(AttributeError):
            mapping.mlx_dtype = "mx.float16"  # type: ignore[misc]


class TestGetDtypeMapping:
    """get_dtype_mapping() lookup function."""

    def test_known_dtype(self) -> None:
        assert get_dtype_mapping("torch.float32") == "mx.float32"

    def test_unknown_dtype_returns_none(self) -> None:
        assert get_dtype_mapping("torch.quint8") is None

    def test_bool_trailing_underscore(self) -> None:
        assert get_dtype_mapping("torch.bool") == "mx.bool_"

    def test_float64_downcast(self) -> None:
        assert get_dtype_mapping("torch.float64") == "mx.float32"

    def test_complex_unsupported(self) -> None:
        assert get_dtype_mapping("torch.complex64") == "unsupported"
        assert get_dtype_mapping("torch.complex128") == "unsupported"
