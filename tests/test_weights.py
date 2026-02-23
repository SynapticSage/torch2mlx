"""Tests for weight_converter and state_dict modules."""

from __future__ import annotations

import numpy as np
import pytest

from torch2mlx.weight_converter import TRANSPOSITION_RULES, available_rules, convert_weight
from torch2mlx.state_dict import flatten, load_safetensors, save_safetensors, unflatten


# ---------------------------------------------------------------------------
# Transposition shape tests
# ---------------------------------------------------------------------------

class TestTranspositionShapes:
    def test_identity(self):
        arr = np.zeros((4, 3))
        assert convert_weight(arr, "identity").shape == (4, 3)

    def test_conv1d(self):
        arr = np.zeros((16, 3, 5))
        assert convert_weight(arr, "conv1d").shape == (16, 5, 3)

    def test_conv2d(self):
        arr = np.zeros((16, 3, 5, 5))
        assert convert_weight(arr, "conv2d").shape == (16, 5, 5, 3)

    def test_conv_transpose1d(self):
        arr = np.zeros((3, 16, 5))
        assert convert_weight(arr, "conv_transpose1d").shape == (16, 5, 3)

    def test_conv_transpose2d(self):
        arr = np.zeros((3, 6, 4, 4))
        assert convert_weight(arr, "conv_transpose2d").shape == (6, 4, 4, 3)

    def test_batch_norm(self):
        arr = np.zeros((64,))
        assert convert_weight(arr, "batch_norm").shape == (64,)


# ---------------------------------------------------------------------------
# Transposition value tests
# ---------------------------------------------------------------------------

class TestTranspositionValues:
    def test_identity_values(self):
        arr = np.arange(12.0).reshape(4, 3)
        result = convert_weight(arr, "identity")
        np.testing.assert_array_equal(result, arr)

    def test_conv1d_values(self):
        # [O=2, I=3, K=4] -> [O=2, K=4, I=3]
        arr = np.arange(24.0).reshape(2, 3, 4)
        result = convert_weight(arr, "conv1d")
        assert result.shape == (2, 4, 3)
        # result[o, k, i] == arr[o, i, k]
        np.testing.assert_array_equal(result, np.swapaxes(arr, 1, 2))

    def test_conv2d_values(self):
        # [O=2, I=3, H=4, W=5] -> [O=2, H=4, W=5, I=3]
        arr = np.arange(120.0).reshape(2, 3, 4, 5)
        result = convert_weight(arr, "conv2d")
        assert result.shape == (2, 4, 5, 3)
        np.testing.assert_array_equal(result, np.moveaxis(arr, 1, -1))

    def test_conv_transpose1d_values(self):
        # [I=3, O=2, K=4] -> [O=2, K=4, I=3]
        arr = np.arange(24.0).reshape(3, 2, 4)
        result = convert_weight(arr, "conv_transpose1d")
        assert result.shape == (2, 4, 3)
        np.testing.assert_array_equal(result, np.transpose(arr, (1, 2, 0)))

    def test_conv_transpose2d_values(self):
        # [I=3, O=2, H=4, W=5] -> [O=2, H=4, W=5, I=3]
        arr = np.arange(120.0).reshape(3, 2, 4, 5)
        result = convert_weight(arr, "conv_transpose2d")
        assert result.shape == (2, 4, 5, 3)
        np.testing.assert_array_equal(result, np.transpose(arr, (1, 2, 3, 0)))

    def test_batch_norm_is_identity(self):
        arr = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(convert_weight(arr, "batch_norm"), arr)


# ---------------------------------------------------------------------------
# convert_weight dispatch + error
# ---------------------------------------------------------------------------

class TestConvertWeight:
    def test_unknown_rule_raises_key_error(self):
        arr = np.zeros((4, 4))
        with pytest.raises(KeyError, match="unknown_rule"):
            convert_weight(arr, "unknown_rule")

    def test_all_registered_rules_are_callable(self):
        for name, fn in TRANSPOSITION_RULES.items():
            assert callable(fn), f"Rule {name!r} is not callable"


# ---------------------------------------------------------------------------
# available_rules
# ---------------------------------------------------------------------------

class TestAvailableRules:
    def test_contains_expected_rules(self):
        rules = available_rules()
        for expected in ["identity", "conv1d", "conv2d", "conv_transpose1d", "conv_transpose2d", "batch_norm"]:
            assert expected in rules, f"Missing rule: {expected!r}"

    def test_returns_list(self):
        assert isinstance(available_rules(), list)


# ---------------------------------------------------------------------------
# Safetensors roundtrip
# ---------------------------------------------------------------------------

class TestSafetensorsRoundtrip:
    def test_float32_roundtrip(self, tmp_path):
        tensors = {
            "weight": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "bias": np.array([0.1, 0.2], dtype=np.float32),
        }
        path = tmp_path / "model.safetensors"
        save_safetensors(tensors, path)
        loaded = load_safetensors(path)

        for key, arr in tensors.items():
            assert key in loaded
            np.testing.assert_array_equal(loaded[key], arr)
            assert loaded[key].dtype == arr.dtype

    def test_float16_roundtrip(self, tmp_path):
        tensors = {"w": np.ones((3, 4), dtype=np.float16)}
        path = tmp_path / "fp16.safetensors"
        save_safetensors(tensors, path)
        loaded = load_safetensors(path)
        np.testing.assert_array_equal(loaded["w"], tensors["w"])
        assert loaded["w"].dtype == np.float16

    def test_path_string_and_pathlib(self, tmp_path):
        tensors = {"x": np.zeros((2,), dtype=np.float32)}
        path = tmp_path / "test.safetensors"
        # save with Path, load with str
        save_safetensors(tensors, path)
        loaded = load_safetensors(str(path))
        np.testing.assert_array_equal(loaded["x"], tensors["x"])


# ---------------------------------------------------------------------------
# Flatten / unflatten
# ---------------------------------------------------------------------------

class TestFlattenUnflatten:
    def test_roundtrip(self):
        nested = {"a": {"b": {"c": np.array([1.0])}, "d": np.array([2.0])}}
        flat = flatten(nested)
        assert set(flat.keys()) == {"a.b.c", "a.d"}
        recovered = unflatten(flat)
        np.testing.assert_array_equal(recovered["a"]["b"]["c"], nested["a"]["b"]["c"])
        np.testing.assert_array_equal(recovered["a"]["d"], nested["a"]["d"])

    def test_empty_dict(self):
        assert flatten({}) == {}
        assert unflatten({}) == {}

    def test_flat_dict_passthrough(self):
        """A dict with no nesting should survive flatten -> unflatten unchanged."""
        arr = np.array([1.0, 2.0])
        flat = flatten({"weight": arr})
        assert flat == {"weight": arr}
        recovered = unflatten(flat)
        np.testing.assert_array_equal(recovered["weight"], arr)

    def test_deeply_nested(self):
        arr = np.zeros((2, 2))
        nested = {"a": {"b": {"c": {"d": arr}}}}
        flat = flatten(nested)
        assert "a.b.c.d" in flat
        recovered = unflatten(flat)
        np.testing.assert_array_equal(recovered["a"]["b"]["c"]["d"], arr)

    def test_multiple_keys_same_prefix(self):
        w = np.ones((4, 4))
        b = np.zeros((4,))
        nested = {"layer": {"weight": w, "bias": b}}
        flat = flatten(nested)
        assert flat == {"layer.weight": w, "layer.bias": b}
