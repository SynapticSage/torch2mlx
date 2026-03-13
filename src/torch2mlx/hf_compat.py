"""HuggingFace compatibility layer for codegen.

Extracts HF-specific patches, stubs, and inject functions from the codegen
core so that the compiler stays framework-agnostic.  All HF awareness lives
here and is applied as a post-processing step on generated source.
"""

from __future__ import annotations

import re as _re
from typing import Any


# ---------------------------------------------------------------------------
# Scalar overrides — emitted during _walk_module's scalar attribute scan
# ---------------------------------------------------------------------------

SCALAR_OVERRIDES: dict[str, str] = {
    "attn_implementation": "'eager'",  # Force eager attention (no SDPA mask prep)
}


# ---------------------------------------------------------------------------
# HF private attributes — defaults injected after super().__init__()
# ---------------------------------------------------------------------------

_HF_ATTR_DEFAULTS: dict[str, str] = {
    "_use_flash_attention_2": "False",
    "_use_sdpa": "False",  # Use vanilla attention path for MLX
    "_gradient_checkpointing_func": "None",
    "_attn_implementation": "'eager'",
    "gradient_checkpointing": "False",
}

_HF_PRIVATE_RE = _re.compile(r"self\.(_\w+)")


def _inject_hf_private_attrs(source: str) -> str:
    """Inject default values for HF attributes (attention dispatch, training, etc.)."""
    all_refs: set[str] = set()
    all_refs.update(_HF_PRIVATE_RE.findall(source))
    all_refs.update(_re.findall(r"self\.(gradient_checkpointing)\b", source))
    needed = all_refs & _HF_ATTR_DEFAULTS.keys()
    if not needed:
        return source

    lines = source.split("\n")
    result: list[str] = []
    for line in lines:
        result.append(line)
        if line.strip() == "super().__init__()":
            attr_lines = [
                f"        self.{attr} = {val}"
                for attr, val in sorted(_HF_ATTR_DEFAULTS.items())
                if attr in needed
            ]
            if attr_lines:
                result.extend(attr_lines)

    return "\n".join(result)


# ---------------------------------------------------------------------------
# HF method stubs — utility methods referenced by forward() bodies
# ---------------------------------------------------------------------------

HF_METHOD_STUBS: dict[str, str] = {
    "get_head_mask": (
        "    @staticmethod\n"
        "    def get_head_mask(head_mask, num_hidden_layers, is_attention_chunked=False):\n"
        "        if head_mask is not None:\n"
        "            return head_mask\n"
        "        return [None] * num_hidden_layers"
    ),
    "warn_if_padding_and_no_attention_mask": (
        "    @staticmethod\n"
        "    def warn_if_padding_and_no_attention_mask(*args, **kwargs):\n"
        "        pass  # Warning-only method, no-op at inference"
    ),
    "invert_attention_mask": (
        "    @staticmethod\n"
        "    def invert_attention_mask(encoder_attention_mask):\n"
        "        return (1.0 - encoder_attention_mask) * mx.array(-1e9)"
    ),
    "get_extended_attention_mask": (
        "    def get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):\n"
        "        if attention_mask.ndim == 3:\n"
        "            extended = mx.expand_dims(attention_mask, 1)\n"
        "        elif attention_mask.ndim == 2:\n"
        "            extended = mx.expand_dims(mx.expand_dims(attention_mask, 1), 2)\n"
        "        else:\n"
        "            extended = attention_mask\n"
        "        extended = (1.0 - extended) * mx.array(-1e9)\n"
        "        return extended"
    ),
}

_HF_METHOD_CALL_RE = _re.compile(r"self\.(\w+)\(")


def _inject_hf_method_stubs(source: str) -> str:
    """Inject stub methods for HF utility calls referenced in generated source."""
    called_methods = set(_HF_METHOD_CALL_RE.findall(source))
    needed = called_methods & HF_METHOD_STUBS.keys()
    if not needed:
        return source

    lines = source.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("def __call__"):
            class_start = None
            for j in range(len(result) - 1, -1, -1):
                if result[j].strip().startswith("class ") and "(nn.Module):" in result[j]:
                    class_start = j
                    break

            if class_start is not None:
                next_class = len(lines)
                for k in range(i + 1, len(lines)):
                    if lines[k].strip().startswith("class ") and "(nn.Module):" in lines[k]:
                        next_class = k
                        break
                class_body = "\n".join(lines[i:next_class])
                class_methods = set(_HF_METHOD_CALL_RE.findall(class_body))
                class_needed = class_methods & needed

                for method_name in sorted(class_needed):
                    result.append("")
                    result.append(HF_METHOD_STUBS[method_name])
                    result.append("")

        result.append(line)
        i += 1

    return "\n".join(result)


# ---------------------------------------------------------------------------
# HF global utility function stubs
# ---------------------------------------------------------------------------

_HF_GLOBAL_STUBS: dict[str, str] = {
    "apply_chunking_to_forward": (
        "def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):\n"
        "    return forward_fn(*input_tensors)\n"
    ),
    "create_position_ids_from_input_ids": (
        "def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):\n"
        "    mask = (input_ids != padding_idx).astype(mx.int32)\n"
        "    incremental_indices = (mx.cumsum(mask, axis=1).astype(mask.dtype) + past_key_values_length) * mask\n"
        "    return incremental_indices.astype(mx.int32) + padding_idx\n"
    ),
}


def _inject_hf_global_stubs(source: str) -> str:
    """Inject global HF utility function stubs before the first class definition."""
    needed = [name for name in _HF_GLOBAL_STUBS if name + "(" in source]
    if not needed:
        return source

    stubs = "\n".join(_HF_GLOBAL_STUBS[n] for n in sorted(needed))
    idx = source.find("\nclass ")
    if idx >= 0:
        source = source[:idx] + "\n" + stubs + "\n" + source[idx:]
    return source


# ---------------------------------------------------------------------------
# HF output dataclass stubs
# ---------------------------------------------------------------------------

_HF_OUTPUT_CLASS = (
    "class _HFOutput:\n"
    "    def __init__(self, **kwargs):\n"
    "        self.__dict__.update(kwargs)\n"
    "        self._fields = list(kwargs.keys())\n"
    "        self._values = list(kwargs.values())\n"
    "    def __getitem__(self, idx):\n"
    "        if isinstance(idx, str):\n"
    "            return getattr(self, idx)\n"
    "        return [v for v in self._values if v is not None][idx]\n"
    "    def __iter__(self):\n"
    "        return iter(v for v in self._values if v is not None)\n"
    "    def __len__(self):\n"
    "        return len([v for v in self._values if v is not None])\n"
)


def _inject_hf_output_stubs(source: str) -> str:
    """Stub HF output dataclasses (BaseModelOutput, etc.)."""
    output_names_raw = _re.findall(r"\b(\w*Output\w*)\(", source)
    if not output_names_raw:
        return source

    output_names = sorted({n for n in set(output_names_raw) if f"{n} = " not in source})
    if not output_names:
        return source

    aliases = "\n".join(f"{n} = _HFOutput" for n in output_names)
    stub_block = _HF_OUTPUT_CLASS + "\n" + aliases + "\n\n"
    first_class = source.find("\nclass ")
    if first_class >= 0:
        source = source[: first_class + 1] + stub_block + source[first_class + 1 :]

    return source


# ---------------------------------------------------------------------------
# Public API: composite post-processor
# ---------------------------------------------------------------------------


def hf_post_process(source: str, model: Any) -> str:
    """Chain all HF-specific inject functions.

    This is the single entry point for HF compatibility post-processing.
    Non-HF models pass through unchanged (all injectors are no-ops when
    no HF patterns are detected).
    """
    source = _inject_hf_private_attrs(source)
    source = _inject_hf_method_stubs(source)
    source = _inject_hf_global_stubs(source)
    source = _inject_hf_output_stubs(source)
    return source
