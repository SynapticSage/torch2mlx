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
    "_prepare_4d_causal_attention_mask_with_cache_position": (
        "    @staticmethod\n"
        "    def _prepare_4d_causal_attention_mask_with_cache_position(\n"
        "        attention_mask, sequence_length, target_length, dtype, cache_position=None,\n"
        "        batch_size=1, **kwargs,\n"
        "    ):\n"
        "        if attention_mask is not None and len(attention_mask.shape) == 4:\n"
        "            return attention_mask\n"
        "        min_dtype = mx.finfo(dtype).min\n"
        "        causal_mask = mx.full((sequence_length, target_length), min_dtype, dtype=dtype)\n"
        "        if sequence_length != 1:\n"
        "            causal_mask = mx.triu(causal_mask, k=1)\n"
        "        if cache_position is not None:\n"
        "            causal_mask = causal_mask * (mx.arange(target_length) > mx.reshape(cache_position, (-1, 1)))\n"
        "        causal_mask = mx.broadcast_to(causal_mask[None, None, :, :], (batch_size, 1, sequence_length, target_length))\n"
        "        if attention_mask is not None:\n"
        "            mask_length = attention_mask.shape[-1]\n"
        "            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]\n"
        "            causal_mask = mx.concatenate([\n"
        "                mx.where(padding_mask == 0, min_dtype, causal_mask[:, :, :, :mask_length]),\n"
        "                causal_mask[:, :, :, mask_length:],\n"
        "            ], axis=-1)\n"
        "        return causal_mask"
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
                # Scan entire class body (including extra methods before __call__)
                class_body = "\n".join(lines[class_start:next_class])
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
    "eager_attention_forward": (
        "def eager_attention_forward(module, query, key, value, attention_mask=None,"
        " scaling=None, dropout=0.0, **kwargs):\n"
        "    if scaling is None:\n"
        "        scaling = query.shape[-1] ** -0.5\n"
        "    # Step-by-step attention (matches HF eager path numerics)\n"
        "    attn_weights = (query @ mx.swapaxes(key, -1, -2)) * scaling\n"
        "    # Apply causal mask if needed (decoder attention)\n"
        "    is_causal = getattr(module, 'is_causal', False)\n"
        "    if is_causal and query.shape[-2] > 1:\n"
        "        L, S = query.shape[-2], key.shape[-2]\n"
        "        causal = mx.tril(mx.ones((L, S)))\n"
        "        mask_value = mx.array(mx.finfo(attn_weights.dtype).min)\n"
        "        attn_weights = mx.where(causal, attn_weights, mask_value)\n"
        "    if attention_mask is not None:\n"
        "        attn_weights = attn_weights + attention_mask\n"
        "    attn_weights = mx.softmax(attn_weights, axis=-1)\n"
        "    attn_output = attn_weights @ value\n"
        "    # Transpose from (B, heads, seq, dim) to (B, seq, heads, dim)\n"
        "    attn_output = mx.transpose(attn_output, (0, 2, 1, 3))\n"
        "    return attn_output, attn_weights\n"
    ),
    "_prepare_4d_causal_attention_mask": (
        "def _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length, sliding_window=None):\n"
        "    bsz, tgt_len = input_shape\n"
        "    src_len = tgt_len + past_key_values_length\n"
        "    # Causal mask: upper-triangle is -inf\n"
        "    causal = mx.full((tgt_len, src_len), -1e9)\n"
        "    mask_cond = mx.arange(src_len)\n"
        "    causal = mx.where(mask_cond[None, :] <= mx.arange(tgt_len)[:, None] + past_key_values_length, 0.0, causal)\n"
        "    causal = mx.reshape(causal, (1, 1, tgt_len, src_len))\n"
        "    causal = mx.broadcast_to(causal, (bsz, 1, tgt_len, src_len))\n"
        "    if attention_mask is not None:\n"
        "        # Expand 2D mask (bsz, src_len) → (bsz, 1, 1, src_len), invert\n"
        "        expanded = mx.reshape(attention_mask, (bsz, 1, 1, src_len))\n"
        "        expanded = (1.0 - expanded) * -1e9\n"
        "        causal = causal + expanded\n"
        "    return causal\n"
    ),
    "_prepare_4d_causal_attention_mask_for_sdpa": (
        "def _prepare_4d_causal_attention_mask_for_sdpa(*args, **kwargs):\n"
        "    raise NotImplementedError('SDPA mask prep not needed with eager attention')\n"
    ),
    "_prepare_4d_attention_mask_for_sdpa": (
        "def _prepare_4d_attention_mask_for_sdpa(*args, **kwargs):\n"
        "    raise NotImplementedError('SDPA mask prep not needed with eager attention')\n"
    ),
    "ALL_ATTENTION_FUNCTIONS": ("ALL_ATTENTION_FUNCTIONS = {'eager': eager_attention_forward}\n"),
}


def _inject_hf_global_stubs(source: str) -> str:
    """Inject global HF utility function stubs before the first class definition."""
    needed = [
        name for name in _HF_GLOBAL_STUBS if _re.search(r"\b" + _re.escape(name) + r"\b", source)
    ]
    if not needed:
        return source

    # Order stubs so dependencies come first (ALL_ATTENTION_FUNCTIONS needs eager_attention_forward)
    _STUB_DEPS = {"ALL_ATTENTION_FUNCTIONS": "eager_attention_forward"}
    ordered = sorted(needed, key=lambda n: (n in _STUB_DEPS, n))
    stubs = "\n".join(_HF_GLOBAL_STUBS[n] for n in ordered)
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


_HF_CACHE_STUB = (
    "class Cache:\n"
    "    def get_seq_length(self, layer_idx=0):\n"
    "        return 0\n"
    "    def get_max_cache_shape(self):\n"
    "        return None\n"
    "class DynamicCache(Cache):\n"
    "    @classmethod\n"
    "    def from_legacy_cache(cls, past=None):\n"
    "        return cls()\n"
    "class StaticCache(Cache):\n"
    "    pass\n"
    "class SlidingWindowCache(Cache):\n"
    "    pass\n"
)


def _inject_hf_output_stubs(source: str) -> str:
    """Stub HF output dataclasses (BaseModelOutput, etc.) and Cache classes."""
    # Cache stubs (needed by decoder models with KV caching)
    if _re.search(r"\bCache\b", source) and "class Cache" not in source:
        first_class = source.find("\nclass ")
        if first_class >= 0:
            source = source[: first_class + 1] + _HF_CACHE_STUB + "\n" + source[first_class + 1 :]

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


def _fix_sdpa_kwargs(source: str) -> str:
    """Fix mx.fast.scaled_dot_product_attention keyword args → positional.

    HF SDPA attention emits: mx.fast.scaled_dot_product_attention(query=Q, key=K, value=V, mask=M)
    MLX expects positional: mx.fast.scaled_dot_product_attention(Q, K, V, scale=s, mask=M)
    """
    pat = _re.compile(
        r"mx\.fast\.scaled_dot_product_attention\("
        r"query=(\w+),\s*key=(\w+),\s*value=(\w+)"
        r"(?:,\s*attn_mask=(\w+))?"
        r"(?:,\s*mask=(\w+))?"
        r"[^)]*\)"
    )

    def _repl(m: _re.Match) -> str:
        q, k, v = m.group(1), m.group(2), m.group(3)
        mask = m.group(4) or m.group(5)
        scale = f"1.0 / ({q}.shape[-1] ** 0.5)"
        if mask:
            return (
                f"mx.fast.scaled_dot_product_attention({q}, {k}, {v}, scale={scale}, mask={mask})"
            )
        return f"mx.fast.scaled_dot_product_attention({q}, {k}, {v}, scale={scale})"

    return pat.sub(_repl, source)


def _inject_nchw_to_nhwc(source: str) -> str:
    """Inject NCHW→NHWC transpose for vision models.

    HF vision models unpack pixel_values as (B, C, H, W) then pass to Conv2d.
    MLX Conv2d expects (B, H, W, C).  Rewrite the unpack + Conv2d pipeline.
    """
    # Pattern: NCHW shape unpack — (batch_size, num_channels, height, width) = pixel_values.shape
    nchw_pat = _re.compile(
        r"^(\s*)\((\w+), (\w+), (\w+), (\w+)\) = (pixel_values)\.shape",
        _re.MULTILINE,
    )
    match = nchw_pat.search(source)
    if match is None:
        return source
    indent = match.group(1)
    b, c, h, w = match.group(2), match.group(3), match.group(4), match.group(5)
    pv = match.group(6)
    # Rewrite to NHWC: transpose then unpack as (B, H, W, C)
    replacement = (
        f"{indent}{pv} = mx.transpose({pv}, (0, 2, 3, 1))  # NCHW → NHWC for MLX Conv2d\n"
        f"{indent}({b}, {h}, {w}, {c}) = {pv}.shape"
    )
    source = source[: match.start()] + replacement + source[match.end() :]

    # Fix the flatten+transpose after Conv2d patch projection:
    # NCHW: flatten(proj, 2) -> (B, C, H*W), then swapaxes(1,2) -> (B, H*W, C)
    # NHWC: proj is (B, H, W, C), need flatten spatial dims then done:
    #        reshape(proj, (B, -1, C)) -> (B, H*W, C)
    source = source.replace(
        "mx.swapaxes(mx.flatten(self.projection(pixel_values), 2), 1, 2)",
        "(lambda _p: mx.reshape(_p, (_p.shape[0], -1, _p.shape[-1])))(self.projection(pixel_values))",
    )

    # Fix channel-dim indexing in child classes: after NCHW→NHWC, channel is
    # at dim -1 (not dim 1).  Rewrite pixel_values.shape[1] → shape[-1].
    source = source.replace("pixel_values.shape[1]", "pixel_values.shape[-1]")

    return source


def _inject_logger_stub(source: str) -> str:
    """Inject logging setup when generated code references `logger.`."""
    if "logger." not in source or "import logging" in source:
        return source
    stub = (
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
        "if not hasattr(logger, 'warning_once'):\n"
        "    logger.warning_once = logger.warning\n"
    )
    idx = source.find("\nclass ")
    if idx >= 0:
        source = source[:idx] + "\n" + stub + source[idx:]
    return source


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
    source = _inject_logger_stub(source)
    source = _inject_nchw_to_nhwc(source)
    source = _fix_sdpa_kwargs(source)
    return source
