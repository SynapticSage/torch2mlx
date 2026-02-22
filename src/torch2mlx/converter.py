"""Main conversion orchestrator.

Wires together all modules to perform end-to-end model conversion:
  1. Walk torch module tree
  2. Build module-to-rule map from registry
  3. Convert state dict (weight transpositions + key restructuring)
  4. Save to safetensors for MLX loading
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from numpy.typing import NDArray

from torch2mlx import analyzer, registry, state_dict, weight_converter

try:
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def build_module_map(
    named_modules: list[tuple[str, str]],
) -> dict[str, str]:
    """Build a prefix-to-rule mapping from named modules.

    Args:
        named_modules: list of (name, class_name) pairs from walking
                       a torch module tree

    Returns:
        dict mapping module name prefix to transposition rule name
    """
    # module_map: "encoder.conv1" -> "conv1d"
    module_map: dict[str, str] = {}
    for name, class_name in named_modules:
        mapping = registry.lookup(class_name)
        if mapping is not None:
            module_map[name] = mapping.weight_transposition
    return module_map


def convert_state_dict(
    flat_state: dict[str, NDArray],
    module_map: dict[str, str],
) -> dict[str, NDArray]:
    """Apply weight transpositions based on module-to-rule mapping.

    Args:
        flat_state: flat state dict with dot-separated keys and numpy arrays
        module_map: maps module prefix (e.g., "encoder.conv1") to transposition
                    rule name (e.g., "conv1d")

    Returns:
        New flat state dict with transposed weight arrays
    """
    result: dict[str, NDArray] = {}
    for key, array in flat_state.items():
        # Only transpose keys where the final segment is "weight"
        # (handles both "conv.weight" and bare "weight" for root modules)
        parts = key.split(".")
        if parts[-1] != "weight":
            result[key] = array
            continue

        # Find longest matching prefix in module_map
        # Key "encoder.conv1.weight" -> prefix "encoder.conv1"
        # Key "weight" (root module) -> prefix ""
        prefix = ".".join(parts[:-1])
        best_prefix: str | None = None
        for candidate in module_map:
            if prefix == candidate or prefix.startswith(candidate + "."):
                if best_prefix is None or len(candidate) > len(best_prefix):
                    best_prefix = candidate

        if best_prefix is not None:
            rule = module_map[best_prefix]
            result[key] = weight_converter.convert_weight(array, rule)
        else:
            result[key] = array

    return result


def convert(
    torch_model_or_state: Any,
    output_path: str | Path,
    *,
    analyze_first: bool = True,
) -> Path:
    """Full end-to-end conversion pipeline.

    Args:
        torch_model_or_state: either a torch.nn.Module or a flat state dict
                               (numpy arrays, dot-separated keys)
        output_path: where to save the converted safetensors file
        analyze_first: if True and input is a Module, run analyzer first
                       and warn if coverage < 1.0

    Returns:
        Path to the saved safetensors file
    """
    output_path = Path(output_path)
    flat_weights: dict[str, NDArray]
    module_map: dict[str, str]

    if HAS_TORCH and isinstance(torch_model_or_state, nn.Module):
        model = torch_model_or_state

        if analyze_first:
            report = analyzer.analyze(model)
            if report.coverage < 1.0:
                unmapped = ", ".join(report.unmapped_layers)
                warnings.warn(
                    f"Model coverage {report.coverage:.1%} — unmapped layers: {unmapped}",
                    stacklevel=2,
                )
            if report.blockers:
                warnings.warn(
                    f"Conversion blockers detected: {'; '.join(report.blockers)}",
                    stacklevel=2,
                )

        # [(name, class_name), ...] — include root ("") so single-layer
        # models (e.g., nn.Conv1d directly) get their transpositions applied.
        # Root class is usually custom and absent from registry, so it's a no-op
        # for multi-layer models.
        named_modules = [
            (name, type(m).__name__)
            for name, m in model.named_modules()
        ]
        # numpy-only from here on
        flat_weights = {k: v.detach().numpy() for k, v in model.state_dict().items()}
        module_map = build_module_map(named_modules)

    elif isinstance(torch_model_or_state, dict):
        # Caller passed a flat state dict; no transpositions without a module map
        flat_weights = torch_model_or_state
        module_map = {}

    else:
        raise TypeError(
            f"Expected torch.nn.Module or dict, got {type(torch_model_or_state).__name__}"
        )

    converted = convert_state_dict(flat_weights, module_map)
    state_dict.save_safetensors(converted, output_path)
    return output_path


def load_converted(path: str | Path) -> dict[str, Any]:
    """Load a converted model as a nested MLX-compatible parameter tree.

    Returns:
        Nested dict suitable for mlx.nn.Module.load_weights().
    """
    flat = state_dict.load_safetensors(path)
    return state_dict.unflatten(flat)
