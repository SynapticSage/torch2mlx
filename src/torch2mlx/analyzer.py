"""Model analyzer: walk a torch Module tree, report convertibility.

Produces a portability report showing:
  - % of layers with registry mappings (auto-convertible)
  - Layers requiring manual port
  - Op-level compatibility of forward() methods
  - Blocker patterns (in-place ops, custom autograd, etc.)

The analyzer is the most valuable piece — it tells you before you start
what percentage of the model is auto-convertible.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

from torch2mlx.registry import lookup

try:
    import torch.nn as nn  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class PortabilityReport:
    """Result of analyzing a torch model for MLX portability."""

    total_layers: int = 0
    mapped_layers: int = 0
    unmapped_layers: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of layers with automatic mappings."""
        if self.total_layers == 0:
            return 0.0
        return self.mapped_layers / self.total_layers


# Blocker patterns: (search_string, human-readable description)
_BLOCKER_PATTERNS: list[tuple[str, str]] = [
    (".copy_(", "In-place operation: .copy_() found in forward()"),
    ("torch.autograd.Function", "Custom autograd: torch.autograd.Function subclass detected"),
    (".item()", "Scalar extraction: .item() found in forward()"),
    ("register_forward_hook", "Hook registration: register_forward_hook found"),
    ("register_backward_hook", "Hook registration: register_backward_hook found"),
]


def detect_blockers(module: object) -> list[str]:
    """Detect patterns that prevent automatic conversion.

    Inspects the module's forward() source for known blocker patterns.
    Silently skips modules where source is unavailable (e.g., built-ins).
    """
    blockers: list[str] = []
    forward = getattr(module, "forward", None)
    if forward is None:
        return blockers

    try:
        source = inspect.getsource(forward)
    except (OSError, TypeError):
        return blockers

    for pattern, description in _BLOCKER_PATTERNS:
        if pattern in source:
            blockers.append(description)

    # Heuristic: tensor in-place index assignment (x[i] = v style)
    # Only flag lines that look like tensor mutations (exclude dict/list patterns)
    for line in source.splitlines():
        stripped = line.strip()
        # Skip comments and obvious non-tensor assignments
        if stripped.startswith("#"):
            continue
        # += on a line that isn't a simple counter (heuristic)
        if (
            "+=" in stripped
            and "total" not in stripped
            and "count" not in stripped
            and "i +=" not in stripped
        ):
            desc = "In-place operation: += assignment found in forward()"
            if desc not in blockers:
                blockers.append(desc)

    return blockers


def _is_fully_composed(module: object) -> bool:
    """Check if an unregistered module is fully composed of convertible children.

    Returns True when the module has children and every direct child is either
    in the registry or is itself fully composed.  Leaf modules (no children)
    return False — they need an explicit registry entry.
    """
    children = list(module.children())
    if not children:
        return False
    return all(lookup(type(c).__name__) is not None or _is_fully_composed(c) for c in children)


def analyze(module: object) -> PortabilityReport:
    """Analyze a torch.nn.Module for MLX portability.

    Args:
        module: a torch.nn.Module instance

    Returns:
        PortabilityReport with coverage stats and blockers
    """
    if not HAS_TORCH:
        raise ImportError("torch is required for model analysis")

    report = PortabilityReport()

    # named_modules() yields (name, module) including the root ("", root)
    # Skip root itself; count only children
    for name, child in module.named_modules():
        if name == "":
            continue  # root module — skip
        class_name = type(child).__name__
        report.total_layers += 1
        if lookup(class_name) is not None or _is_fully_composed(child):
            report.mapped_layers += 1
        else:
            if class_name not in report.unmapped_layers:
                report.unmapped_layers.append(class_name)

    report.blockers = detect_blockers(module)
    return report
