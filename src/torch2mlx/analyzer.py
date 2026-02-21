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

from dataclasses import dataclass, field


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
