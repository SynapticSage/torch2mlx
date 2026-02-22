"""MLX implementation of standard feed-forward (MLP) patterns.

Covers: Linear stacks with activations (GELU, ReLU, SiLU),
dropout, and residual connections.
"""

from __future__ import annotations

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

if HAS_MLX:
    _ACTIVATIONS: dict[str, type] = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
    }

    class MLP(nn.Module):
        """Feed-forward network: stacked Linear layers with activations.

        Args:
            dims: Layer widths e.g. [768, 3072, 768]. len(dims)-1 linear layers.
            activation: One of "gelu", "relu", "silu".
            dropout: Dropout probability applied between (not after) layers. 0 = disabled.
        """

        def __init__(
            self,
            dims: list[int],
            activation: str = "gelu",
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            if activation not in _ACTIVATIONS:
                raise ValueError(
                    f"activation must be one of {list(_ACTIVATIONS)}, got {activation!r}"
                )
            if len(dims) < 2:
                raise ValueError(f"dims must have at least 2 elements, got {dims}")

            n_layers = len(dims) - 1
            self.linears = [nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers)]
            self.activation_fn = _ACTIVATIONS[activation]()
            # Dropout only between layers (not after last), so only matters if n_layers > 1
            self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

        def __call__(self, x: mx.array) -> mx.array:
            last = len(self.linears) - 1
            for i, linear in enumerate(self.linears):
                x = linear(x)
                if i < last:
                    x = self.activation_fn(x)
                    if self.dropout is not None:
                        x = self.dropout(x)
            return x
