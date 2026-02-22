"""MLX implementations of standard architecture patterns.

These are hand-written MLX equivalents that replace auto-translation
of forward() logic. Each template covers a common pattern:
  - transformer: standard transformer block (attention + FFN)
  - cnn: convolutional stack patterns
  - mlp: feed-forward networks
"""

from __future__ import annotations

try:
    from torch2mlx.templates.mlp import MLP
    from torch2mlx.templates.transformer import TransformerBlock
    from torch2mlx.templates.cnn import ConvBlock, ConvStack

    __all__ = ["MLP", "TransformerBlock", "ConvBlock", "ConvStack"]
except ImportError:
    __all__ = []
