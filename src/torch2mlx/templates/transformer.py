"""MLX implementation of standard transformer block.

Covers: multi-head attention + feed-forward network + layer norm,
matching the common PyTorch transformer pattern.
"""

from __future__ import annotations

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

if HAS_MLX:

    class TransformerBlock(nn.Module):
        """Single transformer block: self-attention + position-wise FFN.

        Args:
            d_model: Token embedding dimension.
            n_heads: Number of attention heads.
            d_ff: FFN inner dimension (default: 4 * d_model).
            dropout: Dropout probability (0 = disabled).
            norm_first: Pre-norm (GPT-style) if True; post-norm (BERT-style) if False.
        """

        def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int | None = None,
            dropout: float = 0.0,
            norm_first: bool = True,
        ) -> None:
            super().__init__()
            d_ff = d_ff if d_ff is not None else 4 * d_model

            self.attn = nn.MultiHeadAttention(d_model, n_heads)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

            # FFN: Linear -> GELU -> Linear
            self.ff1 = nn.Linear(d_model, d_ff)
            self.ff2 = nn.Linear(d_ff, d_model)
            self.activation = nn.GELU()

            self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None
            self.norm_first = norm_first

        def _ffn(self, x: mx.array) -> mx.array:
            return self.ff2(self.activation(self.ff1(x)))

        def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
            if self.norm_first:
                # Pre-norm (GPT-style): norm → sublayer → residual
                normed = self.norm1(x)
                attn_out = self.attn(normed, normed, normed, mask)
                if self.dropout is not None:
                    attn_out = self.dropout(attn_out)
                x = x + attn_out

                normed = self.norm2(x)
                ffn_out = self._ffn(normed)
                if self.dropout is not None:
                    ffn_out = self.dropout(ffn_out)
                x = x + ffn_out
            else:
                # Post-norm (BERT-style): sublayer → residual → norm
                attn_out = self.attn(x, x, x, mask)
                if self.dropout is not None:
                    attn_out = self.dropout(attn_out)
                x = self.norm1(x + attn_out)

                ffn_out = self._ffn(x)
                if self.dropout is not None:
                    ffn_out = self.dropout(ffn_out)
                x = self.norm2(x + ffn_out)

            return x
