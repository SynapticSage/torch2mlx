"""Adaptive pooling template for MLX.

MLX has native MaxPool/AvgPool but lacks AdaptiveAvgPool2d.
This template computes kernel size and stride dynamically from the
input spatial dimensions and the desired output size.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class AdaptiveAvgPool2d(nn.Module):
    """Adaptive average pooling that produces a fixed output size.

    Computes kernel/stride dynamically so that any input spatial size
    maps to ``output_size``. Expects channels-last layout: (B, H, W, C).
    """

    def __init__(self, output_size: int | tuple[int, int]) -> None:
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, H, W, C) — channels-last
        H, W = x.shape[1], x.shape[2]
        out_H, out_W = self.output_size
        stride_h = H // out_H
        stride_w = W // out_W
        kernel_h = H - stride_h * (out_H - 1)
        kernel_w = W - stride_w * (out_W - 1)
        pool = nn.AvgPool2d(
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
        )
        return pool(x)
