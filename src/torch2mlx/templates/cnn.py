"""MLX implementation of standard convolutional stack patterns.

Covers: Conv1d/Conv2d stacks with batch norm, pooling, and activations.
Handles the channels-last layout that MLX uses for convolutions.
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

    _CONV: dict[str, type] = {
        "1d": nn.Conv1d,
        "2d": nn.Conv2d,
    }

    class ConvBlock(nn.Module):
        """Single Conv + activation block (channels-last input).

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            padding: Zero-padding on each side.
            conv_type: "1d" for Conv1d, "2d" for Conv2d.
            activation: One of "gelu", "relu", "silu".
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 0,
            conv_type: str = "1d",
            activation: str = "relu",
        ) -> None:
            super().__init__()
            if conv_type not in _CONV:
                raise ValueError(f"conv_type must be one of {list(_CONV)}, got {conv_type!r}")
            if activation not in _ACTIVATIONS:
                raise ValueError(
                    f"activation must be one of {list(_ACTIVATIONS)}, got {activation!r}"
                )

            self.conv = _CONV[conv_type](in_channels, out_channels, kernel_size, stride=stride, padding=padding)
            self.activation_fn = _ACTIVATIONS[activation]()

        def __call__(self, x: mx.array) -> mx.array:
            return self.activation_fn(self.conv(x))

    class ConvStack(nn.Module):
        """Stack of ConvBlock layers (channels-last input).

        Args:
            channels: Channel widths per layer, e.g. [3, 16, 32].
                      len(channels)-1 ConvBlock layers are created.
            kernel_sizes: Kernel size for each layer (int = same for all).
            conv_type: "1d" or "2d".
            activation: Activation applied in each block.
        """

        def __init__(
            self,
            channels: list[int],
            kernel_sizes: list[int] | int = 3,
            conv_type: str = "1d",
            activation: str = "relu",
        ) -> None:
            super().__init__()
            if len(channels) < 2:
                raise ValueError(f"channels must have at least 2 elements, got {channels}")

            n_blocks = len(channels) - 1
            if isinstance(kernel_sizes, int):
                kernel_sizes = [kernel_sizes] * n_blocks
            if len(kernel_sizes) != n_blocks:
                raise ValueError(
                    f"kernel_sizes length {len(kernel_sizes)} must match n_blocks {n_blocks}"
                )

            self.blocks = [
                ConvBlock(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    conv_type=conv_type,
                    activation=activation,
                )
                for i in range(n_blocks)
            ]

        def __call__(self, x: mx.array) -> mx.array:
            for block in self.blocks:
                x = block(x)
            return x
