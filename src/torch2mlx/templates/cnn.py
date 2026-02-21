"""MLX implementation of standard convolutional stack patterns.

Covers: Conv1d/Conv2d stacks with batch norm, pooling, and activations.
Handles the channels-last layout that MLX uses for convolutions.
"""

from __future__ import annotations
