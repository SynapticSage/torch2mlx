"""torch2mlx — Translate PyTorch models to Apple's MLX framework.

Approach: Module-Tree Walk + Weight Convert
  1. Walk torch.nn.Module tree, map each layer via registry
  2. Convert state dict (transpositions, key restructuring)
  3. Load weights into equivalent MLX modules
  4. Verify numerical equivalence
"""

__version__ = "0.1.0"
