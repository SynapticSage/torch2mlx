"""Main conversion orchestrator.

Wires together all modules to perform end-to-end model conversion:
  1. Walk torch module tree
  2. Build equivalent MLX module tree from registry
  3. Convert state dict (transpositions + key restructuring)
  4. Load weights into MLX modules
  5. Verify numerical equivalence
"""

from __future__ import annotations
