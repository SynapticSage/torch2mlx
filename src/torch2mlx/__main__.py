"""CLI entry point: python -m torch2mlx <model> <output_dir>.

Converts PyTorch model weights to MLX-compatible safetensors format.

Accepts:
  .pt / .pth   — torch checkpoint (requires torch installed)
  .safetensors — passthrough with key restructuring (no torch needed)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch2mlx import state_dict as sd
from torch2mlx.converter import convert


def _load_torch_checkpoint(path: Path) -> object:
    """Load a .pt/.pth file, returning a Module or state dict."""
    try:
        import torch
    except ImportError:
        print("error: torch is required to load .pt/.pth files", file=sys.stderr)
        sys.exit(1)

    data = torch.load(path, map_location="cpu", weights_only=False)

    # torch.save() can save a Module directly or a state_dict
    if isinstance(data, torch.nn.Module):
        return data
    if isinstance(data, dict):
        # Convert tensor values to numpy for our pipeline
        return {k: v.detach().cpu().numpy() for k, v in data.items() if hasattr(v, "numpy")}
    raise ValueError(f"Unsupported checkpoint contents: {type(data).__name__}")


def _print_report(report: object) -> None:
    """Format and print a PortabilityReport."""
    print(
        f"  Coverage:  {report.coverage:.1%} ({report.mapped_layers}/{report.total_layers} layers)"
    )
    if report.unmapped_layers:
        print(f"  Unmapped:  {', '.join(report.unmapped_layers)}")
    if report.blockers:
        print("  Blockers:")
        for b in report.blockers:
            print(f"    - {b}")
    if not report.unmapped_layers and not report.blockers:
        print("  No issues found")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="torch2mlx",
        description="Convert PyTorch model weights to MLX-compatible safetensors.",
    )
    parser.add_argument("model_path", type=Path, help="Input model (.pt, .pth, or .safetensors)")
    parser.add_argument("output_dir", type=Path, help="Output directory for converted weights")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Run portability analysis only (requires torch, .pt/.pth input)",
    )
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="Skip pre-conversion analysis",
    )
    parser.add_argument(
        "--codegen",
        action="store_true",
        help="Generate MLX module .py file alongside safetensors (requires torch, .pt/.pth input)",
    )
    args = parser.parse_args(argv)

    model_path: Path = args.model_path
    output_dir: Path = args.output_dir

    if not model_path.exists():
        print(f"error: {model_path} not found", file=sys.stderr)
        sys.exit(1)

    suffix = model_path.suffix.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_path.stem}.safetensors"

    # Load input
    if suffix in (".pt", ".pth"):
        data = _load_torch_checkpoint(model_path)
    elif suffix == ".safetensors":
        data = sd.load_safetensors(model_path)
    else:
        print(
            f"error: unsupported format {suffix!r} (expected .pt, .pth, or .safetensors)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Analyze-only mode
    if args.analyze_only:
        try:
            from torch2mlx.analyzer import analyze
        except ImportError:
            print("error: torch is required for --analyze-only", file=sys.stderr)
            sys.exit(1)

        import torch.nn as nn

        if not isinstance(data, nn.Module):
            print(
                "error: --analyze-only requires a .pt/.pth file containing a torch.nn.Module",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"torch2mlx: analyzing {model_path.name}\n")
        print("Analysis")
        _print_report(analyze(data))
        return

    # Convert
    print(f"torch2mlx: {model_path.name} -> {output_file}")

    # Run analysis if input is a Module and not suppressed
    if not args.no_analyze:
        try:
            import torch.nn as nn
            from torch2mlx.analyzer import analyze

            if isinstance(data, nn.Module):
                print("\nAnalysis")
                _print_report(analyze(data))
                print()
        except ImportError:
            pass  # No torch — skip analysis silently

    convert(data, output_file, analyze_first=False)  # We already analyzed above
    n_params = len(sd.load_safetensors(output_file))
    print(f"Converted {n_params} parameters -> {output_file}")

    # Codegen: generate MLX module .py file
    if args.codegen:
        try:
            import torch.nn as nn  # noqa: F811
            from torch2mlx.codegen import generate_to_file

            if not isinstance(data, nn.Module):
                print(
                    "warning: --codegen requires a .pt/.pth file containing a torch.nn.Module",
                    file=sys.stderr,
                )
            else:
                py_file = output_dir / f"{model_path.stem}.py"
                generate_to_file(data, py_file)
                print(f"Generated MLX module -> {py_file}")
        except ImportError:
            print("warning: torch is required for --codegen", file=sys.stderr)


if __name__ == "__main__":
    main()
