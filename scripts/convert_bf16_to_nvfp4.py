"""Convert a FramePack bfloat16 transformer checkpoint to NVFP4-W4A16 safetensors.

The packed file is written next to the original weights as
``diffusion_pytorch_model-nvfp4.safetensors``. The bfloat16 files stay in place
so a Kohya LoRA merge can still read them.

Example:
    .\\.venv\\Scripts\\python.exe scripts\\convert_bf16_to_nvfp4.py PATH

PATH is a local snapshot directory, a Hugging Face repo id, or one safetensors
file inside the snapshot. When PATH has no safetensors, the original weights are
read from HF_HOME, the studio hf_download folder, or ~/.cache/huggingface.
The conversion packs one linear at a time. It uses the GPU when CUDA is available.
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _prefer_repo_root() -> None:
    """Import this checkout before any installed copy of diffusers_helper."""
    root = str(ROOT)
    sys.path[:] = [entry for entry in sys.path if entry != root]
    sys.path.insert(0, root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        help=(
            "Local snapshot directory, Hugging Face repo id, or a safetensors file. "
            "When this path has no weights, the FramePack transformers are read from the Hugging Face cache."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing NVFP4 safetensors file.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Resolve a repo id only from the local Hugging Face cache.",
    )
    args = parser.parse_args()

    _prefer_repo_root()
    checkpoint_file = ROOT / "diffusers_helper" / "nvfp4_checkpoint.py"
    if not checkpoint_file.is_file():
        raise SystemExit(f"Missing {checkpoint_file}")
    from diffusers_helper.nvfp4_checkpoint import (
        convert_checkpoint,
        resolve_conversion_sources,
        resolve_model_dir,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    if args.path and not Path(args.path).exists():
        model_dirs = [resolve_model_dir(args.path, offline=args.offline)]
    else:
        model_dirs = resolve_conversion_sources(args.path)
    for model_dir in model_dirs:
        print(convert_checkpoint(model_dir, force=args.force))


if __name__ == "__main__":
    main()
