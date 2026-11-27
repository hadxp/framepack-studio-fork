"""Convert a FramePack bfloat16 transformer checkpoint to packed NVFP4 safetensors.

``--variant nvfp4`` writes ``diffusion_pytorch_model-nvfp4.safetensors``.
``--variant nvfp4a16`` writes ``diffusion_pytorch_model-nvfp4a16.safetensors``
next to it. ``--variant both`` writes the two files in one pass. The bfloat16
files stay in place so a Kohya LoRA merge can still read them. When the studio
should load a variant and that file is missing, it runs the same conversion
and then loads the result. NVFP4 quantizes activations. NVFP4-A16 leaves them
in bfloat16.

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
        "--variant",
        choices=["nvfp4", "nvfp4a16", "both"],
        default="nvfp4a16",
        help=(
            "nvfp4 writes diffusion_pytorch_model-nvfp4.safetensors. "
            "nvfp4a16 writes diffusion_pytorch_model-nvfp4a16.safetensors "
            "beside it. both writes the two files. The default is nvfp4a16."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing packed safetensors file for the chosen variant.",
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
        NVFP4,
        NVFP4_A16,
        convert_variants,
        resolve_conversion_sources,
        resolve_model_dir,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    if args.path and not Path(args.path).exists():
        model_dirs = [resolve_model_dir(args.path, offline=args.offline)]
    else:
        model_dirs = resolve_conversion_sources(args.path)
    if args.variant == "both":
        variants = [NVFP4, NVFP4_A16]
    else:
        variants = [NVFP4 if args.variant == NVFP4 else NVFP4_A16]
    for model_dir in model_dirs:
        for path in convert_variants(model_dir, variants, force=args.force):
            print(path)


if __name__ == "__main__":
    main()
