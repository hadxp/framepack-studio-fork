"""Save and load NVFP4 copies of a bfloat16 Hunyuan checkpoint.

Each variant has its own file next to the bfloat16 weights.
``diffusion_pytorch_model-nvfp4.safetensors`` quantizes activations.
``diffusion_pytorch_model-nvfp4a16.safetensors`` leaves activations in
bfloat16. The packed tensors are qdata, an FP8 scale, and a per-tensor
scale. Other tensors stay bfloat16. A Kohya LoRA merge still reads the
original bfloat16 safetensors.
"""

import json
import logging
import os
import struct
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

NVFP4_A16 = "nvfp4a16"
NVFP4 = "nvfp4"
_FORMAT_IDS = {
    NVFP4_A16: "framepack-nvfp4-w4a16-v1",
    NVFP4: "framepack-nvfp4-v1",
}
_FILENAMES = {
    NVFP4_A16: "diffusion_pytorch_model-nvfp4a16.safetensors",
    NVFP4: "diffusion_pytorch_model-nvfp4.safetensors",
}
# Older builds wrote NVFP4-A16 to the NVFP4 filename.
_LEGACY_A16_FILENAME = "diffusion_pytorch_model-nvfp4.safetensors"
_TRANSFORMER_CACHE_REPOS = (
    "models--lllyasviel--FramePackI2V_HY",
    "models--lllyasviel--FramePack_F1_I2V_HY_20250503",
)
_BLOCK_SIZE = 16
_QDATA = "::qdata"
_SCALE = "::scale"
_PER_TENSOR_SCALE = "::per_tensor_scale"


def nvfp4_checkpoint_path(model_dir: Path, variant: str = NVFP4_A16) -> Path:
    """Path of the packed file for ``variant`` inside ``model_dir``."""
    variant = normalize_nvfp4_variant(variant)
    return Path(model_dir) / _FILENAMES[variant]


def is_packed_nvfp4_filename(name: str) -> bool:
    """True for a packed NVFP4 or NVFP4-A16 safetensors name."""
    lower = name.lower()
    return lower.endswith("-nvfp4.safetensors") or lower.endswith(
        "-nvfp4a16.safetensors"
    )


def normalize_nvfp4_variant(value: str | None) -> str:
    """``nvfp4`` quantizes activations. Anything else stays NVFP4-A16."""
    if str(value or "").strip().lower() == NVFP4:
        return NVFP4
    return NVFP4_A16


def nvfp4_variant_label(value: str | None) -> str:
    if normalize_nvfp4_variant(value) == NVFP4:
        return "NVFP4"
    return "NVFP4-A16"


def huggingface_cache_roots() -> list[Path]:
    """HF_HOME when it is set, then the studio hf_download folder and the default cache."""
    roots = []
    env_home = os.environ.get("HF_HOME")
    if env_home:
        roots.append(Path(env_home))
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        roots.append(Path(hub_cache))
    roots.append(Path(__file__).resolve().parents[1] / "hf_download")
    roots.append(Path.home() / ".cache" / "huggingface")
    unique = []
    seen = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def cached_transformer_dirs(roots: list[Path] | None = None) -> list[Path]:
    """Snapshot directories for the FramePack transformers in the Hugging Face cache."""
    found = []
    seen = set()
    for root in huggingface_cache_roots() if roots is None else roots:
        hub = root if root.name == "hub" else root / "hub"
        for repo_name in _TRANSFORMER_CACHE_REPOS:
            snapshot = _snapshot_from_repo_cache(hub / repo_name)
            if snapshot is None:
                continue
            key = str(snapshot.resolve())
            if key in seen:
                continue
            seen.add(key)
            found.append(snapshot)
    return found


def resolve_conversion_sources(path: str | None) -> list[Path]:
    """Use ``path`` when it contains weights. Otherwise use the Hugging Face cache."""
    if path:
        candidate = Path(path)
        if candidate.is_file():
            candidate = candidate.parent
        if candidate.is_dir() and _dir_has_weights(candidate):
            return [candidate]
        logger.info(
            "No safetensors weights in %s. Looking in the Hugging Face cache.",
            path,
        )
    found = cached_transformer_dirs()
    if not found:
        searched = ", ".join(str(root) for root in huggingface_cache_roots())
        raise FileNotFoundError(
            "No FramePack transformer safetensors found. "
            f"Looked for {', '.join(_TRANSFORMER_CACHE_REPOS)} under: {searched}"
        )
    for source in found:
        logger.info("Using cached transformer weights in %s.", source)
    return found


def resolve_model_dir(path: str, offline: bool = False) -> Path:
    """Return the local directory that holds config.json and the weight files."""
    candidate = Path(path)
    if candidate.is_file():
        candidate = candidate.parent
    if candidate.is_dir() and _dir_has_weights(candidate):
        return candidate
    if candidate.is_dir():
        cached = cached_transformer_dirs()
        if len(cached) == 1:
            return cached[0]
        if cached:
            raise FileNotFoundError(
                f"No safetensors weights in {candidate}. "
                "Pass a snapshot directory, or convert each cached transformer. "
                + ", ".join(str(path) for path in cached)
            )
    from huggingface_hub import snapshot_download

    local = snapshot_download(repo_id=path, local_files_only=offline)
    return Path(local)


def original_weight_files(model_dir: Path) -> list[Path]:
    """Bfloat16 safetensors for this checkpoint, excluding the NVFP4 copy."""
    model_dir = Path(model_dir)
    index_path = _weight_index_path(model_dir)
    if index_path is not None:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        files = []
        for name in sorted(set(payload["weight_map"].values())):
            path = model_dir / name
            if is_packed_nvfp4_filename(path.name):
                continue
            if not path.is_file():
                raise FileNotFoundError(f"Missing checkpoint shard {path}")
            files.append(path)
        if not files:
            raise FileNotFoundError(f"No weight files listed in {index_path}")
        return files

    for name in ("diffusion_pytorch_model.safetensors", "model.safetensors"):
        path = model_dir / name
        if path.is_file():
            return [path]

    files = sorted(
        path
        for path in model_dir.glob("*.safetensors")
        if not is_packed_nvfp4_filename(path.name)
    )
    if not files:
        raise FileNotFoundError(f"No safetensors weights in {model_dir}")
    return files


def checkpoint_nbytes(model_dir: Path) -> int:
    """Bytes of tensor data in the original safetensors, read from the headers."""
    return sum(_safetensors_payload_nbytes(path) for path in original_weight_files(model_dir))


def usable_vram_bytes(preservation_gb: float) -> int | None:
    """Free CUDA memory minus the preserved reserve. None when CUDA is absent."""
    if not torch.cuda.is_available():
        return None
    device = torch.device("cuda")
    free_cuda, _total = torch.cuda.mem_get_info(device)
    stats = torch.cuda.memory_stats(device)
    inactive_reserved = (
        stats["reserved_bytes.all.current"] - stats["active_bytes.all.current"]
    )
    available = free_cuda + max(inactive_reserved, 0)
    preserved = int(max(float(preservation_gb), 0.0) * (1024**3))
    return max(available - preserved, 0)


def should_use_nvfp4_checkpoint(
    weight_bytes: int,
    usable_bytes: int | None,
    force: bool = False,
) -> bool:
    """Use the packed file when it is forced, or when bfloat16 exceeds usable VRAM.

    This does not look for the file. The loader creates a missing one before
    reading it.
    """
    if force:
        return True
    return usable_bytes is not None and weight_bytes > usable_bytes


def bf16_fits_in_vram(weight_bytes: int | None, usable_bytes: int | None) -> bool:
    """True when the measured bfloat16 checkpoint fits in usable VRAM."""
    return (
        weight_bytes is not None
        and usable_bytes is not None
        and weight_bytes <= usable_bytes
    )


def should_memory_quantize_bf16(
    force: bool, weight_bytes: int | None, usable_bytes: int | None
) -> bool:
    """Quantize a bfloat16 transformer that was just loaded.

    The checkbox off and a checkpoint that fits leaves the weights in bfloat16.
    """
    if force:
        return True
    return not bf16_fits_in_vram(weight_bytes, usable_bytes)


def linear_weight_keys(model_dir: Path) -> set[str]:
    """Names of linear weights the Hunyuan module would quantize."""
    from diffusers_helper.models.hunyuan_video_packed import (
        HunyuanVideoTransformer3DModelPacked,
    )

    config = HunyuanVideoTransformer3DModelPacked.load_config(str(model_dir))
    with torch.device("meta"):
        model = HunyuanVideoTransformer3DModelPacked.from_config(config)
    keys = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        weight = module.weight
        if weight is None or weight.ndim != 2:
            continue
        if weight.shape[-1] % _BLOCK_SIZE or weight.shape[-2] % _BLOCK_SIZE:
            continue
        keys.add(f"{name}.weight" if name else "weight")
    return keys


def _read_format_id(path: Path) -> str | None:
    """Safetensors format metadata, or None when the file has none."""
    if not path.is_file():
        return None
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    except Exception:
        return None
    return metadata.get("format")


def _accepted_formats(variant: str) -> set[str]:
    return {_FORMAT_IDS[normalize_nvfp4_variant(variant)]}


def find_nvfp4_checkpoint(model_dir: Path, variant: str = NVFP4_A16) -> Path | None:
    """The packed file for ``variant``, including the older A16 filename."""
    variant = normalize_nvfp4_variant(variant)
    direct = nvfp4_checkpoint_path(model_dir, variant)
    found = _usable_checkpoint(direct, variant)
    if found is not None:
        return found
    if variant != NVFP4_A16:
        return None
    legacy = Path(model_dir) / _LEGACY_A16_FILENAME
    if legacy == direct:
        return None
    return _usable_checkpoint(legacy, NVFP4_A16)


def _usable_checkpoint(path: Path, variant: str) -> Path | None:
    if not path.is_file():
        return None
    format_id = _read_format_id(path)
    if format_id is None or format_id in _accepted_formats(variant):
        return path
    return None


def _move_legacy_a16_aside(model_dir: Path) -> None:
    """Keep an older A16 file when the NVFP4 filename is about to be reused."""
    legacy = Path(model_dir) / _LEGACY_A16_FILENAME
    if _read_format_id(legacy) != _FORMAT_IDS[NVFP4_A16]:
        return
    target = nvfp4_checkpoint_path(model_dir, NVFP4_A16)
    if target.is_file():
        return
    legacy.replace(target)
    logger.info("Moved the existing NVFP4-A16 checkpoint to %s.", target)


def _variant_for_filename(name: str) -> str | None:
    for variant, filename in _FILENAMES.items():
        if name == filename:
            return variant
    return None


def _save_packed_file(tensors: dict[str, torch.Tensor], destination: Path, format_id: str) -> None:
    from safetensors.torch import save_file

    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    save_file(
        tensors,
        str(temporary),
        metadata={
            "format": format_id,
            "block_size": str(_BLOCK_SIZE),
            "orig_dtype": "bfloat16",
            "is_swizzled_scales": "true",
            "use_dynamic_per_tensor_scale": "true",
        },
    )
    temporary.replace(destination)


def convert_checkpoint(
    model_dir: Path, force: bool = False, variant: str = NVFP4_A16
) -> Path:
    """Write the packed file for ``variant`` beside the bfloat16 weights."""
    model_dir = Path(model_dir)
    variant = normalize_nvfp4_variant(variant)
    if variant == NVFP4:
        _move_legacy_a16_aside(model_dir)
    existing = find_nvfp4_checkpoint(model_dir, variant)
    if existing is not None and not force:
        logger.info(
            "%s checkpoint already exists: %s",
            nvfp4_variant_label(variant),
            existing,
        )
        return existing

    destination = nvfp4_checkpoint_path(model_dir, variant)
    sources = original_weight_files(model_dir)
    keys = linear_weight_keys(model_dir)
    logger.info(
        "Converting %s bfloat16 file(s) to %s. %s linear weights are eligible.",
        len(sources),
        nvfp4_variant_label(variant),
        len(keys),
    )
    stats = convert_weight_files(
        sources, destination, keys, format_id=_FORMAT_IDS[variant]
    )
    logger.info(
        "Wrote %s (%s packed linears, %s bfloat16 tensors).",
        destination,
        stats["packed"],
        stats["plain"],
    )
    return destination


def convert_variants(
    model_dir: Path, variants: list[str], force: bool = False
) -> list[Path]:
    """Write one packed file per variant. One quantization pass fills every missing file."""
    model_dir = Path(model_dir)
    selected = []
    for variant in variants:
        normalized = normalize_nvfp4_variant(variant)
        if normalized not in selected:
            selected.append(normalized)
    if NVFP4 in selected:
        _move_legacy_a16_aside(model_dir)
    ready: list[Path] = []
    pending: list[str] = []
    for variant in selected:
        existing = find_nvfp4_checkpoint(model_dir, variant)
        if existing is not None and not force:
            logger.info(
                "%s checkpoint already exists: %s",
                nvfp4_variant_label(variant),
                existing,
            )
            ready.append(existing)
        else:
            pending.append(variant)
    if len(pending) == 1:
        ready.append(convert_checkpoint(model_dir, force=True, variant=pending[0]))
    elif len(pending) > 1:
        sources = original_weight_files(model_dir)
        keys = linear_weight_keys(model_dir)
        logger.info(
            "Converting %s bfloat16 file(s) to %s. %s linear weights are eligible.",
            len(sources),
            " and ".join(nvfp4_variant_label(variant) for variant in pending),
            len(keys),
        )
        copies = [
            (nvfp4_checkpoint_path(model_dir, variant), _FORMAT_IDS[variant])
            for variant in pending
        ]
        stats = convert_weight_files(
            sources,
            copies[0][0],
            keys,
            format_id=copies[0][1],
            copies=copies[1:],
        )
        logger.info(
            "Wrote %s packed linears and %s bfloat16 tensors into %s file(s).",
            stats["packed"],
            stats["plain"],
            len(copies),
        )
        ready.extend(path for path, _format_id in copies)
    order = {variant: index for index, variant in enumerate(selected)}
    ready.sort(
        key=lambda path: order.get(_variant_for_filename(path.name), len(selected))
    )
    return ready


def convert_weight_files(
    sources: list[Path],
    destination: Path,
    linear_keys: set[str],
    format_id: str = _FORMAT_IDS[NVFP4_A16],
    copies: list[tuple[Path, str]] | None = None,
) -> dict[str, int]:
    """Pack eligible weights from ``sources`` into one or more NVFP4 safetensors files."""
    tensors: dict[str, torch.Tensor] = {}
    packed = 0
    plain = 0
    seen: set[str] = set()
    for source_index, source in enumerate(sources, start=1):
        logger.info(
            "Reading %s (%s/%s).", source.name, source_index, len(sources)
        )
        from safetensors import safe_open

        with safe_open(source, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in seen:
                    raise RuntimeError(f"Duplicate checkpoint key {key}")
                seen.add(key)
                tensor = handle.get_tensor(key)
                if _should_pack(key, tensor, linear_keys):
                    qdata, scale, per_tensor_scale = pack_bf16_weight(tensor)
                    tensors[key + _QDATA] = qdata
                    tensors[key + _SCALE] = scale
                    tensors[key + _PER_TENSOR_SCALE] = per_tensor_scale
                    packed += 1
                    if packed % 25 == 0:
                        logger.info("Packed %s linear weights.", packed)
                else:
                    tensors[key] = _plain_tensor(tensor)
                    plain += 1
                del tensor

    missing = sorted(linear_keys - seen)
    if missing:
        raise RuntimeError(
            f"Checkpoint is missing {len(missing)} linear weights, including {missing[:8]}"
        )

    _save_packed_file(tensors, destination, format_id)
    for copy_path, copy_format in copies or []:
        _save_packed_file(tensors, copy_path, copy_format)
    return {"packed": packed, "plain": plain}


def pack_bf16_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize one linear weight with the same config as ``apply_nvfp4_w4a16``."""
    from torchao.prototype.mx_formats.inference_workflow import NVFP4WeightOnlyConfig
    from torchao.quantization import quantize_

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight = weight.detach().to(device=device, dtype=torch.bfloat16).contiguous()
    layer = nn.Linear(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        dtype=torch.bfloat16,
        device=device,
    )
    with torch.no_grad():
        layer.weight.copy_(weight)
        quantize_(layer, NVFP4WeightOnlyConfig(use_dynamic_per_tensor_scale=True))
    packed = layer.weight
    qdata = packed.qdata.detach().to(device="cpu").contiguous()
    scale = packed.scale.detach().to(device="cpu").contiguous()
    per_tensor_scale = packed.per_tensor_scale.detach().to(device="cpu").contiguous()
    del layer, weight, packed
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return qdata, scale, per_tensor_scale


def nvfp4_activation_kwargs(variant: str):
    """Activation quantization for NVFP4. NVFP4-A16 leaves activations in bfloat16."""
    if normalize_nvfp4_variant(variant) != NVFP4:
        return None
    from torchao.prototype.mx_formats.nvfp4_tensor import QuantizeTensorToNVFP4Kwargs

    return QuantizeTensorToNVFP4Kwargs(
        block_size=_BLOCK_SIZE,
        is_swizzled_scales=True,
        use_triton_kernel=False,
        use_dynamic_per_tensor_scale=True,
    )


def nvfp4_from_parts(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    variant: str = NVFP4_A16,
):
    """Rebuild the NVFP4 tensor stored by ``pack_bf16_weight``."""
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    if per_tensor_scale.ndim != 0:
        per_tensor_scale = per_tensor_scale.reshape(())
    return NVFP4Tensor(
        qdata.contiguous(),
        scale.contiguous(),
        _BLOCK_SIZE,
        torch.bfloat16,
        per_tensor_scale=per_tensor_scale,
        is_swizzled_scales=True,
        use_triton_kernel=False,
        act_quant_kwargs=nvfp4_activation_kwargs(variant),
    )


def load_nvfp4_weights(
    model: nn.Module, path: Path, variant: str = NVFP4_A16
) -> int:
    """Replace module parameters with tensors from an NVFP4 safetensors file."""
    from safetensors import safe_open

    path = Path(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        format_id = metadata.get("format")
        if format_id not in _accepted_formats(variant):
            raise RuntimeError(
                f"{path} is not a {nvfp4_variant_label(variant)} checkpoint "
                f"(found {format_id!r})."
            )
        grouped: dict[str, dict[str, str]] = {}
        plain_keys = []
        for key in handle.keys():
            base, kind = _split_key(key)
            if kind == "plain":
                plain_keys.append(key)
                continue
            grouped.setdefault(base, {})[kind] = key

        packed = 0
        for base, parts in grouped.items():
            missing = {"qdata", "scale", "per_tensor_scale"} - parts.keys()
            if missing:
                raise RuntimeError(f"{base} is missing NVFP4 tensors: {sorted(missing)}")
            weight = nvfp4_from_parts(
                handle.get_tensor(parts["qdata"]),
                handle.get_tensor(parts["scale"]),
                handle.get_tensor(parts["per_tensor_scale"]),
                variant=variant,
            )
            _assign_tensor(model, base, weight)
            packed += 1

        for key in plain_keys:
            _assign_tensor(model, key, _plain_tensor(handle.get_tensor(key)))

    missing_parameters = [
        name
        for name, param in model.named_parameters()
        if param is None or param.device.type == "meta"
    ]
    if missing_parameters:
        preview = ", ".join(missing_parameters[:8])
        raise RuntimeError(
            f"NVFP4 checkpoint is missing {len(missing_parameters)} parameters, including {preview}"
        )
    for module in model.modules():
        for name, buf in list(module.named_buffers(recurse=False)):
            if buf is not None and buf.device.type == "meta":
                logger.warning("Materializing missing buffer %s on CPU.", name)
                module.register_buffer(
                    name, torch.empty(buf.shape, dtype=buf.dtype, device="cpu")
                )
    return packed


def load_nvfp4_transformer(
    model_dir: Path, nvfp4_path: Path, variant: str = NVFP4_A16
):
    """Build the Hunyuan transformer on the meta device and fill it from ``nvfp4_path``."""
    from diffusers_helper.models.hunyuan_video_packed import (
        HunyuanVideoTransformer3DModelPacked,
    )

    config = HunyuanVideoTransformer3DModelPacked.load_config(str(model_dir))
    with torch.device("meta"):
        model = HunyuanVideoTransformer3DModelPacked.from_config(config)
    variant = normalize_nvfp4_variant(variant)
    packed = load_nvfp4_weights(model, nvfp4_path, variant=variant)
    model.eval()
    model.requires_grad_(False)
    logger.info(
        "Loaded %s %s linear weights from %s.",
        packed,
        nvfp4_variant_label(variant),
        nvfp4_path,
    )
    return model


def load_hunyuan_transformer(
    path: str,
    *,
    preservation_gb: float,
    offline: bool,
    force_nvfp4: bool = False,
    variant: str = NVFP4_A16,
):
    """Load the packed NVFP4 file when forced, or when bfloat16 exceeds usable VRAM.

    A missing packed file is created from the bfloat16 weights, then loaded.
    ``variant`` chooses NVFP4 or NVFP4-A16. A bfloat16 checkpoint that fits,
    with the checkbox off, stays bfloat16.
    """
    variant = normalize_nvfp4_variant(variant)
    model_dir = None
    weight_bytes = None
    usable = None
    try:
        model_dir = resolve_model_dir(path, offline=offline)
    except Exception:
        logger.exception(
            "Could not resolve the transformer directory. Loading the bfloat16 checkpoint."
        )

    if model_dir is not None:
        nvfp4_path = find_nvfp4_checkpoint(model_dir, variant)
        try:
            weight_bytes = checkpoint_nbytes(model_dir)
        except Exception:
            logger.exception("Could not measure the bfloat16 checkpoint.")
            weight_bytes = None
        usable = usable_vram_bytes(preservation_gb)
        if should_use_nvfp4_checkpoint(weight_bytes or 0, usable, force=force_nvfp4):
            if nvfp4_path is None:
                logger.info(
                    "%s checkpoint is missing. Creating it from the bfloat16 weights at %s.",
                    nvfp4_variant_label(variant),
                    nvfp4_checkpoint_path(model_dir, variant),
                )
                nvfp4_path = convert_checkpoint(model_dir, variant=variant)
            if force_nvfp4:
                logger.info(
                    "%s is forced. Loading %s.",
                    nvfp4_variant_label(variant),
                    nvfp4_path,
                )
            else:
                logger.info(
                    "bfloat16 checkpoint is %.2f GB and usable VRAM is %.2f GB. Loading %s.",
                    weight_bytes / (1024**3),
                    usable / (1024**3),
                    nvfp4_path,
                )
            return load_nvfp4_transformer(model_dir, nvfp4_path, variant=variant)
        if bf16_fits_in_vram(weight_bytes, usable) and not force_nvfp4:
            logger.info(
                "bfloat16 checkpoint is %.2f GB and fits in %.2f GB usable VRAM. "
                "Leaving the weights in bfloat16.",
                weight_bytes / (1024**3),
                usable / (1024**3),
            )

    from diffusers_helper.models.hunyuan_video_packed import (
        HunyuanVideoTransformer3DModelPacked,
    )

    source = str(model_dir) if model_dir is not None else path
    model = HunyuanVideoTransformer3DModelPacked.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
    ).cpu()
    model.eval()
    model.requires_grad_(False)
    if should_memory_quantize_bf16(force_nvfp4, weight_bytes, usable):
        model.apply_nvfp4(variant)
    return model


def _dir_has_weights(model_dir: Path) -> bool:
    try:
        return bool(original_weight_files(model_dir))
    except FileNotFoundError:
        return False


def _snapshot_from_repo_cache(repo_dir: Path) -> Path | None:
    if not repo_dir.is_dir():
        return None
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        commit = ref.read_text(encoding="utf-8").strip()
        snapshot = repo_dir / "snapshots" / commit
        if snapshot.is_dir() and _dir_has_weights(snapshot):
            return snapshot
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    found = [
        path
        for path in snapshots.iterdir()
        if path.is_dir() and _dir_has_weights(path)
    ]
    if not found:
        return None
    return max(found, key=lambda path: path.stat().st_mtime)


def _weight_index_path(model_dir: Path) -> Path | None:
    for name in (
        "diffusion_pytorch_model.safetensors.index.json",
        "model.safetensors.index.json",
    ):
        path = model_dir / name
        if path.is_file():
            return path
    return None


def _safetensors_payload_nbytes(path: Path) -> int:
    with Path(path).open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    total = 0
    for key, info in header.items():
        if key == "__metadata__":
            continue
        start, end = info["data_offsets"]
        total += end - start
    return total


def _should_pack(key: str, tensor: torch.Tensor, linear_keys: set[str]) -> bool:
    if key not in linear_keys or not tensor.is_floating_point() or tensor.ndim != 2:
        return False
    return tensor.shape[-1] % _BLOCK_SIZE == 0 and tensor.shape[-2] % _BLOCK_SIZE == 0


def _plain_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().to(device="cpu").contiguous()
    if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
        tensor = tensor.to(dtype=torch.bfloat16)
    return tensor


def _split_key(key: str) -> tuple[str, str]:
    for suffix, kind in (
        (_PER_TENSOR_SCALE, "per_tensor_scale"),
        (_QDATA, "qdata"),
        (_SCALE, "scale"),
    ):
        if key.endswith(suffix):
            return key[: -len(suffix)], kind
    return key, "plain"


def _assign_tensor(model: nn.Module, key: str, tensor: torch.Tensor) -> None:
    parent_name, _, attr = key.rpartition(".")
    try:
        module = model.get_submodule(parent_name) if parent_name else model
    except AttributeError as exc:
        raise RuntimeError(f"NVFP4 checkpoint has no module for {key}") from exc
    if attr in module._parameters:
        module.register_parameter(attr, nn.Parameter(tensor, requires_grad=False))
        return
    if attr in module._buffers:
        module.register_buffer(attr, tensor)
        return
    raise RuntimeError(f"NVFP4 checkpoint key {key} is not a parameter or buffer")
