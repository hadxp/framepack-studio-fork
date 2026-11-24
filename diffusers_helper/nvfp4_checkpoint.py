"""Save and load NVFP4-W4A16 copies of a bfloat16 Hunyuan checkpoint.

The packed file stores each eligible linear as qdata, an FP8 scale, and a
per-tensor scale. Other tensors stay bfloat16. A Kohya LoRA merge still reads
the original bfloat16 safetensors.
"""

import json
import logging
import os
import struct
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

FORMAT_ID = "framepack-nvfp4-w4a16-v1"
NVFP4_FILENAME = "diffusion_pytorch_model-nvfp4.safetensors"
_TRANSFORMER_CACHE_REPOS = (
    "models--lllyasviel--FramePackI2V_HY",
    "models--lllyasviel--FramePack_F1_I2V_HY_20250503",
)
_BLOCK_SIZE = 16
_QDATA = "::qdata"
_SCALE = "::scale"
_PER_TENSOR_SCALE = "::per_tensor_scale"


def nvfp4_checkpoint_path(model_dir: Path) -> Path:
    return Path(model_dir) / NVFP4_FILENAME


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
            if path.name.endswith("-nvfp4.safetensors"):
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
        if not path.name.endswith("-nvfp4.safetensors")
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


def convert_checkpoint(model_dir: Path, force: bool = False) -> Path:
    """Write ``diffusion_pytorch_model-nvfp4.safetensors`` beside the bfloat16 weights."""
    model_dir = Path(model_dir)
    destination = nvfp4_checkpoint_path(model_dir)
    if destination.is_file() and not force:
        logger.info("NVFP4 checkpoint already exists: %s", destination)
        return destination

    sources = original_weight_files(model_dir)
    keys = linear_weight_keys(model_dir)
    logger.info(
        "Converting %s bfloat16 file(s) to NVFP4-W4A16. %s linear weights are eligible.",
        len(sources),
        len(keys),
    )
    stats = convert_weight_files(sources, destination, keys)
    logger.info(
        "Wrote %s (%s packed linears, %s bfloat16 tensors).",
        destination,
        stats["packed"],
        stats["plain"],
    )
    return destination


def convert_weight_files(
    sources: list[Path], destination: Path, linear_keys: set[str]
) -> dict[str, int]:
    """Pack eligible weights from ``sources`` into one NVFP4 safetensors file."""
    from safetensors.torch import save_file

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

    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    save_file(
        tensors,
        str(temporary),
        metadata={
            "format": FORMAT_ID,
            "block_size": str(_BLOCK_SIZE),
            "orig_dtype": "bfloat16",
            "is_swizzled_scales": "true",
            "use_dynamic_per_tensor_scale": "true",
        },
    )
    temporary.replace(destination)
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


def nvfp4_from_parts(
    qdata: torch.Tensor, scale: torch.Tensor, per_tensor_scale: torch.Tensor
):
    """Rebuild the weight-only NVFP4 tensor stored by ``pack_bf16_weight``."""
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
        act_quant_kwargs=None,
    )


def load_nvfp4_weights(model: nn.Module, path: Path) -> int:
    """Replace module parameters with tensors from an NVFP4 safetensors file."""
    from safetensors import safe_open

    path = Path(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != FORMAT_ID:
            raise RuntimeError(
                f"{path} is not a {FORMAT_ID} checkpoint (found {metadata.get('format')!r})."
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


def load_nvfp4_transformer(model_dir: Path, nvfp4_path: Path):
    """Build the Hunyuan transformer on the meta device and fill it from ``nvfp4_path``."""
    from diffusers_helper.models.hunyuan_video_packed import (
        HunyuanVideoTransformer3DModelPacked,
    )

    config = HunyuanVideoTransformer3DModelPacked.load_config(str(model_dir))
    with torch.device("meta"):
        model = HunyuanVideoTransformer3DModelPacked.from_config(config)
    packed = load_nvfp4_weights(model, nvfp4_path)
    model.eval()
    model.requires_grad_(False)
    logger.info("Loaded %s NVFP4 linear weights from %s.", packed, nvfp4_path)
    return model


def load_hunyuan_transformer(
    path: str, *, preservation_gb: float, offline: bool, force_nvfp4: bool = False
):
    """Load the packed NVFP4 file when forced, or when bfloat16 exceeds usable VRAM.

    A missing packed file is created from the bfloat16 weights, then loaded.
    A bfloat16 checkpoint that fits, with the checkbox off, stays bfloat16.
    """
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
        nvfp4_path = nvfp4_checkpoint_path(model_dir)
        try:
            weight_bytes = checkpoint_nbytes(model_dir)
        except Exception:
            logger.exception("Could not measure the bfloat16 checkpoint.")
            weight_bytes = None
        usable = usable_vram_bytes(preservation_gb)
        if should_use_nvfp4_checkpoint(weight_bytes or 0, usable, force=force_nvfp4):
            if not nvfp4_path.is_file():
                logger.info(
                    "NVFP4 checkpoint is missing at %s. Creating it from the bfloat16 weights.",
                    nvfp4_path,
                )
                nvfp4_path = convert_checkpoint(model_dir)
            if force_nvfp4:
                logger.info("NVFP4-W4A16 is forced. Loading %s.", nvfp4_path)
            else:
                logger.info(
                    "bfloat16 checkpoint is %.2f GB and usable VRAM is %.2f GB. Loading %s.",
                    weight_bytes / (1024**3),
                    usable / (1024**3),
                    nvfp4_path,
                )
            return load_nvfp4_transformer(model_dir, nvfp4_path)
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
        model.apply_nvfp4_w4a16()
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
