import logging
import torch
import os  # required for os.path
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import List, Optional, cast
from pathlib import Path
from transformers import BitsAndBytesConfig

from diffusers_helper.memory import DynamicSwapInstaller
from diffusers_helper.streaming import BlockStreamer
from diffusers_helper.models.hunyuan_video_packed import (
    HunyuanVideoTransformer3DModelPacked,
)

from ..settings import Settings
from .model_configuration import ModelConfiguration
from modules import vram_mode
from shared import QuantizationFormat

logger = logging.getLogger(__name__)

# cSpell: ignore loras


class BaseModelGenerator(ABC):
    """
    Base class for model generators.
    This defines the common interface that all model generators must implement.
    """

    quantization_format: QuantizationFormat = QuantizationFormat.DEFAULT
    quantization_config: BitsAndBytesConfig | None = None

    def __init__(
        self,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        vae,
        image_encoder,
        feature_extractor,
        high_vram=False,
        prompt_embedding_cache=None,
        settings: Settings | None = None,
        offline=False,
    ):  # NEW: offline flag
        """
        Initialize the base model generator.

        Args:
            text_encoder: The text encoder model
            text_encoder_2: The second text encoder model
            tokenizer: The tokenizer for the first text encoder
            tokenizer_2: The tokenizer for the second text encoder
            vae: The VAE model
            image_encoder: The image encoder model
            feature_extractor: The feature extractor
            high_vram: Whether high VRAM mode is enabled
            prompt_embedding_cache: Cache for prompt embeddings
            settings: Application settings
            offline: Whether to run in offline mode for model loading
        """
        self.model_name: str
        self.model_path: str
        self.model_repo_id_for_cache: str

        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.vae = vae
        self.image_encoder = image_encoder
        self.feature_extractor = feature_extractor
        self.high_vram = high_vram
        self.prompt_embedding_cache = prompt_embedding_cache or {}
        self.settings: Settings = settings if settings is not None else Settings()
        self.offline = offline
        self.transformer = None
        # Set by place_transformer in low VRAM mode with offload_engine "streamed"
        self.block_streamer: BlockStreamer | None = None
        self.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")

        # quantization is currently global, configured in settings
        # maybe add kwargs if we need this to be more dynamic per job?
        self.quantization_format = self.settings.get(
            "quantization_format", QuantizationFormat.integer_8bit
        )
        if (
            not self.settings.nvfp4_w4a16
            and vram_mode.transformer_stays_on_gpu()
            and self.quantization_format
            == QuantizationFormat.brain_floating_point_16bit
        ):
            # bf16 (26 GB) can't stay on the GPU; fp8 storage (13 GB) can.
            # The NVFP4 checkbox is its own weight format and skips this rewrite.
            self.quantization_format = QuantizationFormat.float8_storage
        self.set_quantization_config()

        self.previous_model_hash: str = ""
        self.previous_model_configuration: ModelConfiguration | None = None

    def set_quantization_config(self):
        if self.quantization_format == QuantizationFormat.brain_floating_point_16bit:
            # BF16 does not require a special config
            pass
        if self.quantization_format == QuantizationFormat.normal_float_4bit:
            # 4-bit NF4 quantization config
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="bfloat16",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        if self.quantization_format == QuantizationFormat.integer_8bit:
            # 8-bit integer quantization config
            self.quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    def apply_weight_storage(self):
        """
        With float8_storage, keep the transformer's weights in fp8 and cast
        each layer to bf16 only while it runs (norms and embeddings stay
        bf16). The model then takes about 13 GB instead of 26 GB.
        """
        if self.quantization_format != QuantizationFormat.float8_storage:
            return
        self.transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
        )
        logger.info(f"{self.model_name} Transformer weights stored in fp8.")

    def place_transformer(self):
        """
        High VRAM and medium modes keep the whole transformer on the GPU.
        Low mode streams it from system RAM: block by block with
        offload_engine "streamed", otherwise weight by weight as it is used.
        """
        # A second load_model on this generator would hook the blocks twice
        self.release_transformer()
        if self.high_vram or vram_mode.transformer_stays_on_gpu():
            self.transformer.to(device=self.gpu)
            return
        if self.settings.get("offload_engine", "legacy") == "streamed":
            if self._has_nvfp4_weights():
                logger.info(
                    "NVFP4 weights use the legacy offload. Block streaming moves plain tensors."
                )
            elif self.quantization_format in (
                QuantizationFormat.brain_floating_point_16bit,
                QuantizationFormat.float8_storage,
            ):
                self.block_streamer = BlockStreamer(self.transformer, self.gpu)
                return
            else:
                # bitsandbytes weights carry state outside .data and can't be
                # moved as plain tensors
                logger.warning(
                    f"Block streaming needs bf16 or float8_storage weights, not "
                    f"{self.quantization_format}; using the legacy offload."
                )
        DynamicSwapInstaller.install_model(self.transformer, device=self.gpu)

    def release_transformer(self):
        """Stop streaming and unpin before this generator is dropped."""
        if self.block_streamer is not None:
            self.block_streamer.close()
            self.block_streamer = None

    def _has_nvfp4_weights(self) -> bool:
        if self.transformer is None:
            return False
        return any(
            type(param).__name__ == "NVFP4Tensor"
            for param in self.transformer.parameters()
        )

    def use_nvfp4_loader(self) -> bool:
        """NVFP4 replaces the dev loader when forced, or when bfloat16 does not fit.

        A missing packed file is created during that load.
        """
        from diffusers_helper.nvfp4_checkpoint import (
            checkpoint_nbytes,
            resolve_model_dir,
            should_use_nvfp4_checkpoint,
            usable_vram_bytes,
        )

        if self.settings.nvfp4_w4a16:
            return True
        path_to_load = self.model_path
        if self.offline:
            path_to_load = self._get_offline_load_path()
        preservation_gb = float(self.settings.get("gpu_memory_preservation", 6) or 0)
        try:
            model_dir = resolve_model_dir(path_to_load, offline=bool(self.offline))
            weight_bytes = checkpoint_nbytes(model_dir)
        except Exception:
            logger.exception(
                "Could not measure the bfloat16 checkpoint. Using the dev loader."
            )
            return False
        return should_use_nvfp4_checkpoint(
            weight_bytes,
            usable_vram_bytes(preservation_gb),
            force=False,
        )

    def _load_transformer_weights(self) -> HunyuanVideoTransformer3DModelPacked:
        """Load a packed NVFP4 file, or quantize bfloat16 when the checkbox requires it."""
        from diffusers_helper.nvfp4_checkpoint import load_hunyuan_transformer

        path_to_load = self.model_path
        if self.offline:
            path_to_load = self._get_offline_load_path()
        preservation_gb = float(self.settings.get("gpu_memory_preservation", 6) or 0)
        model = load_hunyuan_transformer(
            path_to_load,
            preservation_gb=preservation_gb,
            offline=bool(self.offline),
            force_nvfp4=self.settings.nvfp4_w4a16,
            variant=self.settings.nvfp4_variant,
        )
        self.should_memory_quantize_bf16(refresh=True)
        return model

    def load_packed_transformer(self) -> HunyuanVideoTransformer3DModelPacked:
        """Dev weight load, unless NVFP4 should supply the transformer instead."""
        if self.use_nvfp4_loader():
            self.transformer = self._load_transformer_weights()
            return self.transformer

        path_to_load = self.model_path
        if self.offline:
            path_to_load = self._get_offline_load_path()
        self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
            path_to_load,
            torch_dtype=torch.bfloat16,
            quantization_config=self.quantization_config,
        ).cpu()
        self.transformer.eval()
        if self.quantization_format in (
            QuantizationFormat.brain_floating_point_16bit,
            QuantizationFormat.float8_storage,
        ):
            self.transformer.to(dtype=torch.bfloat16)
        self.transformer.requires_grad_(False)
        self.apply_weight_storage()
        self._memory_quantize_bf16 = False
        return self.transformer

    def should_memory_quantize_bf16(self, *, refresh: bool = False) -> bool:
        """True when a loaded bfloat16 transformer should be packed in memory.

        The checkbox off and a checkpoint that fits in usable VRAM stays bfloat16.
        LoRA merges pass ``refresh=True`` so they use the same rule at merge time.
        """
        cached = getattr(self, "_memory_quantize_bf16", None)
        if cached is not None and not refresh:
            return cached
        from diffusers_helper.nvfp4_checkpoint import (
            checkpoint_nbytes,
            resolve_model_dir,
            should_memory_quantize_bf16,
            usable_vram_bytes,
        )

        if self.settings.nvfp4_w4a16:
            self._memory_quantize_bf16 = True
            return True
        path_to_load = self.model_path
        if self.offline:
            path_to_load = self._get_offline_load_path()
        preservation_gb = float(self.settings.get("gpu_memory_preservation", 6) or 0)
        weight_bytes = None
        try:
            model_dir = resolve_model_dir(path_to_load, offline=bool(self.offline))
            weight_bytes = checkpoint_nbytes(model_dir)
        except Exception:
            logger.exception("Could not measure the bfloat16 checkpoint.")
        decision = should_memory_quantize_bf16(
            False, weight_bytes, usable_vram_bytes(preservation_gb)
        )
        self._memory_quantize_bf16 = decision
        return decision

    def apply_selected_nvfp4(self) -> int:
        """Compress the loaded bfloat16 weights to the NVFP4 format in settings."""
        return self.transformer.apply_nvfp4(self.settings.nvfp4_variant)

    @abstractmethod
    def load_model(self) -> HunyuanVideoTransformer3DModelPacked:
        """
        Load the transformer model.
        """
        # this load_model function has the same implementation in all subclasses
        # candidate for consolidation to directly implement here in the base class
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the model.
        This method should be implemented by each specific model generator.
        """
        # this get_model_name function has the same implementation in all subclasses
        # candidate for consolidation to directly implement here in the base class
        pass

    @abstractmethod
    def get_latent_paddings(self, total_latent_sections) -> list[int]:
        raise NotImplementedError(
            "get_latent_paddings must be implemented by the specific model generator subclass."
        )

    @abstractmethod
    def format_position_description(
        self, total_generated_latent_frames, current_pos, original_pos, current_prompt
    ) -> str:
        raise NotImplementedError(
            "format_position_description must be implemented by the specific model generator subclass."
        )

    @abstractmethod
    def get_real_history_latents(
        self, history_latents: torch.Tensor, total_generated_latent_frames: int
    ) -> torch.Tensor:
        """
        Get the real history latents by slicing the history latents tensor.
        """
        raise NotImplementedError(
            "get_real_history_latents must be implemented by the specific model generator subclass."
        )

    @abstractmethod
    def update_history_latents(
        self, history_latents: torch.Tensor, generated_latents: torch.Tensor
    ) -> torch.Tensor:
        """
        Update the history latents with the generated latents.
        This method should be implemented by each specific model generator.

        Args:
            history_latents: The history latents
            generated_latents: The generated latents

        Returns:
            The updated history latents
        """
        raise NotImplementedError(
            "update_history_latents must be implemented by the specific model generator subclass."
        )

    @staticmethod
    def _get_snapshot_hash_from_refs(model_repo_id_for_cache: str) -> str | None:
        """
        Reads the commit hash from the refs/main file for a given model in the HF cache.
        Args:
            model_repo_id_for_cache (str): The model ID formatted for cache directory names
                                           (e.g., "models--lllyasviel--FramePackI2V_HY").
        Returns:
            str: The commit hash if found, otherwise None.
        """
        hf_home_dir = os.environ.get("HF_HOME")
        if not hf_home_dir:
            logger.warning(
                "HF_HOME environment variable not set. Cannot determine snapshot hash."
            )
            return None

        refs_main_path = os.path.join(
            hf_home_dir, "hub", model_repo_id_for_cache, "refs", "main"
        )
        if os.path.exists(refs_main_path):
            try:
                with open(refs_main_path, "r") as f:
                    logger.debug(
                        f"Offline mode: Reading snapshot hash from: {refs_main_path}"
                    )
                    return f.read().strip()
            except Exception as e:
                logger.warning(
                    f"Could not read snapshot hash from {refs_main_path}: {e}"
                )
                return None
        else:
            logger.warning(
                f"refs/main file not found at {refs_main_path}. Cannot determine snapshot hash."
            )
            return None

    def _get_offline_load_path(self) -> str:
        """
        Returns the local snapshot path for offline loading if available.
        Falls back to the default self.model_path if local snapshot can't be found.
        Relies on self.model_repo_id_for_cache and self.model_path being set by subclasses.
        """
        # Ensure necessary attributes are set by the subclass
        if (
            not hasattr(self, "model_repo_id_for_cache")
            or not self.model_repo_id_for_cache
        ):
            logger.warning(
                f"model_repo_id_for_cache not set in {self.__class__.__name__}. Cannot determine offline path."
            )
            # Fallback to model_path if it exists, otherwise None
            return str(getattr(self, "model_path", None))

        if not hasattr(self, "model_path") or not self.model_path:
            logger.warning(
                f"model_path not set in {self.__class__.__name__}. Cannot determine fallback for offline path."
            )
            # raise error instead of returning None?
            # raise ValueError(f"{self.__class__.__name__} must set model_path for offline loading.")
            return None

        snapshot_hash = self._get_snapshot_hash_from_refs(self.model_repo_id_for_cache)
        hf_home = os.environ.get("HF_HOME")

        if snapshot_hash and hf_home:
            specific_snapshot_path = os.path.join(
                hf_home, "hub", self.model_repo_id_for_cache, "snapshots", snapshot_hash
            )
            if os.path.isdir(specific_snapshot_path):
                return specific_snapshot_path

        # If snapshot logic fails or path is not a dir, fallback to the default model path
        return self.model_path

    def unload_loras(self):
        """LoRA Ready bakes the weights into the checkpoint, so nothing is detached here.

        The next load_loras call rebuilds the merged state dict from the base files.
        """
        return

    def __compute_lora_state_hash(self, lora_config: ModelConfiguration) -> str:
        """
        Compute a simple hash representing the current state of LoRA adapters in the transformer.
        This can be used to detect changes in loaded LoRAs.
        """
        import hashlib

        # md5 should be sufficient for this purpose
        m = hashlib.md5()

        if self.transformer is None:
            # Should not happen - return a unique value
            logger.warning("Transformer is None when computing LoRA state hash.")
            from time import time

            m.update(str(time() * 1000).encode("utf-8"))

        import json

        m.update(json.dumps(asdict(lora_config), sort_keys=True).encode("utf-8"))
        return m.hexdigest()

    def load_loras(
        self,
        selected_loras: List[str],
        lora_folder: str,
        lora_loaded_names: List[str],
        lora_values: Optional[List[float]] = None,
    ):
        """
        Load LoRAs into the transformer model and applies their weights.

        Args:
            selected_loras: List of LoRA base names to load (e.g., ["lora_A", "lora_B"]).
            lora_folder: Path to the folder containing the LoRA files.
            lora_loaded_names: The master list of ALL available LoRA names, used for correct weight indexing.
            lora_values: A list of strength values corresponding to lora_loaded_names.
        """
        if not selected_loras:
            # Only unload at this point if no LoRAs are selected
            self.unload_loras()
            logger.info("No LoRAs selected, skipping loading.")
            return

        if self.transformer is None:
            logger.error("Transformer model is None, cannot load LoRAs.")
            return

        if lora_values is None:
            lora_values = []

        selected_lora_values = (
            lora_values
            if len(selected_loras) == len(lora_values)
            else [
                lora_values[lora_loaded_names.index(name)]
                for name in selected_loras
                if name in lora_loaded_names
            ]
        )
        logger.info(
            f"Loading LoRAs: {selected_loras} with values: {selected_lora_values}"
        )

        active_model_configuration = ModelConfiguration.from_lora_names_and_weights(
            self.get_model_name(),
            self.quantization_format,
            selected_loras,
            selected_lora_values,
            self.settings.lora_loader,
            nvfp4_w4a16=self.settings.nvfp4_w4a16,
            nvfp4_variant=self.settings.nvfp4_variant,
        )

        active_model_hash = self.__compute_lora_state_hash(active_model_configuration)
        if active_model_hash == self.previous_model_hash:
            # This can only happen if the model is not changed
            # When the model is loaded we will always have the default previous_model_hash value
            # The only time that this can happen is when settings.reuse_model_instance is True
            # and the model is not changed, and the LoRAs are not changed.
            logger.info("Model configuration unchanged, skipping reload.")
            return

        logger.debug(
            f"Previous LoRA config: {self.previous_model_configuration}, Current LoRA config: {active_model_configuration}"
        )
        logger.debug(
            f"Previous LoRA hash: {self.previous_model_hash}, Current LoRA hash: {active_model_hash}"
        )

        self.previous_model_hash = active_model_hash
        self.previous_model_configuration = active_model_configuration

        lora_dir = Path(lora_folder)

        from diffusers_helper.lora_utils_kohya_ss.lora_loader import (
            load_and_apply_lora,
        )

        logger.info(f"Loading LoRAs using kohya_ss LoRAReady loader from {lora_dir}")

        def _find_model_files(model_path):
            """Get state dictionary file from specified model path
            This is undesirable as it depends on Diffusers implementation."""
            import glob

            model_root = os.environ["HF_HOME"]  # './hf_download'?
            subdir = os.path.join(
                model_root, "hub", "models--" + model_path.replace("/", "--")
            )
            model_files = glob.glob(
                os.path.join(subdir, "**", "*.safetensors"), recursive=True
            ) + glob.glob(os.path.join(subdir, "**", "*.pt"), recursive=True)
            from diffusers_helper.nvfp4_checkpoint import is_packed_nvfp4_filename

            model_files = [
                path
                for path in model_files
                if not is_packed_nvfp4_filename(os.path.basename(path))
            ]
            model_files.sort()
            return model_files

        pack_after_lora = self.settings.nvfp4_w4a16 or self._has_nvfp4_weights()
        try:
            model_files = _find_model_files(self.model_path)
            logger.debug(f"LoRA -> Found model files: {model_files}")
            lora_paths = [
                # not sure why the full path is not passed around and potentially trimmed for the interface display
                str(lora_dir / f"{lora_setting.name}.safetensors")
                if Path(lora_dir / f"{lora_setting.name}.safetensors").exists()
                else str(
                    lora_dir / f"{lora_setting.name}.pt"
                )  # hopefully .pt is the correct extension.
                for lora_setting in active_model_configuration.settings.lora_settings
            ]
            lora_scales: list[float] = [
                lora_setting.weight
                for lora_setting in active_model_configuration.settings.lora_settings
            ]
            logger.debug(f"Lora paths: {lora_paths}")
            if not lora_paths:
                raise ValueError("No valid LoRA paths found for the selected LoRAs.")

            if pack_after_lora and self.settings.get("fp8", False):
                logger.info(
                    "Skipping Kohya fp8 optimization. The merged state dict stays bfloat16, then %s is applied when it does not fit.",
                    self.settings.nvfp4_variant,
                )
            state_dict = load_and_apply_lora(
                model_files=model_files,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
                fp8_enabled=False
                if pack_after_lora
                else cast(bool, self.settings.get("fp8", False)),
                device=self.gpu if torch.cuda.is_available() else self.cpu,
            )
            quantize_merged = pack_after_lora and self.should_memory_quantize_bf16(
                refresh=True
            )
            for key, value in list(state_dict.items()):
                if not torch.is_tensor(value):
                    continue
                if (
                    pack_after_lora
                    and torch.is_floating_point(value)
                    and value.dtype != torch.bfloat16
                ):
                    value = value.to(dtype=torch.bfloat16)
                if quantize_merged and torch.cuda.is_available():
                    value = value.to(device=self.gpu)
                state_dict[key] = value
            if quantize_merged and torch.cuda.is_available():
                logger.info(
                    "Moved merged bfloat16 LoRA weights to CUDA before %s.",
                    self.settings.nvfp4_variant,
                )
            logger.debug("Loading state dict into transformer...")
            # With fp8 storage, copy into the existing fp8 weights (cast,
            # and kept on whatever device they are on); assigning would
            # swap in the merged bf16 tensors from system RAM.
            # NVFP4 must assign, because the packed weights are not fp8 parameters.
            missing_keys, unexpected_keys = self.transformer.load_state_dict(
                state_dict,
                assign=pack_after_lora
                or self.quantization_format != QuantizationFormat.float8_storage,
                strict=True,
            )

            if missing_keys:
                logger.warning(
                    f"Missing keys when loading LoRA state dict: {missing_keys}"
                )
            if unexpected_keys:
                logger.warning(
                    f"Unexpected keys when loading LoRA state dict: {unexpected_keys}"
                )

            state_dict_size: int = 0
            try:
                state_dict_size = sum(
                    param.numel() * param.element_size()
                    for param in state_dict.values()
                    if hasattr(param, "numel")
                )
                logger.debug(
                    f"State dictionary size: {state_dict_size / (1024**3):.2f} GB"
                )
            except Exception:
                pass

            try:
                del state_dict
                import gc

                gc.collect()
                logger.debug(
                    f"Freed state dictionary size: {state_dict_size / (1024**3):.2f} GB"
                )
            except Exception:
                logger.warning("Could not free state dictionary from memory.")

            if quantize_merged:
                self.apply_selected_nvfp4()
            elif pack_after_lora:
                logger.info(
                    "bfloat16 weights fit in usable VRAM. Leaving the LoRA merge in bfloat16."
                )

        except Exception as e:
            logger.exception(f"Error loading LoRAs with kohya_ss LoRAReady loader: {e}")
