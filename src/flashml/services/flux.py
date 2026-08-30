"""FLUX.2 klein 4B (int8) object-removal backend for ``POST /remove``.

Uses the quantized FLUX.2-klein-4B int8 model (image-to-image editing) from
``aydin99/FLUX.2-klein-4B-int8``.

The client is responsible for marking the object on the conditioning image
(e.g. drawing a red semi-transparent overlay over it). This image is passed
straight to the model for editing.
"""

from __future__ import annotations

import io
import logging
import threading
from pathlib import Path

from PIL import Image

from flashml.config import Settings
from flashml.errors import InferenceError, InvalidImageError
from flashml.schemas import ServiceStatus
from flashml.services.proxy import InferenceProxy

logger = logging.getLogger(__name__)

_LORA_NAME = "object_remove"


def _decode_rgb(image_bytes: bytes):
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (OSError, ValueError) as exc:
        raise InvalidImageError("file could not be decoded; use a PNG or JPEG image") from exc


def _pil_to_png(image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _round_dims(width: int, height: int, multiple_of: int) -> tuple[int, int]:
    """Round dimensions up to a multiple of ``multiple_of`` (min 1)."""
    m = max(1, multiple_of)
    return max(m, ((width + m - 1) // m) * m), max(m, ((height + m - 1) // m) * m)


class FluxService:
    backend = "local"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ready = False
        self._lock = threading.Lock()
        self.pipe = None
        self.device = None

    def preload(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            self._load_locked()

    def status(self) -> ServiceStatus:
        return ServiceStatus(
            enabled=True,
            backend=self.backend,
            ready=self._ready,
            detail=str(self.settings.flux_model_dir),
        )

    def remove(self, image_bytes: bytes, *, max_size: int) -> bytes:
        self.preload()
        image = _decode_rgb(image_bytes)

        longest = max(image.size)
        if longest > max_size:
            new_size = (
                max(1, round(image.width * max_size / longest)),
                max(1, round(image.height * max_size / longest)),
            )
            image = image.resize(new_size, Image.BILINEAR)

        with self._lock:
            result = self._infer_locked(image)

        if result.size != image.size:
            result = result.resize(image.size, Image.BILINEAR)

        return _pil_to_png(result)

    def _load_locked(self) -> None:
        import importlib.util
        import json

        import torch
        from accelerate import init_empty_weights
        from diffusers import Flux2KleinPipeline
        from huggingface_hub import hf_hub_download
        from optimum.quanto import requantize
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM

        if self.settings.require_cuda and not torch.cuda.is_available():
            raise InferenceError("CUDA is required for FLUX.2 klein but is not available")

        self.device = torch.device(self.settings.device if torch.cuda.is_available() else "cpu")
        logger.info("Loading FLUX.2-klein-4B int8 on %s", self.device)

        model_path = str(self.settings.flux_model_dir)
        if not Path(model_path).exists():
            raise InferenceError(f"FLUX model not found at {model_path}. Run setup_conda.sh to download weights.")

        wrapper_path = hf_hub_download(
            "aydin99/FLUX.2-klein-4B-int8",
            "quantized_flux2.py",
        )
        spec = importlib.util.spec_from_file_location("quantized_flux2", wrapper_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        QuantizedFlux2Transformer2DModel = module.QuantizedFlux2Transformer2DModel

        qtransformer = QuantizedFlux2Transformer2DModel.from_pretrained(model_path)
        qtransformer.to(device=self.device, dtype=torch.bfloat16)

        config = AutoConfig.from_pretrained(
            f"{model_path}/text_encoder",
            trust_remote_code=True,
        )
        with init_empty_weights():
            text_encoder = Qwen3ForCausalLM(config)

        with open(f"{model_path}/text_encoder/quanto_qmap.json", "r", encoding="utf-8") as f:
            qmap = json.load(f)
        state_dict = load_file(f"{model_path}/text_encoder/model.safetensors")
        requantize(text_encoder, state_dict=state_dict, quantization_map=qmap)
        text_encoder.eval()
        text_encoder.to(self.device, dtype=torch.bfloat16)

        tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer")

        pipe = Flux2KleinPipeline.from_pretrained(
            self.settings.flux_base_model,
            transformer=None,
            text_encoder=None,
            tokenizer=None,
            torch_dtype=torch.bfloat16,
        )

        pipe.transformer = qtransformer._wrapped
        pipe.text_encoder = text_encoder
        pipe.tokenizer = tokenizer
        pipe.to(self.device)
        self._load_lora(pipe)

        self.pipe = pipe
        self._ready = True
        logger.info("FLUX.2-klein-4B ready on %s", self.device)

    def _load_lora(self, pipe) -> None:
        """Attach the object-removal LoRA to the quantised transformer.

        Applied at the transformer rather than through pipe.load_lora_weights:
        every key in this checkpoint converts to a transformer.* target, so
        there is no text-encoder half to route.

        Loading onto int8 weights needs no special handling - quanto's QLinear
        subclasses nn.Linear, so peft wraps the quantised layers directly and
        both the adapter load and the fuse work as they would on bf16.
        """

        lora_path = self.settings.flux_lora_path
        if not lora_path:
            logger.info("No LoRA configured - running the base model")
            return
        if not Path(lora_path).is_file():
            raise InferenceError(f"object-removal LoRA not found at {lora_path}")

        from diffusers import Flux2KleinPipeline

        state_dict = Flux2KleinPipeline.lora_state_dict(str(lora_path))
        if isinstance(state_dict, tuple):
            state_dict = state_dict[0]

        pipe.transformer.load_lora_adapter(state_dict, adapter_name=_LORA_NAME)
        if self.settings.flux_lora_fuse:
            # Only meaningful on an unquantised build. On the int8 weights this
            # is a silent no-op: it returns cleanly, changes nothing, and the
            # model then behaves as if no LoRA were loaded.
            logger.warning("fusing the LoRA has no effect on quantised weights")
            pipe.transformer.fuse_lora(adapter_names=[_LORA_NAME])
        logger.info(
            "object-removal LoRA loaded from %s (%d tensors, fused=%s)",
            lora_path,
            len(state_dict),
            self.settings.flux_lora_fuse,
        )

    def _infer_locked(self, conditioning):
        try:
            prompt = self.settings.flux_prompt
            steps = self.settings.flux_num_inference_steps
            guidance = self.settings.flux_guidance

            # Round the source dims to the model's required multiple so the
            # generated latents and the conditioning image agree.
            multiple_of = self.pipe.vae_scale_factor * 2 if hasattr(self.pipe, "vae_scale_factor") else 32
            width, height = _round_dims(conditioning.width, conditioning.height, multiple_of)

            result = self.pipe(
                image=conditioning,
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
            ).images[0]
            return result
        except Exception as exc:
            logger.exception("FLUX object-removal inference failed")
            raise InferenceError("FLUX object-removal inference failed") from exc


class RemoteFluxService:
    backend = "http"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._proxy = InferenceProxy(
            settings.remove_url or "",
            timeout_s=settings.inference_timeout_s,
            name="FLUX.2-klein-4B",
        )
        self._ready = True

    def preload(self) -> None:
        return None

    def status(self) -> ServiceStatus:
        return ServiceStatus(
            enabled=True,
            backend=self.backend,
            ready=self._ready,
            detail=self.settings.remove_url,
        )

    async def remove_remote(
        self,
        image_bytes: bytes,
        *,
        image_content_type: str | None,
        max_size: int,
    ) -> bytes:
        result = await self._proxy.request(
            "POST",
            "/remove",
            data={"max_size": str(max_size)},
            files={
                "file": ("image", image_bytes, image_content_type or "image/png"),
            },
        )
        return result.content


def build_flux_service(settings: Settings) -> FluxService | RemoteFluxService:
    if settings.remove_url:
        return RemoteFluxService(settings)
    return FluxService(settings)
