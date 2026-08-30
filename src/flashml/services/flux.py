"""FLUX.2 klein 4B (int8) object-removal backend for ``POST /remove``.

Uses the quantized FLUX.2-klein-4B int8 model (image-to-image editing) from
``aydin99/FLUX.2-klein-4B-int8``.

The binary mask region is drawn as a visible highlight overlay on the source
image and the model is prompted to remove the highlighted object, relying on
FLUX.2 klein's built-in editing capability (no LoRA).
"""

from __future__ import annotations

import io
import logging
import threading
from pathlib import Path

from PIL import Image, ImageFilter

from flashml.config import Settings
from flashml.errors import InferenceError, InvalidImageError
from flashml.schemas import ServiceStatus
from flashml.services.proxy import InferenceProxy

logger = logging.getLogger(__name__)


def _decode_rgb(image_bytes: bytes):
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (OSError, ValueError) as exc:
        raise InvalidImageError("file could not be decoded; use a PNG or JPEG image") from exc


def _decode_mask(mask_bytes: bytes):
    try:
        return Image.open(io.BytesIO(mask_bytes)).convert("L")
    except (OSError, ValueError) as exc:
        raise InvalidImageError("mask could not be decoded; use a single-channel PNG") from exc


def _pil_to_png(image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _dilate_mask(mask_image, dilate_size: int):
    if dilate_size <= 0:
        return mask_image

    if dilate_size % 2 == 0:
        dilate_size += 1

    mask_image = mask_image.convert("L")
    mask_image = mask_image.filter(ImageFilter.MaxFilter(dilate_size))
    return mask_image


def _highlight_mask(image, mask, alpha: float = 0.6):
    """Return a copy of ``image`` with the mask region tinted with a visible overlay."""
    tint = Image.new("RGB", image.size, (255, 0, 0))
    mask_l = mask.convert("L").resize(image.size, Image.NEAREST)
    return Image.composite(
        Image.blend(image, tint, alpha),
        image,
        mask_l,
    )


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

    def remove(self, image_bytes: bytes, mask_bytes: bytes, *, max_size: int) -> bytes:
        self.preload()
        image = _decode_rgb(image_bytes)
        mask = _decode_mask(mask_bytes)

        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)

        longest = max(image.size)
        if longest > max_size:
            new_size = (
                max(1, round(image.width * max_size / longest)),
                max(1, round(image.height * max_size / longest)),
            )
            image = image.resize(new_size, Image.BILINEAR)
            mask = mask.resize(new_size, Image.NEAREST)

        mask = _dilate_mask(mask, self.settings.flux_dilate_size)
        highlighted = _highlight_mask(image, mask, alpha=self.settings.flux_highlight_alpha)

        with self._lock:
            result = self._infer_locked(image, highlighted)

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

        self.pipe = pipe
        self._ready = True
        logger.info("FLUX.2-klein-4B ready on %s", self.device)

    def _infer_locked(self, image, highlighted):
        try:
            prompt = self.settings.flux_prompt
            steps = self.settings.flux_num_inference_steps
            guidance = self.settings.flux_guidance

            # Round the source dims to the model's required multiple so the
            # generated latents and the conditioning image agree.
            multiple_of = self.pipe.vae_scale_factor * 2 if hasattr(self.pipe, "vae_scale_factor") else 32
            width, height = _round_dims(highlighted.width, highlighted.height, multiple_of)

            result = self.pipe(
                image=highlighted,
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
        mask_bytes: bytes,
        *,
        image_content_type: str | None,
        mask_content_type: str | None,
        max_size: int,
    ) -> bytes:
        result = await self._proxy.request(
            "POST",
            "/remove",
            data={"max_size": str(max_size)},
            files={
                "file": ("image", image_bytes, image_content_type or "image/png"),
                "mask": ("mask.png", mask_bytes, mask_content_type or "image/png"),
            },
        )
        return result.content


def build_flux_service(settings: Settings) -> FluxService | RemoteFluxService:
    if settings.remove_url:
        return RemoteFluxService(settings)
    return FluxService(settings)
