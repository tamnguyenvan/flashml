"""RORem-mixed inpainting backend for object removal (``POST /remove``).

Uses diffusers with SDXL-inpainting base + RORem-mixed UNet for high-quality inference.
RORem-mixed is trained on mixed resolution (512x512 and 1024x1024) for superior quality.
"""

from __future__ import annotations

import io
import logging
import threading
from pathlib import Path

from flashml.config import Settings
from flashml.errors import InferenceError, InvalidImageError
from flashml.schemas import ServiceStatus
from flashml.services.proxy import InferenceProxy

logger = logging.getLogger(__name__)


def _decode_rgb(image_bytes: bytes):
    from PIL import Image

    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (OSError, ValueError) as exc:
        raise InvalidImageError("file could not be decoded; use a PNG or JPEG image") from exc


def _decode_mask(mask_bytes: bytes):
    from PIL import Image

    try:
        return Image.open(io.BytesIO(mask_bytes)).convert("L")
    except (OSError, ValueError) as exc:
        raise InvalidImageError("mask could not be decoded; use a single-channel PNG") from exc


def _pil_to_png(image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _dilate_mask(mask_image, dilate_size: int):
    import cv2
    import numpy as np
    from PIL import Image

    if dilate_size <= 0:
        return mask_image

    mask = np.array(mask_image)
    mask = mask.astype(np.uint8)
    kernel = np.ones((dilate_size, dilate_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask_image = Image.fromarray(mask)
    mask_image = mask_image.convert("L")
    return mask_image


class RORemService:
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
            detail=str(self.settings.rorem_model_dir),
        )

    def remove(self, image_bytes: bytes, mask_bytes: bytes, *, max_size: int) -> bytes:
        self.preload()
        image = _decode_rgb(image_bytes)
        mask = _decode_mask(mask_bytes)

        from PIL import Image

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

        mask = _dilate_mask(mask, self.settings.rorem_dilate_size)

        with self._lock:
            result = self._infer_locked(image, mask)

        if result.size != image.size:
            result = result.resize(image.size, Image.BILINEAR)

        return _pil_to_png(result)

    def _load_locked(self) -> None:
        import torch
        from diffusers import AutoPipelineForInpainting, UNet2DConditionModel

        if self.settings.require_cuda and not torch.cuda.is_available():
            raise InferenceError("CUDA is required for RORem-mixed but is not available")

        self.device = torch.device(self.settings.device if torch.cuda.is_available() else "cpu")
        logger.info("Loading RORem-mixed on %s", self.device)

        base_model = self.settings.rorem_base_model
        unet_path = self.settings.rorem_unet_path

        pipe = AutoPipelineForInpainting.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        if unet_path and Path(unet_path).exists():
            logger.info("Loading RORem-mixed UNet from %s", unet_path)
            unet = UNet2DConditionModel.from_pretrained(unet_path).to(self.device, dtype=torch.float16)
            pipe.unet = unet
        else:
            logger.warning("RORem UNet path not found: %s, using base model UNet", unet_path)

        pipe.to(self.device)
        self.pipe = pipe
        self._ready = True
        logger.info("RORem-mixed ready on %s", self.device)

    def _infer_locked(self, image, mask):
        try:
            resolution = self.settings.rorem_resolution
            image_resized = image.resize((resolution, resolution), Image.BILINEAR)
            mask_resized = mask.resize((resolution, resolution), Image.NEAREST)

            if not self.settings.rorem_use_cfg:
                prompts = ""
                result = self.pipe(
                    prompt=prompts,
                    height=resolution,
                    width=resolution,
                    image=image_resized,
                    mask_image=mask_resized,
                    guidance_scale=1.0,
                    num_inference_steps=self.settings.rorem_num_inference_steps,
                    strength=0.99,
                ).images[0]
            else:
                prompts = "4K, high quality, masterpiece, Highly detailed, Sharp focus, Professional, photorealistic, realistic"
                negative_prompts = "low quality, worst, bad proportions, blurry, extra finger, Deformed, disfigured, unclear background"
                result = self.pipe(
                    prompt=prompts,
                    negative_prompt=negative_prompts,
                    height=resolution,
                    width=resolution,
                    image=image_resized,
                    mask_image=mask_resized,
                    guidance_scale=1.0,
                    num_inference_steps=self.settings.rorem_num_inference_steps,
                    strength=0.99,
                ).images[0]

            return result
        except Exception as exc:
            logger.exception("RORem-mixed inference failed")
            raise InferenceError("RORem-mixed inference failed") from exc


class RemoteRORemService:
    backend = "http"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._proxy = InferenceProxy(
            settings.remove_url or "",
            timeout_s=settings.inference_timeout_s,
            name="RORem-mixed",
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


def build_rorem_service(settings: Settings) -> RORemService | RemoteRORemService:
    if settings.remove_url:
        return RemoteRORemService(settings)
    return RORemService(settings)