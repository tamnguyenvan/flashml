"""MultiMatte promptable matting backend for ``POST /matte``.

Wraps ``feyninc/multimatte`` through the ``nobg`` library (a LoRA fine-tune of
SAM 3 retrained for continuous alpha mattes). Input is a PIL image plus a text
prompt; output is a binary PNG mask obtained by thresholding the alpha matte.
"""

from __future__ import annotations

import base64
import io
import logging
import threading

from flashml.config import Settings
from flashml.errors import InferenceError, InputValidationError
from flashml.schemas import MatteRequest, MatteResponse, ServiceStatus
from flashml.services.images import pil_rgb_from_payload
from flashml.services.proxy import InferenceProxy

logger = logging.getLogger(__name__)


def _resolve_prompt(request_prompt: str | None, default_prompt: str) -> str:
    prompt = (request_prompt or "").strip() or default_prompt.strip()
    if not prompt:
        raise InputValidationError("prompt must be a non-empty string")
    if len(prompt) > 500:
        raise InputValidationError("prompt must be at most 500 characters")
    return prompt


def _resolve_threshold(request_threshold: float | None, default: float) -> float:
    threshold = default if request_threshold is None else request_threshold
    if not 0.0 < threshold < 1.0:
        raise InputValidationError("threshold must be in (0, 1)")
    return threshold


def _mask_png_base64(binary_mask) -> str:
    import numpy as np
    from PIL import Image

    mask_image = Image.fromarray((np.asarray(binary_mask).astype("uint8")) * 255, mode="L")
    output = io.BytesIO()
    mask_image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


class MultiMatteService:
    backend = "local"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ready = False
        self._lock = threading.Lock()
        self.model = None
        self.processor = None
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
            detail=self.settings.multimatte_model_id,
        )

    def segment(self, request: MatteRequest) -> MatteResponse:
        self.preload()
        image = pil_rgb_from_payload(request.image, max_bytes=self.settings.max_upload_bytes)
        prompt = _resolve_prompt(request.prompt, self.settings.multimatte_default_prompt)
        threshold = _resolve_threshold(request.threshold, self.settings.multimatte_threshold)

        longest = max(image.size)
        max_longest = self.settings.multimatte_max_longest_size
        if longest > max_longest:
            from PIL import Image

            new_size = (
                max(1, round(image.width * max_longest / longest)),
                max(1, round(image.height * max_longest / longest)),
            )
            image = image.resize(new_size, Image.BILINEAR)

        with self._lock:
            binary_mask = self._infer_locked(image, prompt, threshold)

        height, width = image.height, image.width
        if binary_mask.shape != (height, width):
            raise InferenceError("MultiMatte returned a mask with an unexpected shape")
        return MatteResponse(
            mask=_mask_png_base64(binary_mask),
            mask_format="png",
            mask_shape=[height, width],
            prompt=prompt,
            threshold=threshold,
        )

    def _load_locked(self) -> None:
        import torch
        from nobg import AutoModel, AutoProcessor

        if self.settings.require_cuda and not torch.cuda.is_available():
            raise InferenceError("CUDA is required for MultiMatte but is not available")

        self.device = torch.device(self.settings.device if torch.cuda.is_available() else "cpu")
        model_dir = self.settings.multimatte_model_dir
        source = str(model_dir) if model_dir.exists() else self.settings.multimatte_model_id
        logger.info("Loading MultiMatte from %s on %s", source, self.device)
        self.processor = AutoProcessor.from_pretrained(source)
        self.model = AutoModel.from_pretrained(source)
        self.model.to(self.device).eval()
        # Warm up the text path once so the first request pays no tokenizer cost.
        warmup_prompt = self.settings.multimatte_default_prompt.strip() or "object"
        try:
            self.model.predict(self.processor, self._blank_image(), warmup_prompt)
        except Exception as exc:
            logger.exception("MultiMatte warmup failed")
            raise InferenceError("MultiMatte warmup failed") from exc
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._ready = True
        logger.info("MultiMatte ready on %s", self.device)

    @staticmethod
    def _blank_image():
        from PIL import Image

        return Image.new("RGB", (64, 64), (0, 0, 0))

    def _infer_locked(self, image, prompt: str, threshold: float):
        try:
            alpha = self.model.predict(
                self.processor,
                image,
                prompt,
                return_type="alpha",
            )
        except Exception as exc:
            logger.exception("MultiMatte inference failed")
            raise InferenceError("MultiMatte inference failed") from exc
        return (alpha.detach().float().cpu().numpy() >= threshold).astype("uint8")


class RemoteMultiMatteService:
    backend = "http"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._proxy = InferenceProxy(
            settings.matte_url or "",
            timeout_s=settings.inference_timeout_s,
            name="MultiMatte",
        )

    def preload(self) -> None:
        return None

    def status(self) -> ServiceStatus:
        return ServiceStatus(
            enabled=True,
            backend=self.backend,
            ready=True,
            detail=self.settings.matte_url,
        )

    async def segment(self, request: MatteRequest) -> MatteResponse:
        result = await self._proxy.request(
            "POST",
            "/matte",
            json=request.model_dump(),
        )
        return MatteResponse.model_validate_json(result.content)


def build_multimatte_service(settings: Settings) -> MultiMatteService | RemoteMultiMatteService:
    if settings.matte_url:
        return RemoteMultiMatteService(settings)
    return MultiMatteService(settings)
