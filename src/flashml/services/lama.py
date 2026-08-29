"""LaMa inpainting backend for object removal (``POST /remove``).

The local path loads the ``big-lama`` TorchScript model via the
``simple-lama-inpainting`` package. The remote path simply proxies to a
dedicated LaMa GPU worker. This follows the same pattern as the other services.
"""

from __future__ import annotations

import io
import logging
import threading

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


class LamaService:
    backend = "local"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ready = False
        self._lock = threading.Lock()
        self.model = None
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
            detail=str(self.settings.lama_model_dir),
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

        with self._lock:
            result = self._infer_locked(image, mask)

        # simple-lama-inpainting pads dimensions to a multiple of 8 internally.
        # Crop back to the target image dimensions if needed.
        if result.size != image.size:
            result = result.crop((0, 0, image.width, image.height))

        return _pil_to_png(result)

    def _load_locked(self) -> None:
        import torch
        from simple_lama_inpainting import SimpleLama

        if self.settings.require_cuda and not torch.cuda.is_available():
            raise InferenceError("CUDA is required for LaMa but is not available")

        # Point the package at a local TorchScript model if we already downloaded one.
        if (self.settings.lama_model_dir / "big-lama.pt").is_file():
            import os

            os.environ["LAMA_MODEL"] = str(self.settings.lama_model_dir / "big-lama.pt")

        self.device = torch.device(self.settings.device if torch.cuda.is_available() else "cpu")
        logger.info("Loading simple-lama-inpainting big-lama on %s", self.device)
        self.model = SimpleLama(device=self.device)
        self._ready = True
        logger.info("LaMa ready on %s", self.device)

    def _infer_locked(self, image, mask):
        try:
            return self.model(image, mask)  # returns a PIL RGB image
        except Exception as exc:
            logger.exception("LaMa inference failed")
            raise InferenceError("LaMa inference failed") from exc


class RemoteLamaService:
    backend = "http"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._proxy = InferenceProxy(
            settings.remove_url or "",
            timeout_s=settings.inference_timeout_s,
            name="LaMa",
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


def build_lama_service(settings: Settings) -> LamaService | RemoteLamaService:
    if settings.remove_url:
        return RemoteLamaService(settings)
    return LamaService(settings)
