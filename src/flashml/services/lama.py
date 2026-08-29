"""LaMa inpainting backend for object removal (``POST /remove``).

The local path loads the LaMa big model from the ``iopaint``/``lama-cleaner``
package, which must be installed in the dedicated ``flashml-lama`` conda env
(see ``envs/environment-lama.yml`` and ``scripts/setup_conda.sh``). The remote
path simply proxies to a dedicated LaMa GPU worker.
"""

from __future__ import annotations

import logging
import threading

from flashml.config import Settings
from flashml.errors import InferenceError, InvalidImageError
from flashml.schemas import ServiceStatus
from flashml.services.proxy import InferenceProxy

logger = logging.getLogger(__name__)


def _decode_bgr(image_bytes: bytes, *, field: str):
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError(f"{field} could not be decoded; use a PNG or JPEG image")
    return image


def _decode_mask(mask_bytes: bytes, *, field: str = "mask"):
    import cv2
    import numpy as np

    mask = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise InvalidImageError(f"{field} could not be decoded; use a single-channel PNG")
    return mask


def _mask_png(mask) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError("failed to encode inpainted result")
    return buffer.tobytes()


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

    def remove(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        *,
        max_size: int,
    ) -> bytes:
        self.preload()
        image_bgr = _decode_bgr(image_bytes, field="file")
        mask = _decode_mask(mask_bytes)

        height, width = mask.shape[:2]
        longest = max(height, width)
        scale = max_size / longest if longest > max_size else 1.0
        if scale < 1.0:
            import cv2

            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image_bgr = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)

        with self._lock:
            return self._infer_locked(image_bgr, mask)

    def _load_locked(self) -> None:
        import torch

        if self.settings.require_cuda and not torch.cuda.is_available():
            raise InferenceError("CUDA is required for LaMa but is not available")

        self.device = torch.device(self.settings.device if torch.cuda.is_available() else "cpu")
        try:
            from iopaint import LaMa
        except ImportError:
            from lama_cleaner import LaMa
        except Exception as exc:  # pragma: no cover - defensive
            raise InferenceError(
                "iopaint (lama-cleaner) is not installed in the flashml-lama environment"
            ) from exc

        logger.info("Loading LaMa big model from %s", self.settings.lama_model_dir)
        self.model = LaMa(device=self.device, model_dir=str(self.settings.lama_model_dir))
        self._ready = True
        logger.info("LaMa ready on %s", self.device)

    def _infer_locked(self, image_bgr, mask) -> bytes:
        import numpy as np

        try:
            result = self.model(image_bgr, mask)
        except Exception as exc:
            logger.exception("LaMa inference failed")
            raise InferenceError("LaMa inference failed") from exc
        result = np.asarray(result)
        if result.dtype != np.uint8:
            result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        if result.ndim == 2:
            result = np.repeat(result[..., None], 3, axis=2)
        return _mask_png(result)


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
