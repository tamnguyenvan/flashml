from __future__ import annotations

import logging
import threading

from flashml.config import Settings
from flashml.errors import InferenceError
from flashml.schemas import SegmentRequest, SegmentResponse, ServiceStatus, SurfaceMask
from flashml.services.images import mask_png_data_url, pil_rgb_from_payload
from flashml.services.proxy import InferenceProxy

logger = logging.getLogger(__name__)

TARGET_ALIASES = {
    "wall": ("wall",),
    "floor": ("floor",),
    "rug": ("rug", "carpet"),
}


class OneFormerService:
    backend = "local"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ready = False
        self._lock = threading.Lock()
        self.processor = None
        self.model = None
        self.device = None
        self.id_to_label: dict[int, str] = {}

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
            detail=self.settings.oneformer_model_id,
        )

    def segment(self, request: SegmentRequest) -> SegmentResponse:
        self.preload()
        image = pil_rgb_from_payload(request.image, max_bytes=self.settings.max_upload_bytes)
        with self._lock:
            return self._infer_locked(image)

    @staticmethod
    def _normalize_label(value: object) -> str:
        return " ".join(str(value).lower().replace("_", " ").split())

    def _to_device(self, inputs):
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    @staticmethod
    def _blank_image():
        from PIL import Image

        return Image.new("RGB", (64, 64), (0, 0, 0))

    def _load_locked(self) -> None:
        import torch
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

        if self.settings.require_cuda and not torch.cuda.is_available():
            raise InferenceError("CUDA is required for OneFormer but is not available")

        self.device = torch.device(self.settings.device if torch.cuda.is_available() else "cpu")
        model_dir = self.settings.oneformer_model_dir
        source = str(model_dir) if model_dir.exists() else self.settings.oneformer_model_id
        logger.info("Loading OneFormer from %s", source)
        self.processor = OneFormerProcessor.from_pretrained(source)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(source)
        self.model.to(self.device).eval()
        self.id_to_label = {
            int(index): self._normalize_label(label)
            for index, label in self.model.config.id2label.items()
        }
        warmup = self.processor(
            images=self._blank_image(),
            task_inputs=["semantic"],
            return_tensors="pt",
        )
        warmup = self._to_device(warmup)
        with torch.inference_mode():
            self.model(**warmup)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._ready = True
        logger.info("OneFormer ready on %s", self.device)

    def _infer_locked(self, image) -> SegmentResponse:
        import torch

        inputs = self.processor(
            images=image,
            task_inputs=["semantic"],
            return_tensors="pt",
        )
        inputs = self._to_device(inputs)
        try:
            with torch.inference_mode():
                outputs = self.model(**inputs)
            semantic_map = self.processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[image.size[::-1]],
            )[0]
        except Exception as exc:
            logger.exception("OneFormer inference failed")
            raise InferenceError("OneFormer inference failed") from exc

        predicted = semantic_map.detach().cpu().numpy()
        masks: dict[str, list[SurfaceMask]] = {}
        label_ids: dict[str, list[int]] = {}
        for target, aliases in TARGET_ALIASES.items():
            ids = [
                index for index, label in self.id_to_label.items() if label in aliases
            ]
            label_ids[target] = ids
            mask = (predicted[..., None] == ids).any(axis=-1) if ids else predicted == -1
            items: list[SurfaceMask] = []
            if mask.any():
                items.append(SurfaceMask(mask=mask_png_data_url(mask), score=None, box=None))
            masks[target] = items

        return SegmentResponse(
            model=self.settings.oneformer_model_id,
            provider="oneformer",
            image_size_hw=[image.height, image.width],
            label_ids=label_ids,
            masks=masks,
        )


class RemoteOneFormerService:
    backend = "http"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._proxy = InferenceProxy(
            settings.segment_url or "",
            timeout_s=settings.inference_timeout_s,
            name="OneFormer",
        )

    def preload(self) -> None:
        return None

    def status(self) -> ServiceStatus:
        return ServiceStatus(
            enabled=True,
            backend=self.backend,
            ready=True,
            detail=self.settings.segment_url,
        )

    async def segment(self, request: SegmentRequest) -> SegmentResponse:
        result = await self._proxy.request(
            "POST",
            "/segment",
            json=request.model_dump(),
        )
        return SegmentResponse.model_validate_json(result.content)


def build_oneformer_service(settings: Settings) -> OneFormerService | RemoteOneFormerService:
    if settings.segment_url:
        return RemoteOneFormerService(settings)
    return OneFormerService(settings)
