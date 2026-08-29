from __future__ import annotations

import logging
import sys
import threading

from flashml.config import Settings
from flashml.errors import InferenceError, InputValidationError
from flashml.schemas import InteractiveSegmentRequest, InteractiveSegmentResponse, ServiceStatus
from flashml.services.images import mask_png_base64, rgb_from_payload
from flashml.services.proxy import InferenceProxy

logger = logging.getLogger(__name__)


def _sample_points(points: list[list[float]], limit: int) -> list[list[int]]:
    if len(points) <= limit:
        return [[int(round(point[0])), int(round(point[1]))] for point in points]
    step = (len(points) - 1) / (limit - 1)
    indices = [round(index * step) for index in range(limit)]
    return [[int(round(points[index][0])), int(round(points[index][1]))] for index in indices]


def _validate_points(
    value: object,
    width: int,
    height: int,
    field_name: str,
    *,
    required: bool,
    max_points: int,
) -> list[list[int]]:
    if value is None and not required:
        return []
    if not isinstance(value, list) or (required and not value):
        if required:
            raise InputValidationError(f"{field_name} must contain at least one [x, y] point")
        raise InputValidationError(f"{field_name} must be a list of [x, y] points")

    points: list[list[float]] = []
    for index, point in enumerate(value):
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or not all(isinstance(coordinate, (int, float)) for coordinate in point)
        ):
            raise InputValidationError(f"{field_name}[{index}] must be [x, y]")
        x, y = float(point[0]), float(point[1])
        if not (0 <= x < width and 0 <= y < height):
            raise InputValidationError(f"{field_name}[{index}] is outside the image bounds")
        points.append([x, y])
    return _sample_points(points, max_points)


class SimpleClickService:
    backend = "local"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ready = False
        self._lock = threading.Lock()
        self.predictor = None
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
            detail=str(self.settings.simpleclick_checkpoint),
        )

    def segment(self, request: InteractiveSegmentRequest) -> InteractiveSegmentResponse:
        self.preload()
        image_rgb = rgb_from_payload(request.image, max_bytes=self.settings.max_upload_bytes)
        height, width = image_rgb.shape[:2]
        positive_points = _validate_points(
            request.positive_points,
            width,
            height,
            "positive_points",
            required=True,
            max_points=self.settings.simpleclick_max_points,
        )
        negative_points = _validate_points(
            request.negative_points,
            width,
            height,
            "negative_points",
            required=False,
            max_points=self.settings.simpleclick_max_points,
        )
        with self._lock:
            mask = self._infer_locked(image_rgb, positive_points, negative_points, request.threshold)
        return InteractiveSegmentResponse(
            mask=mask_png_base64(mask),
            mask_format="png",
            mask_shape=[height, width],
            positive_points_used=positive_points,
            negative_points_used=negative_points,
            threshold=request.threshold,
        )

    def _load_locked(self) -> None:
        root = str(self.settings.simpleclick_root)
        if root not in sys.path:
            sys.path.insert(0, root)

        import torch
        from isegm.inference import utils
        from isegm.inference.predictors import get_predictor

        if self.settings.require_cuda and not torch.cuda.is_available():
            raise InferenceError("CUDA is required for SimpleClick but is not available")

        checkpoint = self.settings.simpleclick_checkpoint
        if not checkpoint.is_file():
            raise InferenceError(f"SimpleClick checkpoint not found: {checkpoint}")

        self.device = torch.device(self.settings.device if torch.cuda.is_available() else "cpu")
        logger.info("Loading SimpleClick from %s", checkpoint)
        model = utils.load_is_model(
            str(checkpoint),
            self.device,
            eval_ritm=False,
            cpu_dist_maps=True,
        )
        self.predictor = get_predictor(
            model,
            "NoBRS",
            self.device,
            prob_thresh=self.settings.simpleclick_default_threshold,
            with_flip=True,
            zoom_in_params={
                "skip_clicks": -1,
                "target_size": (
                    self.settings.simpleclick_model_input_size,
                    self.settings.simpleclick_model_input_size,
                ),
                "expansion_ratio": 1.4,
            },
            predictor_params={"max_size": self.settings.simpleclick_max_longest_size},
        )
        self._ready = True
        logger.info("SimpleClick ready on %s", self.device)

    def _infer_locked(self, image_rgb, positive_points, negative_points, threshold):
        import torch
        from isegm.inference.clicker import Click, Clicker

        clicker = Clicker(
            init_clicks=[
                *[
                    Click(is_positive=True, coords=(point[1], point[0]))
                    for point in positive_points
                ],
                *[
                    Click(is_positive=False, coords=(point[1], point[0]))
                    for point in negative_points
                ],
            ]
        )
        try:
            with torch.inference_mode():
                self.predictor.set_input_image(image_rgb)
                probabilities = self.predictor.get_prediction(clicker)
        except Exception as exc:
            logger.exception("SimpleClick inference failed")
            raise InferenceError("SimpleClick inference failed") from exc
        return probabilities > threshold


class RemoteSimpleClickService:
    backend = "http"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._proxy = InferenceProxy(
            settings.interactive_segment_url or "",
            timeout_s=settings.inference_timeout_s,
            name="SimpleClick",
        )

    def preload(self) -> None:
        return None

    def status(self) -> ServiceStatus:
        return ServiceStatus(
            enabled=True,
            backend=self.backend,
            ready=True,
            detail=self.settings.interactive_segment_url,
        )

    async def segment(self, request: InteractiveSegmentRequest) -> InteractiveSegmentResponse:
        result = await self._proxy.request(
            "POST",
            "/interactive-segment",
            json=request.model_dump(),
        )
        return InteractiveSegmentResponse.model_validate_json(result.content)


def build_simpleclick_service(settings: Settings) -> SimpleClickService | RemoteSimpleClickService:
    if settings.interactive_segment_url:
        return RemoteSimpleClickService(settings)
    return SimpleClickService(settings)
