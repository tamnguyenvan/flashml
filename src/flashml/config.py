from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RouteName = Literal["reconstruct", "interactive-segment", "segment"]
ALL_ROUTES: tuple[RouteName, ...] = (
    "reconstruct",
    "interactive-segment",
    "segment",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FLASHML_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    json_logs: bool = False
    require_cuda: bool = True
    device: str = "cuda"
    preload: bool = False
    inference_timeout_s: float = 600.0
    max_upload_bytes: int = 25 * 1024 * 1024

    enabled_routes: str = "all"
    reconstruct_url: str | None = None
    interactive_segment_url: str | None = None
    segment_url: str | None = None

    moge_model_repo: str = "Ruicheng/moge-3-vitg"
    moge_source_repo: str = "https://github.com/microsoft/MoGe.git"
    moge_source_revision: str = "74fbce054ebed49800de42d0ad0e83495065719a"
    moge_root: Path = Path("/workspace/flashml/third_party/MoGe")

    simpleclick_root: Path = Path("/workspace/flashml/third_party/SimpleClick")
    simpleclick_checkpoint: Path = Path(
        "/workspace/flashml/weights/simpleclick/cocolvis_vit_huge.pth"
    )
    simpleclick_max_points: int = 24
    simpleclick_max_longest_size: int = 800
    simpleclick_model_input_size: int = 448
    simpleclick_default_threshold: float = 0.49

    oneformer_model_id: str = "shi-labs/oneformer_ade20k_swin_large"
    oneformer_model_dir: Path = Path("/workspace/flashml/weights/oneformer")

    request_id_header: str = "X-Request-ID"

    @field_validator("reconstruct_url", "interactive_segment_url", "segment_url", mode="before")
    @classmethod
    def _empty_url_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @property
    def routes(self) -> frozenset[RouteName]:
        raw_value = self.enabled_routes.strip()
        if raw_value in {"all", "*"}:
            return frozenset(ALL_ROUTES)
        parsed: set[RouteName] = set()
        for raw in raw_value.split(","):
            name = raw.strip()
            if not name:
                continue
            if name not in ALL_ROUTES:
                raise ValueError(
                    f"Unknown FLASHML_ENABLED_ROUTES entry {name!r}; "
                    f"expected all or one of {', '.join(ALL_ROUTES)}"
                )
            parsed.add(name)  # type: ignore[arg-type]
        return frozenset(parsed)

    def is_enabled(self, route: RouteName) -> bool:
        return route in self.routes


@lru_cache
def get_settings() -> Settings:
    return Settings()
