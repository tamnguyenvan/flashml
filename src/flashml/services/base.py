from __future__ import annotations

import logging
from typing import Protocol

from flashml.schemas import ServiceStatus

logger = logging.getLogger(__name__)


class HealthAware(Protocol):
    def preload(self) -> None: ...

    def status(self) -> ServiceStatus: ...
