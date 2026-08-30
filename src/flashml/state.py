from __future__ import annotations

from typing import TYPE_CHECKING

from flashml.config import Settings

if TYPE_CHECKING:
    from flashml.services.moge import MogeService, RemoteMogeService
    from flashml.services.oneformer import OneFormerService, RemoteOneFormerService
    from flashml.services.rorem import RORemService, RemoteRORemService
    from flashml.services.simpleclick import RemoteSimpleClickService, SimpleClickService


class AppState:
    settings: Settings
    moge: MogeService | RemoteMogeService | None = None
    simpleclick: SimpleClickService | RemoteSimpleClickService | None = None
    oneformer: OneFormerService | RemoteOneFormerService | None = None
    rorem: RORemService | RemoteRORemService | None = None
