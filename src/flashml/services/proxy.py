from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from flashml.errors import AppError, DependencyUnavailableError, InferenceError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProxyResult:
    status_code: int
    content: bytes
    content_type: str
    headers: dict[str, str]


class InferenceProxy:
    def __init__(self, base_url: str, *, timeout_s: float, name: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.name = name
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: object | None = None,
        data: dict | None = None,
        files: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> ProxyResult:
        try:
            response = await self._client.request(
                method,
                path,
                json=json,
                data=data,
                files=files,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            logger.exception("%s proxy transport error", self.name)
            raise DependencyUnavailableError(
                f"{self.name} worker is unavailable"
            ) from exc

        content_type = response.headers.get("content-type", "application/octet-stream")
        if response.status_code >= 400:
            self._raise_from_worker(response)
        return ProxyResult(
            status_code=response.status_code,
            content=response.content,
            content_type=content_type,
            headers={
                key: value
                for key, value in response.headers.items()
                if key.lower() in {"content-disposition", "server-timing"}
            },
        )

    def _raise_from_worker(self, response: httpx.Response) -> None:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict) and payload.get("error"):
            raise AppError(
                str(payload["error"]),
                status_code=response.status_code,
                code=str(payload.get("code", "worker_error")),
                details=payload.get("details"),
            )
        raise InferenceError(
            f"{self.name} worker returned HTTP {response.status_code}",
            details=response.text[:300],
        )
