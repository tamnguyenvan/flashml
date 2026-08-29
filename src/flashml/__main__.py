from __future__ import annotations

import argparse

import uvicorn

from flashml.config import get_settings


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the FlashML API with uvicorn")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    args = parser.parse_args()
    uvicorn.run(
        "flashml.app:app",
        host=args.host,
        port=args.port,
        factory=False,
        workers=1,
        timeout_keep_alive=75,
        log_config=None,
    )


if __name__ == "__main__":
    main()
