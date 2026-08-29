"""Client test for ``GET /health`` and ``GET /ready``.

Validates the liveness and readiness contracts: health returns ``200`` with the
device and per-service status, and every response carries an ``X-Request-ID``.

Run:

    python e2e/test_health.py [--base-url URL] [--timeout SEC]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the editable-repo import of `e2e.common` work when run directly as
# `python e2e/test_health.py` from outside the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from e2e.common import auth_headers, build_parser


def check_health(client: httpx.Client, base_url: str, api_key: str) -> None:
    print(f"GET {base_url}/health")
    response = client.get("/health", headers=auth_headers(api_key))
    response.raise_for_status()
    body = response.json()

    assert response.status_code == 200, body
    assert "X-Request-ID" in response.headers, "missing X-Request-ID on /health"

    print(f"  status      : {body['status']}")
    print(f"  version     : {body['version']}")
    print(f"  device      : {body['device']}")
    print(f"  request id  : {response.headers['X-Request-ID']}")
    print("  services    :")
    for name, info in body["services"].items():
        print(f"    - {name}: enabled={info['enabled']} ready={info['ready']} backend={info['backend']}")
    print("  OK")


def check_ready(client: httpx.Client, base_url: str, api_key: str) -> None:
    print(f"GET {base_url}/ready")
    response = client.get("/ready", headers=auth_headers(api_key))

    if response.status_code == 200:
        status = response.json()["status"]
        assert status == "ready", f"expected ready but got {status!r}"
        print(f"  ready      : true ({status})")
    elif response.status_code == 503:
        print("  ready      : false (backends still loading or not ready)")
    else:
        raise AssertionError(f"unexpected /ready status {response.status_code}: {response.text}")

    assert "X-Request-ID" in response.headers, "missing X-Request-ID on /ready"
    print("  OK")


def main() -> int:
    parser = build_parser("Exercise /health and /ready on a running FlashML gateway.")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        try:
            check_health(client, args.base_url, args.api_key)
            check_ready(client, args.base_url, args.api_key)
        except AssertionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        except httpx.HTTPError as exc:
            print(f"REQUEST ERROR: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())