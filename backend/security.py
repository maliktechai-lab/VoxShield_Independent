# Production API authentication helper.
from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, status


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    expected = os.environ.get("VOXSHIELD_API_KEY", "").strip()

    # Local development mode: no configured key.
    if not expected:
        return

    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
