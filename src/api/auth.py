"""API key authentication dependency."""

import os

from fastapi import Header, HTTPException

_API_KEY = os.getenv("API_KEY", "dev-key")


def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key.")
