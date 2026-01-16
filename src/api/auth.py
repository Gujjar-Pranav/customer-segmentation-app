# src/api/auth.py
from __future__ import annotations

import os
from fastapi import Header, HTTPException

from src.main import load_config


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    cfg = load_config("config/config.yaml")
    api_cfg = cfg.get("api", {})

    if not api_cfg.get("require_api_key", False):
        return

    env_name = api_cfg.get("api_key_env", "CUSTOMER_SEG_API_KEY")
    expected = os.getenv(env_name)

    if not expected:
        raise HTTPException(
            status_code=500,
            detail=f"API key auth enabled but env var '{env_name}' is not set.",
        )

    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
