from __future__ import annotations

from typing import Any

import requests

from app.config import settings
from app.utils.logging import get_logger


logger = get_logger(__name__)


class OverpassClient:
    def __init__(self) -> None:
        self.api_url = settings.overpass_api_url
        self.headers = {
            "User-Agent": f"{settings.app_name}/{settings.app_build} (Locus backend prototype)"
        }

    def query(self, query: str) -> list[dict[str, Any]]:
        response = requests.post(
            self.api_url,
            data=query.encode("utf-8"),
            headers={
                **self.headers,
                "Content-Type": "text/plain; charset=utf-8",
            },
            timeout=settings.overpass_timeout_seconds,
        )
        response.raise_for_status()
        return response.json().get("elements", [])
