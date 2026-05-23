"""
Serper search backend.

Uses:
    https://google.serper.dev/search

Required environment variables:
    SERPER_API_KEY:
        API key for Serper.

Example .env:
    SERPER_API_KEY=your_serper_api_key
"""

from typing import Any, Dict, List, Optional

import requests

from lib.config import require_env
from lib.web.searchers.base import SearchClient, SearchResult

SERPER_SEARCH_ENDPOINT = "https://google.serper.dev/search"

class SerperSearchClient(SearchClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 20,
    ):
        self.api_key = api_key or require_env("SERPER_API_KEY")
        self.endpoint = SERPER_SEARCH_ENDPOINT
        self.timeout = timeout

    def search(
        self,
        query: str,
        num_results: int = 10,
        **kwargs: Any,
    ) -> List[SearchResult]:
        payload: Dict[str, Any] = {
            "q": query,
            "num": num_results,
        }

        # Optional Serper params
        for key in ["gl", "hl", "location", "filter", "page"]:
            value = kwargs.get(key)
            if value is not None:
                payload[key] = value

        resp = requests.post(
            self.endpoint,
            headers={
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        data = resp.json()
        results: List[SearchResult] = []

        for rank, item in enumerate(data.get("organic", []), start=1):
            url = item.get("link") or item.get("url")
            if not url:
                continue

            results.append({
                "source_block": "organic",
                "rank": rank,
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("snippet", ""),
                "raw": item,
            })

        return results