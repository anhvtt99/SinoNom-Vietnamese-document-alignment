"""
SerpAPI search backend.

Uses:
    GET https://serpapi.com/search?engine=google&...

Required environment variables:
    SERPAPI_KEY:
        API key for SerpAPI.

Example .env:
    SERPAPI_KEY=your_serpapi_key
"""

from typing import Any, List, Optional

import requests

from lib.config import require_env
from lib.web.searchers.base import SearchClient, SearchResult

SERPAPI_ENDPOINT = "https://serpapi.com/search"


class SerpApiSearchClient(SearchClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 20,
    ):
        self.api_key = api_key or require_env("SERPAPI_KEY")
        self.timeout = timeout

    def search(
        self,
        query: str,
        num_results: int = 10,
        **kwargs: Any,
    ) -> List[SearchResult]:
        params: dict = {
            "engine": "google",
            "q": query,
            "num": num_results,
            "api_key": self.api_key,
        }

        for key in ["gl", "hl", "location", "filter"]:
            value = kwargs.get(key)
            if value is not None:
                params[key] = value

        resp = requests.get(SERPAPI_ENDPOINT, params=params, timeout=self.timeout)
        resp.raise_for_status()

        results: List[SearchResult] = []
        for item in resp.json().get("organic_results", []):
            url = item.get("link")
            if not url:
                continue

            results.append({
                "source_block": "organic",
                "rank": item.get("position"),
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("snippet", ""),
                "raw": item,
            })

        return results
