"""
Base interface for web search backends.
"""

from typing import Any, Dict, List


SearchResult = Dict[str, Any]


class SearchClient:
    def search(
        self,
        query: str,
        num_results: int = 10,
        **kwargs: Any,
    ) -> List[SearchResult]:
        raise NotImplementedError