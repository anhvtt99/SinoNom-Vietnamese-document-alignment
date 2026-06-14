from lib.web.searchers.base import SearchClient


def create_search_client(
    backend: str = "serper",
    **kwargs,
) -> SearchClient:
    backend = backend.lower().strip()

    if backend == "serper":
        from lib.web.searchers.serper import SerperSearchClient
        return SerperSearchClient(**kwargs)

    if backend == "serpapi":
        from lib.web.searchers.serpapi import SerpApiSearchClient
        return SerpApiSearchClient(**kwargs)

    raise ValueError(f"Unknown search backend: {backend}")