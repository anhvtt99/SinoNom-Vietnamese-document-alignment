"""
Run web search from query JSON files and save raw URL results.

Input:
    JSON files produced by build_query.py

Output:
    JSON files under url_dir containing flat raw search URL results.

Required env:
    SERPER_API_KEY
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

from lib.config import load_project_env
from lib.web.searchers import create_search_client


def random_sleep(base_sec: float, jitter_sec: float = 0.0) -> None:
    if base_sec <= 0 and jitter_sec <= 0:
        return

    delay = base_sec
    if jitter_sec > 0:
        delay += random.uniform(0, jitter_sec)

    if delay > 0:
        time.sleep(delay)


def load_query_file(query_path: Path) -> Dict[str, Any]:
    with query_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_query_items(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract query records from build_query.py output.

    Expected input format:
        {
            "doc_id": "...",
            "results": [
                {
                    "query_id": "global",
                    "query": "...",
                    ...
                }
            ]
        }
    """
    query_items: List[Dict[str, Any]] = []

    for item in data.get("results", []):
        query = str(item.get("query", "")).strip()
        if not query:
            continue

        query_items.append({
            "query_id": item.get("query_id", ""),
            "query": query,
            "rerank_source": item.get("rerank_source", ""),
        })

    return query_items


def build_search_kwargs(args) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}

    if args.gl:
        kwargs["gl"] = args.gl

    if args.hl:
        kwargs["hl"] = args.hl

    if args.location:
        kwargs["location"] = args.location

    if args.include_omitted:
        kwargs["filter"] = 0

    return kwargs


def search_one_query_file(
    query_path: Path,
    output_path: Path,
    search_client,
    args,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_query_file(query_path)

    doc_id = data.get("doc_id", query_path.stem)
    query_items = extract_query_items(data)
    if args.top_queries is not None:
        query_items = query_items[:args.top_queries]
    search_kwargs = build_search_kwargs(args)

    urls: List[Dict[str, Any]] = []
    query_errors: List[Dict[str, Any]] = []

    if args.verbose:
        print(f"\n=== Searching URLs for {doc_id} ===")
        print(f"Query file: {query_path}")
        print(f"Queries: {len(query_items)}")
        print(f"Search kwargs: {search_kwargs}")

    for idx, item in enumerate(query_items, start=1):
        query_id = item["query_id"]
        query = item["query"]
        if args.verbose:
            print(f"\n[{idx}/{len(query_items)}] {doc_id}/{query_id}")
            print(query)

        rerank_source = item.get("rerank_source", "")
        is_weak_query = not str(rerank_source).startswith("wikisource")
        num_results = args.num_results
        if is_weak_query and args.fallback_num_results is not None:
            num_results = args.fallback_num_results
                
        try:
            results = search_client.search(
                query=query,
                num_results=num_results,
                **search_kwargs,
            )

        except Exception as e:
            error_msg = str(e)
            query_errors.append({
                "query_id": query_id,
                "query": query,
                "error": error_msg,
            })

            if args.verbose:
                print(f"[Search error] {doc_id}/{query_id}: {error_msg}")

            results = []

        for result in results:
            url = str(result.get("url", "")).strip()
            if not url:
                continue

            urls.append({
                "doc_id": doc_id,
                "query_id": query_id,
                "query": query,
                "rerank_source": item.get("rerank_source", ""),
                "requested_num_results": num_results,
                "source_block": result.get("source_block", ""),
                "rank": result.get("rank"),
                "title": result.get("title", ""),
                "url": url,
                "snippet": result.get("snippet", ""),
                "raw": result.get("raw", {}),
            })

        if args.verbose:
            print(f"Results: {len(results)}")

        if idx < len(query_items):
            random_sleep(args.sleep_sec, args.sleep_jitter)

    output = {
        "doc_id": doc_id,
        "source_query_path": str(query_path),
        "search_backend": args.search_backend,
        "search_params": {
            "num_results": args.num_results,
            "gl": args.gl,
            "hl": args.hl,
            "location": args.location,
            "include_omitted": args.include_omitted,
            "filter": 0 if args.include_omitted else None,
        },
        "num_queries": len(query_items),
        "num_raw_urls": len(urls),
        "num_query_errors": len(query_errors),
        "urls": urls,
        "query_errors": query_errors,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if args.verbose:
        print(f"\nSaved URL file to: {output_path}")
        print(f"Raw URLs: {len(urls)}")
        print(f"Query errors: {len(query_errors)}")


def collect_query_files(input_dir: Path, recursive: bool = False) -> List[Path]:
    if recursive:
        return sorted(input_dir.rglob("*.json"))
    return sorted(input_dir.glob("*.json"))


def main():
    load_project_env()

    parser = argparse.ArgumentParser(
        description="Search URLs from query JSON files"
    )

    # I/O config
    parser.add_argument("--query_path", type=str, default=None)
    parser.add_argument("--query_dir", type=str, default=None)
    parser.add_argument("--url_path", type=str, default=None)
    parser.add_argument("--url_dir", type=str, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--top_queries",
        type=int,
        default=None,
        help="Only search the first N queries in each query JSON file. If None, search all queries.",
    )

    # Search config
    parser.add_argument(
        "--search_backend",
        type=str,
        default="serper",
        help="Search backend name, e.g. serper",
    )

    parser.add_argument(
        "--num_results",
        type=int,
        default=10,
        help="Number of results per query",
    )

    parser.add_argument(
        "--fallback_num_results",
        type=int,
        default=None,
        help="Number of results for queries not reranked by Wikisource. If None, use --num_results.",
    )

    parser.add_argument(
        "--gl",
        type=str,
        default="vn",
        help="Google country code passed to search backend",
    )

    parser.add_argument(
        "--hl",
        type=str,
        default="zh-cn",
        help="Google UI language passed to search backend",
    )

    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="Optional search location",
    )

    parser.add_argument(
        "--include_omitted",
        action="store_true",
        help="Use filter=0 to ask Google/Serper to include omitted similar results",
    )

    parser.add_argument(
        "--sleep_sec",
        type=float,
        default=1.0,
        help="Base sleep between queries",
    )

    parser.add_argument(
        "--sleep_jitter",
        type=float,
        default=0.5,
        help="Random extra sleep between queries",
    )

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if bool(args.query_path) == bool(args.query_dir):
        raise ValueError("Provide exactly one of --query_path or --query_dir")

    if args.query_path and not args.url_path:
        raise ValueError("--url_path is required when using --query_path")

    if args.query_dir and not args.url_dir:
        raise ValueError("--url_dir is required when using --query_dir")

    search_client = create_search_client(args.search_backend)

    if args.query_path:
        search_one_query_file(
            query_path=Path(args.query_path),
            output_path=Path(args.url_path),
            search_client=search_client,
            args=args,
        )
        return

    query_dir = Path(args.query_dir)
    url_dir = Path(args.url_dir)
    url_dir.mkdir(parents=True, exist_ok=True)

    query_files = collect_query_files(query_dir, recursive=args.recursive)

    if not query_files:
        raise FileNotFoundError(f"No JSON files found in {query_dir}")

    if args.verbose:
        print(f"Found {len(query_files)} query JSON files")

    for idx, query_file in enumerate(query_files, start=1):
        rel_path = query_file.relative_to(query_dir)
        output_path = url_dir / rel_path.with_suffix(".json")

        if args.verbose:
            print(f"\n[{idx}/{len(query_files)}] {query_file} -> {output_path}")

        search_one_query_file(
            query_path=query_file,
            output_path=output_path,
            search_client=search_client,
            args=args,
        )


if __name__ == "__main__":
    main()
