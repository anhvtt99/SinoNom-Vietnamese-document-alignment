"""
Fetch pages from URL JSON files and save extracted text into JSON.

Input:
    JSON files produced by search_urls.py

Output:
    JSON files containing fetched pages and extracted text.

Supported content:
    - HTML: trafilatura first, BeautifulSoup fallback
    - text/plain: direct response text
    - PDF: PyMuPDF

Async:
    Uses aiohttp with a configurable semaphore (--max_concurrent) so
    multiple URLs are fetched concurrently per document.

Retry:
    Transient errors (network failures, 5xx, 429) are retried with
    exponential backoff (--max_retries, --retry_backoff).

Incremental:
    Output files that already exist are skipped by default.
    Use --force to re-fetch everything.

Required packages:
    aiohttp
    trafilatura
    beautifulsoup4
    lxml
    pymupdf
"""

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlparse, urlunparse

import aiohttp
import fitz  # PyMuPDF
import trafilatura
from bs4 import BeautifulSoup

from lib.utils import random_sleep_seconds


DROP_QUERY_PARAMS_PREFIX = ("utm_",)
DROP_QUERY_PARAMS = {
    "fbclid",
    "gclid",
    "yclid",
}


DEFAULT_SLEEP_RANGE = (1.0, 2.0)   # polite but fast enough for academic sites
DEFAULT_TIMEOUT = 20
DEFAULT_MIN_TEXT_LEN = 200
DEFAULT_USER_AGENT = "HistoricalTextFetcher/1.0 (research use)"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 2.0
DEFAULT_MAX_CONCURRENT = 5      # max concurrent URLs per document
DEFAULT_MAX_DOC_CONCURRENT = 3  # max documents processed in parallel


# =============================================================================
# URL HELPERS
# =============================================================================

def canonicalize_url(url: str) -> str:
    """
    Light generic URL canonicalization before fetching.

    Only removes obvious duplicates:
    - fragment
    - tracking params
    - trailing slash
    - lowercase scheme/domain
    """
    url = str(url).strip()
    if not url:
        return ""

    parsed = urlparse(url)

    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()

    path = unquote(parsed.path or "")
    path = re.sub(r"/+$", "", path)

    kept_params = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lower_key = key.lower()

        if lower_key in DROP_QUERY_PARAMS:
            continue

        if any(lower_key.startswith(prefix) for prefix in DROP_QUERY_PARAMS_PREFIX):
            continue

        kept_params.append((key, value))

    query = urlencode(kept_params, doseq=True)
    path = quote(path, safe="/:%")

    return urlunparse((scheme, netloc, path, "", query, ""))


def iter_url_records(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Yield flat URL records from search_urls.py output.

    Supports new grouped format:
        {
          "query_results": [
            {
              "query_id": "...",
              "query": "...",
              "rerank_source": "...",
              "requested_num_results": 10,
              "urls": [...]
            }
          ]
        }
    """
    if "query_results" in data:
        for query_result in data.get("query_results", []):
            query_id = query_result.get("query_id", "")
            query = query_result.get("query", "")
            rerank_source = query_result.get("rerank_source", "")
            requested_num_results = query_result.get("requested_num_results")

            for url_item in query_result.get("urls", []):
                record = dict(url_item)
                record["query_id"] = query_id
                record["query"] = query
                record["rerank_source"] = rerank_source
                record["requested_num_results"] = requested_num_results
                yield record

def group_duplicate_urls(url_records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group duplicate URLs by canonical_url before fetching.

    Keeps source query traces.
    """
    grouped: Dict[str, Dict[str, Any]] = {}

    for record in url_records:
        url = str(record.get("url", "")).strip()
        if not url:
            continue

        canonical_url = canonicalize_url(url)
        if not canonical_url:
            continue

        if canonical_url not in grouped:
            grouped[canonical_url] = {
                "url": url,
                "canonical_url": canonical_url,
                "title": record.get("title", ""),
                "snippet": record.get("snippet", ""),
                "source_block": record.get("source_block", ""),
                "source_queries": [],
            }

        grouped[canonical_url]["source_queries"].append({
            "query_id": record.get("query_id", ""),
            "query": record.get("query", ""),
            "rank": record.get("rank"),
            "rerank_source": record.get("rerank_source", ""),
            "requested_num_results": record.get("requested_num_results"),
            "source_block": record.get("source_block", ""),
        })

    return list(grouped.values())


# =============================================================================
# CONTENT DETECTION
# =============================================================================

def detect_content_kind(url: str, content_type: str, content_bytes: bytes) -> str:
    """
    Detect content kind.

    Returns:
        "pdf", "html", "text", or "unknown"
    """
    lower_url = str(url).lower()
    lower_type = str(content_type).lower()

    if "application/pdf" in lower_type or lower_url.endswith(".pdf"):
        return "pdf"

    if "text/plain" in lower_type or lower_url.endswith(".txt"):
        return "text"

    if "text/html" in lower_type:
        return "html"

    head = content_bytes[:500].lower()

    if head.startswith(b"%pdf"):
        return "pdf"

    if b"<html" in head or b"<!doctype html" in head:
        return "html"

    return "unknown"


def _detect_charset(content_type: str, default: str = "utf-8") -> str:
    """Parse charset from Content-Type header, falling back to utf-8."""
    m = re.search(r"charset=([^\s;\"']+)", content_type, re.IGNORECASE)
    if m:
        charset = m.group(1).strip("\"'")
        try:
            "".encode(charset)
            return charset
        except LookupError:
            pass
    return default


# =============================================================================
# EXTRACTORS
# =============================================================================

def extract_title_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    if soup.title and soup.title.string:
        return soup.title.string.strip()

    h1 = soup.find("h1")
    if h1:
        return h1.get_text(" ", strip=True)

    return ""


def extract_html_with_bs4(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    lines = [
        line.strip()
        for line in soup.get_text("\n").splitlines()
        if line.strip()
    ]

    return "\n".join(lines)


def extract_plain_text_content(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    return {
        "text": text,
        "text_len": len(text),
        "extractor": "plain_text",
        "html_title": "",
        "needs_ocr": False,
        "assets": [],
    }


def extract_html_content(
    html: str,
    url: str = "",
    min_text_len: int = DEFAULT_MIN_TEXT_LEN,
) -> Dict[str, Any]:
    html_title = extract_title_from_html(html)

    text = trafilatura.extract(
        html,
        url=url or None,
        include_comments=False,
        include_tables=True,
        favor_precision=False,
    )

    if text and len(text.strip()) >= min_text_len:
        text = text.strip()
        extractor = "trafilatura"
    else:
        text = extract_html_with_bs4(html).strip()
        extractor = "beautifulsoup"

    return {
        "text": text,
        "text_len": len(text),
        "extractor": extractor,
        "html_title": html_title,
        "needs_ocr": False,
        "assets": [],
    }


def extract_pdf_content(
    pdf_bytes: bytes,
    min_text_len: int = DEFAULT_MIN_TEXT_LEN,
    collect_assets: bool = False,
) -> Dict[str, Any]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []
    assets = []

    for page_idx, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text:
            pages.append(text.strip())

        # Future OCR hook: record page assets without rendering images yet.
        if collect_assets:
            assets.append({
                "type": "pdf_page",
                "page_num": page_idx,
                "needs_ocr": True,
            })

    text = "\n\n".join(pages).strip()
    needs_ocr = len(text) < min_text_len

    return {
        "text": text,
        "text_len": len(text),
        "extractor": "pymupdf",
        "html_title": "",
        "needs_ocr": needs_ocr,
        "assets": assets if needs_ocr else [],
    }


def _extract_from_bytes(
    content_kind: str,
    url: str,
    final_url: str,
    content_bytes: bytes,
    charset: str,
    min_text_len: int = DEFAULT_MIN_TEXT_LEN,
    collect_assets: bool = False,
) -> Dict[str, Any]:
    """Synchronous extraction from raw bytes. Safe to run in a thread executor."""
    if content_kind == "pdf":
        return extract_pdf_content(
            pdf_bytes=content_bytes,
            min_text_len=min_text_len,
            collect_assets=collect_assets,
        )

    text = content_bytes.decode(charset, errors="replace")

    if content_kind == "text":
        return extract_plain_text_content(text)

    return extract_html_content(html=text, url=final_url, min_text_len=min_text_len)


def _error_result(error: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "http_status": None,
        "final_url": "",
        "content_type": "",
        "content_kind": "unknown",
        "extractor": "",
        "html_title": "",
        "text": "",
        "text_len": 0,
        "needs_ocr": False,
        "assets": [],
        "error": error,
    }


# =============================================================================
# ASYNC FETCH
# =============================================================================

async def _do_fetch(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
    timeout: int,
    sleep_range: Tuple[float, float],
) -> Dict[str, Any]:
    """
    Single low-level fetch attempt.

    Acquires the semaphore, fetches the URL, then sleeps (politely) before
    releasing the slot so the effective request rate stays bounded.
    """
    async with semaphore:
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout),
                allow_redirects=True,
            ) as resp:
                result = {
                    "status_code": resp.status,
                    "final_url": str(resp.url),
                    "content_type": resp.headers.get("Content-Type", ""),
                    "content_bytes": await resp.read(),
                }
        finally:
            # Polite delay held inside the semaphore slot so the effective
            # request rate is bounded even with high concurrency.
            await asyncio.sleep(random_sleep_seconds(sleep_range))

    return result


async def fetch_and_extract_async(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
    sleep_range: Tuple[float, float] = DEFAULT_SLEEP_RANGE,
    timeout: int = DEFAULT_TIMEOUT,
    min_text_len: int = DEFAULT_MIN_TEXT_LEN,
    collect_assets: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> Dict[str, Any]:
    """
    Fetch one URL and extract content with retry and exponential backoff.

    Retries on:
        - Network / connection errors (aiohttp.ClientError)
        - Timeouts (asyncio.TimeoutError)
        - HTTP 5xx (server errors)
        - HTTP 429 (rate limited, longer backoff)
    """
    last_error = "unknown error"

    for attempt in range(max_retries + 1):
        try:
            raw = await _do_fetch(session, url, semaphore, timeout, sleep_range)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                await asyncio.sleep(retry_backoff ** attempt)
            continue
        except Exception as exc:
            return _error_result(str(exc))

        http_status = raw["status_code"]

        if http_status == 429:
            last_error = "HTTP 429 (rate limited)"
            if attempt < max_retries:
                await asyncio.sleep(retry_backoff ** (attempt + 2))
            continue

        if http_status >= 500:
            last_error = f"HTTP {http_status}"
            if attempt < max_retries:
                await asyncio.sleep(retry_backoff ** attempt)
            continue

        # Successful response — extract content in thread pool so CPU-bound
        # work (trafilatura, fitz) does not block the event loop.
        try:
            content_kind = detect_content_kind(
                url=raw["final_url"],
                content_type=raw["content_type"],
                content_bytes=raw["content_bytes"],
            )
            if content_kind == "unknown":
                content_kind = "html"

            charset = _detect_charset(raw["content_type"])

            extracted = await asyncio.to_thread(
                _extract_from_bytes,
                content_kind,
                url,
                raw["final_url"],
                raw["content_bytes"],
                charset,
                min_text_len,
                collect_assets,
            )

            return {
                "status": "ok",
                "http_status": http_status,
                "final_url": raw["final_url"],
                "content_type": raw["content_type"],
                "content_kind": content_kind,
                "extractor": extracted.get("extractor", ""),
                "html_title": extracted.get("html_title", ""),
                "text": extracted.get("text", ""),
                "text_len": extracted.get("text_len", 0),
                "needs_ocr": extracted.get("needs_ocr", False),
                "assets": extracted.get("assets", []),
                "error": "",
            }
        except Exception as exc:
            return _error_result(str(exc))

    return _error_result(last_error)


# =============================================================================
# FILE RUNNER
# =============================================================================

async def fetch_one_url_file_async(
    url_path: Path,
    output_path: Path,
    sleep_range: Tuple[float, float] = DEFAULT_SLEEP_RANGE,
    timeout: int = DEFAULT_TIMEOUT,
    min_text_len: int = DEFAULT_MIN_TEXT_LEN,
    user_agent: str = DEFAULT_USER_AGENT,
    collect_assets: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    skip_existing: bool = True,
    verbose: bool = False,
) -> None:
    if skip_existing and output_path.exists():
        if verbose:
            print(f"[skip] {output_path.name} (already exists, use --force to re-fetch)")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with url_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    doc_id = data.get("doc_id", url_path.stem)
    url_records = list(iter_url_records(data))
    grouped_urls = group_duplicate_urls(url_records)

    if verbose:
        print(f"\n=== Fetching pages for {doc_id} ===")
        print(f"URL file: {url_path}")
        print(f"Raw URL records: {len(url_records)}")
        print(f"Unique canonical URLs: {len(grouped_urls)}")
        print(f"Max concurrent: {max_concurrent}, Max retries: {max_retries}")
        print(f"Output file: {output_path}")

    semaphore = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(limit=max_concurrent * 2)
    headers = {"User-Agent": user_agent}

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        tasks = [
            fetch_and_extract_async(
                session=session,
                url=item["canonical_url"],
                semaphore=semaphore,
                sleep_range=sleep_range,
                timeout=timeout,
                min_text_len=min_text_len,
                collect_assets=collect_assets,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
            )
            for item in grouped_urls
        ]
        fetch_results = await asyncio.gather(*tasks)

    pages: List[Dict[str, Any]] = []

    for idx, (item, fetch_result) in enumerate(zip(grouped_urls, fetch_results), start=1):
        canonical_url = item["canonical_url"]
        page_id = f"{idx:04d}"

        if verbose:
            status = fetch_result.get("status", "error")
            err = fetch_result.get("error", "")
            suffix = f" ({err})" if err else ""
            print(f"[{idx}/{len(grouped_urls)}] {status.upper()}{suffix} {canonical_url}")

        domain = urlparse(canonical_url).netloc.lower()

        page = {
            "page_id": page_id,
            "doc_id": doc_id,

            "url": item.get("url", ""),
            "canonical_url": canonical_url,
            "final_url": fetch_result["final_url"],
            "domain": domain,

            "title": item.get("title", ""),
            "html_title": fetch_result["html_title"],
            "snippet": item.get("snippet", ""),
            "source_block": item.get("source_block", ""),

            "status": fetch_result["status"],
            "http_status": fetch_result["http_status"],
            "content_type": fetch_result["content_type"],
            "content_kind": fetch_result["content_kind"],
            "extractor": fetch_result["extractor"],

            "text_len": fetch_result["text_len"],
            "needs_ocr": fetch_result["needs_ocr"],
            "assets": fetch_result.get("assets", []),
            "error": fetch_result["error"],

            "source_queries": item.get("source_queries", []),
            "text": fetch_result["text"],
        }

        pages.append(page)

    output = {
        "doc_id": doc_id,
        "source_url_path": str(url_path),
        "num_raw_url_records": len(url_records),
        "num_unique_urls": len(grouped_urls),
        "num_pages": len(pages),
        "num_fetch_ok": sum(1 for p in pages if p["status"] == "ok"),
        "num_fetch_error": sum(1 for p in pages if p["status"] != "ok"),
        "num_needs_ocr": sum(1 for p in pages if p.get("needs_ocr")),
        "num_assets": sum(len(p.get("assets", [])) for p in pages),
        "pages": pages,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\nSaved fetched pages to: {output_path}")
        print(f"Pages: {len(pages)}")
        print(f"Fetch ok: {output['num_fetch_ok']}")
        print(f"Fetch error: {output['num_fetch_error']}")
        print(f"Assets: {output['num_assets']}")
        print(f"Needs OCR: {output['num_needs_ocr']}")


def collect_json_files(input_dir: Path, recursive: bool = False) -> List[Path]:
    if recursive:
        return sorted(input_dir.rglob("*.json"))
    return sorted(input_dir.glob("*.json"))


# =============================================================================
# ASYNC MAIN
# =============================================================================

async def _async_main(args: argparse.Namespace) -> None:
    sleep_range = tuple(args.sleep_range)
    skip_existing = not args.force

    if args.url_path:
        await fetch_one_url_file_async(
            url_path=Path(args.url_path),
            output_path=Path(args.output_path),
            sleep_range=sleep_range,
            timeout=args.timeout,
            min_text_len=args.min_text_len,
            collect_assets=args.collect_assets,
            user_agent=args.user_agent,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
            max_concurrent=args.max_concurrent,
            skip_existing=skip_existing,
            verbose=args.verbose,
        )
    else:
        url_dir = Path(args.url_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        url_files = collect_json_files(url_dir, recursive=args.recursive)

        if not url_files:
            raise FileNotFoundError(f"No JSON files found in {url_dir}")

        print(f"Found {len(url_files)} URL JSON files "
              f"(max_doc_concurrent={args.max_doc_concurrent})")

        # Document-level semaphore: limits how many docs are fetched in parallel.
        # Each doc independently uses its own per-URL semaphore (max_concurrent).
        doc_sem = asyncio.Semaphore(args.max_doc_concurrent)

        async def _fetch_doc(idx: int, url_file: Path) -> None:
            rel_path = url_file.relative_to(url_dir)
            output_path = output_dir / rel_path.with_suffix(".json")
            async with doc_sem:
                if args.verbose:
                    print(f"[{idx}/{len(url_files)}] {url_file.name}")
                await fetch_one_url_file_async(
                    url_path=url_file,
                    output_path=output_path,
                    sleep_range=sleep_range,
                    timeout=args.timeout,
                    min_text_len=args.min_text_len,
                    collect_assets=args.collect_assets,
                    user_agent=args.user_agent,
                    max_retries=args.max_retries,
                    retry_backoff=args.retry_backoff,
                    max_concurrent=args.max_concurrent,
                    skip_existing=skip_existing,
                    verbose=args.verbose,
                )

        await asyncio.gather(*[
            _fetch_doc(idx, url_file)
            for idx, url_file in enumerate(url_files, start=1)
        ])


def main():
    parser = argparse.ArgumentParser(
        description="Fetch pages from URL JSON files and extract text"
    )

    parser.add_argument("--url_path", type=str, default=None)
    parser.add_argument("--url_dir", type=str, default=None)

    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--recursive", action="store_true")

    parser.add_argument(
        "--sleep_range",
        type=float,
        nargs=2,
        default=DEFAULT_SLEEP_RANGE,
        metavar=("MIN", "MAX"),
        help="Random sleep range (seconds) between fetch requests per slot",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--min_text_len", type=int, default=DEFAULT_MIN_TEXT_LEN)

    parser.add_argument(
        "--user_agent",
        type=str,
        default=DEFAULT_USER_AGENT,
    )

    parser.add_argument(
        "--collect_assets",
        action="store_true",
        help="Collect asset metadata for future OCR/image extraction.",
    )

    # Concurrency & retry
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help="Maximum number of concurrent fetch requests per document (default: 5)",
    )
    parser.add_argument(
        "--max_doc_concurrent",
        type=int,
        default=DEFAULT_MAX_DOC_CONCURRENT,
        help="Maximum number of documents fetched in parallel (default: 3)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retry attempts on transient errors (default: 3)",
    )
    parser.add_argument(
        "--retry_backoff",
        type=float,
        default=DEFAULT_RETRY_BACKOFF,
        help="Exponential backoff base in seconds for retries (default: 2.0)",
    )

    # Incremental processing
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if the output file already exists (default: skip existing)",
    )

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if bool(args.url_path) == bool(args.url_dir):
        raise ValueError("Provide exactly one of --url_path or --url_dir")

    if args.url_path and not args.output_path:
        raise ValueError("--output_path is required when using --url_path")

    if args.url_dir and not args.output_dir:
        raise ValueError("--output_dir is required when using --url_dir")

    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
