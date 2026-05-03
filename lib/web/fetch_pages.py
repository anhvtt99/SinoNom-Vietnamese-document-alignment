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

Required packages:
    requests
    trafilatura
    beautifulsoup4
    lxml
    pymupdf
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlparse, urlunparse

import fitz  # PyMuPDF
import requests
import trafilatura
from bs4 import BeautifulSoup

from lib.utils import random_sleep_seconds


DROP_QUERY_PARAMS_PREFIX = ("utm_",)
DROP_QUERY_PARAMS = {
    "fbclid",
    "gclid",
    "yclid",
}


DEFAULT_SLEEP_RANGE = (1.0, 2.0)
DEFAULT_TIMEOUT = 20
DEFAULT_MIN_TEXT_LEN = 200
DEFAULT_USER_AGENT = "HistoricalTextFetcher/1.0 (research use)"


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


def decode_response_text(resp: requests.Response) -> str:
    """
    Decode response text using requests, with apparent encoding fallback.
    """
    if not resp.encoding:
        resp.encoding = resp.apparent_encoding

    return resp.text or ""


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


def extract_content(
    content_kind: str,
    url: str,
    resp: requests.Response,
    min_text_len: int = DEFAULT_MIN_TEXT_LEN,
    collect_assets: bool = False,
) -> Dict[str, Any]:
    """
    Dispatch content extraction by content kind.

    Returns a normalized extraction dict:
        text, text_len, extractor, html_title, needs_ocr, assets
    """
    if content_kind == "pdf":
        return extract_pdf_content(
            pdf_bytes=resp.content or b"",
            min_text_len=min_text_len,
            collect_assets=collect_assets,
        )

    if content_kind == "text":
        return extract_plain_text_content(decode_response_text(resp))

    # html / unknown
    html = decode_response_text(resp)
    return extract_html_content(
        html=html,
        url=resp.url or url,
        min_text_len=min_text_len,
    )

# =============================================================================
# FETCH
# =============================================================================

def fetch_and_extract(
    session: requests.Session,
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    min_text_len: int = DEFAULT_MIN_TEXT_LEN,
    collect_assets: bool = False,
) -> Dict[str, Any]:
    """
    Fetch one URL and extract content.

    This function only handles network + dispatch.
    Actual extraction lives in extractor functions.
    """
    try:
        resp = session.get(
            url,
            timeout=timeout,
            allow_redirects=True,
        )

        content_type = resp.headers.get("Content-Type", "")
        content_bytes = resp.content or b""

        content_kind = detect_content_kind(
            url=resp.url or url,
            content_type=content_type,
            content_bytes=content_bytes,
        )

        if content_kind == "unknown":
            content_kind = "html"

        extracted = extract_content(
            content_kind=content_kind,
            url=url,
            resp=resp,
            min_text_len=min_text_len,
            collect_assets=collect_assets,
        )

        return {
            "status": "ok",
            "http_status": resp.status_code,
            "final_url": resp.url,
            "content_type": content_type,
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
            "error": str(exc),
        }

# =============================================================================
# FILE RUNNER
# =============================================================================

def fetch_one_url_file(
    url_path: Path,
    output_path: Path,
    sleep_range: Tuple[float, float] = DEFAULT_SLEEP_RANGE,
    timeout: int = DEFAULT_TIMEOUT,
    min_text_len: int = DEFAULT_MIN_TEXT_LEN,
    user_agent: str = DEFAULT_USER_AGENT,
    collect_assets: bool = False,
    verbose: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with url_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    doc_id = data.get("doc_id", url_path.stem)
    url_records = list(iter_url_records(data))
    grouped_urls = group_duplicate_urls(url_records)

    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
    })

    pages: List[Dict[str, Any]] = []

    if verbose:
        print(f"\n=== Fetching pages for {doc_id} ===")
        print(f"URL file: {url_path}")
        print(f"Raw URL records: {len(url_records)}")
        print(f"Unique canonical URLs: {len(grouped_urls)}")
        print(f"Output file: {output_path}")

    for idx, item in enumerate(grouped_urls, start=1):
        canonical_url = item["canonical_url"]
        page_id = f"{idx:04d}"

        if verbose:
            print(f"[{idx}/{len(grouped_urls)}] {canonical_url}")

        fetch_result = fetch_and_extract(
            session=session,
            url=canonical_url,
            timeout=timeout,
            min_text_len=min_text_len,
            collect_assets=collect_assets,
        )

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

        if idx < len(grouped_urls):
            time.sleep(random_sleep_seconds(sleep_range))

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
        help="Random sleep range between fetch requests",
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
        help="Collect asset metadata for future OCR/image extraction. Does not perform OCR.",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if bool(args.url_path) == bool(args.url_dir):
        raise ValueError("Provide exactly one of --url_path or --url_dir")

    if args.url_path and not args.output_path:
        raise ValueError("--output_path is required when using --url_path")

    if args.url_dir and not args.output_dir:
        raise ValueError("--output_dir is required when using --url_dir")

    sleep_range = tuple(args.sleep_range)

    if args.url_path:
        fetch_one_url_file(
            url_path=Path(args.url_path),
            output_path=Path(args.output_path),
            sleep_range=sleep_range,
            timeout=args.timeout,
            min_text_len=args.min_text_len,
            collect_assets=args.collect_assets,
            user_agent=args.user_agent,
            verbose=args.verbose,
        )
    else:
        url_dir = Path(args.url_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        url_files = collect_json_files(url_dir, recursive=args.recursive)

        if not url_files:
            raise FileNotFoundError(f"No JSON files found in {url_dir}")

        if args.verbose:
            print(f"Found {len(url_files)} URL JSON files")

        for idx, url_file in enumerate(url_files, start=1):
            rel_path = url_file.relative_to(url_dir)
            output_path = output_dir / rel_path.with_suffix(".json")

            if args.verbose:
                print(f"\n[{idx}/{len(url_files)}] {url_file} -> {output_path}")

            fetch_one_url_file(
                url_path=url_file,
                output_path=output_path,
                sleep_range=sleep_range,
                timeout=args.timeout,
                min_text_len=args.min_text_len,
                collect_assets=args.collect_assets,
                user_agent=args.user_agent,
                verbose=args.verbose,
            )


if __name__ == "__main__":
    main()