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
import hashlib
import json
import re
from datetime import datetime, timezone
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
DEFAULT_MAX_CONCURRENT = 10      # max concurrent URL fetches (global)

# Output layout under the page_dir (output_dir):
#   content/   actual fetched content, deduped globally by canonical-url hash
#                <hash>.txt  extracted text (web + pdf text layer)
#                <hash>.pdf  raw file kept only for PDFs (re-extract/OCR later)
#   docs/      lightweight per-doc index JSON (references content, no inline text)
#   _registry.json  canonical_url -> fetch result (cross-doc dedup + incremental)
CONTENT_SUBDIR = "content"
DOCS_SUBDIR = "docs"
REGISTRY_NAME = "_registry.json"


def url_hash(canonical_url: str) -> str:
    """Stable 16-char hex hash of a canonical URL — used as a dedup key and
    short disambiguating suffix on content filenames."""
    return hashlib.sha1(canonical_url.encode("utf-8")).hexdigest()[:16]


# Domains to skip entirely — no fetch, no text extraction.
# Match is exact on the lowercased host after stripping "www.".
# Add entries here to extend the blocklist.
BLOCKED_DOMAINS: frozenset = frozenset([
    # Video platforms
    "youtube.com", "youtu.be",
    "tiktok.com",
    "vimeo.com",
    "dailymotion.com",
    "twitch.tv",
    "instagram.com",
    "facebook.com", "fb.watch",
    "bilibili.com", "m.bilibili.com",
    "nicovideo.jp",
    "rumble.com",
    "odysee.com",
    "iqiyi.com",
    "youku.com",
    "v.qq.com",
])


def is_blocked_url(url: str) -> bool:
    """True when the URL's host is in BLOCKED_DOMAINS."""
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host in BLOCKED_DOMAINS


import unicodedata as _ud

def content_stem(
    url: str,
    content_kind: str,
    html_title: str,
    hash_str: str,
) -> str:
    """
    Build a human-readable filename stem for a content file.

    - PDF / binary: use the last path segment of the URL (original filename).
    - Web / text:   use the page title, sanitised + short hash suffix so two
                    pages with the same title don't collide.

    The stem is used for both the .txt and (for PDFs) the raw file.
    """
    def _sanitize(s: str, max_len: int = 80) -> str:
        s = _ud.normalize("NFKC", s)
        # Replace filesystem-unsafe characters with underscores.
        s = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", s)
        # Collapse runs of underscores/spaces.
        s = re.sub(r"[\s_]+", "_", s).strip("_. ")
        return s[:max_len].strip("_. ") or "untitled"

    if content_kind == "pdf":
        # e.g. https://example.com/docs/vhtt2023.pdf  →  vhtt2023
        path_part = urlparse(url).path.rstrip("/")
        name = Path(path_part).stem if path_part else ""
        stem = _sanitize(name) if name else hash_str
    else:
        title = (html_title or "").strip()
        # Remove common site-name suffixes to keep names short.
        title = re.sub(r"\s*[-|–]\s*(Wikipedia|Wikisource|Wikivoyage|Wikibooks"
                       r"|Wikimedia|Wiktionary|YouTube|Bilibili).*$",
                       "", title, flags=re.IGNORECASE)
        stem = (_sanitize(title) + "__" + hash_str[:8]) if title else hash_str

    return stem


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


def _extract_and_save(
    content_kind: str,
    url: str,
    final_url: str,
    content_bytes: bytes,
    charset: str,
    content_dir: Path,
    hash_str: str,
    min_text_len: int = DEFAULT_MIN_TEXT_LEN,
    collect_assets: bool = False,
) -> Dict[str, Any]:
    """
    Extract text from raw bytes and write content files to disk.

    Safe to run in a thread executor. Writes:
      - content/<stem>.txt   extracted text (web: title-based name, pdf: url-filename-based)
      - content/<stem>.pdf   raw bytes (PDF only — kept for OCR/re-extract)

    Returns extraction metadata plus relative paths of written files (or None).
    Large content is never returned to the caller — it stays on disk only.
    """
    if content_kind == "pdf":
        extracted = extract_pdf_content(
            pdf_bytes=content_bytes,
            min_text_len=min_text_len,
            collect_assets=collect_assets,
        )
    else:
        decoded = content_bytes.decode(charset, errors="replace")
        if content_kind == "text":
            extracted = extract_plain_text_content(decoded)
        else:
            extracted = extract_html_content(html=decoded, url=final_url, min_text_len=min_text_len)

    html_title = extracted.get("html_title", "") or ""
    text = extracted.get("text", "") or ""

    # Build a human-readable filename stem (title-based for web, original
    # filename for PDFs), with a short hash suffix to prevent collisions.
    stem = content_stem(url=url, content_kind=content_kind,
                        html_title=html_title, hash_str=hash_str)

    text_file = None
    raw_file = None

    if text.strip():
        content_dir.mkdir(parents=True, exist_ok=True)
        (content_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
        text_file = f"{CONTENT_SUBDIR}/{stem}.txt"

    # PDF: also keep the raw file for future OCR / re-extraction.
    if content_kind == "pdf":
        content_dir.mkdir(parents=True, exist_ok=True)
        (content_dir / f"{stem}.pdf").write_bytes(content_bytes)
        raw_file = f"{CONTENT_SUBDIR}/{stem}.pdf"

    return {
        "extractor": extracted.get("extractor", ""),
        "html_title": html_title,
        "text_len": extracted.get("text_len", 0),
        "needs_ocr": extracted.get("needs_ocr", False),
        "assets": extracted.get("assets", []),
        "text_file": text_file,
        "raw_file": raw_file,
    }


def _error_entry(canonical_url: str, hash_str: str, error: str) -> Dict[str, Any]:
    """Registry entry for a URL that failed to fetch/extract."""
    return {
        "hash": hash_str,
        "canonical_url": canonical_url,
        "final_url": "",
        "status": "error",
        "http_status": None,
        "content_type": "",
        "content_kind": "unknown",
        "extractor": "",
        "html_title": "",
        "text_len": 0,
        "needs_ocr": False,
        "text_file": None,
        "raw_file": None,
        "error": error,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
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


async def fetch_and_save_async(
    session: aiohttp.ClientSession,
    canonical_url: str,
    content_dir: Path,
    semaphore: asyncio.Semaphore,
    sleep_range: Tuple[float, float] = DEFAULT_SLEEP_RANGE,
    timeout: int = DEFAULT_TIMEOUT,
    min_text_len: int = DEFAULT_MIN_TEXT_LEN,
    collect_assets: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> Dict[str, Any]:
    """
    Fetch one URL, extract + write content to disk, and return a registry entry.

    The returned dict contains only metadata + relative content paths (no inline
    text/bytes), so it can be stored in _registry.json and shared across docs.

    Retries on:
        - Network / connection errors (aiohttp.ClientError)
        - Timeouts (asyncio.TimeoutError)
        - HTTP 5xx (server errors)
        - HTTP 429 (rate limited, longer backoff)
    """
    hash_str = url_hash(canonical_url)

    if is_blocked_url(canonical_url):
        return _error_entry(canonical_url, hash_str, "skipped:blocked_domain")

    last_error = "unknown error"

    for attempt in range(max_retries + 1):
        try:
            raw = await _do_fetch(session, canonical_url, semaphore, timeout, sleep_range)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                await asyncio.sleep(retry_backoff ** attempt)
            continue
        except Exception as exc:
            return _error_entry(canonical_url, hash_str, str(exc))

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

        # Successful response — extract + write files in a thread pool so the
        # CPU-bound work (trafilatura, fitz) and disk I/O do not block the loop.
        try:
            content_kind = detect_content_kind(
                url=raw["final_url"],
                content_type=raw["content_type"],
                content_bytes=raw["content_bytes"],
            )
            if content_kind == "unknown":
                content_kind = "html"

            charset = _detect_charset(raw["content_type"])

            saved = await asyncio.to_thread(
                _extract_and_save,
                content_kind,
                canonical_url,
                raw["final_url"],
                raw["content_bytes"],
                charset,
                content_dir,
                hash_str,
                min_text_len,
                collect_assets,
            )

            return {
                "hash": hash_str,
                "canonical_url": canonical_url,
                "final_url": raw["final_url"],
                "status": "ok",
                "http_status": http_status,
                "content_type": raw["content_type"],
                "content_kind": content_kind,
                "extractor": saved["extractor"],
                "html_title": saved["html_title"],
                "text_len": saved["text_len"],
                "needs_ocr": saved["needs_ocr"],
                "assets": saved["assets"],
                "text_file": saved["text_file"],
                "raw_file": saved["raw_file"],
                "error": "",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            return _error_entry(canonical_url, hash_str, str(exc))

    return _error_entry(canonical_url, hash_str, last_error)


# =============================================================================
# REGISTRY (cross-doc dedup + incremental)
# =============================================================================

def load_registry(path: Path) -> Dict[str, Any]:
    """Load the canonical_url -> fetch-entry registry, or {} if missing/corrupt."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_registry(path: Path, registry: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# PER-DOC INDEX
# =============================================================================

def build_doc_index(
    doc_id: str,
    direction: str,
    source_url_path: str,
    grouped_items: List[Dict[str, Any]],
    registry: Dict[str, Any],
    *,
    full_output: bool = False,
) -> Dict[str, Any]:
    """
    Build a lightweight per-doc index referencing content files in the store.

    Each page record carries doc-specific metadata (title/snippet/source_queries
    from search) plus the shared fetch result (status, content_kind, text_file,
    raw_file, ...) looked up from the global registry. No inline page text.
    """
    pages: List[Dict[str, Any]] = []

    for idx, item in enumerate(grouped_items, start=1):
        cu = item["canonical_url"]
        reg = registry.get(cu, {})
        domain = urlparse(cu).netloc.lower()

        page = {
            "page_id": f"{idx:04d}",
            "doc_id": doc_id,
            "url": item.get("url", ""),
            "canonical_url": cu,
            "final_url": reg.get("final_url", ""),
            "domain": domain,
            "title": item.get("title", ""),
            "html_title": reg.get("html_title", ""),
            "status": reg.get("status", "error"),
            "content_kind": reg.get("content_kind", "unknown"),
            "extractor": reg.get("extractor", ""),
            "needs_ocr": reg.get("needs_ocr", False),
            "error": reg.get("error", "" if reg else "not_fetched"),
            "content_hash": reg.get("hash", url_hash(cu)),
            "text_file": reg.get("text_file"),
            "raw_file": reg.get("raw_file"),
            "source_queries": item.get("source_queries", []),
        }

        if full_output:
            page.update({
                "http_status": reg.get("http_status"),
                "content_type": reg.get("content_type", ""),
                "text_len": reg.get("text_len", 0),
                "snippet": item.get("snippet", ""),
                "source_block": item.get("source_block", ""),
                "assets": reg.get("assets", []),
                "fetched_at": reg.get("fetched_at"),
            })

        pages.append(page)

    return {
        "doc_id": doc_id,
        "direction": direction,
        "source_url_path": source_url_path,
        "num_pages": len(pages),
        "num_fetch_ok": sum(1 for p in pages if p["status"] == "ok"),
        "num_fetch_error": sum(1 for p in pages if p["status"] != "ok"),
        "num_needs_ocr": sum(1 for p in pages if p.get("needs_ocr")),
        "pages": pages,
    }


def collect_json_files(input_dir: Path, recursive: bool = False) -> List[Path]:
    if recursive:
        return sorted(input_dir.rglob("*.json"))
    return sorted(input_dir.glob("*.json"))


# =============================================================================
# ASYNC MAIN
# =============================================================================

async def _async_main(args: argparse.Namespace) -> None:
    sleep_range = tuple(args.sleep_range)

    page_dir = Path(args.output_dir)
    content_dir = page_dir / CONTENT_SUBDIR
    docs_dir = page_dir / DOCS_SUBDIR
    registry_path = page_dir / REGISTRY_NAME
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Collect input URL JSON files (output of search_urls.py).
    if args.url_path:
        url_root = Path(args.url_path).parent
        url_files = [Path(args.url_path)]
    else:
        url_root = Path(args.url_dir)
        url_files = collect_json_files(url_root, recursive=args.recursive)
        if not url_files:
            raise FileNotFoundError(f"No JSON files found in {url_root}")

    # ── Phase 1: load every doc, group URLs per doc, collect the global set ──
    docs: List[Dict[str, Any]] = []
    global_urls: Dict[str, None] = {}

    for url_file in url_files:
        with url_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        doc_id = data.get("doc_id", url_file.stem)
        direction = data.get("direction", "vi2zh")
        grouped = group_duplicate_urls(iter_url_records(data))

        for item in grouped:
            global_urls.setdefault(item["canonical_url"], None)

        if args.url_path:
            out_path = docs_dir / f"{doc_id}.json"
        else:
            out_path = docs_dir / url_file.relative_to(url_root).with_suffix(".json")

        docs.append({
            "doc_id": doc_id,
            "direction": direction,
            "source_url_path": str(url_file),
            "grouped": grouped,
            "out_path": out_path,
        })

    # ── Phase 2: fetch each unique URL once (cross-doc dedup + incremental) ──
    registry = load_registry(registry_path)
    pending = [cu for cu in global_urls if args.force or cu not in registry]

    print(
        f"Docs: {len(docs)} | unique URLs: {len(global_urls)} | "
        f"to fetch: {len(pending)} | cached: {len(global_urls) - len(pending)}"
    )

    if pending:
        semaphore = asyncio.Semaphore(args.max_concurrent)
        connector = aiohttp.TCPConnector(limit=args.max_concurrent * 2)
        headers = {"User-Agent": args.user_agent}

        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            tasks = [
                fetch_and_save_async(
                    session=session,
                    canonical_url=cu,
                    content_dir=content_dir,
                    semaphore=semaphore,
                    sleep_range=sleep_range,
                    timeout=args.timeout,
                    min_text_len=args.min_text_len,
                    collect_assets=args.collect_assets,
                    max_retries=args.max_retries,
                    retry_backoff=args.retry_backoff,
                )
                for cu in pending
            ]

            done = 0
            for fut in asyncio.as_completed(tasks):
                entry = await fut
                registry[entry["canonical_url"]] = entry
                done += 1
                if args.verbose:
                    err = f" ({entry['error']})" if entry.get("error") else ""
                    print(f"[{done}/{len(pending)}] {entry['status'].upper()}{err} "
                          f"{entry['canonical_url']}")

        save_registry(registry_path, registry)

    # ── Phase 3: write per-doc index JSON referencing the content store ──
    for d in docs:
        output = build_doc_index(
            doc_id=d["doc_id"],
            direction=d["direction"],
            source_url_path=d["source_url_path"],
            grouped_items=d["grouped"],
            registry=registry,
            full_output=args.full_output,
        )
        d["out_path"].parent.mkdir(parents=True, exist_ok=True)
        with d["out_path"].open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        if args.verbose:
            print(f"[index] {d['out_path']} "
                  f"(pages={output['num_pages']}, ok={output['num_fetch_ok']}, "
                  f"err={output['num_fetch_error']})")

    print(f"Done. Registry: {len(registry)} URLs | content: {content_dir} | index: {docs_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch pages from URL JSON files and extract text"
    )

    parser.add_argument("--url_path", type=str, default=None)
    parser.add_argument("--url_dir", type=str, default=None)

    # Output is always a directory (the "page_dir"): it holds content/, docs/
    # and _registry.json. Works for both --url_path and --url_dir.
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
        help="Maximum number of concurrent URL fetches, globally (default: 5)",
    )
    parser.add_argument(
        "--max_doc_concurrent",
        type=int,
        default=3,
        help="Deprecated/ignored: fetching is now global across docs, not per-doc.",
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
        help="Re-fetch every URL even if already in the registry (default: skip cached).",
    )

    parser.add_argument(
        "--full_output",
        action="store_true",
        help=(
            "Add diagnostic fields to the per-doc index (http_status, content_type, "
            "text_len, snippet, source_block, assets, fetched_at). Default keeps only "
            "the fields the next step (export_clean_txt.py) reads."
        ),
    )

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if bool(args.url_path) == bool(args.url_dir):
        raise ValueError("Provide exactly one of --url_path or --url_dir")

    if not args.output_dir:
        raise ValueError("--output_dir is required")

    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
