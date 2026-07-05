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
    - DOCX (Word / OOXML): zip + word/document.xml (stdlib)

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
import io
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlparse, urlunparse
from xml.etree import ElementTree as ET

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
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
_BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "application/pdf,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
}
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

    if content_kind in ("pdf", "docx"):
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

    # Check actual bytes first — prevents mis-classifying HTML error pages
    # that the server served for a .pdf/.docx URL (e.g. 403 redirect).
    head = content_bytes[:500].lower()

    if content_bytes[:4] == b"%PDF" or head.startswith(b"%pdf"):
        return "pdf"

    if content_bytes[:4] == b"PK\x03\x04":
        return "docx"

    if b"<html" in head or b"<!doctype html" in head:
        return "html"

    # Fall back to Content-Type header, then URL extension.
    if "application/pdf" in lower_type:
        return "pdf"
    if "wordprocessingml.document" in lower_type:
        return "docx"
    if "text/plain" in lower_type:
        return "text"
    if "text/html" in lower_type:
        return "html"

    if lower_url.endswith(".pdf"):
        return "pdf"
    if lower_url.endswith(".docx"):
        return "docx"
    if lower_url.endswith(".txt"):
        return "text"

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


# Layout-aware PDF extraction is the default; it can be turned off globally by
# flipping this flag (extract_pdf_content then falls back to plain get_text).
PDF_LAYOUT_CLEAN = True

# When True, use the pymupdf4llm backend (footnote-precise but ~30x slower).
# Default False: the block extractor is the right speed/quality trade-off for
# embedding-based alignment. Flip for a high-fidelity one-off extraction.
PDF_HIGH_FIDELITY = False

_PDF_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")
_PDF_END_SENT_RE = re.compile(r"[.!?…:;]\s*(?:[“”\"')\]])?\s*$")
_PDF_FOLIO_RE = re.compile(r"\[\d+[ab]\]")
_PDF_LEADING_FN_RE = re.compile(r"^\d{1,3}\s")
# A short digit run rendered in a much smaller font than the body is a footnote
# reference marker (used together with the superscript flag at span level).
_PDF_FN_NUM_RE = re.compile(r"^\d{1,4}$")
# PyMuPDF span flag bits.
_PDF_FLAG_SUPERSCRIPT = 1 << 0
_PDF_FLAG_BOLD = 1 << 4
_PDF_FN_SIZE_RATIO = 0.75   # span smaller than body * this, and digits-only -> footnote


def _sample_page_indices(total: int, start: int, n: int) -> list:
    available = total - start
    if available <= 0:
        return []
    step = max(1, available // min(n, available))
    return list(range(start, total, step))[:n]


def _detect_pdf_bounds(doc, sample_pages=20, min_repeat_ratio=0.3):
    """Y-position header/footer bounds via page sampling."""
    top_cands, bot_cands = [], []
    n_sampled = 0
    for i in _sample_page_indices(len(doc), 0, sample_pages):
        page = doc[i]
        h = page.rect.height
        blocks = page.get_text("blocks")
        if not blocks:
            continue
        n_sampled += 1
        for b in blocks:
            y0, text = b[1], str(b[4]).strip()
            if not text:
                continue
            yr = round(y0 / 5) * 5
            norm = re.sub(r"\d+", "#", text).strip()[:50]
            if y0 < h * 0.15:
                top_cands.append((yr, norm))
            if y0 > h * 0.85:
                bot_cands.append((yr, norm))

    if not n_sampled:
        return None, None

    min_rep = max(2, n_sampled * min_repeat_ratio)
    from collections import Counter

    upper = None
    hits = [(y, c) for (y, _), c in Counter(top_cands).items() if c >= min_rep]
    if hits:
        upper = max(y for y, _ in hits) + 20

    lower = None
    hits = [(y, c) for (y, _), c in Counter(bot_cands).items() if c >= min_rep]
    if hits:
        lower = min(y for y, _ in hits) - 10

    return upper, lower


def _detect_font_hierarchy(doc, start_page=2, sample_pages=20):
    """Map font sizes to heading levels; 0 = body."""
    from collections import Counter
    size_ctr = Counter()
    for i in _sample_page_indices(len(doc), start_page, sample_pages):
        page = doc[i]
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    sz = round(float(span.get("size", 0)), 1)
                    if text and len(text) > 1:
                        size_ctr[sz] += len(text)
    if not size_ctr:
        return {}, 10.0
    body = size_ctr.most_common(1)[0][0]
    headings = sorted([s for s in size_ctr if s > body], reverse=True)
    mapping = {s: i + 1 for i, s in enumerate(headings)}
    for s in size_ctr:
        if s <= body:
            mapping[s] = 0
    return mapping, body


def _is_footnote_span(span, body_size):
    """
    True if a span is a footnote reference marker.

    PyMuPDF exposes per-span ``size`` and ``flags``; a footnote ref is either
    flagged superscript, or a short digit run set in a much smaller font than
    the body (some PDFs raise the baseline without setting the superscript bit).
    Detecting it at span level lets us drop the marker WITHOUT gluing it to the
    adjacent number/word -- e.g. "năm thứ 1" + superscript "7" stays "năm thứ 1"
    instead of the corrupted "năm thứ 17".
    """
    flags = int(span.get("flags", 0))
    if flags & _PDF_FLAG_SUPERSCRIPT:
        return True
    txt = span.get("text", "").strip()
    size = float(span.get("size", 0))
    if body_size and size < body_size * _PDF_FN_SIZE_RATIO and _PDF_FN_NUM_RE.match(txt):
        return True
    return False


def _assemble_block(block, body_size, font_map=None):
    """
    Assemble a dict-block's text from its spans, dropping footnote-reference
    spans. Returns ``(cleaned_text, max_body_font_size)`` or ``(None, 0.0)``.
    ``max_body_font_size`` (over the kept spans) drives heading-level mapping.
    """
    line_texts = []
    max_size = 0.0
    for line in block.get("lines", []):
        parts = []
        for span in line.get("spans", []):
            txt = span.get("text", "")
            if not txt:
                continue
            if _is_footnote_span(span, body_size):
                continue
            parts.append(txt)
            sz = float(span.get("size", 0))
            if sz > max_size:
                max_size = sz
        if parts:
            line_texts.append("".join(parts))
    if not line_texts:
        return None, 0.0
    return _clean_block(" ".join(line_texts), font_map), max_size


def _clean_block(text, font_map=None):
    """
    Normalize an assembled block: strip invisible chars, apply a font map, and
    remove folio markers. Footnote-reference removal happens earlier at span
    level (see :func:`_is_footnote_span`), so no inline digit-stripping is done
    here -- that avoids corrupting legitimate word+digit tokens.
    """
    text = re.sub(r"[​‌‍﻿­]", "", text)
    text = text.replace(" ", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    if font_map:
        for broken, correct in font_map.items():
            text = text.replace(broken, correct)
    text = _PDF_FOLIO_RE.sub("", text)
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def _merge_blocks(blocks):
    """Join consecutive body blocks whose predecessor lacks a sentence boundary."""
    merged = []
    for text, level in blocks:
        if merged and merged[-1][1] == 0 and level == 0:
            prev, _ = merged[-1]
            if not _PDF_END_SENT_RE.search(prev):
                merged[-1] = (prev.rstrip() + " " + text.lstrip(), 0)
                continue
        merged.append((text, level))
    return merged


def _pdf_layout_text(
    doc,
    *,
    font_map=None,
    start_page=0,
    sample_pages=20,
    min_repeat_ratio=0.3,
):
    """
    Layout-aware plain-text extraction, span-aware:
      - Y-position header/footer removal (sampling-based)
      - Font-hierarchy detection (headings output as plain text, no markers)
      - Footnote-reference removal at span level (superscript / tiny-font digits),
        so markers never glue onto adjacent numbers or words
      - Folio-marker strip, page-number / footnote-block skip
      - Continuation-block merge within page, pending buffer across pages
    """
    upper, lower = _detect_pdf_bounds(
        doc, sample_pages=sample_pages, min_repeat_ratio=min_repeat_ratio
    )
    size_map, body_size = _detect_font_hierarchy(
        doc, start_page=max(start_page, 2), sample_pages=sample_pages
    )
    out = []
    pending = ""

    for pi in range(start_page, len(doc)):
        page = doc[pi]
        blocks = [b for b in page.get_text("dict").get("blocks", []) if "lines" in b]
        blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        page_blocks = []

        for b in blocks:
            y0 = float(b["bbox"][1])
            if upper is not None and y0 < upper:
                continue
            if lower is not None and y0 > lower:
                continue

            cleaned, max_size = _assemble_block(b, body_size, font_map)
            if not cleaned or _PDF_NUM_RE.match(cleaned) or _PDF_LEADING_FN_RE.match(cleaned):
                continue

            level = size_map.get(round(max_size, 1), 0)
            page_blocks.append((cleaned, level))

        page_blocks = _merge_blocks(page_blocks)

        if pending and page_blocks:
            first_text, first_level = page_blocks[0]
            if first_level == 0:
                page_blocks[0] = (pending.rstrip() + " " + first_text.lstrip(), 0)
            else:
                out.append(pending)
            pending = ""
        elif pending:
            continue

        if page_blocks:
            last_text, last_level = page_blocks[-1]
            if last_level == 0 and not _PDF_END_SENT_RE.search(last_text):
                pending = last_text
                page_blocks = page_blocks[:-1]

        out.extend(t for t, _ in page_blocks)

    if pending:
        out.append(pending)

    return "\n\n".join(out).strip()


# ---------------------------------------------------------------------------
# pymupdf4llm-based PDF extraction (preferred backend when installed)
#
# pymupdf4llm renders superscript footnote references as <sup>N</sup>, so they
# can be removed cleanly without gluing the surrounding words together or
# corrupting adjacent digits ("nam thu 1<sup>7</sup>" stays "nam thu 1", where
# raw block text gives the corrupted "nam thu 17").
# ---------------------------------------------------------------------------

_MD_SUP_RE = re.compile(r"<sup>.*?</sup>", re.DOTALL)
# Remaining inline HTML tag markers (e.g. <u>, </u>, <br>) — drop the tag but
# keep inner text: underline usually marks a proper noun in these translations.
_MD_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+")
_MD_EMPH_RE = re.compile(r"(\*\*\*|\*\*|\*|___|__|_)(?=\S)(.+?)(?<=\S)\1")
_MD_HR_RE = re.compile(r"^\s*[-*_]{3,}\s*$")
_MD_BOLD_LINE_RE = re.compile(r"^\*\*[^*]{1,80}\*\*\s*$")
_MD_LEAD_PUNCT_RE = re.compile(r"^[.,;:]\s+")
# A 3-4 digit number glued to the end of a word is a footnote counter (752, 1026)
# whose superscript styling was lost; 1-2 digit gluings are usually real content
# with a dropped space ("thang8" = "thang 8") and must be kept.
_MD_GLUED_FN_RE = re.compile(r"(?<=[^\W\d_])\d{3,4}(?=[\s.,;:)\]]|$)", re.UNICODE)


def _md_clean_line(line: str, font_map=None) -> str:
    """One markdown line -> plain text (drop sup refs, md marks, folio markers)."""
    line = _MD_SUP_RE.sub("", line)
    line = _MD_TAG_RE.sub("", line)
    line = _MD_IMAGE_RE.sub("", line)
    line = _MD_LINK_RE.sub(r"\1", line)
    line = _MD_HEADING_RE.sub("", line)
    line = _MD_EMPH_RE.sub(r"\2", line)
    line = line.replace("`", "").replace("|", " ")
    # invisible characters
    line = re.sub(r"[​‌‍﻿­]", "", line)
    line = line.replace(" ", " ")
    if font_map:
        for broken, correct in font_map.items():
            line = line.replace(broken, correct)
    line = _PDF_FOLIO_RE.sub("", line)
    line = _MD_GLUED_FN_RE.sub("", line)
    line = re.sub(r"\s+", " ", line).strip()
    line = _MD_LEAD_PUNCT_RE.sub("", line)
    return line


def _md_line_key(line: str) -> str:
    """Normalized fingerprint used to detect per-page repeated header/footer lines."""
    key = _md_clean_line(line)
    key = re.sub(r"\d+", "#", key).strip().lower()
    return key[:60]


def _pdf_markdown_text(doc, *, font_map=None, min_repeat_ratio=0.3) -> str:
    """
    Plain-text extraction through pymupdf4llm markdown:

      - running headers/footers: lines near a page's edges whose normalized text
        repeats on >= ``min_repeat_ratio`` of pages are removed everywhere
      - <sup>..</sup> footnote references, folio markers ([1a], [2b]), markdown
        formatting and page-number-only lines are stripped
      - footnote paragraphs (starting with a bare footnote number) are dropped
      - a paragraph cut by a page break is joined with the next page's first
        paragraph when it does not end with sentence punctuation

    Raises ImportError when pymupdf4llm is not installed (caller falls back to
    the block-based extractor).
    """
    import pymupdf4llm
    from collections import Counter

    # The ML layout engine (pymupdf.layout, ONNX) costs ~0.14s/page and adds
    # nothing for prose books; the legacy heuristic engine (~4x faster) keeps
    # the <sup> footnote markers we rely on. Paragraph reflow lost by the
    # legacy engine is restored by the continuation merge below.
    try:
        pymupdf4llm.use_layout(False)
    except AttributeError:
        pass  # older pymupdf4llm without the layout switch

    chunks = pymupdf4llm.to_markdown(doc, page_chunks=True, show_progress=False)
    pages_lines = [str(ch.get("text", "")).split("\n") for ch in chunks]

    # Repeated edge lines across pages -> running headers/footers.
    EDGE = 4
    edge_ctr = Counter()
    n_pages = 0
    for lines in pages_lines:
        nz = [l for l in lines if l.strip() and not _MD_HR_RE.match(l)]
        if not nz:
            continue
        n_pages += 1
        for key in {k for k in map(_md_line_key, nz[:EDGE] + nz[-EDGE:]) if k}:
            edge_ctr[key] += 1
    min_rep = max(2, n_pages * min_repeat_ratio)
    banned = {k for k, c in edge_ctr.items() if c >= min_rep}

    out = []
    pending = ""
    for lines in pages_lines:
        # paragraphs = blank-line-separated runs; headings end a paragraph too
        paras = []          # (text, is_heading)
        buf: list = []
        def _flush():
            if buf:
                paras.append((" ".join(buf), False))
                buf.clear()
        for raw in lines:
            if not raw.strip() or _MD_HR_RE.match(raw):
                _flush()
                continue
            if _md_line_key(raw) in banned:
                _flush()
                continue
            is_heading = bool(
                _MD_HEADING_RE.match(raw.lstrip())
                or _MD_BOLD_LINE_RE.match(raw.strip())
            )
            cleaned = _md_clean_line(raw, font_map)
            if not cleaned or _PDF_NUM_RE.match(cleaned):
                continue
            if is_heading:
                _flush()
                paras.append((cleaned, True))
            else:
                buf.append(cleaned)
        _flush()

        # drop footnote paragraphs (bare footnote number + text)
        paras = [(t, h) for (t, h) in paras if not _PDF_LEADING_FN_RE.match(t)]
        if not paras:
            continue

        # merge consecutive body paragraphs when the previous one has no
        # sentence-ending punctuation (spurious blank lines / column breaks)
        merged: list = []
        for t, h in paras:
            if (merged and not h and not merged[-1][1]
                    and not _PDF_END_SENT_RE.search(merged[-1][0])):
                merged[-1] = (merged[-1][0].rstrip() + " " + t.lstrip(), False)
            else:
                merged.append((t, h))
        paras = merged

        # join a paragraph cut by the page break
        if pending:
            first_text, first_heading = paras[0]
            if not first_heading:
                paras[0] = (pending.rstrip() + " " + first_text.lstrip(), False)
            else:
                out.append(pending)
            pending = ""

        last_text, last_heading = paras[-1]
        if not last_heading and not _PDF_END_SENT_RE.search(last_text):
            pending = last_text
            paras = paras[:-1]

        out.extend(t for t, _ in paras)

    if pending:
        out.append(pending)

    return "\n\n".join(out).strip()


def extract_pdf_content(
    pdf_bytes: bytes,
    min_text_len: int = DEFAULT_MIN_TEXT_LEN,
    collect_assets: bool = False,
    layout_clean: bool = None,
    font_map: Dict[str, str] = None,
    high_fidelity: bool = False,
) -> Dict[str, Any]:
    """
    Extract text from a PDF.

    Default backend is the span-aware block extractor ``_pdf_layout_text``
    (plain PyMuPDF): it strips running headers/footers and folio markers, and
    removes footnote-reference markers at span level (superscript / tiny-font
    digits) so they never corrupt the adjacent text -- "năm thứ 1" + superscript
    "7" stays "năm thứ 1", not "năm thứ 17". It runs ~30x faster than the
    pymupdf4llm backend and, unlike the old inline-digit regex, does not delete
    legitimate glued numbers (month/day counts, "q1"-style citations).

    The residual gap vs pymupdf4llm is footnote numbers that the source PDF
    fused into a body-size text run (no separate span to drop) -- a few hundred
    digits in a 1600-page book, negligible for embedding/alignment. Set
    ``high_fidelity=True`` (or PDF_HIGH_FIDELITY) to use the pymupdf4llm backend,
    which resolves more of these via <sup> tags at the cost of the runtime.
    """
    if layout_clean is None:
        layout_clean = PDF_LAYOUT_CLEAN
    if not high_fidelity:
        high_fidelity = PDF_HIGH_FIDELITY

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    assets = (
        [{"type": "pdf_page", "page_num": i + 1, "needs_ocr": True} for i in range(len(doc))]
        if collect_assets else []
    )

    extractor = "pymupdf"
    text = ""
    if layout_clean and high_fidelity:
        # Opt-in backend: pymupdf4llm (clean footnote refs via <sup>, ~30x slower).
        try:
            text = _pdf_markdown_text(doc, font_map=font_map)
            if text:
                extractor = "pymupdf4llm"
        except ImportError:
            text = ""  # not installed -> fall through to the block extractor
        except Exception:
            text = ""

    if layout_clean and not text:
        # Default backend: fast block-based layout extractor.
        try:
            text = _pdf_layout_text(doc, font_map=font_map)
            if text:
                extractor = "pymupdf_layout"
        except Exception:
            text = ""  # fall back to plain extraction below

    if not text:
        pages = [p.get_text("text").strip() for p in doc]
        text = "\n\n".join(t for t in pages if t).strip()

    needs_ocr = len(text) < min_text_len

    return {
        "text": text,
        "text_len": len(text),
        "extractor": extractor,
        "html_title": "",
        "needs_ocr": needs_ocr,
        "assets": assets if needs_ocr else [],
    }


def extract_docx_content(
    docx_bytes: bytes,
    min_text_len: int = DEFAULT_MIN_TEXT_LEN,
) -> Dict[str, Any]:
    """
    Extract text from a Word .docx (OOXML) file.

    A .docx is a ZIP whose main text lives in ``word/document.xml``. We take the
    text of every paragraph (``w:p``) by concatenating its runs (``w:t``); this
    also captures table cells, which are paragraphs nested inside ``w:tbl``.

    Non-Word OOXML (xlsx/pptx) has no ``word/document.xml`` and yields empty text.
    """
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    empty = {
        "text": "", "text_len": 0, "extractor": "docx",
        "html_title": "", "needs_ocr": False, "assets": [],
    }

    try:
        zf = zipfile.ZipFile(io.BytesIO(docx_bytes))
    except zipfile.BadZipFile:
        return empty

    if "word/document.xml" not in zf.namelist():
        return empty  # not a Word document (e.g. xlsx/pptx)

    try:
        root = ET.fromstring(zf.read("word/document.xml"))
    except ET.ParseError:
        return empty

    paras: List[str] = []
    for p in root.iter(f"{W}p"):
        line = "".join(t.text for t in p.iter(f"{W}t") if t.text).strip()
        if line:
            paras.append(line)

    text = "\n".join(paras).strip()
    return {
        "text": text,
        "text_len": len(text),
        "extractor": "docx",
        "html_title": "",
        "needs_ocr": False,
        "assets": [],
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
    elif content_kind == "docx":
        extracted = extract_docx_content(
            docx_bytes=content_bytes,
            min_text_len=min_text_len,
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

    # PDF / DOCX: also keep the raw file for future OCR / re-extraction.
    if content_kind in ("pdf", "docx"):
        content_dir.mkdir(parents=True, exist_ok=True)
        (content_dir / f"{stem}.{content_kind}").write_bytes(content_bytes)
        raw_file = f"{CONTENT_SUBDIR}/{stem}.{content_kind}"

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
        parsed = urlparse(url)
        referer = f"{parsed.scheme}://{parsed.netloc}/"
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout),
                allow_redirects=True,
                headers={**_BROWSER_HEADERS, "Referer": referer},
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

        if http_status >= 400:
            return _error_entry(canonical_url, hash_str, f"HTTP {http_status}")

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
