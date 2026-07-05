"""
One-command corpus pipeline: URL JSONs -> final clean TXT corpus.

Replaces the four-step chain (fetch_pages -> export_raw_txt -> filter_corpus ->
export_clean_txt) with a single command and a single, human-readable output
tree. All proven logic (extractors, cleaning, filter gates, dedup) is reused
from the existing modules; only orchestration and the output layer are new.

    python -m lib.web.corpus_pipeline \
        --url_dir ./urls --output_dir ./corpus --direction zh2vi

Output tree::

    corpus/
      index.tsv        # 1 row per unique URL: status, kept, reason, files, url
      report.txt       # human summary: per source doc + drop-reason histogram
      txt/             # FINAL corpus (flat): {gid:04d}__{title}__{domain}__{hash}.txt
      _raw/            # original bytes (stable names, re-extract without re-crawl)
      _meta/           # per source doc: {doc_id}.jsonl (1 line per page, full trace)
      _cache/
        fetch.json     # canonical_url -> fetch entry (resume cache)
        text/          # extracted-text cache ({urlhash}.json)

Stages (each skippable/cached):

    1. LOAD     url JSONs (search_urls output) -> source docs + unique URL set
    2. FETCH    raw bytes for unfetched URLs (async, polite, retries)
    3. EXTRACT  text from _raw bytes (pdf/docx/html/txt) -> text cache
    4. FILTER   clean + per-page gates + exact/near dedup + keyword gate
    5. WRITE    txt/ + _meta/*.jsonl + index.tsv + report.txt

Useful flags:

    --force            re-fetch everything
    --refetch-errors   re-fetch only URLs whose cached status is an error
    --reextract        ignore the text cache (after changing extractor code)
    --skip-fetch       never hit the network (offline re-filter / re-extract)
    --only C001,C002   restrict to some source docs
    --import-legacy DIR  seed _cache/_raw from an old fetch_pages output dir
"""

import argparse
import asyncio
import json
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import unquote, urlparse

from lib.web.fetch_pages import (
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MIN_TEXT_LEN,
    DEFAULT_RETRY_BACKOFF,
    DEFAULT_SLEEP_RANGE,
    DEFAULT_TIMEOUT,
    DEFAULT_USER_AGENT,
    _detect_charset,
    _do_fetch,
    detect_content_kind,
    extract_docx_content,
    extract_html_content,
    extract_pdf_content,
    extract_plain_text_content,
    group_duplicate_urls,
    is_blocked_url,
    iter_url_records,
    url_hash,
)
from lib.web.export_clean_txt import (
    calculate_jaccard,
    clean_text_for_export,
    extreme_clean_for_compare,
    get_character_ngrams,
    is_vertical_ocr_text,
    language_ratios,
    safe_filename,
    sha1_text,
    short_hash,
    traditional_score,
)
# Bump when extractor code changes so cached texts are recomputed on next run
# (or pass --reextract to force it once).
EXTRACTOR_VERSION = 1

TXT_SUBDIR = "txt"
RAW_SUBDIR = "_raw"
META_SUBDIR = "_meta"
CACHE_SUBDIR = "_cache"

_KIND_EXT = {"pdf": "pdf", "docx": "docx", "html": "html", "text": "txt"}

# A document whose mean line length exceeds this has lost its paragraph
# structure (headers inlined / whole doc on a few lines); such a copy is
# de-prioritised when choosing a duplicate group's representative. Healthy
# prose paragraphs average well under this; >1000 chars/line is pathological.
_DEDUP_MAX_AVG_LINE = 1000.0

_QUOTED_RE = re.compile(r'"([^"]+)"')
_WS_RE = re.compile(r"\s+")


# =============================================================================
# Small text helpers (previously in export_raw_txt / filter_corpus)
# =============================================================================

def structural_metrics(text: str) -> Dict[str, float]:
    """
    Cheap, format-aware facts used by the degenerate-text gate and dedup.

    digit/letter ratios are format-invariant (robust to line re-wrapping);
    short_pct/avg_line describe line structure.
    """
    n = len(text) or 1
    lines = text.splitlines() or [""]
    nl = len(lines)
    short = sum(1 for L in lines if len(L.strip()) < 15)
    return {
        "short_pct": round(short / nl, 4),
        "avg_line": round(sum(len(L) for L in lines) / nl, 2),
        "digit_ratio": round(sum(c.isdigit() for c in text) / n, 4),
        "letter_ratio": round(sum(c.isalpha() for c in text) / n, 4),
    }


def collapse(s: str) -> str:
    """NFC + lowercase + single-spaced — for anchor substring matching."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", s).lower()).strip()


def extract_anchors(query: str) -> List[str]:
    """Pull quoted anchor phrases from a search query string."""
    phrases = _QUOTED_RE.findall(query or "") or ([query] if query else [])
    return [collapse(p) for p in phrases if p and p.strip()]


# =============================================================================
# Layout helpers
# =============================================================================

class Layout:
    """Paths of the output tree."""

    def __init__(self, root: Path):
        self.root = root
        self.txt = root / TXT_SUBDIR
        self.raw = root / RAW_SUBDIR
        self.meta = root / META_SUBDIR
        self.cache = root / CACHE_SUBDIR
        self.text_cache = self.cache / "text"
        self.fetch_cache = self.cache / "fetch.json"
        self.index_tsv = root / "index.tsv"
        self.report = root / "report.txt"

    def make_dirs(self) -> None:
        for d in (self.txt, self.raw, self.meta, self.text_cache):
            d.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _slug_from_url(url: str) -> str:
    """Readable last-path-segment of a URL (percent-decoded)."""
    name = Path(urlparse(url).path).name or urlparse(url).netloc
    name = unquote(unquote(name))
    name = re.sub(r"\.(pdf|docx?|html?|txt|php|aspx?)$", "", name, flags=re.I)
    return safe_filename(name, max_len=60) or "page"


def raw_stem(canonical_url: str, title: str, hash_str: str) -> str:
    """
    Stable, readable stem for files in _raw/ (no global id -- ids are only
    assigned to the final txt corpus): {domain}__{slug}__{hash8}.
    """
    domain = safe_filename(urlparse(canonical_url).netloc.lower() or "unknown", max_len=40)
    slug = safe_filename(title, max_len=60) if title else ""
    if not slug:
        slug = _slug_from_url(canonical_url)
    return f"{domain}__{slug}__{hash_str[:8]}"


# =============================================================================
# Stage 1 — load URL JSONs
# =============================================================================

def load_source_docs(
    url_files: List[Path],
    only: Optional[Set[str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Read search_urls JSONs. Returns (docs, url_map):

      docs:    [{doc_id, direction, url_file, items: [grouped url records]}]
      url_map: canonical_url -> representative record (title, source_queries
               merged across docs) — the fetch/extract unit.
    """
    docs: List[Dict[str, Any]] = []
    url_map: Dict[str, Dict[str, Any]] = {}

    for uf in url_files:
        data = load_json(uf, {})
        doc_id = str(data.get("doc_id") or uf.stem)
        if only and doc_id not in only:
            continue
        grouped = group_duplicate_urls(iter_url_records(data))
        docs.append({
            "doc_id": doc_id,
            "direction": data.get("direction", "zh2vi"),
            "url_file": str(uf),
            "items": grouped,
        })
        for it in grouped:
            cu = it["canonical_url"]
            rec = url_map.setdefault(cu, {
                "canonical_url": cu,
                "title": "",
                "source_queries": [],
                "docs": [],
            })
            if not rec["title"] and it.get("title"):
                rec["title"] = str(it["title"])
            rec["source_queries"].extend(it.get("source_queries") or [])
            rec["docs"].append(doc_id)

    return docs, url_map


# =============================================================================
# Stage 2 — fetch raw bytes
# =============================================================================

async def _fetch_raw_one(
    session,
    canonical_url: str,
    title: str,
    layout: Layout,
    semaphore: asyncio.Semaphore,
    args,
) -> Dict[str, Any]:
    """Fetch one URL and save the raw bytes. Returns a fetch-cache entry."""
    import aiohttp

    hash_str = url_hash(canonical_url)
    now = datetime.now(timezone.utc).isoformat()

    def _err(error: str, http_status=None) -> Dict[str, Any]:
        return {
            "canonical_url": canonical_url, "hash": hash_str, "status": "error",
            "http_status": http_status, "content_type": "", "content_kind": "unknown",
            "final_url": "", "charset": "utf-8", "raw_file": None,
            "error": error, "fetched_at": now,
        }

    if is_blocked_url(canonical_url):
        return _err("skipped:blocked_domain")

    last_error = "unknown error"
    for attempt in range(args.max_retries + 1):
        try:
            raw = await _do_fetch(
                session, canonical_url, semaphore, args.timeout, tuple(args.sleep_range)
            )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.max_retries:
                await asyncio.sleep(args.retry_backoff ** attempt)
            continue
        except Exception as exc:
            return _err(str(exc))

        http_status = raw["status_code"]
        if http_status == 429:
            last_error = "HTTP 429 (rate limited)"
            if attempt < args.max_retries:
                await asyncio.sleep(args.retry_backoff ** (attempt + 2))
            continue
        if http_status >= 500:
            last_error = f"HTTP {http_status}"
            if attempt < args.max_retries:
                await asyncio.sleep(args.retry_backoff ** attempt)
            continue
        if http_status >= 400:
            return _err(f"HTTP {http_status}", http_status)

        content_bytes = raw["content_bytes"]
        kind = detect_content_kind(
            url=raw["final_url"], content_type=raw["content_type"],
            content_bytes=content_bytes,
        )
        ext = _KIND_EXT.get(kind, "bin")
        stem = raw_stem(canonical_url, title, hash_str)
        raw_rel = f"{RAW_SUBDIR}/{stem}.{ext}"
        (layout.root / raw_rel).parent.mkdir(parents=True, exist_ok=True)
        (layout.root / raw_rel).write_bytes(content_bytes)

        return {
            "canonical_url": canonical_url, "hash": hash_str, "status": "ok",
            "http_status": http_status, "content_type": raw["content_type"],
            "content_kind": kind, "final_url": raw["final_url"],
            "charset": _detect_charset(raw["content_type"]),
            "raw_file": raw_rel, "error": "", "fetched_at": now,
        }

    return _err(last_error)


async def fetch_stage(
    url_map: Dict[str, Dict[str, Any]],
    layout: Layout,
    args,
) -> Dict[str, Dict[str, Any]]:
    """Fetch every unique URL not already in the cache. Returns the cache."""
    import aiohttp

    cache: Dict[str, Dict[str, Any]] = load_json(layout.fetch_cache, {})

    def _needs_fetch(cu: str) -> bool:
        if args.force:
            return True
        e = cache.get(cu)
        if e is None:
            return True
        if args.refetch_errors and e.get("status") != "ok":
            return True
        return False

    pending = [cu for cu in url_map if _needs_fetch(cu)]
    print(f"[fetch] unique URLs: {len(url_map)} | to fetch: {len(pending)} "
          f"| cached: {len(url_map) - len(pending)}")

    if args.skip_fetch or not pending:
        if pending and args.skip_fetch:
            print(f"[fetch] --skip-fetch: leaving {len(pending)} URL(s) unfetched")
        return cache

    semaphore = asyncio.Semaphore(args.max_concurrent)
    connector = aiohttp.TCPConnector(limit=args.max_concurrent * 2)
    async with aiohttp.ClientSession(
        connector=connector, headers={"User-Agent": args.user_agent}
    ) as session:
        tasks = [
            _fetch_raw_one(session, cu, url_map[cu].get("title", ""), layout, semaphore, args)
            for cu in pending
        ]
        done = 0
        for fut in asyncio.as_completed(tasks):
            entry = await fut
            cache[entry["canonical_url"]] = entry
            done += 1
            if args.verbose:
                err = f" ({entry['error']})" if entry.get("error") else ""
                print(f"  [{done}/{len(pending)}] {entry['status'].upper()}{err} "
                      f"{entry['canonical_url']}")
            if done % 25 == 0:
                dump_json(layout.fetch_cache, cache)  # checkpoint

    dump_json(layout.fetch_cache, cache)
    return cache


# =============================================================================
# Stage 3 — extract text (with cache)
# =============================================================================

def _extract_from_raw(
    entry: Dict[str, Any],
    layout: Layout,
    min_text_len: int,
    high_fidelity: bool = False,
) -> Dict[str, Any]:
    """Extract text for one fetched URL from its raw bytes."""
    raw_rel = entry.get("raw_file")
    kind = entry.get("content_kind", "unknown")
    raw_path = (layout.root / raw_rel) if raw_rel else None
    if raw_path is None or not raw_path.exists():
        return {"text": "", "extractor": "", "html_title": "",
                "needs_ocr": False, "error": "raw_file_missing"}

    content = raw_path.read_bytes()
    try:
        if kind == "pdf":
            ex = extract_pdf_content(content, min_text_len=min_text_len,
                                     high_fidelity=high_fidelity)
        elif kind == "docx":
            ex = extract_docx_content(content, min_text_len=min_text_len)
        else:
            decoded = content.decode(entry.get("charset") or "utf-8", errors="replace")
            if kind == "text":
                ex = extract_plain_text_content(decoded)
            else:
                ex = extract_html_content(
                    html=decoded, url=entry.get("final_url", ""), min_text_len=min_text_len
                )
    except Exception as exc:
        return {"text": "", "extractor": "", "html_title": "",
                "needs_ocr": False, "error": f"extract_failed: {exc}"}

    return {
        "text": ex.get("text", "") or "",
        "extractor": ex.get("extractor", ""),
        "html_title": ex.get("html_title", "") or "",
        "needs_ocr": bool(ex.get("needs_ocr", False)),
        "error": "",
    }


def _extract_job(payload: Tuple[str, Dict[str, Any], str, int]) -> Tuple[str, Dict[str, Any]]:
    """
    Module-level extraction worker (must be picklable for ProcessPoolExecutor).

    payload = (canonical_url, fetch_entry, output_root, min_text_len). Extracts
    text from the entry's raw bytes, stamps the extractor version, writes the
    text-cache file, and returns (canonical_url, result). Writing per-URL cache
    files means there is no shared state between workers.
    """
    cu, entry, root_str, min_text_len, high_fidelity = payload
    layout = Layout(Path(root_str))
    ex = _extract_from_raw(entry, layout, min_text_len, high_fidelity=high_fidelity)
    ex["v"] = EXTRACTOR_VERSION
    dump_json(layout.text_cache / f"{entry['hash']}.json", ex)
    return cu, ex


def extract_stage(
    url_map: Dict[str, Dict[str, Any]],
    cache: Dict[str, Dict[str, Any]],
    layout: Layout,
    args,
) -> Dict[str, Dict[str, Any]]:
    """
    Ensure every fetched URL has an up-to-date text-cache entry.

    Returns canonical_url -> {text, extractor, html_title, needs_ocr, error}.
    Cache files live at _cache/text/{urlhash}.json and are stamped with
    EXTRACTOR_VERSION so extractor changes invalidate them.

    The default block PDF backend is fast (~3s/book), so extraction runs in a
    thread pool. When ``--high_fidelity`` selects the slower pymupdf4llm backend,
    pass ``--extract_mode process`` to parallelise it past the GIL (measured
    ~1.7x on 4 workers); ``--extract_workers`` sets the pool size.
    """
    results: Dict[str, Dict[str, Any]] = {}
    todo: List[str] = []

    for cu in url_map:
        entry = cache.get(cu)
        if not entry or entry.get("status") != "ok":
            continue
        tc_path = layout.text_cache / f"{entry['hash']}.json"
        cached = load_json(tc_path, None)
        # Legacy-imported HTML/text pages have no stored raw bytes (the old
        # fetch_pages kept only the extracted .txt). They can never be
        # re-extracted, so always reuse their cached text -- even under
        # --reextract, which only re-runs extractors that have raw bytes.
        if (cached is not None and cached.get("v") == "legacy"
                and not entry.get("raw_file")):
            results[cu] = cached
            continue
        if not args.reextract and cached is not None and cached.get("v") == EXTRACTOR_VERSION:
            results[cu] = cached
            continue
        todo.append(cu)

    if not todo:
        print(f"[extract] cached: {len(results)} | to extract: 0")
        return results

    # Heaviest first so a slow 1600-page PDF starts early and overlaps the rest.
    def _weight(cu: str) -> int:
        rf = cache[cu].get("raw_file")
        try:
            return (layout.root / rf).stat().st_size if rf else 0
        except OSError:
            return 0
    todo.sort(key=_weight, reverse=True)

    root_str = str(layout.root)
    payloads = [(cu, cache[cu], root_str, args.min_text_len, args.high_fidelity)
                for cu in todo]

    n_workers = max(1, args.extract_workers)
    use_process = args.extract_mode == "process" and n_workers > 1 and len(todo) > 1
    Pool = ProcessPoolExecutor if use_process else ThreadPoolExecutor
    mode = "process" if use_process else "thread"
    print(f"[extract] cached: {len(results)} | to extract: {len(todo)} "
          f"| {mode} x{n_workers}")

    with Pool(max_workers=n_workers) as pool:
        for i, (cu, ex) in enumerate(pool.map(_extract_job, payloads), 1):
            results[cu] = ex
            if args.verbose or i % 20 == 0:
                print(f"  [extract {i}/{len(todo)}] {ex.get('extractor','?'):14} "
                      f"len={len(ex.get('text','')):>9,}  {cu[:70]}")

    return results


# =============================================================================
# Stage 4 — clean + filter (per-page gates, dedup, keyword gate)
# =============================================================================

def filter_stage(
    docs: List[Dict[str, Any]],
    url_map: Dict[str, Dict[str, Any]],
    cache: Dict[str, Dict[str, Any]],
    texts: Dict[str, Dict[str, Any]],
    args,
) -> Dict[str, Dict[str, Any]]:
    """
    Decide keep/drop per unique URL. Returns canonical_url -> record with
    cleaned text, metrics and a decision {kept: bool, reason: str}.
    """
    target_lang = "vi" if args.direction == "zh2vi" else "zh"
    trusted = {d.strip().lower() for d in (args.trusted_domains or "").split(",") if d.strip()}

    opencc = None
    if target_lang == "zh":
        try:
            from opencc import OpenCC
            opencc = OpenCC("t2s")
        except ImportError:
            pass

    records: Dict[str, Dict[str, Any]] = {}

    # ---- per-page gates -------------------------------------------------
    for cu, meta in url_map.items():
        entry = cache.get(cu)
        rec: Dict[str, Any] = {
            "canonical_url": cu,
            "hash": (entry or {}).get("hash", url_hash(cu)),
            "title": meta.get("title", ""),
            "docs": sorted(set(meta["docs"])),
            "status": (entry or {}).get("status", "not_fetched"),
            "http_status": (entry or {}).get("http_status"),
            "content_kind": (entry or {}).get("content_kind", "unknown"),
            "raw_file": (entry or {}).get("raw_file"),
            "error": (entry or {}).get("error", "" if entry else "not_fetched"),
            "domain": urlparse(cu).netloc.lower(),
            "extractor": "", "html_title": "", "needs_ocr": False,
            "text": "", "text_len": 0, "lang_chars": 0, "lang_ratio": 0.0,
            "content_hash": "", "kept": False, "reason": "",
        }
        records[cu] = rec

        if rec["status"] != "ok":
            rec["reason"] = f"fetch:{rec['error'] or rec['status']}"
            continue

        ex = texts.get(cu)
        if ex is None or ex.get("error"):
            rec["reason"] = f"extract:{(ex or {}).get('error', 'missing')}"
            continue
        rec["extractor"] = ex.get("extractor", "")
        rec["html_title"] = ex.get("html_title", "")
        rec["needs_ocr"] = bool(ex.get("needs_ocr"))

        if args.skip_ocr and rec["needs_ocr"]:
            rec["reason"] = "needs_ocr"
            continue

        raw_text = ex.get("text", "")
        if not raw_text.strip():
            rec["reason"] = "empty_text"
            continue

        text = clean_text_for_export(raw_text, annotation_mode=args.annotation_mode)
        if args.max_text_chars and args.max_text_chars > 0:
            text = text[: args.max_text_chars].strip()

        han, latin, h_ratio, l_ratio = language_ratios(text)
        lang_chars, lang_ratio = (latin, l_ratio) if target_lang == "vi" else (han, h_ratio)
        rec.update(text=text, text_len=len(text),
                   lang_chars=lang_chars, lang_ratio=round(lang_ratio, 4))

        if len(text) < args.min_text_len:
            rec["reason"] = f"quality_minlen({len(text)})"
            continue
        if lang_chars < args.min_chars:
            rec["reason"] = f"quality_fewchars({lang_chars})"
            continue
        if lang_ratio < args.min_ratio:
            rec["reason"] = f"quality_ratio({lang_ratio:.2f})"
            continue

        if args.drop_vertical_ocr:
            vertical, _stats = is_vertical_ocr_text(text)
            if vertical:
                rec["reason"] = "vertical_ocr_text"
                continue

        s = structural_metrics(text)
        rec["structural"] = s
        why = []
        if s["avg_line"] < args.degen_avg:
            why.append(f"avg={s['avg_line']}")
        if s["digit_ratio"] > args.degen_digit:
            why.append(f"digit={s['digit_ratio']}")
        if s["letter_ratio"] < args.degen_letter:
            why.append(f"letter={s['letter_ratio']}")
        if why:
            rec["reason"] = f"degenerate({','.join(why)})"
            continue

        signature = extreme_clean_for_compare(text, target_lang=target_lang,
                                              opencc_converter=opencc)
        if not signature:
            rec["reason"] = "empty_signature"
            continue
        rec["content_hash"] = sha1_text(signature)
        rec["_signature"] = signature
        rec["trad_score"] = round(traditional_score(text), 4)
        rec["kept"] = True
        rec["reason"] = "kept"

    alive = [r for r in records.values() if r["kept"]]

    def _drop(rec: Dict[str, Any], reason: str) -> None:
        rec["kept"] = False
        rec["reason"] = reason

    def _quality(rec: Dict[str, Any]):
        """
        Pick the cleanest representative of a duplicate group, NOT the longest.

        Raw length is a bad tiebreaker: the noisiest copy is often the largest,
        because leaked running headers / boilerplate (which survive the per-page
        gates when they are in the target language) inflate its length. Instead
        rank by, in order:

          1. trusted domain
          2. retained line structure -- a pathological ``avg_line`` (whole doc
             merged into a few giant lines, headers inlined) loses the paragraph
             breaks the downstream sentence-snapper relies on
          3. cleanliness: higher target-language ratio, then fewer stray digits
             (footnote/nav numbers), bucketed so tiny differences don't dominate
          4. more actual content (target-language chars) as the final tiebreaker
        """
        s = rec.get("structural", {})
        avg_line = s.get("avg_line", 0.0)
        structured = 1 if avg_line <= _DEDUP_MAX_AVG_LINE else 0
        return (
            rec["domain"] in trusted,
            structured,
            round(rec.get("lang_ratio", 0.0), 2),
            -round(s.get("digit_ratio", 0.0), 3),
            rec.get("lang_chars", 0),
        )

    # ---- exact dedup -----------------------------------------------------
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in alive:
        buckets[r["content_hash"]].append(r)
    for h, grp in buckets.items():
        if len(grp) <= 1:
            continue
        rep = max(grp, key=_quality)
        for r in grp:
            if r is not rep:
                _drop(r, f"exact_dup(rep={rep['hash'][:8]})")
    alive = [r for r in alive if r["kept"]]

    # ---- near dedup (char n-gram Jaccard, size-sorted prune) -------------
    sigs = {r["canonical_url"]: get_character_ngrams(r.pop("_signature"), n=args.ngram_n)
            for r in alive}
    for r in records.values():
        r.pop("_signature", None)

    parent = {r["canonical_url"]: r["canonical_url"] for r in alive}
    link_jac: Dict[str, float] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    order = sorted([r for r in alive if sigs[r["canonical_url"]]],
                   key=lambda r: len(sigs[r["canonical_url"]]))
    for i, a in enumerate(order):
        ca = a["canonical_url"]
        sa = sigs[ca]
        na = len(sa)
        for j in range(i + 1, len(order)):
            cb = order[j]["canonical_url"]
            nb = len(sigs[cb])
            if na / nb < args.near_threshold:
                break
            jac = calculate_jaccard(sa, sigs[cb])
            if jac >= args.near_threshold:
                ra, rb = find(ca), find(cb)
                if ra != rb:
                    parent[ra] = rb
                link_jac[ca] = max(link_jac.get(ca, 0.0), jac)
                link_jac[cb] = max(link_jac.get(cb, 0.0), jac)

    clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in alive:
        clusters[find(r["canonical_url"])].append(r)
    for grp in clusters.values():
        if len(grp) <= 1:
            continue
        rep = max(grp, key=_quality)
        for r in grp:
            if r is not rep:
                _drop(r, f"near_dup(rep={rep['hash'][:8]},"
                         f"jac={link_jac.get(r['canonical_url'], 0.0):.3f})")
    alive = [r for r in alive if r["kept"]]

    # ---- keyword gate (anchors from search queries) ----------------------
    if args.min_distinct > 0:
        source_vocab: Dict[str, Set[str]] = defaultdict(set)
        for cu, meta in url_map.items():
            for sq in meta.get("source_queries", []):
                q = sq.get("query", "") if isinstance(sq, dict) else str(sq)
                for d in meta["docs"]:
                    source_vocab[d].update(extract_anchors(q))
        global_vocab: Set[str] = (
            set().union(*source_vocab.values()) if source_vocab else set()
        )
        if global_vocab:
            for r in alive:
                low = collapse(r["text"])
                present = {a for a in global_vocab if a in low}
                best, bsrc = 0, ""
                for src, vocab in source_vocab.items():
                    k = len(present & vocab)
                    if k > best:
                        best, bsrc = k, src
                r["best_distinct"] = best
                r["best_src"] = bsrc
                if best < args.min_distinct:
                    _drop(r, f"keyword(dist={best}<{args.min_distinct})")

    return records


# =============================================================================
# Stage 5 — write txt/, meta, index.tsv, report.txt
# =============================================================================

def _txt_filename(gid: int, rec: Dict[str, Any]) -> str:
    title = (rec.get("html_title") or rec.get("title") or "untitled").strip()
    title_part = safe_filename(title, max_len=55) or "untitled"
    domain_part = safe_filename(rec["domain"] or "unknown", max_len=40)
    name = f"{gid:04d}__{title_part}__{domain_part}__{rec['content_hash'][:10]}.txt"
    if len(name) > 180:
        name = (f"{gid:04d}__{safe_filename(title_part, max_len=25)}-"
                f"{short_hash(title, 6)}__{domain_part}__{rec['content_hash'][:10]}.txt")
    return name


_INDEX_COLS = [
    "gid", "kept", "reason", "status", "content_kind", "extractor",
    "text_len", "lang_chars", "lang_ratio", "needs_ocr",
    "domain", "title", "docs", "txt_file", "raw_file", "url",
]

_TSV_BAD_RE = re.compile(r"[\t\r\n]+")


def _tsv_cell(value: Any) -> str:
    """One TSV cell: tabs/newlines collapsed so a row is always one line."""
    return _TSV_BAD_RE.sub(" ", str(value))


def write_stage(
    docs: List[Dict[str, Any]],
    url_map: Dict[str, Dict[str, Any]],
    records: Dict[str, Dict[str, Any]],
    layout: Layout,
    args,
) -> None:
    # -- final txt corpus (fresh directory so stale files never linger) ----
    if layout.txt.exists():
        shutil.rmtree(layout.txt)
    layout.txt.mkdir(parents=True, exist_ok=True)

    doc_order = {d["doc_id"]: i for i, d in enumerate(docs)}
    keepers = sorted(
        (r for r in records.values() if r["kept"]),
        key=lambda r: (min(doc_order.get(d, 9999) for d in r["docs"]), r["canonical_url"]),
    )
    for gid, rec in enumerate(keepers, start=1):
        rec["gid"] = gid
        rec["txt_file"] = f"{TXT_SUBDIR}/{_txt_filename(gid, rec)}"
        (layout.root / rec["txt_file"]).write_text(rec["text"], encoding="utf-8")

    # -- per source doc meta jsonl -----------------------------------------
    layout.meta.mkdir(parents=True, exist_ok=True)
    for d in docs:
        rows = []
        for it in d["items"]:
            rec = records.get(it["canonical_url"])
            if rec is None:
                continue
            row = {k: v for k, v in rec.items() if k != "text"}
            row["source_queries"] = it.get("source_queries", [])
            rows.append(row)
        out = layout.meta / f"{d['doc_id']}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # -- index.tsv ----------------------------------------------------------
    with layout.index_tsv.open("w", encoding="utf-8") as f:
        f.write("\t".join(_INDEX_COLS) + "\n")
        for rec in sorted(records.values(),
                          key=lambda r: (not r["kept"], r.get("gid", 99999), r["canonical_url"])):
            row = {
                "gid": rec.get("gid", ""),
                "kept": int(rec["kept"]),
                "reason": rec["reason"],
                "status": rec["status"],
                "content_kind": rec["content_kind"],
                "extractor": rec["extractor"],
                "text_len": rec["text_len"],
                "lang_chars": rec["lang_chars"],
                "lang_ratio": rec["lang_ratio"],
                "needs_ocr": int(bool(rec["needs_ocr"])),
                "domain": rec["domain"],
                "title": (rec.get("html_title") or rec.get("title") or "")[:80],
                "docs": ",".join(rec["docs"]),
                "txt_file": rec.get("txt_file", ""),
                "raw_file": rec.get("raw_file") or "",
                "url": rec["canonical_url"],
            }
            f.write("\t".join(_tsv_cell(row[c]) for c in _INDEX_COLS) + "\n")

    # -- report.txt ----------------------------------------------------------
    lines: List[str] = []
    lines.append(f"Corpus pipeline report — {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"direction={args.direction}  output={layout.root}")
    lines.append("")
    lines.append(f"{'doc':10} {'urls':>5} {'fetched':>8} {'errors':>7} {'kept':>5}")
    lines.append("-" * 42)
    for d in docs:
        recs = [records[i["canonical_url"]] for i in d["items"]
                if i["canonical_url"] in records]
        n_ok = sum(1 for r in recs if r["status"] == "ok")
        n_err = sum(1 for r in recs if r["status"] != "ok")
        n_kept = sum(1 for r in recs if r["kept"])
        lines.append(f"{d['doc_id']:10} {len(recs):>5} {n_ok:>8} {n_err:>7} {n_kept:>5}")

    total_kept = sum(1 for r in records.values() if r["kept"])
    lines.append("-" * 42)
    lines.append(f"unique URLs: {len(records)}  kept: {total_kept}")
    lines.append("")

    reasons = Counter(re.sub(r"\(.*", "", r["reason"]) for r in records.values()
                      if not r["kept"])
    lines.append("drop reasons:")
    for reason, n in reasons.most_common():
        lines.append(f"  {n:>5}  {reason}")
    lines.append("")

    errors = [r for r in records.values() if r["status"] != "ok"]
    if errors:
        lines.append(f"fetch errors ({len(errors)}):")
        for r in sorted(errors, key=lambda r: r["error"]):
            lines.append(f"  {_tsv_cell(r['error']):40.40}  {r['canonical_url']}")
    layout.report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[write] txt: {total_kept} file(s) -> {layout.txt}")
    print(f"[write] index: {layout.index_tsv}")
    print(f"[write] report: {layout.report}")


# =============================================================================
# Legacy import — seed _cache/_raw from an old fetch_pages output dir
# =============================================================================

def import_legacy(pages_dir: Path, layout: Layout) -> None:
    """
    Seed the pipeline caches from an old fetch_pages output directory
    (pages/_registry.json + pages/content/) so previously crawled data is
    reused without hitting the network:

      - pdf/docx raw files  -> _raw/ (re-extracted with the current extractor)
      - html/text pages     -> extracted .txt goes to the text cache with a
                               "legacy" stamp (raw HTML was never stored)
      - fetch entries       -> _cache/fetch.json (errors kept: --refetch-errors)
    """
    registry = load_json(pages_dir / "_registry.json", {})
    if not registry:
        print(f"[legacy] no registry at {pages_dir}/_registry.json")
        return

    layout.make_dirs()
    cache: Dict[str, Dict[str, Any]] = load_json(layout.fetch_cache, {})
    n_raw = n_txt = n_err = 0

    for cu, e in registry.items():
        if not isinstance(e, dict):
            continue
        if cu in cache:
            continue
        hash_str = e.get("hash") or url_hash(cu)
        status = e.get("status", "error")
        entry = {
            "canonical_url": cu, "hash": hash_str, "status": status,
            "http_status": e.get("http_status"),
            "content_type": e.get("content_type", ""),
            "content_kind": e.get("content_kind", "unknown"),
            "final_url": e.get("final_url", ""),
            "charset": _detect_charset(e.get("content_type", "")),
            "raw_file": None,
            "error": e.get("error", ""),
            "fetched_at": e.get("fetched_at", ""),
        }

        if status != "ok":
            cache[cu] = entry
            n_err += 1
            continue

        raw_rel = e.get("raw_file")
        if raw_rel:
            src = pages_dir / raw_rel
            if src.exists():
                ext = src.suffix.lstrip(".") or _KIND_EXT.get(entry["content_kind"], "bin")
                stem = raw_stem(cu, e.get("html_title", ""), hash_str)
                dst_rel = f"{RAW_SUBDIR}/{stem}.{ext}"
                dst = layout.root / dst_rel
                if not dst.exists():
                    shutil.copy2(src, dst)
                entry["raw_file"] = dst_rel
                n_raw += 1

        if not entry["raw_file"]:
            # html/text page: keep the old extracted text as a legacy cache hit
            txt_rel = e.get("text_file")
            src = (pages_dir / txt_rel) if txt_rel else None
            if src and src.exists():
                dump_json(layout.text_cache / f"{hash_str}.json", {
                    "v": "legacy",
                    "text": src.read_text(encoding="utf-8", errors="ignore"),
                    "extractor": e.get("extractor", "legacy"),
                    "html_title": e.get("html_title", ""),
                    "needs_ocr": bool(e.get("needs_ocr", False)),
                    "error": "",
                })
                n_txt += 1
            else:
                entry["status"] = "error"
                entry["error"] = "legacy:text_file_missing"
                n_err += 1

        cache[cu] = entry

    dump_json(layout.fetch_cache, cache)
    print(f"[legacy] imported {len(registry)} entries: "
          f"{n_raw} raw file(s), {n_txt} legacy text(s), {n_err} error(s)")


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="One-command corpus pipeline: URL JSONs -> clean TXT corpus."
    )
    p.add_argument("--url_dir", type=str, default=None,
                   help="Directory of search_urls JSON files (one per source doc).")
    p.add_argument("--url_path", type=str, default=None, help="Single URL JSON file.")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--output_dir", type=str, required=True, help="Corpus output tree.")
    p.add_argument("--direction", choices=["vi2zh", "zh2vi"], default="zh2vi")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated doc ids to process (e.g. C001,C002).")

    # stage toggles
    p.add_argument("--force", action="store_true", help="Re-fetch everything.")
    p.add_argument("--refetch-errors", action="store_true",
                   help="Re-fetch only URLs whose cached status is an error.")
    p.add_argument("--reextract", action="store_true",
                   help="Ignore the text cache (after extractor changes).")
    p.add_argument("--skip-fetch", action="store_true",
                   help="Never hit the network (offline re-filter/re-extract).")
    p.add_argument("--import-legacy", type=str, default=None, metavar="PAGES_DIR",
                   help="Seed caches from an old fetch_pages output dir, then run.")

    # fetch behaviour
    p.add_argument("--sleep_range", type=float, nargs=2, default=list(DEFAULT_SLEEP_RANGE))
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES)
    p.add_argument("--retry_backoff", type=float, default=DEFAULT_RETRY_BACKOFF)
    p.add_argument("--max_concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    p.add_argument("--user_agent", type=str, default=DEFAULT_USER_AGENT)
    p.add_argument("--extract_workers", type=int, default=None,
                   help="Extraction pool size (default: os.cpu_count(), capped at 8).")
    p.add_argument("--extract_mode", choices=["process", "thread"], default="thread",
                   help="'thread' (default) suits the fast block PDF backend; use "
                        "'process' with --high_fidelity to parallelise pymupdf4llm past the GIL.")
    p.add_argument("--high_fidelity", action="store_true",
                   help="Use the pymupdf4llm PDF backend (footnote-precise but ~30x "
                        "slower). Default off: the block backend's only gap is a handful "
                        "of footnote digits per book, irrelevant to alignment.")

    # cleaning + per-page gates
    p.add_argument("--annotation_mode", choices=["off", "light", "aggressive"],
                   default="light")
    p.add_argument("--max_text_chars", type=int, default=None)
    p.add_argument("--skip_ocr", action=argparse.BooleanOptionalAction, default=True,
                   help="Drop pages flagged needs_ocr (default on).")
    p.add_argument("--drop_vertical_ocr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min_text_len", type=int, default=500)
    p.add_argument("--min_chars", type=int, default=100,
                   help="Min target-language chars (Han for vi2zh, Latin for zh2vi).")
    p.add_argument("--min_ratio", type=float, default=0.55,
                   help="Min target-language char ratio.")

    # structural gate
    p.add_argument("--degen_avg", type=float, default=18.0)
    p.add_argument("--degen_digit", type=float, default=0.15)
    p.add_argument("--degen_letter", type=float, default=0.50)

    # dedup
    p.add_argument("--near_threshold", type=float, default=0.92)
    p.add_argument("--ngram_n", type=int, default=3)
    p.add_argument("--trusted_domains", type=str, default="")

    # keyword gate
    p.add_argument("--min_distinct", type=int, default=3,
                   help="Min distinct query anchors a page must contain (0 = off).")

    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    import os

    args = build_parser().parse_args()
    if args.extract_workers is None:
        args.extract_workers = min(8, os.cpu_count() or 4)
    layout = Layout(Path(args.output_dir))
    layout.make_dirs()

    if args.import_legacy:
        import_legacy(Path(args.import_legacy), layout)

    # Stage 1 — inputs
    if args.url_path:
        url_files = [Path(args.url_path)]
    elif args.url_dir:
        root = Path(args.url_dir)
        url_files = sorted(root.rglob("*.json") if args.recursive else root.glob("*.json"))
    else:
        print("[!] Provide --url_dir or --url_path")
        sys.exit(1)
    if not url_files:
        print("[!] No URL JSON files found")
        sys.exit(1)

    only = ({x.strip() for x in args.only.split(",") if x.strip()}
            if args.only else None)
    docs, url_map = load_source_docs(url_files, only)
    print(f"[load] source docs: {len(docs)} | unique URLs: {len(url_map)}")
    if not docs:
        sys.exit(0)

    # Stage 2 — fetch
    cache = asyncio.run(fetch_stage(url_map, layout, args))

    # Stage 3 — extract
    texts = extract_stage(url_map, cache, layout, args)

    # Stage 4 — filter
    records = filter_stage(docs, url_map, cache, texts, args)

    # Stage 5 — write
    write_stage(docs, url_map, records, layout, args)


if __name__ == "__main__":
    main()
