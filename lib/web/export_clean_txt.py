#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export clean TXT corpus from fetch_pages.py JSON outputs.

Input:
    JSON files produced by fetch_pages.py, each containing:
        {
          "doc_id": "...",
          "pages": [
            {
              "status": "ok",
              "text": "...",
              "text_len": 123,
              "needs_ocr": false,
              "url": "...",
              "canonical_url": "...",
              "final_url": "...",
              "domain": "...",
              "title": "...",
              "html_title": "...",
              "source_queries": [...]
            }
          ]
        }

Output:
    output_dir/
      _summary.json
      <doc_id>/
        0001__domain__hash.txt
        0002__domain__hash.txt
        _manifest.json

Main features:
    - clean text lightly
    - filter noisy pages by Han-character ratio
    - skip OCR-needed pages by default
    - exact dedup by Han-only signature
    - optional near dedup by character n-gram containment
    - export pure .txt for local aligner
    - keep URL/title/query trace in manifest JSON
"""

import argparse
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# =============================================================================
# Regex patterns
# =============================================================================

# Han/CJK Unified Ideographs:
# - Basic:        U+4E00–U+9FFF
# - Extension A:  U+3400–U+4DBF
# - Extension B:  U+20000–U+2A6DF
# - Compatibility Ideographs: U+F900–U+FAFF
HAN_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\U00020000-\U0002A6DF\uF900-\uFAFF]")

# Vietnamese/Latin letters. Used only for diagnostics.
LATIN_RE = re.compile(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF]")

WHITESPACE_RE = re.compile(r"\s+")


# =============================================================================
# Data structure
# =============================================================================

@dataclass
class Candidate:
    doc_id: str
    source_page_json: str
    page_id: str

    text: str
    text_len: int
    han_chars: int
    latin_chars: int
    han_ratio: float
    latin_ratio: float

    content_hash: str
    cjk_signature_len: int

    url: str
    canonical_url: str
    final_url: str
    domain: str
    title: str
    html_title: str
    content_kind: str
    extractor: str
    needs_ocr: bool
    source_queries: List[Dict[str, Any]]

    # Filled when exported
    txt_path: str = ""


# =============================================================================
# Basic helpers
# =============================================================================

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def safe_filename(name: str, max_len: int = 120) -> str:
    """
    Make a safe filename while preserving readable Unicode.
    """
    name = unicodedata.normalize("NFKC", str(name or "unknown"))
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = name.strip(". ")
    if not name:
        name = "unknown"
    return name[:max_len]


def short_hash(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def iter_json_files(input_dir: Path, recursive: bool = False) -> List[Path]:
    if recursive:
        return sorted(input_dir.rglob("*.json"))
    return sorted(input_dir.glob("*.json"))


# =============================================================================
# Text statistics
# =============================================================================

def count_han_chars(text: str) -> int:
    return len(HAN_RE.findall(text or ""))


def count_latin_chars(text: str) -> int:
    return len(LATIN_RE.findall(text or ""))


def count_nonspace_chars(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def language_ratios(text: str) -> Tuple[int, int, float, float]:
    """
    Compute Han and Latin ratios over Han + Latin characters.

    This behaves like the old pipeline:
        zh_ratio = han / (han + latin)
        vi_ratio = latin / (han + latin)

    It ignores punctuation, digits, spaces, etc.
    """
    han = count_han_chars(text)
    latin = count_latin_chars(text)
    total = han + latin

    if total == 0:
        return han, latin, 0.0, 0.0

    return han, latin, han / total, latin / total


# =============================================================================
# Cleaning
# =============================================================================

def normalize_unicode(text: str, form: str = "NFC") -> str:
    if not text:
        return ""
    return unicodedata.normalize(form, text)


def remove_zero_width_chars(text: str) -> str:
    return re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text or "")


def remove_ctext_pattern(text: str) -> str:
    """
    Clean ctext.org table-like extracted lines.

    Examples:
        | 8 | some text |
    """
    if not text:
        return ""

    text = re.sub(r"^\|\s*\d+\s*\|", "", text, flags=re.MULTILINE)
    text = re.sub(r"\|\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]+|[ \t]+$", "", text, flags=re.MULTILINE)

    return text


def remove_navigation_elements(text: str) -> str:
    """
    Remove common Wikisource-like navigation lines.
    """
    if not text:
        return ""

    # Lines containing prev/next arrows.
    nav_line_pattern = re.compile(r"^[ \t]*\|?.*?[◄►].*?\|?[ \t]*$", re.MULTILINE)
    text = nav_line_pattern.sub("", text)

    # Stray arrow fragments.
    text = re.sub(r"[ \t]*[◄►].*?(\n|$)", "", text)

    return text


def remove_common_boilerplate_lines(text: str) -> str:
    """
    Conservative line-level boilerplate removal.
    """
    if not text:
        return ""

    drop_patterns = [
        r"^\s*$",
        r"^返回.*$",
        r"^上一页$",
        r"^下一页$",
        r"^上一頁$",
        r"^下一頁$",
        r"^目录$",
        r"^目錄$",
        r"^编辑$",
        r"^編輯$",
        r"^本页面最后修订于.*$",
        r"^本頁面最後修訂於.*$",
        r"^此页面最后编辑于.*$",
        r"^此頁面最後編輯於.*$",
        r"^维基文库.*$",
        r"^維基文庫.*$",
        r"^Wikisource.*$",
        r"^Chinese Text Project.*$",
        r"^中國哲學書電子化計劃.*$",
        r"^Copyright.*$",
        r"^版权所有.*$",
        r"^版權所有.*$",
        r"^登录$",
        r"^登入$",
        r"^注册$",
        r"^搜尋$",
        r"^搜索$",
    ]
    compiled = [re.compile(p, flags=re.IGNORECASE) for p in drop_patterns]

    kept = []
    for line in text.splitlines():
        s = line.strip()
        if any(p.search(s) for p in compiled):
            continue
        kept.append(line)

    return "\n".join(kept).strip()


def remove_inline_annotations(text: str, mode: str = "light") -> str:
    """
    Remove inline annotations.

    mode:
        none:
            do nothing

        light:
            remove modern numeric references and annotation brackets:
                [1], (12), 〔...〕, 【...】

        aggressive:
            also remove:
                （...）, (...), [...], {...}, 〈...〉, 《...》

    For alignment corpus, "light" is usually safer.
    """
    if not text:
        return ""

    if mode == "none":
        return text

    # Numeric refs.
    text = re.sub(r"\[\d+\]|\(\d+\)", "", text)

    # Common annotation brackets.
    text = re.sub(r"〔.*?〕|【.*?】", "", text)

    if mode == "aggressive":
        text = re.sub(r"\[.*?\]|\(.*?\)|\{.*?\}|〈.*?〉|《.*?》|（.*?）", "", text)

    return text


def remove_historical_noise(text: str) -> str:
    """
    Remove non-semantic punctuation often introduced by editions.
    Keep this conservative.
    """
    if not text:
        return ""
    return re.sub(r"[〇「」『』]", "", text)


def clean_whitespaces(text: str) -> str:
    if not text:
        return ""

    # Normalize horizontal spaces/tabs without destroying newlines.
    text = re.sub(r"[ \t]+", " ", text)

    # Clean spaces around newlines.
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)

    # Collapse 3+ newlines into 2 newlines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_text_for_export(
    text: str,
    annotation_mode: str = "light",
    unicode_form: str = "NFC",
) -> str:
    """
    Main cleaning pipeline.
    """
    text = text or ""

    text = normalize_unicode(text, form=unicode_form)
    text = remove_zero_width_chars(text)

    # Source-specific structural cleanup.
    text = remove_ctext_pattern(text)
    text = remove_navigation_elements(text)
    text = remove_common_boilerplate_lines(text)

    # Annotation/noise cleanup.
    text = remove_inline_annotations(text, mode=annotation_mode)
    text = remove_historical_noise(text)

    # Final whitespace cleanup.
    text = clean_whitespaces(text)

    return text


# =============================================================================
# Dedup signatures
# =============================================================================

def extreme_clean_for_compare(text: str, target_lang: str = "zh") -> str:
    """
    Strip formatting/punctuation/noise and return only target-language chars.
    Used for dedup signature, not for exported text.
    """
    if not text:
        return ""

    # Remove bracketed notes for comparison only.
    text = re.sub(r"\[.*?\]|\(.*?\)|\{.*?\}|〈.*?〉|《.*?》|〔.*?〕|【.*?】|（.*?）", "", text)

    if target_lang == "zh":
        return "".join(HAN_RE.findall(text))

    # Latin/Vietnamese fallback.
    return "".join(LATIN_RE.findall(text)).lower()


def get_character_ngrams(text: str, n: int = 3) -> Set[str]:
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def calculate_containment(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Containment over the smaller set:
        |A ∩ B| / min(|A|, |B|)
    """
    if not set_a or not set_b:
        return 0.0
    return len(set_a.intersection(set_b)) / min(len(set_a), len(set_b))


# =============================================================================
# Candidate extraction
# =============================================================================

def page_to_candidate(
    page: Dict[str, Any],
    page_json_path: Path,
    doc_id: str,
    min_text_len: int,
    min_han_chars: int,
    min_han_ratio: float,
    max_text_chars: Optional[int],
    skip_ocr: bool,
    annotation_mode: str,
    unicode_form: str,
) -> Tuple[Optional[Candidate], Optional[Dict[str, Any]]]:
    """
    Convert a page record into a clean Candidate or a rejection reason.
    """
    page_id = str(page.get("page_id") or "")
    if not page_id:
        page_id = "unknown"

    url = page.get("canonical_url") or page.get("final_url") or page.get("url") or ""

    status = page.get("status")
    if status != "ok":
        return None, {
            "page_id": page_id,
            "reason": "status_not_ok",
            "url": url,
            "status": status,
            "error": page.get("error", ""),
        }

    needs_ocr = bool(page.get("needs_ocr"))
    if skip_ocr and needs_ocr:
        return None, {
            "page_id": page_id,
            "reason": "needs_ocr",
            "url": url,
        }

    raw_text = page.get("text") or ""
    if not raw_text.strip():
        return None, {
            "page_id": page_id,
            "reason": "empty_text",
            "url": url,
        }

    text = clean_text_for_export(
        raw_text,
        annotation_mode=annotation_mode,
        unicode_form=unicode_form,
    )

    if max_text_chars is not None and max_text_chars > 0:
        text = text[:max_text_chars].strip()

    text_len = len(text)
    han, latin, h_ratio, l_ratio = language_ratios(text)

    if text_len < min_text_len:
        return None, {
            "page_id": page_id,
            "reason": "text_too_short",
            "url": url,
            "text_len": text_len,
            "han_chars": han,
            "latin_chars": latin,
            "han_ratio": h_ratio,
            "latin_ratio": l_ratio,
        }

    if han < min_han_chars:
        return None, {
            "page_id": page_id,
            "reason": "too_few_han_chars",
            "url": url,
            "text_len": text_len,
            "han_chars": han,
            "latin_chars": latin,
            "han_ratio": h_ratio,
            "latin_ratio": l_ratio,
        }

    if h_ratio < min_han_ratio:
        return None, {
            "page_id": page_id,
            "reason": "low_han_ratio",
            "url": url,
            "text_len": text_len,
            "han_chars": han,
            "latin_chars": latin,
            "han_ratio": h_ratio,
            "latin_ratio": l_ratio,
        }

    signature = extreme_clean_for_compare(text, target_lang="zh")
    if not signature:
        return None, {
            "page_id": page_id,
            "reason": "empty_han_signature",
            "url": url,
            "text_len": text_len,
            "han_chars": han,
            "latin_chars": latin,
            "han_ratio": h_ratio,
            "latin_ratio": l_ratio,
        }

    content_hash = sha1_text(signature)

    candidate = Candidate(
        doc_id=doc_id,
        source_page_json=str(page_json_path),
        page_id=page_id,

        text=text,
        text_len=text_len,
        han_chars=han,
        latin_chars=latin,
        han_ratio=h_ratio,
        latin_ratio=l_ratio,

        content_hash=content_hash,
        cjk_signature_len=len(signature),

        url=str(page.get("url") or ""),
        canonical_url=str(page.get("canonical_url") or ""),
        final_url=str(page.get("final_url") or ""),
        domain=str(page.get("domain") or ""),
        title=str(page.get("title") or ""),
        html_title=str(page.get("html_title") or ""),
        content_kind=str(page.get("content_kind") or ""),
        extractor=str(page.get("extractor") or ""),
        needs_ocr=needs_ocr,
        source_queries=list(page.get("source_queries") or []),
    )

    return candidate, None


def collect_candidates_from_file(
    page_json_path: Path,
    args: argparse.Namespace,
) -> Tuple[str, List[Candidate], List[Dict[str, Any]]]:
    data = read_json(page_json_path)
    doc_id = str(data.get("doc_id") or page_json_path.stem)
    pages = data.get("pages") or []

    candidates: List[Candidate] = []
    rejected: List[Dict[str, Any]] = []

    for idx, page in enumerate(pages, start=1):
        if "page_id" not in page:
            page = dict(page)
            page["page_id"] = f"{idx:04d}"

        cand, rej = page_to_candidate(
            page=page,
            page_json_path=page_json_path,
            doc_id=doc_id,
            min_text_len=args.min_text_len,
            min_han_chars=args.min_han_chars,
            min_han_ratio=args.min_han_ratio,
            max_text_chars=args.max_text_chars,
            skip_ocr=not args.keep_ocr_needed,
            annotation_mode=args.annotation_mode,
            unicode_form=args.unicode_form,
        )

        if cand is not None:
            candidates.append(cand)
        elif rej is not None:
            rejected.append(rej)

    return doc_id, candidates, rejected


# =============================================================================
# Candidate ranking and dedup
# =============================================================================

def candidate_quality_key(c: Candidate, trusted_domains: Set[str]) -> Tuple[float, float, int, int]:
    """
    Higher is better.

    Preference:
      1. trusted domain
      2. high Han ratio
      3. more Han chars
      4. longer text
    """
    domain_score = 1.0 if c.domain.lower() in trusted_domains else 0.0
    return (
        domain_score,
        c.han_ratio,
        c.han_chars,
        c.text_len,
    )


def exact_dedup_candidates(
    candidates: List[Candidate],
    scope: str,
    trusted_domains: Set[str],
) -> Tuple[List[Candidate], List[Dict[str, Any]]]:
    """
    Dedup by exact Han-only content hash.

    scope:
      - global: all candidates together
      - doc: within each doc_id only
    """
    buckets: Dict[Tuple[str, str], List[Candidate]] = {}

    for c in candidates:
        key = (c.content_hash, c.doc_id) if scope == "doc" else (c.content_hash, "__global__")
        buckets.setdefault(key, []).append(c)

    kept: List[Candidate] = []
    rejected: List[Dict[str, Any]] = []

    for _, group in buckets.items():
        if len(group) == 1:
            kept.append(group[0])
            continue

        sorted_group = sorted(
            group,
            key=lambda x: candidate_quality_key(x, trusted_domains),
            reverse=True,
        )

        winner = sorted_group[0]
        kept.append(winner)

        for loser in sorted_group[1:]:
            rejected.append({
                "page_id": loser.page_id,
                "doc_id": loser.doc_id,
                "reason": "duplicate_exact",
                "url": loser.canonical_url or loser.final_url or loser.url,
                "content_hash": loser.content_hash,
                "kept_doc_id": winner.doc_id,
                "kept_page_id": winner.page_id,
                "kept_url": winner.canonical_url or winner.final_url or winner.url,
            })

    return kept, rejected


def near_dedup_candidates(
    candidates: List[Candidate],
    threshold: float,
    ngram_n: int,
    scope: str,
    trusted_domains: Set[str],
    verbose: bool = False,
) -> Tuple[List[Candidate], List[Dict[str, Any]]]:
    """
    Near dedup by character n-gram containment.

    This is O(n^2), so use only after exact dedup and quality filtering.
    """
    if not candidates:
        return [], []

    # Sort best-first. When duplicates are found, keep the earlier/better candidate.
    ordered = sorted(
        candidates,
        key=lambda x: candidate_quality_key(x, trusted_domains),
        reverse=True,
    )

    signatures: Dict[int, Set[str]] = {}
    for i, c in enumerate(ordered):
        sig = extreme_clean_for_compare(c.text, target_lang="zh")
        signatures[i] = get_character_ngrams(sig, n=ngram_n)

    keep_indices: List[int] = []
    dropped_indices: Set[int] = set()
    rejected: List[Dict[str, Any]] = []

    for i, cand_i in enumerate(ordered):
        if i in dropped_indices:
            continue

        keep_indices.append(i)

        for j in range(i + 1, len(ordered)):
            if j in dropped_indices:
                continue

            cand_j = ordered[j]

            if scope == "doc" and cand_i.doc_id != cand_j.doc_id:
                continue

            score = calculate_containment(signatures[i], signatures[j])

            if score >= threshold:
                dropped_indices.add(j)

                rejected.append({
                    "page_id": cand_j.page_id,
                    "doc_id": cand_j.doc_id,
                    "reason": "duplicate_near",
                    "url": cand_j.canonical_url or cand_j.final_url or cand_j.url,
                    "near_score": score,
                    "threshold": threshold,
                    "kept_doc_id": cand_i.doc_id,
                    "kept_page_id": cand_i.page_id,
                    "kept_url": cand_i.canonical_url or cand_i.final_url or cand_i.url,
                })

                if verbose:
                    print(
                        f"[near-dup] DROP {cand_j.doc_id}/{cand_j.page_id} "
                        f"KEEP {cand_i.doc_id}/{cand_i.page_id} "
                        f"score={score:.3f}"
                    )

    kept = [ordered[i] for i in keep_indices if i not in dropped_indices]
    return kept, rejected


# =============================================================================
# Export
# =============================================================================

def export_candidates(
    candidates: List[Candidate],
    rejected_by_doc: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    flat_output: bool = False,
) -> Dict[str, Any]:
    """
    Write TXT files and manifests.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    by_doc: Dict[str, List[Candidate]] = {}
    for c in candidates:
        by_doc.setdefault(c.doc_id, []).append(c)

    docs_summary = []
    total_exported = 0

    for doc_id, doc_candidates in sorted(by_doc.items(), key=lambda x: x[0]):
        if flat_output:
            doc_dir = output_dir
        else:
            doc_dir = output_dir / safe_filename(doc_id)

        doc_dir.mkdir(parents=True, exist_ok=True)

        exported_records = []

        # stable order: page_id then domain
        doc_candidates = sorted(doc_candidates, key=lambda c: (c.page_id, c.domain, c.content_hash))

        for idx, c in enumerate(doc_candidates, start=1):
            domain = safe_filename(c.domain or "unknown", max_len=60)
            h = c.content_hash[:10]

            if flat_output:
                filename = f"{safe_filename(doc_id, 80)}__{idx:04d}__{domain}__{h}.txt"
            else:
                filename = f"{idx:04d}__{domain}__{h}.txt"

            txt_path = doc_dir / filename
            txt_path.write_text(c.text, encoding="utf-8")

            c.txt_path = str(txt_path)

            record = asdict(c)
            # Do not duplicate full text inside manifest.
            record.pop("text", None)
            exported_records.append(record)

        manifest = {
            "doc_id": doc_id,
            "num_exported": len(exported_records),
            "num_rejected": len(rejected_by_doc.get(doc_id, [])),
            "exported": exported_records,
            "rejected": rejected_by_doc.get(doc_id, []),
        }

        manifest_path = doc_dir / "_manifest.json"
        write_json(manifest_path, manifest)

        docs_summary.append({
            "doc_id": doc_id,
            "num_exported": len(exported_records),
            "num_rejected": len(rejected_by_doc.get(doc_id, [])),
            "manifest_path": str(manifest_path),
        })

        total_exported += len(exported_records)

    return {
        "num_docs": len(docs_summary),
        "total_exported": total_exported,
        "docs": docs_summary,
    }


def add_rejections_by_doc(
    rejected_by_doc: Dict[str, List[Dict[str, Any]]],
    rejections: Iterable[Dict[str, Any]],
    fallback_doc_id: Optional[str] = None,
) -> None:
    for r in rejections:
        doc_id = str(r.get("doc_id") or fallback_doc_id or "unknown")
        rejected_by_doc.setdefault(doc_id, []).append(r)


# =============================================================================
# CLI
# =============================================================================

def parse_trusted_domains(raw: str) -> Set[str]:
    if not raw:
        return set()
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean fetch_pages.py JSON outputs, filter by Han ratio, deduplicate, and export .txt files."
    )

    parser.add_argument("--page_dir", type=str, required=True, help="Directory containing fetch_pages.py JSON outputs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write clean .txt corpus.")
    parser.add_argument("--recursive", action="store_true", help="Recursively read JSON files under --page_dir.")

    # Quality gate
    parser.add_argument("--min_text_len", type=int, default=500, help="Minimum cleaned text length.")
    parser.add_argument("--min_han_chars", type=int, default=100, help="Minimum number of Han characters.")
    parser.add_argument(
        "--min_han_ratio",
        type=float,
        default=0.60,
        help="Minimum Han ratio over Han+Latin characters. Recommended for zh corpus: 0.45-0.70.",
    )
    parser.add_argument(
        "--max_text_chars",
        type=int,
        default=None,
        help="Optional cap for exported text length per page. Default: no cap.",
    )
    parser.add_argument(
        "--keep_ocr_needed",
        action="store_true",
        help="Keep pages marked needs_ocr=True. Default: skip them.",
    )

    # Cleaning
    parser.add_argument(
        "--annotation_mode",
        type=str,
        choices=["none", "light", "aggressive"],
        default="light",
        help="How aggressively to remove inline annotations.",
    )
    parser.add_argument(
        "--unicode_form",
        type=str,
        choices=["NFC", "NFKC"],
        default="NFC",
        help="Unicode normalization form for exported text.",
    )

    # Dedup
    parser.add_argument(
        "--dedup_scope",
        type=str,
        choices=["global", "doc"],
        default="global",
        help="global = dedup across all docs; doc = dedup inside each doc only.",
    )
    parser.add_argument(
        "--disable_exact_dedup",
        action="store_true",
        help="Disable exact dedup by Han-only content hash.",
    )
    parser.add_argument(
        "--near_dedup",
        action="store_true",
        help="Enable slower near-duplicate detection by character n-gram containment.",
    )
    parser.add_argument(
        "--near_threshold",
        type=float,
        default=0.85,
        help="Containment threshold for near dedup.",
    )
    parser.add_argument(
        "--ngram_n",
        type=int,
        default=3,
        help="Character n-gram size for near dedup.",
    )

    # Output
    parser.add_argument(
        "--flat_output",
        action="store_true",
        help="Write all TXT files directly under output_dir instead of per-doc folders.",
    )
    parser.add_argument(
        "--trusted_domains",
        type=str,
        default="ctext.org,zh.wikisource.org,wikisource.org,shidianguji.com,guoxuemi.com",
        help="Comma-separated domains preferred when duplicates are detected.",
    )

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    page_dir = Path(args.page_dir)
    output_dir = Path(args.output_dir)

    if not page_dir.exists():
        raise FileNotFoundError(f"page_dir not found: {page_dir}")

    json_files = iter_json_files(page_dir, recursive=args.recursive)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {page_dir}")

    trusted_domains = parse_trusted_domains(args.trusted_domains)

    all_candidates: List[Candidate] = []
    rejected_by_doc: Dict[str, List[Dict[str, Any]]] = {}

    if args.verbose:
        print("=" * 80)
        print("[*] EXPORT CLEAN TXT CORPUS")
        print(f"[*] Input page_dir: {page_dir}")
        print(f"[*] Output dir:     {output_dir}")
        print(f"[*] JSON files:     {len(json_files)}")
        print(f"[*] min_han_ratio:  {args.min_han_ratio}")
        print(f"[*] exact dedup:    {not args.disable_exact_dedup}")
        print(f"[*] near dedup:     {args.near_dedup}")
        print("=" * 80)

    # Phase 1: collect + filter + clean
    for path in json_files:
        doc_id, candidates, rejections = collect_candidates_from_file(path, args)

        all_candidates.extend(candidates)
        add_rejections_by_doc(rejected_by_doc, rejections, fallback_doc_id=doc_id)

        if args.verbose:
            raw_count = len((read_json(path).get("pages") or []))
            print(
                f"[collect] {path.name}: "
                f"pages={raw_count}, keep={len(candidates)}, reject={len(rejections)}"
            )

    num_after_quality = len(all_candidates)

    # Phase 2: exact dedup
    exact_rejections: List[Dict[str, Any]] = []
    if not args.disable_exact_dedup:
        all_candidates, exact_rejections = exact_dedup_candidates(
            all_candidates,
            scope=args.dedup_scope,
            trusted_domains=trusted_domains,
        )
        add_rejections_by_doc(rejected_by_doc, exact_rejections)

    num_after_exact = len(all_candidates)

    # Phase 3: near dedup
    near_rejections: List[Dict[str, Any]] = []
    if args.near_dedup:
        all_candidates, near_rejections = near_dedup_candidates(
            all_candidates,
            threshold=args.near_threshold,
            ngram_n=args.ngram_n,
            scope=args.dedup_scope,
            trusted_domains=trusted_domains,
            verbose=args.verbose,
        )
        add_rejections_by_doc(rejected_by_doc, near_rejections)

    num_after_near = len(all_candidates)

    # Phase 4: export
    export_summary = export_candidates(
        candidates=all_candidates,
        rejected_by_doc=rejected_by_doc,
        output_dir=output_dir,
        flat_output=args.flat_output,
    )

    summary = {
        "page_dir": str(page_dir),
        "output_dir": str(output_dir),
        "num_input_json_files": len(json_files),
        "num_candidates_after_quality_filter": num_after_quality,
        "num_candidates_after_exact_dedup": num_after_exact,
        "num_candidates_after_near_dedup": num_after_near,
        "total_exported": export_summary["total_exported"],
        "total_rejected": sum(len(v) for v in rejected_by_doc.values()),
        "num_exact_duplicates": len(exact_rejections),
        "num_near_duplicates": len(near_rejections),
        "config": {
            "min_text_len": args.min_text_len,
            "min_han_chars": args.min_han_chars,
            "min_han_ratio": args.min_han_ratio,
            "max_text_chars": args.max_text_chars,
            "skip_ocr": not args.keep_ocr_needed,
            "annotation_mode": args.annotation_mode,
            "unicode_form": args.unicode_form,
            "dedup_scope": args.dedup_scope,
            "exact_dedup": not args.disable_exact_dedup,
            "near_dedup": args.near_dedup,
            "near_threshold": args.near_threshold,
            "ngram_n": args.ngram_n,
            "trusted_domains": sorted(trusted_domains),
            "flat_output": args.flat_output,
        },
        "docs": export_summary["docs"],
    }

    summary_path = output_dir / "_summary.json"
    write_json(summary_path, summary)

    print("Done.")
    print(f"Input JSON files: {len(json_files)}")
    print(f"After quality filter: {num_after_quality}")
    print(f"After exact dedup: {num_after_exact}")
    print(f"After near dedup: {num_after_near}")
    print(f"Exported TXT: {summary['total_exported']}")
    print(f"Rejected total: {summary['total_rejected']}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
