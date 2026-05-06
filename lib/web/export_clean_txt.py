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


try:
    from opencc import OpenCC
except ImportError:
    OpenCC = None


# =============================================================================
# Regex patterns
# =============================================================================

# Han/CJK Unified Ideographs:
# - Basic:        U+4E00–U+9FFF
# - Extension A:  U+3400–U+4DBF
# - Extension B:  U+20000–U+2A6DF
# - Compatibility Ideographs: U+F900–U+FAFF
HAN_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\U00020000-\U0002A6DF\uF900-\uFAFF]")

# A lightweight marker set to estimate whether a text is Traditional or Simplified.
# This is not a full converter; it is only used as a tie-breaker when choosing
# which duplicate source to keep. OpenCC is used for actual dedup normalization.
TRAD_MARKER_RE = re.compile(r"[欽鑑綱紀記書國為爲舊開寶寶閭條維輯體廣萬與義實錄學東龍龍門師後臺臺萬]")
SIMP_MARKER_RE = re.compile(r"[钦鉴纲纪记书国为旧开宝闾条维辑体广万与义实录学东龙门师后台台]")

# Vietnamese/Latin letters. Used only for diagnostics.
LATIN_RE = re.compile(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF]")

WHITESPACE_RE = re.compile(r"\s+")

# Keep filenames Windows-safe.
# Windows commonly has MAX_PATH issues around 260 chars for the whole path.
# So keep each filename conservative.
MAX_FILENAME_LEN = 180
MAX_DOC_PART_LEN = 50
MAX_TITLE_PART_LEN = 55
MAX_DOMAIN_PART_LEN = 45


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
    traditional_score: float

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

    Windows-invalid characters are replaced:
        < > : " / \\ | ? *
    Also removes control characters and trims trailing spaces/dots.
    """
    name = unicodedata.normalize("NFKC", str(name or "unknown"))

    # Remove control chars.
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)

    # Replace Windows-invalid filename chars.
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)

    # Collapse whitespace.
    name = re.sub(r"\s+", " ", name).strip()

    # Windows dislikes trailing dots/spaces.
    name = name.strip(". ")

    if not name:
        name = "unknown"

    return name[:max_len].strip(". ")


def short_hash(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def choose_title_for_filename(c: "Candidate", max_len: int = MAX_TITLE_PART_LEN) -> str:
    """
    Pick a readable but short title for the output TXT filename.

    Priority:
      1. html_title extracted from page
      2. search-result title
      3. doc_id

    Long CText titles can contain many works joined together, so keep this short.
    The full title is still preserved in _manifest.json.
    """
    title = (c.html_title or c.title or c.doc_id or "untitled").strip()

    # Remove common site suffixes to keep filenames shorter.
    title = re.sub(r"\s*[-_｜|]\s*Wikisource.*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*[-_｜|]\s*Chinese Text Project.*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*[-_｜|]\s*中國哲學書電子化計劃.*$", "", title, flags=re.IGNORECASE)

    return safe_filename(title, max_len=max_len)


def build_output_filename(
    c: "Candidate",
    idx: int,
    doc_id: str,
    flat_output: bool = False,
    flat_name_mode: str = "global_id",
) -> str:
    """
    Build a readable but Windows-safe filename.

    Non-flat per-doc output:
        <local_id>__<title>__<domain>__<content_hash>.txt

    Flat global output:
        <global_id>__<title>__<domain>__<content_hash>.txt

    If the filename is still too long, title is shortened further and
    a title hash is added so different long titles remain distinguishable.
    """
    doc_part = safe_filename(doc_id, max_len=MAX_DOC_PART_LEN)
    title_raw = (c.html_title or c.title or c.doc_id or "untitled").strip()
    title_part = choose_title_for_filename(c, max_len=MAX_TITLE_PART_LEN)
    title_hash = short_hash(title_raw, 6)
    domain_part = safe_filename(c.domain or "unknown", max_len=MAX_DOMAIN_PART_LEN)
    content_hash = c.content_hash[:10]

    if flat_output and flat_name_mode == "doc_id":
        filename = f"{doc_part}__{idx:04d}__{title_part}__{domain_part}__{content_hash}.txt"
    else:
        # Default for both:
        #   flat/global    -> global id
        #   per-doc folder -> local id
        filename = f"{idx:04d}__{title_part}__{domain_part}__{content_hash}.txt"

    if len(filename) <= MAX_FILENAME_LEN:
        return filename

    # Emergency shorter title version.
    title_part = safe_filename(title_part, max_len=30)

    if flat_output and flat_name_mode == "doc_id":
        filename = f"{doc_part}__{idx:04d}__{title_part}-{title_hash}__{domain_part}__{content_hash}.txt"
    else:
        filename = f"{idx:04d}__{title_part}-{title_hash}__{domain_part}__{content_hash}.txt"

    if len(filename) <= MAX_FILENAME_LEN:
        return filename

    # Last-resort minimal version.
    doc_part = safe_filename(doc_id, max_len=35)
    domain_part = safe_filename(c.domain or "unknown", max_len=30)

    if flat_output and flat_name_mode == "doc_id":
        return f"{doc_part}__{idx:04d}__{title_hash}__{domain_part}__{content_hash}.txt"

    return f"{idx:04d}__{title_hash}__{domain_part}__{content_hash}.txt"

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

def traditional_score(text: str) -> float:
    """
    Heuristic score for preferring Traditional Chinese sources.

    Returns:
        trad_marker_count / (trad_marker_count + simp_marker_count)

    If there are no markers, returns 0.5 as neutral.
    """
    if not text:
        return 0.5

    trad = len(TRAD_MARKER_RE.findall(text))
    simp = len(SIMP_MARKER_RE.findall(text))
    total = trad + simp

    if total == 0:
        return 0.5

    return trad / total

def vertical_ocr_stats(text: str) -> Dict[str, Any]:
    """
    Detect PDF vertical text-layer / OCR extraction noise.

    Typical bad extraction:
        一
        河
        北
        京
        津
        文
        獻
        第
        四
        十
        三
        期
        目
        錄

    This can come from OCR or from a real PDF text layer with vertical layout.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    n_lines = len(lines)

    if n_lines == 0:
        return {
            "n_lines": 0,
            "one_han_lines": 0,
            "short_han_lines": 0,
            "vertical_line_ratio": 0.0,
            "short_han_line_ratio": 0.0,
            "avg_nonspace_len": 0.0,
            "max_short_han_run": 0,
            "max_one_han_run": 0,
        }

    one_han_lines = 0
    short_han_lines = 0
    total_nonspace_len = 0

    cur_short_run = 0
    max_short_run = 0

    cur_one_run = 0
    max_one_run = 0

    for ln in lines:
        compact = re.sub(r"\s+", "", ln)
        nonspace_len = len(compact)
        han_count = count_han_chars(ln)

        total_nonspace_len += nonspace_len

        is_one_han = han_count == 1 and nonspace_len <= 2
        is_short_han = 1 <= han_count <= 2 and nonspace_len <= 3

        if is_one_han:
            one_han_lines += 1
            cur_one_run += 1
            max_one_run = max(max_one_run, cur_one_run)
        else:
            cur_one_run = 0

        if is_short_han:
            short_han_lines += 1
            cur_short_run += 1
            max_short_run = max(max_short_run, cur_short_run)
        else:
            cur_short_run = 0

    return {
        "n_lines": n_lines,
        "one_han_lines": one_han_lines,
        "short_han_lines": short_han_lines,
        "vertical_line_ratio": one_han_lines / n_lines,
        "short_han_line_ratio": short_han_lines / n_lines,
        "avg_nonspace_len": total_nonspace_len / n_lines,
        "max_short_han_run": max_short_run,
        "max_one_han_run": max_one_run,
    }


def is_vertical_ocr_text(
    text: str,
    *,
    min_lines: int = 30,
    ratio_threshold: float = 0.60,
    avg_len_threshold: float = 3.0,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Return True if text looks like vertical PDF/OCR extraction.

    Triggers:
      1. Whole-file vertical pattern:
         many single-Han lines and very short average line length.

      2. Long vertical run:
         many consecutive 1-2 Han-character lines.
         This catches PDFs where only part of the file is vertical front matter
         or where the whole-file ratio is diluted by later normal text.
    """
    stats = vertical_ocr_stats(text)

    if stats["n_lines"] < min_lines:
        return False, stats

    whole_file_vertical = (
        (
            stats["vertical_line_ratio"] >= ratio_threshold
            or stats["short_han_line_ratio"] >= ratio_threshold
        )
        and stats["avg_nonspace_len"] <= avg_len_threshold
    )

    # Consecutive vertical-layout lines are a strong signal even if the
    # whole file later contains normal paragraphs.
    long_vertical_run = (
        stats["max_one_han_run"] >= min_lines
        or stats["max_short_han_run"] >= min_lines
    )

    return whole_file_vertical or long_vertical_run, stats


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


def remove_wikisource_edit_markers(text: str) -> str:
    """
    Remove Wikisource/Wikipedia UI markers that trafilatura/BS4 may leave behind.

    Examples:
        [编辑]先皇帝       -> 先皇帝
        [編輯]先皇帝       -> 先皇帝
        [编辑]| 维基百科条目：丁先皇 -> removed as a line
    """
    if not text:
        return ""

    # Remove inline edit markers.
    text = re.sub(r"\[(编辑|編輯|edit)\]", "", text, flags=re.IGNORECASE)

    # Remove Wikipedia pointer lines.
    text = re.sub(
        r"^\s*\|?\s*(维基百科条目|維基百科條目|Wikipedia article)\s*[:：].*$",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

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
    text = remove_wikisource_edit_markers(text)
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

def extreme_clean_for_compare(
    text: str,
    target_lang: str = "zh",
    opencc_converter: Optional[Any] = None,
) -> str:
    """
    Strip formatting/punctuation/noise and return only target-language chars.
    Used for dedup signature, not for exported text.

    If opencc_converter is provided, Simplified/Traditional normalization is
    applied before generating the signature. This lets dedup catch pairs like:
        欽定越史通鑑綱目
        钦定越史通鉴纲目
    """
    if not text:
        return ""

    # Remove bracketed notes for comparison only.
    text = re.sub(r"\[.*?\]|\(.*?\)|\{.*?\}|〈.*?〉|《.*?》|〔.*?〕|【.*?】|（.*?）", "", text)

    if opencc_converter is not None:
        text = opencc_converter.convert(text)

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

def calculate_jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Jaccard similarity: |A ∩ B| / |A ∪ B|
    """
    if not set_a and not set_b:
        return 1.0  # Both empty = identical
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def calculate_containment_asymmetric(
    set_a: Set[str], 
    set_b: Set[str]
) -> Tuple[float, float]:
    """
    Returns (containment of A in B, containment of B in A)
    
    containment_a_in_b: What fraction of A appears in B?
    containment_b_in_a: What fraction of B appears in A?
    """
    if not set_a or not set_b:
        return 0.0, 0.0
    intersection = len(set_a & set_b)
    return (
        intersection / len(set_a),  # A in B
        intersection / len(set_b),  # B in A
    )


def calculate_similarity_metrics(
    set_a: Set[str], 
    set_b: Set[str]
) -> Dict[str, float]:
    """
    Calculate all similarity metrics at once.
    """
    if not set_a and not set_b:
        return {"jaccard": 1.0, "containment_a_in_b": 1.0, "containment_b_in_a": 1.0}
    if not set_a or not set_b:
        return {"jaccard": 0.0, "containment_a_in_b": 0.0, "containment_b_in_a": 0.0}
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return {
        "jaccard": intersection / union if union > 0 else 0.0,
        "containment_a_in_b": intersection / len(set_a),
        "containment_b_in_a": intersection / len(set_b),
    }

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
    opencc_converter: Optional[Any] = None,
    drop_vertical_ocr: bool = True,
    vertical_min_lines: int = 30,
    vertical_ratio_threshold: float = 0.60,
    vertical_avg_len_threshold: float = 3.0,
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

    if drop_vertical_ocr:
        is_vertical, vertical_stats = is_vertical_ocr_text(
            text,
            min_lines=vertical_min_lines,
            ratio_threshold=vertical_ratio_threshold,
            avg_len_threshold=vertical_avg_len_threshold,
        )
        if is_vertical:
            text_len_tmp = len(text)
            han_tmp, latin_tmp, h_ratio_tmp, l_ratio_tmp = language_ratios(text)
            return None, {
                "page_id": page_id,
                "reason": "vertical_ocr_text",
                "url": url,
                "text_len": text_len_tmp,
                "han_chars": han_tmp,
                "latin_chars": latin_tmp,
                "han_ratio": h_ratio_tmp,
                "latin_ratio": l_ratio_tmp,
                "vertical_stats": vertical_stats,
            }

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

    signature = extreme_clean_for_compare(
        text,
        target_lang="zh",
        opencc_converter=opencc_converter,
    )
    trad_score = traditional_score(text)

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
        traditional_score=trad_score,

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
    opencc_converter: Optional[Any] = None,
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
            opencc_converter=opencc_converter,
            drop_vertical_ocr=not args.keep_vertical_ocr,
            vertical_min_lines=args.vertical_min_lines,
            vertical_ratio_threshold=args.vertical_ratio_threshold,
            vertical_avg_len_threshold=args.vertical_avg_len_threshold,
        )

        if cand is not None:
            candidates.append(cand)
        elif rej is not None:
            rejected.append(rej)

    return doc_id, candidates, rejected


# =============================================================================
# Candidate ranking and dedup
# =============================================================================

def candidate_quality_key(
    c: Candidate,
    trusted_domains: Set[str],
    prefer_traditional: bool = False,
    prefer_longer: bool = False,
) -> Tuple[float, float, int, float, int]:
    """
    Higher is better.
    Default: domain > trad > han_ratio > han_chars > text_len
    With prefer_longer: domain > trad > text_len > han_chars > han_ratio
    """
    domain_score = 1.0 if c.domain.lower() in trusted_domains else 0.0
    trad_score = c.traditional_score if prefer_traditional else 0.0
    if prefer_longer:
        return (domain_score, trad_score, c.text_len, c.han_chars, c.han_ratio)
    return (domain_score, trad_score, c.han_ratio, c.han_chars, c.text_len)

def exact_dedup_candidates(
    candidates: List[Candidate],
    scope: str,
    trusted_domains: Set[str],
    prefer_traditional: bool = False,
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
            key=lambda x: candidate_quality_key(x, trusted_domains, prefer_traditional=prefer_traditional),
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
    prefer_traditional: bool = False,
    prefer_longer: bool = True,  # Default True now
    opencc_converter: Optional[Any] = None,
    similarity_mode: str = "jaccard",  # "jaccard", "containment", "containment_asym"
    verbose: bool = False,
) -> Tuple[List[Candidate], List[Dict[str, Any]]]:
    """
    Near dedup by character n-gram similarity.
    
    similarity_mode:
        - "jaccard": Symmetric Jaccard similarity
        - "containment": Min-containment (original behavior)
        - "containment_asym": Asymmetric containment, keeps longer doc
    """
    if not candidates:
        return [], []

    # Sort best-first based on quality
    ordered = sorted(
        candidates,
        key=lambda x: candidate_quality_key(
            x, 
            trusted_domains, 
            prefer_traditional=prefer_traditional,
            prefer_longer=prefer_longer,
        ),
        reverse=True,
    )

    # Pre-compute signatures and ngram sets
    signatures: Dict[int, str] = {}
    ngram_sets: Dict[int, Set[str]] = {}
    
    for i, c in enumerate(ordered):
        sig = extreme_clean_for_compare(
            c.text,
            target_lang="zh",
            opencc_converter=opencc_converter,
        )
        signatures[i] = sig
        ngram_sets[i] = get_character_ngrams(sig, n=ngram_n)

    keep_indices: List[int] = []
    dropped_indices: Set[int] = set()
    rejected: List[Dict[str, Any]] = []

    for i, cand_i in enumerate(ordered):
        if i in dropped_indices:
            continue
        
        keep_indices.append(i)
        set_i = ngram_sets[i]
        
        for j in range(i + 1, len(ordered)):
            if j in dropped_indices:
                continue
            
            cand_j = ordered[j]
            
            if scope == "doc" and cand_i.doc_id != cand_j.doc_id:
                continue
            
            set_j = ngram_sets[j]
            
            # Calculate similarity based on mode
            is_duplicate = False
            score = 0.0
            drop_reason = ""
            
            if similarity_mode == "jaccard":
                score = calculate_jaccard(set_i, set_j)
                is_duplicate = score >= threshold
                drop_reason = f"jaccard={score:.3f}"
                
            elif similarity_mode == "containment":
                # Original min-containment
                score = calculate_containment(set_i, set_j)
                is_duplicate = score >= threshold
                drop_reason = f"containment={score:.3f}"
                
            elif similarity_mode == "containment_asym":
                # Asymmetric: check if j is contained in i
                # Since i is "better" (sorted first), we keep i and drop j if j ⊂ i
                cont_i_in_j, cont_j_in_i = calculate_containment_asymmetric(set_i, set_j)
                
                # j is mostly contained in i → drop j (the shorter/worse one)
                if cont_j_in_i >= threshold:
                    is_duplicate = True
                    score = cont_j_in_i
                    drop_reason = f"j_in_i={cont_j_in_i:.3f}"
                # Or high mutual overlap
                elif min(cont_i_in_j, cont_j_in_i) >= threshold * 0.9:
                    is_duplicate = True
                    score = min(cont_i_in_j, cont_j_in_i)
                    drop_reason = f"mutual={score:.3f}"
            
            if is_duplicate:
                dropped_indices.add(j)
                rejected.append({
                    "page_id": cand_j.page_id,
                    "doc_id": cand_j.doc_id,
                    "reason": "duplicate_near",
                    "similarity_mode": similarity_mode,
                    "similarity_score": score,
                    "threshold": threshold,
                    "url": cand_j.canonical_url or cand_j.final_url or cand_j.url,
                    "text_len": cand_j.text_len,
                    "kept_doc_id": cand_i.doc_id,
                    "kept_page_id": cand_i.page_id,
                    "kept_url": cand_i.canonical_url or cand_i.final_url or cand_i.url,
                    "kept_text_len": cand_i.text_len,
                })
                if verbose:
                    drop_title = cand_j.html_title or cand_j.title or cand_j.url
                    keep_title = cand_i.html_title or cand_i.title or cand_i.url
                    print(
                        f"[near-dup] DROP: {drop_title[:60]}\n"
                        f"           (len={cand_j.text_len:,}, {cand_j.domain})\n"
                        f"     KEEP: {keep_title[:60]}\n"
                        f"           (len={cand_i.text_len:,}, {cand_i.domain})\n"
                        f"     {drop_reason} >= {threshold}\n"
                    )                

    kept = [ordered[i] for i in keep_indices]
    return kept, rejected

# =============================================================================
# Export
# =============================================================================

def export_candidates(
    candidates: List[Candidate],
    rejected_by_doc: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    flat_output: bool = False,
    flat_name_mode: str = "global_id",
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

    global_idx = 0

    for doc_id, doc_candidates in sorted(by_doc.items(), key=lambda x: x[0]):
        if flat_output:
            txt_dir = output_dir
            manifest_dir = output_dir / "_manifests"
        else:
            txt_dir = output_dir / safe_filename(doc_id)
            manifest_dir = txt_dir

        txt_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)

        exported_records = []

        # stable order: page_id then domain
        doc_candidates = sorted(doc_candidates, key=lambda c: (c.page_id, c.domain, c.content_hash))

        for local_idx, c in enumerate(doc_candidates, start=1):
            global_idx += 1
            filename_idx = global_idx if flat_output else local_idx

            filename = build_output_filename(
                c,
                idx=filename_idx,
                doc_id=doc_id,
                flat_output=flat_output,
                flat_name_mode=flat_name_mode,
            )

            txt_path = txt_dir / filename
            txt_path.write_text(c.text, encoding="utf-8")

            c.txt_path = str(txt_path)

            record = asdict(c)
            record["global_id"] = global_idx
            record["local_id"] = local_idx

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

        if flat_output:
            manifest_path = manifest_dir / f"{safe_filename(doc_id, 100)}__manifest.json"
        else:
            manifest_path = manifest_dir / "_manifest.json"

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
    parser.add_argument(
        "--keep_vertical_ocr",
        action="store_true",
        help=(
            "Keep pages that look like vertical PDF/OCR extraction, e.g. one Han "
            "character per line. Default is to drop them."
        ),
    )
    parser.add_argument(
        "--vertical_min_lines",
        type=int,
        default=30,
        help="Minimum non-empty lines, also used as minimum consecutive vertical-line run length.",
    )
    parser.add_argument(
        "--vertical_ratio_threshold",
        type=float,
        default=0.60,
        help="Drop if this fraction of non-empty lines are single-Han-character lines.",
    )
    parser.add_argument(
        "--vertical_avg_len_threshold",
        type=float,
        default=3.0,
        help="Drop vertical OCR only if average non-space line length is <= this value.",
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
        "--dedup_opencc",
        type=str,
        choices=["none", "s2t", "t2s"],
        default="none",
        help=(
            "Normalize Simplified/Traditional Chinese before dedup only. "
            "Use s2t if you prefer Traditional output candidates."
        ),
    )
    parser.add_argument(
        "--prefer_traditional",
        action="store_true",
        help="When duplicates are found, prefer the candidate that looks more Traditional Chinese.",
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

    parser.add_argument(
        "--similarity_mode",
        type=str,
        choices=["jaccard", "containment", "containment_asym"],
        default="jaccard",
        help="Near-dedup similarity metric (default: jaccard).",
    )
    
    parser.add_argument(
        "--prefer_longer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer longer text when deduplicating (default: True).",
    )

    # Output
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

    opencc_converter = None
    if args.dedup_opencc != "none":
        if OpenCC is None:
            raise ImportError(
                "OpenCC is required for --dedup_opencc. "
                "Install it with: pip install opencc-python-reimplemented"
            )
        opencc_converter = OpenCC(args.dedup_opencc)

    # Auto output layout:
    #   --dedup_scope global -> flat corpus, filenames use global ids:
    #       0001__title__domain__hash.txt
    #
    #   --dedup_scope doc -> per-doc folders, filenames keep local ids:
    #       <doc_id>/0001__title__domain__hash.txt
    flat_output = args.dedup_scope == "global"
    flat_name_mode = "global_id" if flat_output else "doc_id"

    all_candidates: List[Candidate] = []
    rejected_by_doc: Dict[str, List[Dict[str, Any]]] = {}

    if args.verbose:
        print("=" * 80)
        print("[*] EXPORT CLEAN TXT CORPUS")
        print(f"[*] Input page_dir: {page_dir}")
        print(f"[*] Output dir:     {output_dir}")
        print(f"[*] JSON files:     {len(json_files)}")
        print(f"[*] min_han_ratio:  {args.min_han_ratio}")
        print(f"[*] drop vertical:  {not args.keep_vertical_ocr}")
        print(f"[*] exact dedup:    {not args.disable_exact_dedup}")
        print(f"[*] near dedup:     {args.near_dedup}")
        print(f"[*] dedup_opencc:   {args.dedup_opencc}")
        print(f"[*] prefer trad:    {args.prefer_traditional}")
        print(f"[*] output layout:  {'flat/global-id' if flat_output else 'per-doc/local-id'}")
        print("=" * 80)

    # Phase 1: collect + filter + clean
    for path in json_files:
        doc_id, candidates, rejections = collect_candidates_from_file(
            path,
            args,
            opencc_converter=opencc_converter,
        )

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
            prefer_traditional=args.prefer_traditional,
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
            prefer_traditional=args.prefer_traditional,
            prefer_longer=args.prefer_longer,
            opencc_converter=opencc_converter,
            similarity_mode=args.similarity_mode,
            verbose=args.verbose,
        )
        add_rejections_by_doc(rejected_by_doc, near_rejections)

    num_after_near = len(all_candidates)

    # Phase 4: export
    export_summary = export_candidates(
        candidates=all_candidates,
        rejected_by_doc=rejected_by_doc,
        output_dir=output_dir,
        flat_output=flat_output,
        flat_name_mode=flat_name_mode,
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
            "drop_vertical_ocr": not args.keep_vertical_ocr,
            "vertical_min_lines": args.vertical_min_lines,
            "vertical_ratio_threshold": args.vertical_ratio_threshold,
            "vertical_avg_len_threshold": args.vertical_avg_len_threshold,
            "annotation_mode": args.annotation_mode,
            "unicode_form": args.unicode_form,
            "dedup_scope": args.dedup_scope,
            "dedup_opencc": args.dedup_opencc,
            "prefer_traditional": args.prefer_traditional,
            "exact_dedup": not args.disable_exact_dedup,
            "near_dedup": args.near_dedup,
            "near_threshold": args.near_threshold,
            "ngram_n": args.ngram_n,
            "trusted_domains": sorted(trusted_domains),
            "flat_output": flat_output,
            "flat_name_mode": flat_name_mode,
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