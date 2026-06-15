from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union

import torch

_DEFAULT_MAX_SENTENCES_LENGTH = 10000

# split after end-of-sentence punctuation.
# VI: split after sentence-ending punct OR at newlines
_SPLIT_VI = re.compile(r"(?:(?<=[\.\!\?…]|[。！？…])\s*|\n+)")
# ZH: split after CJK punct (optionally followed by closing quotes) OR at newlines
_SPLIT_ZH = re.compile(r"(?:(?<=[。！？…])(?:[」』”’》〉）\]\}]+)?\s*|\n+)")

_DROP_CHARS = str.maketrans({
    "「": "", "」": "", "『": "", "』": "",
    "〈": "", "〉": "", "《": "", "》": "",
    "“": "", "”": "", "‘": "", "’": "",
    "\"": "", "'": "",
})

# Characters dropped by normalization plus the BOM.
# Note: str.maketrans keys are ordinals (ints), so convert back to chars.
_DROP_SET = {chr(k) for k in _DROP_CHARS} | {"﻿"}

_PUNCT_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)

# A single chunk metadata record: one per embedding row.
ChunkRecord = Dict[str, Union[int, str]]


# =============================================================================
# Normalization with offset tracking (normalized char -> raw char span)
# =============================================================================

def _normalize_with_offsets(raw: str) -> Tuple[str, List[int], List[int]]:
    """
    Reproduce ``_normalize`` exactly while tracking, for each normalized
    character, the [start, end) span in the *raw* input it originated from.

    Returns:
        (norm_text, starts, ends) where len(starts) == len(ends) == len(norm_text)
        and norm_text[i] came from raw[starts[i]:ends[i]].
    """
    # Each item: (char, raw_start, raw_end_exclusive)
    items: List[Tuple[str, int, int]] = [(c, i, i + 1) for i, c in enumerate(raw)]

    # 1) lstrip leading BOM (mirror text.lstrip("﻿"))
    lead = 0
    while lead < len(items) and items[lead][0] == "﻿":
        lead += 1
    items = items[lead:]

    # 2) \r\n -> \n and lone \r -> \n
    merged: List[Tuple[str, int, int]] = []
    i = 0
    while i < len(items):
        ch, rs, re_ = items[i]
        if ch == "\r":
            if i + 1 < len(items) and items[i + 1][0] == "\n":
                merged.append(("\n", rs, items[i + 1][2]))
                i += 2
            else:
                merged.append(("\n", rs, re_))
                i += 1
        else:
            merged.append((ch, rs, re_))
            i += 1
    items = merged

    # 3) drop quote / BOM characters (mirror text.translate(_DROP_CHARS))
    items = [it for it in items if it[0] not in _DROP_SET]

    # 4) collapse runs of spaces/tabs into a single space
    collapsed: List[Tuple[str, int, int]] = []
    i = 0
    while i < len(items):
        ch, rs, re_ = items[i]
        if ch in (" ", "\t"):
            j = i
            last_end = re_
            while j < len(items) and items[j][0] in (" ", "\t"):
                last_end = items[j][2]
                j += 1
            collapsed.append((" ", rs, last_end))
            i = j
        else:
            collapsed.append((ch, rs, re_))
            i += 1
    items = collapsed

    # 5) strip leading/trailing whitespace (mirror text.strip())
    lo, hi = 0, len(items)
    while lo < hi and items[lo][0].isspace():
        lo += 1
    while hi > lo and items[hi - 1][0].isspace():
        hi -= 1
    items = items[lo:hi]

    norm = "".join(it[0] for it in items)
    starts = [it[1] for it in items]
    ends = [it[2] for it in items]
    return norm, starts, ends


def _map_norm_span_to_raw(
    starts: List[int],
    ends: List[int],
    ns: int,
    ne: int,
) -> Tuple[int, int]:
    """
    Map a normalized-text span [ns, ne) to a raw-text span [raw_start, raw_end).

    Empty spans (ne <= ns) or out-of-range indices return (-1, -1).
    """
    if not starts or ne <= ns or ns < 0 or ne > len(starts):
        return -1, -1
    return starts[ns], ends[ne - 1]


def _normalize(text: str) -> str:
    """Backward-compatible normalizer (string only)."""
    norm, _, _ = _normalize_with_offsets(text)
    return norm


# =============================================================================
# Sentence splitting
# =============================================================================

def _split_by_punct_spans(norm: str, lang: str) -> List[Tuple[str, int, int]]:
    """
    Split normalized text into sentences, returning (sentence, start, end)
    where [start, end) are character offsets into ``norm``.
    """
    if not norm:
        return []

    pattern = _SPLIT_ZH if lang == "zh" else _SPLIT_VI

    spans: List[Tuple[int, int]] = []
    prev = 0
    for m in pattern.finditer(norm):
        if m.start() > prev:
            spans.append((prev, m.start()))
        prev = m.end()
    if prev < len(norm):
        spans.append((prev, len(norm)))

    out: List[Tuple[str, int, int]] = []
    for a, b in spans:
        seg = norm[a:b]
        left = len(seg) - len(seg.lstrip())
        right = len(seg) - len(seg.rstrip())
        ns, ne = a + left, b - right
        if ne <= ns:
            continue
        stripped = norm[ns:ne]
        s2 = re.sub(r"\s+", "", stripped)
        if not s2 or _PUNCT_ONLY.match(s2):
            continue
        out.append((stripped, ns, ne))
    return out


def _sentence_spans(
    file_path: Union[str, Path],
    lang: str,
    max_len: Optional[int],
) -> Tuple[str, str, List[int], List[int], List[Tuple[str, int, int]]]:
    """
    Load + normalize a document and split it into sentences with raw offsets.

    Returns:
        (raw_text, norm_text, starts, ends, sentences) where each sentence is
        (text, norm_start, norm_end). ``starts``/``ends`` map norm -> raw.
    """
    lang = (lang or "").strip().lower()
    if lang not in {"vi", "zh"}:
        raise ValueError("lang must be 'vi' or 'zh'")

    raw = Path(file_path).read_text(encoding="utf-8", errors="replace")
    norm, starts, ends = _normalize_with_offsets(raw)
    sentences = _split_by_punct_spans(norm, lang)

    if max_len is not None:
        if max_len <= 0:
            raise ValueError("max_len must be a positive int or None")
        truncated: List[Tuple[str, int, int]] = []
        for text, ns, ne in sentences:
            if len(text) > max_len:
                ne = ns + max_len
                text = norm[ns:ne]
            truncated.append((text, ns, ne))
        sentences = truncated

    return raw, norm, starts, ends, sentences


def sent_split(file_path: Union[str, Path], lang: str, max_len: Optional[int] = _DEFAULT_MAX_SENTENCES_LENGTH) -> List[str]:
    _, _, _, _, sentences = _sentence_spans(file_path, lang, max_len)
    return [s for s, _, _ in sentences]


def _empty_features() -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.empty((0, 0), dtype=torch.long),
        "attention_mask": torch.empty((0, 0), dtype=torch.long),
    }


def sent_split_tkn(
    file_path: Union[str, Path],
    tokenizer,
    lang: str,
    *,
    max_len: Optional[int] = _DEFAULT_MAX_SENTENCES_LENGTH,
    num_of_sent: int = 1,
    overlap_sent: int = 0,
    max_tokens: Optional[int] = None,
    add_special_tokens: bool = True,
    return_metadata: bool = False,
) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], List[ChunkRecord]]]:
    """
    Tokenize a document into (grouped) sentence rows.

    When ``return_metadata=True`` also returns one ChunkRecord per embedding row
    with raw-text char offsets:
        {"chunk_idx", "char_start", "char_end", "text"}
    """
    raw, norm, starts, ends, sentences = _sentence_spans(file_path, lang, max_len)

    if not sentences:
        return (_empty_features(), []) if return_metadata else _empty_features()

    # Group consecutive sentences (with optional overlap). Track each group's
    # span as (first sentence start, last sentence end) in norm coordinates.
    if num_of_sent > 1:
        grouped_text: List[str] = []
        grouped_span: List[Tuple[int, int]] = []
        stride = max(1, num_of_sent - overlap_sent)
        for i in range(0, len(sentences), stride):
            group = sentences[i:i + num_of_sent]
            if group:
                grouped_text.append(" ".join(g[0] for g in group))
                grouped_span.append((group[0][1], group[-1][2]))
            if i + num_of_sent >= len(sentences):
                break
    else:
        grouped_text = [s[0] for s in sentences]
        grouped_span = [(s[1], s[2]) for s in sentences]

    enc = tokenizer(
        grouped_text,
        add_special_tokens=add_special_tokens,
        padding=True,
        truncation=(max_tokens is not None),
        max_length=max_tokens,
        return_tensors="pt",
        return_attention_mask=True,
    )

    features = {
        "input_ids": enc["input_ids"].long(),
        "attention_mask": enc["attention_mask"].long(),
    }

    if not return_metadata:
        return features

    records: List[ChunkRecord] = []
    for idx, (ns, ne) in enumerate(grouped_span):
        rs, re_ = _map_norm_span_to_raw(starts, ends, ns, ne)
        records.append({
            "chunk_idx": idx,
            "char_start": rs,
            "char_end": re_,
            "text": raw[rs:re_] if rs >= 0 else grouped_text[idx],
        })

    assert len(records) == features["input_ids"].shape[0], (
        f"sentence metadata mismatch: {len(records)} records "
        f"vs {features['input_ids'].shape[0]} rows"
    )
    return features, records


# =============================================================================
# Chunk splitting (token windows)
# =============================================================================

def chunk_split(
    file_path: Union[str, Path],
    tokenizer,
    chunk_size: int,
    overlap_size: int,
    *,
    max_tokens: int = 512,
    return_metadata: bool = False,
) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], List[ChunkRecord]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap_size < 0:
        raise ValueError("overlap_size must be >= 0")
    if overlap_size >= chunk_size:
        raise ValueError("overlap_size must be < chunk_size")

    raw = Path(file_path).read_text(encoding="utf-8", errors="replace")
    norm, starts, ends = _normalize_with_offsets(raw)

    if not norm:
        return (_empty_features(), []) if return_metadata else _empty_features()

    # Tokenize full doc once. Request offset_mapping for char offsets (fast
    # tokenizers only); degrade gracefully if unavailable.
    offsets: Optional[List[Tuple[int, int]]] = None
    if return_metadata:
        try:
            enc = tokenizer(
                norm, add_special_tokens=False, truncation=False,
                return_offsets_mapping=True,
            )
            ids = enc["input_ids"]
            offsets = [tuple(o) for o in enc["offset_mapping"]]
        except Exception:
            ids = tokenizer(norm, add_special_tokens=False, truncation=False)["input_ids"]
            offsets = None
    else:
        ids = tokenizer(norm, add_special_tokens=False, truncation=False)["input_ids"]

    if not ids:
        return (_empty_features(), []) if return_metadata else _empty_features()

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    chunk_size = min(chunk_size, max_tokens - 2)
    step = chunk_size - overlap_size
    if step <= 0:
        raise ValueError("overlap_size too large (step <= 0)")

    chunks: List[List[int]] = []
    spans: List[Tuple[int, int]] = []  # token index ranges [start, end)
    for start in range(0, len(ids), step):
        piece = ids[start:start + chunk_size]
        if not piece:
            break
        chunks.append([cls_id] + piece + [sep_id])
        spans.append((start, start + len(piece)))
        if start + chunk_size >= len(ids):
            break

    if not chunks:
        return (_empty_features(), []) if return_metadata else _empty_features()

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    batch_size = len(chunks)
    chunk_length = max(len(c) for c in chunks)

    input_ids = torch.full((batch_size, chunk_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, chunk_length), dtype=torch.long)
    for i, c in enumerate(chunks):
        input_ids[i, :len(c)] = torch.tensor(c, dtype=torch.long)
        attention_mask[i, :len(c)] = 1

    features = {"input_ids": input_ids, "attention_mask": attention_mask}

    if not return_metadata:
        return features

    records: List[ChunkRecord] = []
    for idx, (t_start, t_end) in enumerate(spans):
        rs, re_ = -1, -1
        if offsets is not None and t_end > t_start:
            # First/last token offsets with a real span (ce > cs).
            cs = next((offsets[t][0] for t in range(t_start, t_end) if offsets[t][1] > offsets[t][0]), None)
            ce = next((offsets[t][1] for t in range(t_end - 1, t_start - 1, -1) if offsets[t][1] > offsets[t][0]), None)
            if cs is not None and ce is not None and ce > cs:
                rs, re_ = _map_norm_span_to_raw(starts, ends, cs, ce)
        records.append({
            "chunk_idx": idx,
            "char_start": rs,
            "char_end": re_,
            "text": raw[rs:re_] if rs >= 0 else "",
        })

    assert len(records) == features["input_ids"].shape[0], (
        f"chunk metadata mismatch: {len(records)} records "
        f"vs {features['input_ids'].shape[0]} rows"
    )
    return features, records
