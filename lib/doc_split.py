from pathlib import Path
import re
from typing import Dict, List, Optional, Union
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

_PUNCT_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)

def sent_split(file_path: Union[str, Path], lang: str, max_len: Optional[int] = _DEFAULT_MAX_SENTENCES_LENGTH) -> List[str]:
    lang = (lang or "").strip().lower()
    if lang not in {"vi", "zh"}:
        raise ValueError("lang must be 'vi' or 'zh'")
    
    # read file
    text = Path(file_path).read_text(encoding="utf-8", errors="replace")
    text = _normalize(text)

    # split
    sents = _split_by_punct(text, lang)

    # clean
    sents = [s.strip() for s in sents if s and s.strip()]
    sents = [s for s in sents if not _is_junk_sent(s)]

    # truncate
    if max_len is not None:
        if max_len <= 0:
            raise ValueError("max_len must be a positive int or None")
        sents = [s[:max_len] for s in sents]

    return sents

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
) -> Dict[str, torch.Tensor]:
    # split sentences
    sents = sent_split(file_path, lang=lang, max_len=max_len)
    if not sents:
        return {
            "input_ids": torch.empty((0, 0), dtype=torch.long),
            "attention_mask": torch.empty((0, 0), dtype=torch.long),
        }

    if num_of_sent > 1:
        grouped_sents = []
        # (stride) = num_of_sent - overlap_sent
        stride = max(1, num_of_sent - overlap_sent)
        for i in range(0, len(sents), stride):
            group = sents[i : i + num_of_sent]
            if group:
                grouped_sents.append(" ".join(group))
            if i + num_of_sent >= len(sents):
                break
        sents = grouped_sents
    
    # tokenize batch + pad
    enc = tokenizer(
        sents,
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

    return features

def chunk_split(file_path: Union[str, Path], tokenizer, chunk_size: int, overlap_size: int, *, max_tokens: int = 512) -> Dict[str, torch.Tensor]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap_size < 0:
        raise ValueError("overlap_size must be >= 0")
    if overlap_size >= chunk_size:
        raise ValueError("overlap_size must be < chunk_size")
    
    # Read file
    text = Path(file_path).read_text(encoding="utf-8", errors="replace").strip()
    text = _normalize(text)
    if not text:
        return {
            "input_ids": torch.empty((0, 0), dtype=torch.long),
            "attention_mask": torch.empty((0, 0), dtype=torch.long),
        }
    
    # Tokenize full doc once (no truncation)
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    if not ids:
        return {
            "input_ids": torch.empty((0, 0), dtype=torch.long),
            "attention_mask": torch.empty((0, 0), dtype=torch.long),
        }
    
    # Split (content tokens) + add special tokens later
    cls_id = tokenizer.cls_token_id 
    sep_id = tokenizer.sep_token_id
    chunk_size = min(chunk_size, max_tokens - 2)
    step = chunk_size - overlap_size
    if step <= 0:
        raise ValueError("overlap_size too large (step <= 0)")
    chunks: List[List[int]] = []
    for start in range(0, len(ids), step):
        piece = ids[start:start + chunk_size]
        if not piece:
            break
        piece = [cls_id] + piece + [sep_id]
        chunks.append(piece)
        if start + chunk_size >= len(ids):
            break

    if not chunks:
        return {
            "input_ids": torch.empty((0, 0), dtype=torch.long),
            "attention_mask": torch.empty((0, 0), dtype=torch.long),
        }
    
    # Padding + attention_mask
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    
    batch_size = len(chunks)
    chunk_length = max(len(c) for c in chunks) # chunk_size + cls_id + sep_id
    
    input_ids = torch.full((batch_size, chunk_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, chunk_length), dtype=torch.long)
    
    for i, c in enumerate(chunks):
        l = len(c)
        input_ids[i, :l] = torch.tensor(c, dtype=torch.long)
        attention_mask[i, :l] = 1

    return {"input_ids": input_ids,
            "attention_mask": attention_mask}
       
def _normalize(text: str) -> str:
    text = text.lstrip("\ufeff")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_DROP_CHARS)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def _split_by_punct(text: str, lang: str) -> List[str]:
    if not text:
        return []

    # primary split
    if lang == "zh":
        parts = _SPLIT_ZH.split(text)
    else:
        parts = _SPLIT_VI.split(text)

    return [p for p in parts if p and p.strip()]
    
def _is_junk_sent(s: str) -> bool:
    s2 = re.sub(r"\s+", "", s or "")
    if not s2:
        return True
    return bool(_PUNCT_ONLY.match(s2))
