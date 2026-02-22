from pathlib import Path
import re
import sys
import argparse
from typing import Dict, List, Optional, Union

import logging
logger = logging.getLogger(__name__)

_DEFAULT_MAX_SENTENCES_LENGTH = 10000

_ZH_PUNCT_SPLIT = re.compile(r"(?<=[。！？…])")
_FALLBACK_SPLIT_WS = re.compile(r"(?<=[\.\!\?…。！？])\s+")
_FALLBACK_SPLIT_NO_WS = re.compile(r"(?<=[\.\!\?…。！？])")

_DROP_CHARS = str.maketrans({
    "「":"", "」":"", "『":"", "』":"",
    "〈":"", "〉":"", "《":"", "》":"",
    "“":"", "”":"", "‘":"", "’":"",
    "\"":"", "'":""
})
_PUNCT_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)

_STANZA_PIPES: Dict[str, object] = {}

def split(file_path: Union[str, Path], 
          lang: str, 
          max_len: Optional[int] = _DEFAULT_MAX_SENTENCES_LENGTH) -> List[str]:
    lang = (lang or "").strip().lower()
    if lang not in {"vi", "zh"}:
        raise ValueError("lang must be 'vi' or 'zh'")

    text = Path(file_path).read_text(encoding="utf-8", errors="replace")
    text = _normalize(text)

    sents = _split_with_stanza(text, lang)

    if lang == "zh" and sents:
        sents = _post_split_zh(sents)

    if not sents:
        logger.warning("stanza returned no sentences -> regex fallback | file=%s lang=%s", file_path, lang)
        sents = _split_fallback_regex(text)

    sents = [s.strip() for s in sents if s and s.strip()]
    sents = [s for s in sents if not _is_junk_sent(s)]
    # Truncate very long sentences; max_len=None => unlimited
    if max_len is not None:
        if max_len <= 0:
            raise ValueError("max_len must be a positive int or None")
        sents = [s[:max_len] for s in sents]

    return sents

def _normalize(text: str) -> str:
    text = text.lstrip("\ufeff")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_DROP_CHARS)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def _get_stanza_pipe(lang: str):
    if lang in _STANZA_PIPES:
        return _STANZA_PIPES[lang]

    import stanza  # type: ignore

    pipe = stanza.Pipeline(
        lang=lang,
        processors="tokenize",
        tokenize_no_ssplit=False,
        verbose=False,
    )
    _STANZA_PIPES[lang] = pipe
    return pipe


def _split_with_stanza(text: str, lang: str) -> Optional[List[str]]:
    try:
        pipe = _get_stanza_pipe(lang)
        doc = pipe(text)
        sents = [s.text for s in getattr(doc, "sentences", [])]
        return sents or None
    except Exception:
        logger.debug("stanza split failed (%s): %s", lang, e)
        return None

def _post_split_zh(sents: List[str]) -> List[str]:
    out: List[str] = []
    for s in sents:
        parts = _ZH_PUNCT_SPLIT.split(s)
        out.extend([p.strip() for p in parts if p and p.strip()])
    return out

def _split_fallback_regex(text: str) -> List[str]:
    parts = _FALLBACK_SPLIT_WS.split(text)
    if len(parts) <= 1:
        parts = _FALLBACK_SPLIT_NO_WS.split(text)
    return [p for p in parts if p and p.strip()]
    
def _is_junk_sent(s: str) -> bool:
    s2 = re.sub(r"\s+", "", s or "")
    if not s2:
        return True
    return bool(_PUNCT_ONLY.match(s2))
    
def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sentence splitter (vi/zh) using stanza with regex fallback.")
    parser.add_argument("-f", "--file", required=True, help="Input text file path (utf-8 recommended).")
    parser.add_argument("-l", "--lang", required=True, choices=["vi", "zh"], help="Language: vi or zh.")
    args = parser.parse_args(argv)

    out = split(args.file, args.lang)
    sys.stdout.write(out)
    if out and not out.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())