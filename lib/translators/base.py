"""
Translation Cache & Batch Translator
Simple, single direction design
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

from lib.utils import normalize_vietnamese_phrase

TranslateFn = Callable[[List[str]], Dict[str, str]]

class TranslationCache:
    """
    Simple translation cache: src → tgt
    
    Usage:
        cache = TranslationCache("vn", "zh", cache_dir="cache/")
        cache.add("Hùng_Vương", "雄王")
        cache.get("Hùng_Vương")  # → "雄王"
    """
    
    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        cache_dir: Optional[str] = None,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._data: Dict[str, str] = {}
        
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self.load()
    
    @property
    def filepath(self) -> Optional[Path]:
        if not self._cache_dir:
            return None
        return self._cache_dir / f"{self.src_lang}_to_{self.tgt_lang}.json"
    
    @property
    def direction(self) -> str:
        return f"{self.src_lang}→{self.tgt_lang}"
    
    # ───────────────────────────────────────────────────────────────────────────
    # OPERATIONS
    # ───────────────────────────────────────────────────────────────────────────
    
    def add(self, src: str, tgt: str):
        if src:
            self._data[normalize_vietnamese_phrase(src)] = "" if tgt is None else str(tgt).strip()
    
    def add_batch(self, pairs: Dict[str, str]):
        for src, tgt in pairs.items():
            self.add(src, tgt)
    
    def get(self, src: str) -> Optional[str]:
        return self._data.get(normalize_vietnamese_phrase(src))
    
    def has(self, src: str) -> bool:
        return normalize_vietnamese_phrase(src) in self._data
    
    def lookup(self, terms: List[str]) -> Tuple[Dict[str, str], List[str]]:
        """Returns (found, missing). Empty-string translations are valid cache hits."""
        found, missing = {}, []

        for t in terms:
            if self.has(t):
                found[t] = self.get(t) or ""
            else:
                missing.append(t)

        return found, missing
    
    # ───────────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ───────────────────────────────────────────────────────────────────────────
    
    def save(self):
        if self.filepath:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
    
    def load(self):
        if self.filepath and self.filepath.exists():
            with open(self.filepath, "r", encoding="utf-8") as f:
                self._data = json.load(f)
    
    # ───────────────────────────────────────────────────────────────────────────
    # UTILS
    # ───────────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._data)
    
    def __repr__(self) -> str:
        return f"TranslationCache({self.direction}, n={len(self)})"


class BatchTranslator:
    """
    Batch translator with cache.

    Usage:
        translator = BatchTranslator(cache, translate_fn)
        results = translator.translate(["term1", "term2"])
    """
    
    def __init__(
        self,
        cache: TranslationCache,
        translate_fn: Optional[TranslateFn] = None,
        batch_size: int = 20,
        sleep_sec: float = 1.0,
        max_retries: int = 3,
    ):
        self.cache = cache
        self.translate_fn = translate_fn
        self.batch_size = batch_size
        self.sleep_sec = sleep_sec
        self.max_retries = max_retries
    
    def translate(
        self,
        terms: List[str],
        save_cache: bool = True,
        verbose: bool = False,
    ) -> Dict[str, str]:
        """Translate terms with cache lookup."""
        if not terms:
            return {}
        
        cached, missing = self.cache.lookup(terms)
        
        if verbose:
            empty_hit = sum(1 for v in cached.values() if v == "")
            print(
                f"[{self.cache.direction}] "
                f"Hit: {len(cached)}, Empty hit: {empty_hit}, Miss: {len(missing)}"
            )
        
        if not missing or not self.translate_fn:
            return cached
        
        translated = self._batch_translate(missing, verbose)
        self.cache.add_batch(translated)
        
        if save_cache:
            self.cache.save()
        
        return {**cached, **translated}
    
    def _batch_translate(self, terms: List[str], verbose: bool) -> Dict[str, str]:
        results = {}
        batches = [terms[i:i + self.batch_size] for i in range(0, len(terms), self.batch_size)]
        
        for i, batch in enumerate(batches):
            if verbose:
                print(f"Batch {i+1}/{len(batches)} ({len(batch)} terms)")
            
            result = self._call_with_retry(batch)
            results.update(result)
            
            if i < len(batches) - 1:
                time.sleep(self.sleep_sec)
        
        return results
    
    def _call_with_retry(self, terms: List[str]) -> Dict[str, str]:
        for attempt in range(self.max_retries):
            try:
                return self.translate_fn(terms)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    print(f"Retry {attempt+1}: {e}")
                else:
                    print(f"Failed: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def create_translator(
    src_lang: str,
    tgt_lang: str,
    translate_fn: Optional[TranslateFn] = None,
    cache_dir: str = "cache/",
    batch_size: int = 20,
) -> BatchTranslator:
    """Create translator with cache."""
    cache = TranslationCache(src_lang, tgt_lang, cache_dir)
    return BatchTranslator(cache, translate_fn, batch_size)