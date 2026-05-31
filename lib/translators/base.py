"""
Translation Cache & Batch Translator
SQLite-backed, single direction design.

Cache file: {cache_dir}/translation.db
Schema:
    translation_cache(src_lang, tgt_lang, src_text, tgt_text, model, created_at)
    PRIMARY KEY (src_lang, tgt_lang, src_text)

Notes:
  - tgt_text is nullable: NULL = never attempted, "" = attempted but no valid translation
  - WAL mode for safe concurrent writes
  - In-memory dict keeps lookups O(1) after initial load
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

from lib.utils import normalize_vietnamese_phrase, normalize_chinese_phrase

TranslateFn = Callable[[List[str]], Dict[str, str]]

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS translation_cache (
    src_lang    TEXT NOT NULL,
    tgt_lang    TEXT NOT NULL,
    src_text    TEXT NOT NULL,
    tgt_text    TEXT,
    model       TEXT,
    created_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (src_lang, tgt_lang, src_text)
)
"""


class TranslationCache:
    """
    SQLite-backed translation cache: src → tgt.

    Usage:
        cache = TranslationCache("vi", "zh", cache_dir="cache/")
        cache.add("Hùng Vương", "雄王")
        cache.get("Hùng Vương")  # → "雄王"

    The SQLite file lives at {cache_dir}/translation.db.
    An empty-string translation ("") is a valid hit meaning
    "this term was attempted but has no useful translation".
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

        # The cache key is the SOURCE term, so the normalizer must match the
        # source language: Chinese has no word boundaries (strip whitespace,
        # NFC), Vietnamese is lowercased with underscores → spaces.
        self._normalize = (
            normalize_chinese_phrase
            if src_lang == "zh"
            else normalize_vietnamese_phrase
        )

        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._init_db()
            self._migrate_from_json()
            self.load()

    # ───────────────────────────────────────────────────────────────────────────
    # PATHS / PROPERTIES
    # ───────────────────────────────────────────────────────────────────────────

    @property
    def db_path(self) -> Optional[Path]:
        if not self._cache_dir:
            return None
        return self._cache_dir / "translation.db"

    @property
    def direction(self) -> str:
        return f"{self.src_lang}→{self.tgt_lang}"

    # ───────────────────────────────────────────────────────────────────────────
    # DB INIT & MIGRATION
    # ───────────────────────────────────────────────────────────────────────────

    def _init_db(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE)
            conn.commit()

    def _migrate_from_json(self):
        """
        One-time migration: if an old JSON cache file exists at
        {cache_dir}/{src_lang}_to_{tgt_lang}.json, import its entries
        into SQLite and rename the file to .json.bak.
        """
        if not self._cache_dir:
            return

        old_json = self._cache_dir / f"{self.src_lang}_to_{self.tgt_lang}.json"
        if not old_json.exists():
            return

        try:
            with open(old_json, "r", encoding="utf-8") as f:
                old_data: Dict[str, str] = json.load(f)

            if old_data:
                rows = [
                    (self.src_lang, self.tgt_lang, src, tgt)
                    for src, tgt in old_data.items()
                    if src
                ]
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.executemany(
                        """
                        INSERT OR IGNORE INTO translation_cache
                            (src_lang, tgt_lang, src_text, tgt_text)
                        VALUES (?, ?, ?, ?)
                        """,
                        rows,
                    )
                    conn.commit()

            old_json.rename(old_json.with_suffix(".json.bak"))
            print(
                f"[TranslationCache] Migrated {len(old_data)} entries "
                f"from {old_json.name} → translation.db"
            )

        except Exception as e:
            print(f"[TranslationCache] Migration warning: {e}")

    # ───────────────────────────────────────────────────────────────────────────
    # OPERATIONS
    # ───────────────────────────────────────────────────────────────────────────

    def add(self, src: str, tgt: str):
        if not src:
            return
        key = self._normalize(src)
        value = "" if tgt is None else str(tgt).strip()
        self._data[key] = value

        if self.db_path:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO translation_cache
                        (src_lang, tgt_lang, src_text, tgt_text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (self.src_lang, self.tgt_lang, key, value),
                )
                conn.commit()

    def add_batch(self, pairs: Dict[str, str]):
        """Batch insert — uses executemany for efficiency."""
        if not pairs:
            return

        rows = []
        for src, tgt in pairs.items():
            if src:
                key = self._normalize(src)
                value = "" if tgt is None else str(tgt).strip()
                self._data[key] = value
                rows.append((self.src_lang, self.tgt_lang, key, value))

        if rows and self.db_path:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO translation_cache
                        (src_lang, tgt_lang, src_text, tgt_text)
                    VALUES (?, ?, ?, ?)
                    """,
                    rows,
                )
                conn.commit()

    def get(self, src: str) -> Optional[str]:
        return self._data.get(self._normalize(src))

    def has(self, src: str) -> bool:
        return self._normalize(src) in self._data

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

    def load(self):
        """Load all entries for this language pair into the in-memory dict."""
        if not self.db_path or not self.db_path.exists():
            return

        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                """
                SELECT src_text, tgt_text
                FROM translation_cache
                WHERE src_lang = ? AND tgt_lang = ?
                """,
                (self.src_lang, self.tgt_lang),
            ).fetchall()

        # NULL tgt_text → "" so callers always get a string
        self._data = {
            src: (tgt if tgt is not None else "")
            for src, tgt in rows
        }

    def save(self):
        """No-op: SQLite writes happen immediately in add() / add_batch()."""
        pass

    # ───────────────────────────────────────────────────────────────────────────
    # EXPORT / IMPORT (for manual editing)
    # ───────────────────────────────────────────────────────────────────────────

    def export_to_json(self, path) -> int:
        """
        Dump this direction's cache to an editable JSON file: {src_text: tgt_text}.

        Intended for manual review/correction of cached translations. Keys are
        the normalized source terms (exactly as stored); edit the values, then
        re-apply with import_from_json(). Returns the number of entries written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = dict(sorted(self._data.items()))
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return len(data)

    def import_from_json(self, path) -> int:
        """
        Load a {src_text: tgt_text} JSON file back into the cache and SQLite DB.

        Used to apply manual edits. Keys are re-normalized via the source-language
        normalizer (so edits are idempotent for already-normalized keys), and a
        None value is stored as "" (attempted, no useful translation). Returns
        the number of entries imported.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected a JSON object mapping src -> tgt, got {type(data).__name__}"
            )

        self.add_batch({
            str(src): ("" if tgt is None else str(tgt))
            for src, tgt in data.items()
            if str(src).strip()
        })

        return len(data)

    # ───────────────────────────────────────────────────────────────────────────
    # UTILS
    # ───────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"TranslationCache({self.direction}, n={len(self)}, db={self.db_path})"


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
        # add_batch writes to SQLite immediately; save_cache flag is kept for
        # API compatibility but has no effect.
        self.cache.add_batch(translated)

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

def create_cache_translator(
    src_lang: str,
    tgt_lang: str,
    translate_fn: Optional[TranslateFn] = None,
    cache_dir: str = "cache/",
    batch_size: int = 20,
) -> BatchTranslator:
    """Create a SQLite-backed batch translator."""
    cache = TranslationCache(src_lang, tgt_lang, cache_dir)
    return BatchTranslator(
        cache=cache,
        translate_fn=translate_fn,
        batch_size=batch_size,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — export/import the translation cache for manual editing
# ═══════════════════════════════════════════════════════════════════════════════

def _cli() -> None:
    """
    Export or import a translation cache as an editable JSON file.

    Examples:
        # Dump the vi->zh cache to a JSON file you can edit by hand
        python -m lib.translators.base export \\
            --src_lang vi --tgt_lang zh --cache_dir ./cache --file vi_zh.json

        # Apply your edits back into the SQLite cache
        python -m lib.translators.base import \\
            --src_lang vi --tgt_lang zh --cache_dir ./cache --file vi_zh.json
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Export/import a translation cache as editable JSON",
    )
    parser.add_argument("action", choices=["export", "import"])
    parser.add_argument("--src_lang", required=True, help="Source language, e.g. vi or zh")
    parser.add_argument("--tgt_lang", required=True, help="Target language, e.g. zh or vi")
    parser.add_argument("--cache_dir", default="./cache", help="Directory holding translation.db")
    parser.add_argument("--file", required=True, help="JSON file to write (export) or read (import)")

    args = parser.parse_args()

    cache = TranslationCache(args.src_lang, args.tgt_lang, args.cache_dir)

    if args.action == "export":
        n = cache.export_to_json(args.file)
        print(f"Exported {n} entries ({cache.direction}) -> {args.file}")
    else:
        n = cache.import_from_json(args.file)
        print(f"Imported {n} entries -> {cache.direction} (db: {cache.db_path})")


if __name__ == "__main__":
    _cli()
