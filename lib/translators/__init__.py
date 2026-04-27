"""
Translator factory.

Use this module from pipeline code instead of importing backend-specific code
inside scripts like build_query.py.
"""

from typing import Optional

from lib.config import get_env, require_env

from .base import BatchTranslator, TranslationCache, TranslateFn, create_cache_translator


def create_translator(
    src_lang: str,
    tgt_lang: str,
    backend: str = "cache",
    cache_dir: str = "cache/",
    batch_size: int = 20,
    verbose: bool = False,
    gemini_model_name: Optional[str] = None,
) -> BatchTranslator:
    """
    Create a cache-backed translator with an optional translation backend.

    backend:
        - "cache" / "none": cache lookup only
        - "gemini": use Gemini for missing terms
    """
    backend = (backend or "cache").lower().strip()
    translate_fn: Optional[TranslateFn] = None

    if backend in {"cache", "none", "off"}:
        translate_fn = None

    elif backend == "gemini":
        from .gemini import create_gemini_translate_fn

        translate_fn = create_gemini_translate_fn(
            api_key=require_env("GEMINI_API_KEY"),
            model_name=gemini_model_name or get_env("GEMINI_MODEL_NAME", "models/gemini-2.5-pro"),
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            verbose=verbose,
        )

    else:
        raise ValueError(f"Unknown translation backend: {backend}")

    return create_cache_translator(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        cache_dir=cache_dir,
        batch_size=batch_size,
        translate_fn=translate_fn,
    )


__all__ = [
    "BatchTranslator",
    "TranslationCache",
    "TranslateFn",
    "create_translator",
]
