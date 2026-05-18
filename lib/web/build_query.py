"""
Build web search queries from extracted keywords.

Pipeline:
1. Load keyword JSON.
2. Build one global keyword group, optional local keyword groups, or global variants.
3. Translate keywords.
4. Select Han anchors using Wikisource reranking.
5. Build search queries.
6. Save query JSON.
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.utils import random_sleep_seconds
from lib.web.VnKeywordExtractor import VnKeywordExtractor
from lib.web.wikisource_reranker import (
    score_anchor_candidates,
    get_wikisource_stats,
    get_wikisource_auth_status,
    init_wikisource_session,
    WikisourceRateLimitError,
)
from lib.translators import create_translator
from lib.config import load_project_env, get_env

REQUEST_SLEEP_RANGE = (0.25, 0.35)
GROUP_SLEEP_RANGE = (0.5, 0.5)


def has_wikisource_support(item: Dict[str, Any]) -> bool:
    return (
        str(item.get("rerank_source", "")).startswith("wikisource")
        and int(item.get("support_count", 0) or 0) > 0
    )


def select_query_terms(
    ranked_candidates: List[Dict[str, Any]],
    num_query_terms: int,
    allow_semantic_fallback: bool = False,
) -> List[Dict[str, Any]]:
    """
    Select final query terms after Wikisource reranking.

    Policy:
    - If Wikisource-supported terms exist, use only supported terms by default.
    - If allow_semantic_fallback=True, fill remaining slots with unsupported terms.
    - If no Wikisource-supported term exists, fallback to ranked candidates.
    """
    supported = [
        x for x in ranked_candidates
        if has_wikisource_support(x)
    ]

    if supported:
        if not allow_semantic_fallback:
            return supported[:num_query_terms]

        selected = supported[:num_query_terms]

        if len(selected) < num_query_terms:
            selected_keys = {
                (str(x.get("word", "")), str(x.get("han_word", "")))
                for x in selected
            }

            fallback = [
                x for x in ranked_candidates
                if (str(x.get("word", "")), str(x.get("han_word", ""))) not in selected_keys
            ]

            selected.extend(fallback[: num_query_terms - len(selected)])

        return selected

    return ranked_candidates[:num_query_terms]


def split_exact_and_loose_terms(
    query_terms: List[Dict[str, Any]],
    num_exact_anchors: int = 3,
    preferred_exact_pos: Optional[set[str]] = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split selected query terms into exact quoted anchors and loose terms.

    If Wikisource support metadata is available, exact anchors prefer
    supported terms. This prevents unsupported fallback terms from being quoted.
    """
    preferred_exact_pos = preferred_exact_pos or {"Np"}

    supported_preferred = []
    supported_other = []
    fallback = []

    any_support_metadata = any(
        "support_count" in item or str(item.get("rerank_source", "")).startswith("wikisource")
        for item in query_terms
    )

    for item in query_terms:
        pos_tags = set(str(item.get("pos", "")).split())
        is_supported = has_wikisource_support(item)

        if any_support_metadata:
            if is_supported and (pos_tags & preferred_exact_pos):
                supported_preferred.append(item)
            elif is_supported:
                supported_other.append(item)
            else:
                fallback.append(item)
        else:
            # Semantic-only mode: preserve old behavior.
            if pos_tags & preferred_exact_pos:
                supported_preferred.append(item)
            else:
                fallback.append(item)

    exact = (supported_preferred + supported_other + fallback)[:num_exact_anchors]

    exact_keys = {
        (str(x.get("word", "")), str(x.get("han_word", "")))
        for x in exact
    }

    loose = [
        x for x in query_terms
        if (str(x.get("word", "")), str(x.get("han_word", ""))) not in exact_keys
    ]

    return exact, loose

def build_mixed_query(
    query_terms: List[Dict[str, Any]],
    num_exact_anchors: int = 3,
    sites: Optional[List[str]] = None,
    preferred_exact_pos: Optional[set[str]] = None,
    anchor_only: bool = False,
) -> str:
    exact_items, loose_items = split_exact_and_loose_terms(
        query_terms=query_terms,
        num_exact_anchors=num_exact_anchors,
        preferred_exact_pos=preferred_exact_pos,
    )

    parts = []

    for item in exact_items:
        term = str(item.get("han_word", "") or item.get("trans", "")).strip()
        if term:
            parts.append(f'"{term}"')

    if not anchor_only:
        for item in loose_items:
            term = str(item.get("han_word", "") or item.get("trans", "")).strip()
            if term:
                parts.append(term)

    if sites:
        if len(sites) == 1:
            parts.append(f"site:{sites[0]}")
        else:
            parts.append("(" + " OR ".join(f"site:{site}" for site in sites) + ")")

    return " ".join(parts)


def keyword_tuples_to_dicts(kws):
    return [
        {"word": kw, "score": score, "pos": pos}
        for kw, score, pos in kws
    ]


def shifted_keyword_window(pool, start_idx: int, window_size: int):
    """
    Create a global-query variant by taking a shifted window
    from the global keyword pool.

    If the pool is too short, wrap around to still produce a window.
    """
    if not pool:
        return []

    if len(pool) <= window_size:
        shift = start_idx % len(pool)
        rotated = pool[shift:] + pool[:shift]
        return rotated[:window_size]

    start_idx = start_idx % len(pool)
    end_idx = start_idx + window_size

    if end_idx <= len(pool):
        return pool[start_idx:end_idx]

    return pool[start_idx:] + pool[: end_idx - len(pool)]

def build_queries_for_keyword_file(
    keyword_path: Path,
    output_path: Path,
    args,
    translator,
    allowed_pos,
    preferred_exact_pos,
    sites,
    runtime_state,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with keyword_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    doc_id = data.get("doc_id", keyword_path.stem)
    chunks = data.get("chunks", [])
    num_chunks = len(chunks)

    if num_chunks == 0:
        raise ValueError(f"No chunks found in {keyword_path}")
    
    if args.verbose:
        print(f"\n=== Building queries for {doc_id} ===")
        print(f"Keyword file: {keyword_path}")
        print(f"Chunks: {num_chunks}")
        print(f"Use local query: {args.use_local_query}")
        
    if args.use_local_query:
        num_local_query = max(1, math.ceil(math.sqrt(num_chunks)))
    else:
        num_local_query = 0

    chunk_keywords_for_agg = [
        [
            (x["word"], x["score"], x["pos"])
            for x in chunk.get("keywords", [])
        ]
        for chunk in chunks
    ]

    results = []

    # =========================================================================
    # 1. GLOBAL QUERY GROUP
    # =========================================================================
    extended_global_top_n = args.top_n_keywords + args.min_total_query * args.num_query_terms

    extended_global_kws = VnKeywordExtractor.aggregate(
        chunk_keywords_for_agg,
        top_n=extended_global_top_n,
        rarity_bias=args.rarity_bias,
    )

    global_kws = extended_global_kws[:args.top_n_keywords]

    results.append({
        "query_id": "global",
        "start_chunk_idx": 0,
        "end_chunk_idx": num_chunks - 1,
        "keywords": keyword_tuples_to_dicts(global_kws),
    })

    # =========================================================================
    # 2. OPTIONAL LOCAL QUERY GROUPS
    # =========================================================================
    if args.use_local_query:
        for i in range(num_local_query):
            start_idx = math.floor(i * num_chunks / num_local_query)
            end_idx = math.floor((i + 1) * num_chunks / num_local_query)

            if start_idx >= end_idx:
                continue

            sub_chunk_keywords = chunk_keywords_for_agg[start_idx:end_idx]

            local_kws = VnKeywordExtractor.aggregate(
                sub_chunk_keywords,
                top_n=args.top_n_keywords,
                rarity_bias=args.rarity_bias,
            )

            results.append({
                "query_id": f"local_{i}",
                "start_chunk_idx": start_idx,
                "end_chunk_idx": end_idx - 1,
                "keywords": keyword_tuples_to_dicts(local_kws),
            })
    elif args.verbose:
        print("Local query groups disabled; using global variants to reach min_total_query")

    # =========================================================================
    # 2.1. GLOBAL VARIANTS
    #      - If local queries are enabled, this only pads small documents.
    #      - If local queries are disabled, this becomes the main variant source.
    # =========================================================================
    missing_query_count = max(0, args.min_total_query - len(results))

    if missing_query_count > 0:
        shift_step = max(1, args.num_query_terms)

        for j in range(missing_query_count):
            start_offset = (j + 1) * shift_step

            variant_kws = shifted_keyword_window(
                pool=extended_global_kws,
                start_idx=start_offset,
                window_size=args.top_n_keywords,
            )

            if not variant_kws:
                continue

            results.append({
                "query_id": f"global_variant_{j}",
                "start_chunk_idx": 0,
                "end_chunk_idx": num_chunks - 1,
                "keywords": keyword_tuples_to_dicts(variant_kws),
            })
    
    if args.verbose:
        print(f"Query groups: {len(results)}")
        print(f"Unique terms before translation: {sum(len(item['keywords']) for item in results)} raw occurrences")

    # =========================================================================
    # 3. TRANSLATION
    # =========================================================================
    all_terms = []
    seen_terms = set()

    for item in results:
        for kw in item["keywords"]:
            term = kw["word"]
            if term and term not in seen_terms:
                seen_terms.add(term)
                all_terms.append(term)

    if args.verbose:
        print(f"Unique terms to translate/cache lookup: {len(all_terms)}")

    translated_all = translator.translate(all_terms, verbose=args.verbose)

    for item in results:
        for kw in item["keywords"]:
            kw["trans"] = translated_all.get(kw["word"], "")

        item["translation_hit"] = sum(
            1 for kw in item["keywords"]
            if kw.get("trans")
        )

    if args.verbose:
        total_hit = sum(item.get("translation_hit", 0) for item in results)
        total_kw = sum(len(item["keywords"]) for item in results)
        empty_terms = [term for term, trans in translated_all.items() if trans == ""]

        print(f"Translation hits in groups: {total_hit}/{total_kw}")

    # =========================================================================
    # 4. ANCHOR SELECTION + QUERY BUILD
    # =========================================================================
    for item in results:
        candidates = []

        for kw in item["keywords"]:
            trans = str(kw.get("trans", "")).strip()
            pos = kw.get("pos", "")

            if not trans:
                continue

            candidates.append({
                "word": kw["word"],
                "han_word": trans,
                "score": kw.get("score", 0.0),
                "pos": pos,
            })

        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

        should_use_wikisource = (
            args.use_wikisource_rerank
            and not runtime_state.get("wikisource_disabled", False)
            and (
                args.wikisource_rerank_scope == "all"
                or item["query_id"] == "global"
            )
        )

        if should_use_wikisource:
            try:
                if args.verbose:
                    print(
                        f"{doc_id}/{item['query_id']}: "
                        f"Wikisource rerank top_k={args.wikisource_rerank_top_k}, "
                        f"scope={args.wikisource_rerank_scope}"
                    )

                ranked_candidates, pair_matrix = score_anchor_candidates(
                    keywords=item["keywords"],
                    trans_field="trans",
                    min_han_len=2,
                    allowed_pos=allowed_pos,
                    min_pair_hit=1,
                    top_k=args.wikisource_rerank_top_k,
                    sleep_sec=random_sleep_seconds(REQUEST_SLEEP_RANGE),
                )

                item["pair_matrix"] = [
                    {"word1": k[0], "word2": k[1], "hits": v}
                    for k, v in pair_matrix.items()
                ]
                item["rerank_source"] = "wikisource"

                time.sleep(random_sleep_seconds(GROUP_SLEEP_RANGE))

            except WikisourceRateLimitError as e:
                stats = get_wikisource_stats()

                print(f"[Wikisource disabled] {e}")
                print(
                    "[Wikisource stats] "
                    f"requests={stats['requests']}, "
                    f"cache_hits={stats['cache_hits']}, "
                    f"rate_limits={stats['rate_limits']}"
                )
                print("[Wikisource disabled] Fallback to semantic ranking for the rest of this run.")

                runtime_state["wikisource_disabled"] = True

                ranked_candidates = candidates
                item["pair_matrix"] = []
                item["rerank_source"] = "semantic_fallback"
                item["rerank_error"] = "wikisource_rate_limited"

        else:
            ranked_candidates = candidates
            item["pair_matrix"] = []

            if runtime_state.get("wikisource_disabled", False):
                item["rerank_source"] = "semantic_fallback_wikisource_disabled"
            elif args.use_wikisource_rerank:
                item["rerank_source"] = "semantic_fallback_scope_skipped"
            else:
                item["rerank_source"] = "semantic"

        # Final query build: always run, no matter where ranked_candidates came from.
        if should_use_wikisource and item.get("rerank_source") == "wikisource":
            query_terms = select_query_terms(
                ranked_candidates=ranked_candidates,
                num_query_terms=args.num_query_terms,
                allow_semantic_fallback=args.allow_semantic_fallback_terms,
            )
        else:
            query_terms = ranked_candidates[:args.num_query_terms]

        item["query_terms"] = query_terms

        exact_items, loose_items = split_exact_and_loose_terms(
            query_terms=query_terms,
            num_exact_anchors=args.num_exact_anchors,
            preferred_exact_pos=preferred_exact_pos,
        )

        item["exact_terms"] = exact_items
        item["loose_terms"] = [] if args.anchor_only else loose_items
        item["dropped_loose_terms"] = loose_items if args.anchor_only else []
        item["query_plan"] = {
            "mode": "anchor_only" if args.anchor_only else "mixed",
            "anchor_only": args.anchor_only,
            "num_exact_anchors": args.num_exact_anchors,
            "num_query_terms": args.num_query_terms,
            "exact_han_terms": [
                str(x.get("han_word", "") or x.get("trans", "")).strip()
                for x in exact_items
                if str(x.get("han_word", "") or x.get("trans", "")).strip()
            ],
            "loose_han_terms": [] if args.anchor_only else [
                str(x.get("han_word", "") or x.get("trans", "")).strip()
                for x in loose_items
                if str(x.get("han_word", "") or x.get("trans", "")).strip()
            ],
            "dropped_loose_han_terms": [
                str(x.get("han_word", "") or x.get("trans", "")).strip()
                for x in loose_items
                if args.anchor_only and str(x.get("han_word", "") or x.get("trans", "")).strip()
            ],
        }

        item["query"] = build_mixed_query(
            query_terms=query_terms,
            num_exact_anchors=args.num_exact_anchors,
            sites=sites,
            preferred_exact_pos=preferred_exact_pos,
            anchor_only=args.anchor_only,
        )

        if args.verbose:
            print(
                f"{doc_id}/{item['query_id']}: "
                f"{len(candidates)} translated candidates, "
                f"{len(query_terms)} query terms, "
                f"rerank_source={item['rerank_source']}"
            )

    # =========================================================================
    # 5. SAVE
    # =========================================================================
    output = {
        "doc_id": doc_id,
        "source_keyword_path": str(keyword_path),
        "num_chunks": num_chunks,
        "use_local_query": args.use_local_query,
        "num_local_query": num_local_query,
        "num_query_groups": len(results),
        "min_total_query": args.min_total_query,
        "use_wikisource_rerank": args.use_wikisource_rerank,
        "wikisource_rerank_scope": args.wikisource_rerank_scope,
        "wikisource_rerank_top_k": args.wikisource_rerank_top_k,
        "allow_semantic_fallback_terms": args.allow_semantic_fallback_terms,
        "anchor_only": args.anchor_only,
        "wikisource_disabled": runtime_state.get("wikisource_disabled", False),
        "wikisource_auth": get_wikisource_auth_status() if args.use_wikisource_rerank else None,
        "use_site_restriction": args.use_site_restriction,
        "results": results,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if args.verbose:
        print(f"Saved query file to: {output_path}")


def main():
    load_project_env()

    parser = argparse.ArgumentParser(description="Build search queries from keyword JSON")

    # I/O config
    parser.add_argument("--keyword_path", type=str, default=None, help="Path to one keyword JSON")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing keyword JSON files")
    parser.add_argument("--output_path", type=str, default=None, help="Path to one output query JSON")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save output query JSON files")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search JSON files in --input_dir",
    )

    # Translation cache
    parser.add_argument("--src_lang", type=str, default="vi")
    parser.add_argument("--tgt_lang", type=str, default="zh")
    parser.add_argument("--translation_cache_dir", type=str, default="./cache/translation")

    # Keyword aggregation
    parser.add_argument(
        "--top_n_keywords",
        type=int,
        default=15,
        help="Number of aggregated keywords kept as the initial candidate pool for each query group",
    )
    parser.add_argument(
        "--rarity_bias",
        type=float,
        default=0.2,
        help="Aggregation config: higher value favors rarer keywords across chunks",
    )

    # Query planning
    parser.add_argument(
        "--min_total_query",
        type=int,
        default=4,
        help="Minimum number of query groups per document, including global query",
    )
    parser.add_argument(
        "--use_local_query",
        action="store_true",
        help=(
            "Enable local query groups built from chunk ranges. "
            "If not set, local groups are disabled and global_variant_* "
            "queries are generated from the global keyword pool instead."
        ),
    )
    parser.add_argument(
        "--num_query_terms",
        type=int,
        default=6,
        help="Number of translated keywords used in each final query",
    )

    parser.add_argument(
        "--num_exact_anchors",
        type=int,
        default=3,
        help="Number of top query terms quoted as exact anchors",
    )
    parser.add_argument(
        "--anchor_only",
        action="store_true",
        help="Build queries using only exact quoted anchors; omit all loose terms.",
    )

    parser.add_argument(
        "--preferred_exact_pos",
        nargs="*",
        default=["Np", "N"],
        help="POS tags preferred for exact quoted anchors in the final query",
    )

    # Translation config
    # gemini
    parser.add_argument(
        "--translate_gemini",
        action="store_true",
        help="Use Gemini to translate terms missing from the translation cache",
    )

    parser.add_argument(
        "--gemini_model_name",
        type=str,
        default=get_env("GEMINI_MODEL_NAME", "models/gemini-2.5-pro"),
        help="Gemini model used only when --translate_gemini is set",
    )

    parser.add_argument(
        "--translation_batch_size",
        type=int,
        default=50,
        help="Number of missing terms translated per Gemini request",
    )

    # Optional reranking / restriction
    parser.add_argument(
        "--use_wikisource_rerank",
        action="store_true",
        help="Use Wikisource co-occurrence reranking for anchor selection",
    )
    parser.add_argument(
        "--wikisource_rerank_top_k",
        type=int,
        default=8,
        help="Only the top K eligible translated keywords are sent to Wikisource reranking",
    )

    parser.add_argument(
        "--allowed_pos",
        nargs="*",
        default=["Np", "N"],
        help="POS tags used only when --use_wikisource_rerank is enabled",
    )
    parser.add_argument(
        "--wikisource_rerank_scope",
        type=str,
        default="global",
        choices=["global", "all"],
        help="Which query groups use Wikisource reranking",
    )
    parser.add_argument(
        "--allow_semantic_fallback_terms",
        action="store_true",
        help=(
            "When Wikisource rerank is used and supported terms exist, "
            "allow unsupported semantic terms to fill remaining query slots."
        ),
    )
    parser.add_argument(
        "--use_site_restriction",
        action="store_true",
        help="Add site-restricted queries using --sites",
    )
    parser.add_argument(
        "--sites",
        nargs="*",
        default=["zh.wikisource.org", "ctext.org"],
        help="Domain-restricted search sites; only used when --use_site_restriction is set",
    )

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if bool(args.keyword_path) == bool(args.input_dir):
        raise ValueError("Provide exactly one of --keyword_path or --input_dir")

    if args.keyword_path and not args.output_path:
        raise ValueError("--output_path is required when using --keyword_path")

    if args.input_dir and not args.output_dir:
        raise ValueError("--output_dir is required when using --input_dir")

    translation_backend = "gemini" if args.translate_gemini else "cache"

    translator = create_translator(
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        backend=translation_backend,
        cache_dir=args.translation_cache_dir,
        batch_size=args.translation_batch_size,
        verbose=args.verbose,
        gemini_model_name=args.gemini_model_name,
    )

    allowed_pos = set(args.allowed_pos) if args.allowed_pos else None
    preferred_exact_pos = set(args.preferred_exact_pos) if args.preferred_exact_pos else None
    sites = args.sites if args.use_site_restriction else []
    runtime_state = {
        "wikisource_disabled": False,
    }

    if args.use_wikisource_rerank:
        init_wikisource_session(verbose=args.verbose)
        if args.verbose:
            auth_status = get_wikisource_auth_status()
            mode = "bot" if auth_status["logged_in"] else "anonymous"
            print(f"[Wikisource auth] mode={mode}, user={auth_status['username']}")
    if args.keyword_path:
        build_queries_for_keyword_file(
            keyword_path=Path(args.keyword_path),
            output_path=Path(args.output_path),
            args=args,
            translator=translator,
            allowed_pos=allowed_pos,
            preferred_exact_pos=preferred_exact_pos,
            sites=sites,
            runtime_state=runtime_state,
        )
    else:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.recursive:
            keyword_files = sorted(input_dir.rglob("*.json"))
        else:
            keyword_files = sorted(input_dir.glob("*.json"))

        if not keyword_files:
            raise FileNotFoundError(f"No JSON files found in {input_dir}")

        if args.verbose:
            print(f"Found {len(keyword_files)} keyword JSON files")

        for idx, keyword_file in enumerate(keyword_files, start=1):
            rel_path = keyword_file.relative_to(input_dir)
            out_path = output_dir / rel_path.with_suffix(".json")

            if args.verbose:
                print(f"\n[{idx}/{len(keyword_files)}] {keyword_file} -> {out_path}")

            build_queries_for_keyword_file(
                keyword_path=keyword_file,
                output_path=out_path,
                args=args,
                translator=translator,
                allowed_pos=allowed_pos,
                preferred_exact_pos=preferred_exact_pos,
                sites=sites,
                runtime_state=runtime_state,
            )


if __name__ == "__main__":
    main()