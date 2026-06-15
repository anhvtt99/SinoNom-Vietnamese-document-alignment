"""
Local-retrieval parent-window span localization for cross-lingual document pairs.

After the document-level pipeline selects candidate (source, target) document
pairs, this module finds, for each pair, the continuous interval in the *parent*
document that best matches the whole *child* document.

This step intentionally does NOT do sentence alignment, monotonic alignment, or
1:N / N:1 matching -- those are handled later by Vecalign. Here we only output
the best ``[start_chunk, end_chunk]`` window in the parent.

Method per document pair:
  1. Two directional document-level Bimax scores decide which side is the child
     (the side more fully covered by the other) and which is the parent.
  2. Local retrieval between the whole child and the current parent only: cosine
     similarities ``child @ parent.T`` (no corpus-level index, no querying the
     whole corpus and filtering by document).
  3. For each child chunk, keep its top-k parent chunk hits.
  4. Generate candidate parent windows whose ends are hit positions.
  5. Rank candidate windows preliminarily by hit coverage (how many distinct
     child chunks have a hit inside the window), keeping the top ones.
  6. Score the kept windows with the symmetric mean Bimax
     (``0.5 * (child->window + window->child)``, ``trim_ratio=1.0``) and pick the
     highest -- ``max`` is used only inside each direction to find a chunk's best
     counterpart, never to pick the span.
  7. Pad the winning window by ``span_context_chunks`` on the parent side for
     export (Vecalign); scores are never recomputed after padding.

Local retrieval only proposes candidate positions; the two-directional Bimax mean
is the only score used to choose the final window. No quantiles, clustering,
RANSAC, Smith-Waterman, or DP.

Chunk ranges are mapped to character offsets and raw text is exported as before.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Dataclass
# =============================================================================

@dataclass
class LocalizedSpan:
    """
    The best parent window for a (source, target) document pair.

    The child document is whole; the parent document is the localized window.
    ``child_side`` records which of source/target is the child.

    Score span: the raw Bimax-winning window; all scores are computed on this.
    Export span: score span padded by ``span_context_chunks`` on both ends (parent
    side only). ``source_start_chunk``/``target_start_chunk`` hold the export span;
    the score span is preserved in the ``*_score_*`` fields for debug.
    """
    # Export span (padded) — written to TSV and passed to Vecalign
    source_start_chunk: int
    source_end_chunk: int
    target_start_chunk: int
    target_end_chunk: int
    # Score span (raw Bimax winner, before context padding)
    src_score_start_chunk: int
    src_score_end_chunk: int
    tar_score_start_chunk: int
    tar_score_end_chunk: int
    span_context_chunks: int     # chunks padded on each side of the parent window
    # Bimax scores (computed on score span, never recomputed after padding)
    span_score: float            # 0.5 * (child_to_parent + parent_to_child)
    child_side: str              # "src" or "tar"
    child_to_parent: float       # mean best-match of child chunks into the window
    parent_to_child: float       # mean best-match of window chunks into the child
    # Character offsets for the export span
    source_start_char: int = -1
    source_end_char: int = -1
    target_start_char: int = -1
    target_end_char: int = -1


# =============================================================================
# Bimax scoring helpers (delegate to the existing retrieval functions)
# =============================================================================

def _directional_scores(
    embs_a: np.ndarray,
    embs_b: np.ndarray,
    *,
    normalize: bool,
    device: str,
) -> Tuple[float, float]:
    """
    Directional Bimax coverage between two embedding matrices using the existing
    ``compute_bidirectional_scores`` (trim_ratio=1.0, no aggregation).

    Returns (cover_a_by_b, cover_b_by_a):
      - cover_a_by_b: mean over a-chunks of their best match in b.
      - cover_b_by_a: mean over b-chunks of their best match in a.
    """
    import torch
    from lib.aligner.retrieval import compute_bidirectional_scores

    A = torch.from_numpy(np.ascontiguousarray(embs_a)).float().to(device)
    B = torch.from_numpy(np.ascontiguousarray(embs_b)).float().to(device)
    s_ab, s_ba = compute_bidirectional_scores(A, B, normalize=normalize, trim_ratio=1.0)
    return float(s_ab), float(s_ba)


def _score_window(child_t, parent_t, start: int, end: int, normalize: bool) -> Tuple[float, float]:
    """
    Bimax(child, parent[start:end]) directional components, on pre-built tensors.

    Returns (child_to_window, window_to_child). The span score is their average.
    """
    from lib.aligner.retrieval import compute_bidirectional_scores
    a, b = compute_bidirectional_scores(
        child_t, parent_t[start:end], normalize=normalize, trim_ratio=1.0
    )
    return float(a), float(b)


def score_candidates(
    child_t,
    parent_t,
    candidates: Sequence[Tuple[int, int]],
    normalize: bool,
) -> List[Tuple[float, int, int, float, float]]:
    """
    Score each candidate window with the symmetric mean Bimax and sort descending.

    Returns a list of (score, start, end_exclusive, child_to_window,
    window_to_child); the first element is the best window.
    """
    scored: List[Tuple[float, int, int, float, float]] = []
    for s, e in candidates:
        a, b = _score_window(child_t, parent_t, s, e, normalize)
        scored.append((0.5 * (a + b), s, e, a, b))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# =============================================================================
# Local retrieval (child vs current parent only — no corpus index)
# =============================================================================

def local_retrieval_hits(
    child_t,
    parent_t,
    *,
    top_k: int,
    min_similarity: Optional[float],
    normalize: bool,
) -> List[Tuple[int, int, float]]:
    """
    For each child chunk, its top-k most similar parent chunks, computed directly
    between this child and this parent (a single matmul; no FAISS, no corpus).

    Returns a list of (child_chunk_idx, parent_chunk_idx, similarity).
    """
    import torch
    import torch.nn.functional as F

    if child_t.shape[0] == 0 or parent_t.shape[0] == 0:
        return []

    C = F.normalize(child_t, p=2, dim=1) if normalize else child_t
    P = F.normalize(parent_t, p=2, dim=1) if normalize else parent_t

    sims = C @ P.T  # [child_len, parent_len]
    k = min(top_k, int(P.shape[0]))
    vals, idx = torch.topk(sims, k, dim=1)
    vals = vals.detach().cpu().numpy()
    idx = idx.detach().cpu().numpy()

    hits: List[Tuple[int, int, float]] = []
    for ci in range(idx.shape[0]):
        for j in range(k):
            sim = float(vals[ci, j])
            if min_similarity is not None and sim < min_similarity:
                continue
            hits.append((ci, int(idx[ci, j]), sim))
    return hits


# =============================================================================
# Candidate windows + hit-coverage ranking
# =============================================================================

def _window_coverage(pos_children: Dict[int, set], start: int, end: int) -> int:
    """Distinct child chunks that have a hit at a parent position in [start, end)."""
    seen: set = set()
    for p, cs in pos_children.items():
        if start <= p < end:
            seen |= cs
    return len(seen)


def propose_and_rank_windows(
    hits: Sequence[Tuple[int, int, float]],
    parent_len: int,
    *,
    min_len: int,
    max_len: int,
    top_windows: int,
) -> List[Tuple[int, int, int]]:
    """
    Generate candidate parent windows whose ends are hit positions, rank them by
    hit coverage (distinct child chunks with a hit inside), and keep the best.

    A sliding pair of hit positions (a, b) forms the window [a, b+1) when its
    length is in [min_len, max_len]; coverage is accumulated incrementally as the
    right end extends. Ties are broken toward the shorter window.

    Returns up to ``top_windows`` items as (start, end_exclusive, coverage), best
    coverage first.
    """
    pos_children: Dict[int, set] = defaultdict(set)
    for ci, pp, _sim in hits:
        pos_children[pp].add(ci)
    pos = sorted(pos_children)
    n = len(pos)

    scored: List[Tuple[int, int, int, int]] = []  # (coverage, -length, start, end_excl)
    for i in range(n):
        a = pos[i]
        seen: set = set()
        for j in range(i, n):
            length = pos[j] - a + 1
            if length > max_len:
                break  # sorted: further j only longer
            seen |= pos_children[pos[j]]
            if length >= min_len:
                scored.append((len(seen), -length, a, pos[j] + 1))

    if not scored:
        # Guarantee at least one bounded window anchored on each hit position.
        L = min(min_len, parent_len)
        for a in pos:
            s = max(0, min(a, parent_len - L))
            e = s + L
            if s < e:
                scored.append((_window_coverage(pos_children, s, e), -(e - s), s, e))

    scored.sort(reverse=True)
    return [(s, e, cov) for (cov, _neg_len, s, e) in scored[:top_windows]]


# =============================================================================
# Document-pair localization
# =============================================================================

def _char_span(
    records: Optional[Sequence[Dict[str, Any]]],
    start_chunk: int,
    end_chunk: int,
) -> Tuple[int, int]:
    """Map an inclusive [start_chunk, end_chunk] range to a [char_start, char_end) span."""
    if not records or start_chunk < 0 or end_chunk >= len(records) or end_chunk < start_chunk:
        return -1, -1
    cs = records[start_chunk].get("char_start", -1)
    ce = records[end_chunk].get("char_end", -1)
    try:
        return int(cs), int(ce)
    except (TypeError, ValueError):
        return -1, -1


def _empty_debug() -> Dict[str, Any]:
    return {
        "child_side": "",
        "hit_count": 0,
        "child_coverage": 0.0,
        "raw_min_chunk": -1,
        "raw_max_chunk": -1,
        "candidate_window_count": 0,
        "best_coverage": -1,
        "best_score_start_chunk": -1,
        "best_score_end_chunk": -1,
        "best_window_length": 0,
        "reason": "",
    }


def localize_document_pair(
    src_embs: np.ndarray,
    tar_embs: np.ndarray,
    *,
    src_meta: Optional[Sequence[Dict[str, Any]]] = None,
    tar_meta: Optional[Sequence[Dict[str, Any]]] = None,
    top_k_hits: int = 3,
    hit_min_similarity: Optional[float] = None,
    min_child_length_ratio: float = 0.8,
    max_child_length_ratio: float = 3.0,
    top_windows: int = 10,
    context_chunks: int = 1,
    normalize: bool = True,
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[Optional[LocalizedSpan], Dict[str, Any]]:
    """
    Localize the best parent window for one (source, target) document pair.

    Directional document-level Bimax picks the child (more fully covered) and the
    parent. Local retrieval (child vs this parent only) proposes parent positions;
    candidate windows are formed from those positions, ranked by hit coverage, and
    the kept windows are scored with symmetric mean Bimax. The parent side is
    padded by ``context_chunks`` for export (scores untouched).

    Returns ``(LocalizedSpan, debug)`` on success, or ``(None, debug)`` when the
    pair is skipped (debug always carries a ``reason``).
    """
    import torch

    debug = _empty_debug()

    n_src = int(src_embs.shape[0]) if src_embs.ndim == 2 else 0
    n_tar = int(tar_embs.shape[0]) if tar_embs.ndim == 2 else 0
    if n_src == 0 or n_tar == 0:
        debug["reason"] = "empty_document"
        return None, debug

    # 1. Directional document-level coverage -> child / parent.
    cover_src_by_tar, cover_tar_by_src = _directional_scores(
        src_embs, tar_embs, normalize=normalize, device=device
    )
    src_is_child = cover_src_by_tar >= cover_tar_by_src
    if src_is_child:
        child_embs, parent_embs, child_side = src_embs, tar_embs, "src"
    else:
        child_embs, parent_embs, child_side = tar_embs, src_embs, "tar"
    debug["child_side"] = child_side

    child_len = int(child_embs.shape[0])
    parent_len = int(parent_embs.shape[0])

    child_t = torch.from_numpy(np.ascontiguousarray(child_embs)).float().to(device)
    parent_t = torch.from_numpy(np.ascontiguousarray(parent_embs)).float().to(device)

    if verbose:
        print(
            f"    [pair] cover(src|tar)={cover_src_by_tar:.3f} "
            f"cover(tar|src)={cover_tar_by_src:.3f} -> child={child_side}"
        )

    # 2/3. Local retrieval: top-k parent hits per child chunk.
    hits = local_retrieval_hits(
        child_t, parent_t, top_k=top_k_hits,
        min_similarity=hit_min_similarity, normalize=normalize,
    )
    if not hits:
        debug["reason"] = "no_hits"
        return None, debug

    positions = [p for (_, p, _) in hits]
    distinct_children = len({c for (c, _, _) in hits})
    debug["hit_count"] = len(hits)
    debug["child_coverage"] = round(distinct_children / max(1, child_len), 4)
    debug["raw_min_chunk"] = int(min(positions))
    debug["raw_max_chunk"] = int(max(positions))

    # 4/5/6. Candidate windows (length bounded to the child length) ranked by
    #        hit coverage; keep the top ones.
    min_len = max(1, int(round(child_len * min_child_length_ratio)))
    max_len = int(round(child_len * max_child_length_ratio))
    min_len = min(min_len, parent_len)
    max_len = max(min_len, min(max_len, parent_len))

    top = propose_and_rank_windows(
        hits, parent_len, min_len=min_len, max_len=max_len, top_windows=top_windows
    )
    debug["candidate_window_count"] = len(top)
    if not top:
        debug["reason"] = "no_valid_candidates"
        return None, debug

    # 7. Symmetric mean Bimax over the kept windows; pick the best.
    cand_windows = [(s, e) for (s, e, _cov) in top]
    scored = score_candidates(child_t, parent_t, cand_windows, normalize)
    best_score, best_s, best_e, c2w, w2c = scored[0]

    debug["best_coverage"] = next((cov for (s, e, cov) in top if s == best_s and e == best_e), -1)
    debug["best_score_start_chunk"] = best_s
    debug["best_score_end_chunk"] = best_e - 1
    debug["best_window_length"] = best_e - best_s
    debug["reason"] = "ok"

    if verbose:
        print(
            f"    [window] child={child_len} parent={parent_len} hits={len(hits)} "
            f"child_cov={distinct_children / max(1, child_len):.2f} "
            f"cands={len(top)} len_bounds=[{min_len},{max_len}]"
        )
        for sc, s, e, a, b in scored[:5]:
            print(f"      cand [{s}..{e - 1}] len={e - s} score={sc:.4f} c2p={a:.3f} p2c={b:.3f}")

    # 8. Score span -> export span (pad parent only; child stays whole).
    score_start, score_end_excl = best_s, best_e
    export_start = max(0, score_start - context_chunks)
    export_end = min(parent_len, score_end_excl + context_chunks)

    if src_is_child:
        # src = child (whole), tar = parent
        src_score_s, src_score_e = 0, child_len - 1
        tar_score_s, tar_score_e = score_start, score_end_excl - 1
        src_start_chunk, src_end_chunk = 0, child_len - 1
        tar_start_chunk, tar_end_chunk = export_start, export_end - 1
    else:
        # tar = child (whole), src = parent
        tar_score_s, tar_score_e = 0, child_len - 1
        src_score_s, src_score_e = score_start, score_end_excl - 1
        tar_start_chunk, tar_end_chunk = 0, child_len - 1
        src_start_chunk, src_end_chunk = export_start, export_end - 1

    src_start_char, src_end_char = _char_span(src_meta, src_start_chunk, src_end_chunk)
    tar_start_char, tar_end_char = _char_span(tar_meta, tar_start_chunk, tar_end_chunk)

    span = LocalizedSpan(
        source_start_chunk=src_start_chunk,
        source_end_chunk=src_end_chunk,
        target_start_chunk=tar_start_chunk,
        target_end_chunk=tar_end_chunk,
        src_score_start_chunk=src_score_s,
        src_score_end_chunk=src_score_e,
        tar_score_start_chunk=tar_score_s,
        tar_score_end_chunk=tar_score_e,
        span_context_chunks=context_chunks,
        span_score=float(best_score),
        child_side=child_side,
        child_to_parent=float(c2w),
        parent_to_child=float(w2c),
        source_start_char=src_start_char,
        source_end_char=src_end_char,
        target_start_char=tar_start_char,
        target_end_char=tar_end_char,
    )
    return span, debug


# =============================================================================
# Batch orchestration over alignment pairs (Step 2 of the pipeline)
# =============================================================================

def localize_spans_for_pairs(
    pairs: List[Tuple[int, int, float]],
    *,
    src_emb_path: Path,
    trg_emb_path: Path,
    src_meta_df,
    trg_meta_df,
    args,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    For each source document, localize the best parent window against its top
    target candidates.

    Retrieval is local per pair (child vs that parent only) -- no corpus FAISS
    index is built. Returns (rows, debug_records).
    """
    from lib.utils import AlignerIO, get_filename_only

    src_emb_dir = src_emb_path / "embeddings"
    trg_emb_dir = trg_emb_path / "embeddings"
    src_cm_dir = src_emb_path / "chunk_metadata"
    trg_cm_dir = trg_emb_path / "chunk_metadata"

    verbose = getattr(args, "verbose", False)
    save_debug = getattr(args, "save_span_debug", False)
    device = getattr(args, "device", "cpu") or "cpu"

    # Group candidate targets per source document, keep top-k by document score.
    by_src: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for s_idx, t_idx, score in pairs:
        by_src[s_idx].append((t_idx, score))

    rows: List[Dict[str, Any]] = []
    debug: List[Dict[str, Any]] = []
    n_localized = 0

    for s_idx, cands in by_src.items():
        cands.sort(key=lambda x: x[1], reverse=True)
        cands = cands[: args.localize_top_k_pairs]

        src_embs = AlignerIO.get_emb_by_idx(src_meta_df, src_emb_dir, s_idx)
        if src_embs is None or src_embs.shape[0] == 0:
            continue

        src_records = AlignerIO.get_chunk_metadata_by_idx(src_meta_df, src_cm_dir, s_idx) or None
        s_path = AlignerIO.get_path_by_idx(src_meta_df, s_idx)
        s_name = get_filename_only(s_path) if s_path else str(s_idx)

        for t_idx, doc_score in cands:
            tar_embs = AlignerIO.get_emb_by_idx(trg_meta_df, trg_emb_dir, t_idx)
            if tar_embs is None or tar_embs.shape[0] == 0:
                continue

            tar_records = AlignerIO.get_chunk_metadata_by_idx(trg_meta_df, trg_cm_dir, t_idx) or None
            t_path = AlignerIO.get_path_by_idx(trg_meta_df, t_idx)
            t_name = get_filename_only(t_path) if t_path else str(t_idx)

            span, fdbg = localize_document_pair(
                src_embs, tar_embs,
                src_meta=src_records,
                tar_meta=tar_records,
                top_k_hits=args.span_top_k_hits,
                hit_min_similarity=args.span_hit_min_similarity,
                min_child_length_ratio=args.span_min_child_length_ratio,
                max_child_length_ratio=args.span_max_child_length_ratio,
                top_windows=args.span_top_windows,
                context_chunks=getattr(args, "span_context_chunks", 1),
                normalize=True,
                device=device,
                verbose=verbose,
            )

            if span is None:
                if verbose:
                    print(f"    [skip] {s_name} -> {t_name}: {fdbg.get('reason')}")
                if save_debug:
                    debug.append({
                        "src_doc": s_name, "tar_doc": t_name,
                        "document_score": float(doc_score), **fdbg,
                    })
                continue

            rows.append({
                # TSV columns
                "src_doc": s_name,
                "tar_doc": t_name,
                "document_score": float(doc_score),
                "span_score": span.span_score,
                "child_side": span.child_side,
                "child_to_parent": span.child_to_parent,
                "parent_to_child": span.parent_to_child,
                "span_context_chunks": span.span_context_chunks,
                "src_total_chunks": int(src_embs.shape[0]),
                # score span (raw Bimax winner)
                "src_score_start_chunk": span.src_score_start_chunk,
                "src_score_end_chunk": span.src_score_end_chunk,
                # export span (score + padding; used by Vecalign)
                "src_start_chunk": span.source_start_chunk,
                "src_end_chunk": span.source_end_chunk,
                "tar_total_chunks": int(tar_embs.shape[0]),
                "tar_score_start_chunk": span.tar_score_start_chunk,
                "tar_score_end_chunk": span.tar_score_end_chunk,
                "tar_start_chunk": span.target_start_chunk,
                "tar_end_chunk": span.target_end_chunk,
                "src_start_char": span.source_start_char,
                "src_end_char": span.source_end_char,
                "tar_start_char": span.target_start_char,
                "tar_end_char": span.target_end_char,
                # Internal fields for span export (not written to TSV)
                "_src_file_path": s_path,
                "_tar_file_path": t_path,
            })
            n_localized += 1

            if save_debug:
                debug.append({
                    "src_doc": s_name,
                    "tar_doc": t_name,
                    "document_score": float(doc_score),
                    "span_score": span.span_score,
                    "child_to_parent": span.child_to_parent,
                    "parent_to_child": span.parent_to_child,
                    "span_context_chunks": span.span_context_chunks,
                    "score_boundaries": {
                        "src_chunks": [span.src_score_start_chunk, span.src_score_end_chunk],
                        "tar_chunks": [span.tar_score_start_chunk, span.tar_score_end_chunk],
                    },
                    "export_boundaries": {
                        "src_chunks": [span.source_start_chunk, span.source_end_chunk],
                        "tar_chunks": [span.target_start_chunk, span.target_end_chunk],
                        "src_chars": [span.source_start_char, span.source_end_char],
                        "tar_chars": [span.target_start_char, span.target_end_char],
                    },
                    **fdbg,
                })

    print(f"[*] Localized spans for {n_localized} document pair(s).")
    return rows, debug


# =============================================================================
# Span text export
# =============================================================================

_SPAN_COLS = [
    "src_doc", "tar_doc", "document_score", "span_score", "child_side",
    "child_to_parent", "parent_to_child", "span_context_chunks",
    "src_total_chunks",
    "src_score_start_chunk", "src_score_end_chunk",  # raw Bimax winner
    "src_start_chunk", "src_end_chunk",               # export (padded)
    "tar_total_chunks",
    "tar_score_start_chunk", "tar_score_end_chunk",
    "tar_start_chunk", "tar_end_chunk",
    "src_start_char", "src_end_char", "tar_start_char", "tar_end_char",
]


def _fmt_span_cell(value: Any) -> str:
    """Format a span TSV cell: 4-decimal floats, plain str otherwise."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _extract_span_text(
    file_path: Optional[str],
    char_start: int,
    char_end: int,
    label: str = "",
) -> str:
    """Load a file and slice [char_start:char_end]. Returns a fallback string on error."""
    if not file_path:
        return f"[{label}: file path unavailable]"
    if char_start < 0 or char_end < 0:
        return f"[{label}: char offsets not available — re-run generate_embeddings to produce chunk_metadata]"
    try:
        from lib.utils import AlignerIO
        raw = AlignerIO.load_document_text(file_path)
        span = raw[char_start:char_end]
        if not span.strip():
            return f"[{label}: empty span at {char_start}..{char_end}]"
        return span
    except Exception as e:
        return f"[{label}: could not read {file_path} — {e}]"


def export_span_files(
    rows: List[Dict[str, Any]],
    spans_dir: Path,
) -> int:
    """
    Extract and save text spans to a directory tree.

    Layout::

        spans_dir/
            {src_stem}/
                1_{tar_stem}.txt   # rank 1 by document_score (highest)
                2_{tar_stem}.txt
                ...

    Each ``.txt`` file contains the source span and target span separated by a
    divider, so you can read both sides at a glance. Rows without valid char
    offsets (``-1``) note "no char offset" for that side.

    Returns the number of files written.
    """
    by_src: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_src[r["src_doc"]].append(r)

    n_written = 0
    for src_name, group in by_src.items():
        group = sorted(group, key=lambda x: x["document_score"], reverse=True)

        src_stem = Path(src_name).stem
        src_dir = spans_dir / src_stem
        src_dir.mkdir(parents=True, exist_ok=True)

        for rank, r in enumerate(group, start=1):
            tar_stem = Path(r["tar_doc"]).stem
            out_file = src_dir / f"{rank}_{tar_stem}.txt"

            src_text = _extract_span_text(
                r.get("_src_file_path"), r["src_start_char"], r["src_end_char"], label="src"
            )
            tar_text = _extract_span_text(
                r.get("_tar_file_path"), r["tar_start_char"], r["tar_end_char"], label="tar"
            )

            # Score/export window on the parent side (child side needs no window info)
            if r["child_side"] == "src":
                score_w = f"[{r['tar_score_start_chunk']}..{r['tar_score_end_chunk']}]"
                export_w = f"[{r['tar_start_chunk']}..{r['tar_end_chunk']}]"
            else:
                score_w = f"[{r['src_score_start_chunk']}..{r['src_score_end_chunk']}]"
                export_w = f"[{r['src_start_chunk']}..{r['src_end_chunk']}]"

            header_line1 = (
                f"doc_score={r['document_score']:.4f}  "
                f"span_score={r['span_score']:.4f}  "
                f"child_side={r['child_side']}  "
                f"c2p={r['child_to_parent']:.3f}  p2c={r['parent_to_child']:.3f}"
            )
            header_line2 = (
                f"score_window={score_w}  "
                f"export_window={export_w}  "
                f"context_chunks={r['span_context_chunks']}"
            )
            divider = "─" * 60
            content = (
                f"{divider}\n"
                f"  {header_line1}\n"
                f"  {header_line2}\n"
                f"{divider}\n"
                f"SOURCE  {r['src_doc']}  "
                f"chunks [{r['src_start_chunk']}..{r['src_end_chunk']}]/{r['src_total_chunks']}  "
                f"chars [{r['src_start_char']}..{r['src_end_char']}]\n"
                f"{divider}\n"
                f"{src_text}\n"
                f"{divider}\n"
                f"TARGET  {r['tar_doc']}  "
                f"chunks [{r['tar_start_chunk']}..{r['tar_end_chunk']}]/{r['tar_total_chunks']}  "
                f"chars [{r['tar_start_char']}..{r['tar_end_char']}]\n"
                f"{divider}\n"
                f"{tar_text}\n"
            )
            out_file.write_text(content, encoding="utf-8")
            n_written += 1

    return n_written


# =============================================================================
# CLI (standalone Step 2): read alignment TSV -> write localized spans
# =============================================================================

def _build_config_tag(args) -> str:
    if args.split_mode == "sentence":
        return f"{args.split_mode}_n{args.num_of_sent}_o{args.overlap_sent}"
    return f"{args.split_mode}_s{args.chunk_size}_r{args.overlap_rate}"


def _setup_paths(args) -> Tuple[Path, Path, str]:
    base_path = Path(args.emb_base_path)
    config_tag = _build_config_tag(args)
    return base_path / config_tag / args.src_lang, base_path / config_tag / args.tar_lang, config_tag


def load_alignment_pairs(tsv_path: Path) -> List[Tuple[int, int, float]]:
    """
    Read (src_doc_idx, tar_doc_idx, score) triples from an alignment TSV produced
    by ``aligner.save_alignment_tsv``.
    """
    pairs: List[Tuple[int, int, float]] = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            si = header.index("src_doc_idx")
            ti = header.index("tar_doc_idx")
            sci = header.index("score")
        except ValueError:
            raise ValueError(
                f"Alignment TSV missing required columns "
                f"(src_doc_idx, tar_doc_idx, score): {tsv_path}"
            )
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            pairs.append((int(cols[si]), int(cols[ti]), float(cols[sci])))
    return pairs


def build_span_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="Localize the best parent window for document pairs produced by the aligner."
    )
    # Paths / config (must match the embedding + aligner run)
    parser.add_argument("--emb_base_path", type=str, required=True, help="Base path containing generated embeddings")
    parser.add_argument("--src_lang", type=str, default="vi")
    parser.add_argument("--tar_lang", type=str, default="zh")
    parser.add_argument("--split_mode", type=str, choices=["sentence", "chunk"], default="sentence")
    parser.add_argument("--num_of_sent", type=int, default=1)
    parser.add_argument("--overlap_sent", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--overlap_rate", type=float, default=0.5)

    # Input alignment + output
    parser.add_argument("--alignment_tsv", type=str, default=None,
                        help="Alignment TSV from the aligner. Defaults to "
                             "<output_path|cwd>/alignment_<config>.tsv")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Directory for localized_spans_<config>.tsv (default: cwd)")

    # Pair selection
    parser.add_argument("--localize_top_k_pairs", type=int, default=None,
                        help="Per source doc, localize at most this many top target candidates. "
                             "Default None: use all pairs from the alignment TSV (already bounded "
                             "by --top_k_pairs in the aligner step).")

    # Local retrieval + candidate windows
    parser.add_argument("--span_top_k_hits", type=int, default=3,
                        help="Top parent chunks kept per child chunk during local retrieval.")
    parser.add_argument("--span_hit_min_similarity", type=float, default=None,
                        help="Drop retrieval hits below this cosine similarity (default: no floor).")
    parser.add_argument("--span_min_child_length_ratio", type=float, default=0.8,
                        help="Minimum candidate window length as a fraction of the child length.")
    parser.add_argument("--span_max_child_length_ratio", type=float, default=3.0,
                        help="Maximum candidate window length as a fraction of the child length.")
    parser.add_argument("--span_top_windows", type=int, default=10,
                        help="Number of coverage-ranked candidate windows kept for Bimax scoring.")
    parser.add_argument("--span_context_chunks", type=int, default=1,
                        help="Chunks to pad on each side of the score window for the "
                             "export span passed to Vecalign. Scores are not recomputed "
                             "after padding. Set 0 to disable padding.")
    parser.add_argument("--device", type=str, default=None,
                        help="torch device for retrieval/Bimax (default: cuda if available else cpu).")

    parser.add_argument("--save_span_debug", action="store_true",
                        help="Dump per-pair span debug info to span_debug_<config>.jsonl.")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-pair logging.")

    # Span text export
    parser.add_argument("--spans_dir", type=str, default=None,
                        help="Directory to export extracted span text files. "
                             "Defaults to <output_path>/spans. Pass --no_export_spans to disable.")
    parser.add_argument("--no_export_spans", action="store_true",
                        help="Skip writing span text files (only write the TSV).")
    return parser


def main():
    import json
    import sys
    from lib.utils import AlignerIO, cuda_available

    args = build_span_parser().parse_args()
    if not args.device:
        args.device = "cuda" if cuda_available(verbose=False) else "cpu"

    src_emb_path, trg_emb_path, config_tag = _setup_paths(args)

    if not src_emb_path.exists() or not trg_emb_path.exists():
        print(f"[!] Embeddings directory not found for config '{config_tag}'.")
        if not src_emb_path.exists():
            print(f"    -> Missing Source: {src_emb_path}")
        if not trg_emb_path.exists():
            print(f"    -> Missing Target: {trg_emb_path}")
        sys.exit(1)

    out_dir = Path(args.output_path) if args.output_path else Path.cwd()
    align_tsv = Path(args.alignment_tsv) if args.alignment_tsv else (out_dir / f"alignment_{config_tag}.tsv")
    if not align_tsv.exists():
        print(f"[!] Alignment TSV not found: {align_tsv}")
        print("    Run `python -m lib.aligner.aligner ... --save_results` first.")
        sys.exit(1)

    pairs = load_alignment_pairs(align_tsv)
    print(f"[*] Loaded {len(pairs)} alignment pairs from {align_tsv}")
    print(f"[*] Device: {args.device}")

    src_meta_df = AlignerIO.load_metadata(src_emb_path / "metadata").set_index("doc_idx")
    trg_meta_df = AlignerIO.load_metadata(trg_emb_path / "metadata").set_index("doc_idx")

    rows, debug = localize_spans_for_pairs(
        pairs,
        src_emb_path=src_emb_path,
        trg_emb_path=trg_emb_path,
        src_meta_df=src_meta_df,
        trg_meta_df=trg_meta_df,
        args=args,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    span_out_path = out_dir / f"localized_spans_{config_tag}.tsv"
    with open(span_out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(_SPAN_COLS) + "\n")
        for r in rows:
            f.write("\t".join(_fmt_span_cell(r[c]) for c in _SPAN_COLS) + "\n")
    print(f"[+] Localized spans dumped to: {span_out_path} ({len(rows)} spans)")

    if getattr(args, "save_span_debug", False):
        debug_path = out_dir / f"span_debug_{config_tag}.jsonl"
        with open(debug_path, "w", encoding="utf-8") as f:
            for rec in debug:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[+] Span debug dumped to: {debug_path}")

    # --- EXPORT SPAN TEXT FILES ---
    if not getattr(args, "no_export_spans", False):
        spans_dir = Path(args.spans_dir) if getattr(args, "spans_dir", None) else (out_dir / "spans")
        n_written = export_span_files(rows, spans_dir)
        print(f"[+] Span text files written to: {spans_dir}/ ({n_written} files)")


if __name__ == "__main__":
    main()
