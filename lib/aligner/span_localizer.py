"""
Parent-window span localization for cross-lingual document pairs.

After the document-level pipeline selects candidate (source, target) document
pairs, this module finds, for each pair, the single continuous interval in the
*parent* document whose Bimax score against the whole *child* document is
highest.

This step intentionally does NOT do sentence alignment, monotonic alignment, or
1:N / N:1 matching -- those are handled later by Vecalign. Here we only output
the best ``[start_chunk, end_chunk]`` window in the parent.

Method per document pair:
  1. Two directional document-level Bimax scores decide which side is the child
     (the side more fully covered by the other) and which is the parent.
  2. The child is kept whole.
  3. Continuous candidate windows are generated in the parent, with lengths
     scaled to the child length (parent may be longer due to translation,
     annotation, or commentary).
  4. Each window is scored with the existing Bimax (``aggregation="avg"``,
     ``trim_ratio=1.0``): the mean of (child->window) and (window->child)
     best-match coverages. ``max`` is used only inside each direction to find a
     chunk's best counterpart; it is never used to pick the span.
  5. A coarse-to-fine search (coarse scan with large stride over several window
     lengths -> keep best candidates -> refine start/end) returns the highest
     scoring window.

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


# =============================================================================
# Window generation + coarse-to-fine search
# =============================================================================

def generate_window_lengths(
    child_len: int,
    parent_len: int,
    multipliers: Sequence[float],
) -> List[int]:
    """
    Candidate window lengths (in chunks), scaled to the child length and clipped
    to [1, parent_len]. Deduplicated and sorted.
    """
    lengths = set()
    for m in multipliers:
        L = int(round(float(m) * child_len))
        L = max(1, min(L, parent_len))
        lengths.add(L)
    return sorted(lengths)


def _coarse_scan(
    child_t,
    parent_t,
    lengths: Sequence[int],
    stride_ratio: float,
    num_candidates: int,
    normalize: bool,
) -> List[Tuple[float, int, int]]:
    """
    Coarse scan: for each window length, slide a window with a large stride and
    score it. Returns the top ``num_candidates`` as (score, start, end_exclusive).
    """
    P = int(parent_t.shape[0])
    cands: List[Tuple[float, int, int]] = []
    seen = set()
    for L in lengths:
        stride = max(1, int(round(L * stride_ratio)))
        starts = list(range(0, P - L + 1, stride))
        if not starts:
            starts = [0]
        last = P - L
        if last >= 0 and last not in starts:
            starts.append(last)  # always include the trailing window
        for s in starts:
            e = s + L
            if (s, e) in seen:
                continue
            seen.add((s, e))
            a, b = _score_window(child_t, parent_t, s, e, normalize)
            cands.append((0.5 * (a + b), s, e))
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[:num_candidates]


def _refine(
    child_t,
    parent_t,
    candidates: Sequence[Tuple[float, int, int]],
    radius: int,
    min_window: int,
    normalize: bool,
) -> Optional[Tuple[float, int, int, float, float]]:
    """
    Fine search: around each coarse candidate, vary start and end by +/- radius
    (stride 1) and re-score. Returns the global best as
    (score, start, end_exclusive, child_to_window, window_to_child).
    """
    P = int(parent_t.shape[0])
    best: Optional[Tuple[float, int, int, float, float]] = None
    seen = set()
    for _, s0, e0 in candidates:
        for s in range(s0 - radius, s0 + radius + 1):
            for e in range(e0 - radius, e0 + radius + 1):
                if s < 0 or e > P or (e - s) < min_window:
                    continue
                if (s, e) in seen:
                    continue
                seen.add((s, e))
                a, b = _score_window(child_t, parent_t, s, e, normalize)
                score = 0.5 * (a + b)
                if best is None or score > best[0]:
                    best = (score, s, e, a, b)
    return best


def localize_best_window(
    child_embs: np.ndarray,
    parent_embs: np.ndarray,
    *,
    window_multipliers: Sequence[float],
    coarse_stride_ratio: float = 0.5,
    num_candidates: int = 5,
    refine_radius: int = 3,
    min_window: int = 1,
    normalize: bool = True,
    device: str = "cpu",
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Find the continuous parent window maximizing Bimax(child, window) (avg).

    Returns a dict {start, end (exclusive), span_score, child_to_parent,
    parent_to_child} or None when either document is empty.
    """
    import torch

    C = int(child_embs.shape[0]) if child_embs.ndim == 2 else 0
    P = int(parent_embs.shape[0]) if parent_embs.ndim == 2 else 0
    if C == 0 or P == 0:
        return None

    child_t = torch.from_numpy(np.ascontiguousarray(child_embs)).float().to(device)
    parent_t = torch.from_numpy(np.ascontiguousarray(parent_embs)).float().to(device)

    lengths = generate_window_lengths(C, P, window_multipliers)
    if not lengths:
        return None

    coarse = _coarse_scan(child_t, parent_t, lengths, coarse_stride_ratio, num_candidates, normalize)
    if verbose:
        print(f"    [window] child={C} parent={P} lengths={lengths} coarse_top={len(coarse)}")
    if not coarse:
        return None

    best = _refine(child_t, parent_t, coarse, refine_radius, min_window, normalize)
    if best is None:
        return None

    score, s, e, a, b = best
    if verbose:
        print(f"    [window] best=[{s}..{e - 1}] score={score:.4f} c2p={a:.3f} p2c={b:.3f}")
    return {
        "start": s,
        "end": e,  # exclusive
        "span_score": float(score),
        "child_to_parent": float(a),
        "parent_to_child": float(b),
    }


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


def localize_document_pair(
    src_embs: np.ndarray,
    tar_embs: np.ndarray,
    *,
    src_meta: Optional[Sequence[Dict[str, Any]]] = None,
    tar_meta: Optional[Sequence[Dict[str, Any]]] = None,
    window_multipliers: Sequence[float] = (0.8, 1.0, 1.25, 1.5, 2.0),
    coarse_stride_ratio: float = 0.5,
    num_candidates: int = 5,
    refine_radius: int = 3,
    min_window: int = 1,
    normalize: bool = True,
    context_chunks: int = 1,
    device: str = "cpu",
    verbose: bool = False,
) -> Optional[LocalizedSpan]:
    """
    Localize the best parent window for one (source, target) document pair.

    Directional document-level Bimax picks the child (more fully covered) and the
    parent. The child is kept whole; the best continuous window in the parent is
    found by coarse-to-fine Bimax search.

    After the score window is determined, ``context_chunks`` chunks are added to
    each side of the parent window to produce the export window (for Vecalign).
    Scores are never recomputed after padding. The child side is never padded.
    """
    n_src = int(src_embs.shape[0]) if src_embs.ndim == 2 else 0
    n_tar = int(tar_embs.shape[0]) if tar_embs.ndim == 2 else 0
    if n_src == 0 or n_tar == 0:
        return None

    # 1. Directional document-level coverage -> child / parent.
    cover_src_by_tar, cover_tar_by_src = _directional_scores(
        src_embs, tar_embs, normalize=normalize, device=device
    )
    src_is_child = cover_src_by_tar >= cover_tar_by_src
    if src_is_child:
        child_embs, parent_embs, child_side = src_embs, tar_embs, "src"
    else:
        child_embs, parent_embs, child_side = tar_embs, src_embs, "tar"

    if verbose:
        print(
            f"    [pair] cover(src|tar)={cover_src_by_tar:.3f} "
            f"cover(tar|src)={cover_tar_by_src:.3f} -> child={child_side}"
        )

    # 2/3/4. Best continuous window in the parent.
    win = localize_best_window(
        child_embs, parent_embs,
        window_multipliers=window_multipliers,
        coarse_stride_ratio=coarse_stride_ratio,
        num_candidates=num_candidates,
        refine_radius=refine_radius,
        min_window=min_window,
        normalize=normalize,
        device=device,
        verbose=verbose,
    )
    if win is None:
        return None

    child_len = int(child_embs.shape[0])
    parent_len = int(parent_embs.shape[0])
    win_start, win_end_excl = win["start"], win["end"]

    # 5a. Score span (raw Bimax winner, parent-local coordinates).
    score_start = win_start
    score_end_excl = win_end_excl

    # 5b. Export span: pad score span on both ends (parent only; child stays whole).
    export_start = max(0, score_start - context_chunks)
    export_end = min(parent_len, score_end_excl + context_chunks)

    # 5c. Remap to source/target chunk coordinates.
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

    return LocalizedSpan(
        source_start_chunk=src_start_chunk,
        source_end_chunk=src_end_chunk,
        target_start_chunk=tar_start_chunk,
        target_end_chunk=tar_end_chunk,
        src_score_start_chunk=src_score_s,
        src_score_end_chunk=src_score_e,
        tar_score_start_chunk=tar_score_s,
        tar_score_end_chunk=tar_score_e,
        span_context_chunks=context_chunks,
        span_score=win["span_score"],
        child_side=child_side,
        child_to_parent=win["child_to_parent"],
        parent_to_child=win["parent_to_child"],
        source_start_char=src_start_char,
        source_end_char=src_end_char,
        target_start_char=tar_start_char,
        target_end_char=tar_end_char,
    )


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
    target candidates. No FAISS index is needed: Bimax is computed directly
    between the candidate document embeddings.

    Returns (rows, debug_records).
    """
    from lib.utils import AlignerIO, get_filename_only

    src_emb_dir = src_emb_path / "embeddings"
    trg_emb_dir = trg_emb_path / "embeddings"
    src_cm_dir = src_emb_path / "chunk_metadata"
    trg_cm_dir = trg_emb_path / "chunk_metadata"

    verbose = getattr(args, "verbose", False)
    save_debug = getattr(args, "save_span_debug", False)
    device = getattr(args, "device", "cpu") or "cpu"
    win_mult = getattr(args, "span_window_multipliers", (0.8, 1.0, 1.25, 1.5, 2.0))

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

            span = localize_document_pair(
                src_embs, tar_embs,
                src_meta=src_records,
                tar_meta=tar_records,
                window_multipliers=win_mult,
                coarse_stride_ratio=args.span_coarse_stride,
                num_candidates=args.span_num_candidates,
                refine_radius=args.span_refine_radius,
                min_window=args.span_min_window,
                normalize=True,
                context_chunks=getattr(args, "span_context_chunks", 1),
                device=device,
                verbose=verbose,
            )
            if span is None:
                continue

            t_path = AlignerIO.get_path_by_idx(trg_meta_df, t_idx)
            t_name = get_filename_only(t_path) if t_path else str(t_idx)

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
                    "child_side": span.child_side,
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

    # Window search hyperparameters
    parser.add_argument("--span_window_multipliers", type=float, nargs="+",
                        default=[0.8, 1.0, 1.25, 1.5, 2.0],
                        help="Window lengths as multipliers of the child length "
                             "(parent may be longer due to translation/commentary).")
    parser.add_argument("--span_coarse_stride", type=float, default=0.5,
                        help="Coarse scan stride as a fraction of the window length.")
    parser.add_argument("--span_num_candidates", type=int, default=5,
                        help="Number of best coarse windows kept for refinement.")
    parser.add_argument("--span_refine_radius", type=int, default=3,
                        help="Refine start/end by +/- this many chunks (stride 1).")
    parser.add_argument("--span_min_window", type=int, default=1,
                        help="Minimum window length in chunks.")
    parser.add_argument("--span_context_chunks", type=int, default=1,
                        help="Chunks to pad on each side of the score window for the "
                             "export span passed to Vecalign. Scores are not recomputed "
                             "after padding. Set 0 to disable padding.")
    parser.add_argument("--device", type=str, default=None,
                        help="torch device for Bimax (default: cuda if available else cpu).")

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
