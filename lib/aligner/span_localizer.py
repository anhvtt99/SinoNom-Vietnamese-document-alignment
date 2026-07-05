"""
Anchor-chain parent-window span localization for cross-lingual document pairs.

After the document-level pipeline selects candidate (source, target) document
pairs, this module finds, for each pair, the continuous interval in the *parent*
document that best matches the whole *child* document.

This step intentionally does NOT do sentence alignment or 1:N / N:1 matching --
those are handled later by Vecalign. Here we only output the best
``[start_chunk, end_chunk]`` window in the parent.

Method per document pair:
  1. Two directional document-level Bimax scores decide which side is the child
     (the side more fully covered by the other) and which is the parent.
  2. Anchors: cosine similarities ``child @ parent.T``; for each child chunk keep
     its top-k parent chunk hits.
  3. Max-weight monotone chain (DP over anchors): the highest-total-weight chain
     of anchors with strictly increasing child index and non-decreasing parent
     index. Anchor weight is ``sim - anchor_base_sim`` so weak anchors cannot pay
     for stretching the chain, and parent jumps beyond ``slope_cap`` chunks per
     child step are charged ``gap_penalty`` per excess chunk -- a translation is
     monotone with a roughly bounded expansion ratio, so spurious hits (tables of
     contents, thematically similar neighbouring chapters, repeated names) fall
     off the chain instead of stretching the window.
  4. Window boundaries: the chain's parent extent, extrapolated to child chunk 0
     and child chunk C-1 using the local slope at each chain end (the head and
     tail of a document can expand at different rates than its bulk).
  5. Confidence: ``chain_coverage`` (fraction of child chunks on the chain) and
     ``chain_sim`` (mean anchor similarity along the chain). Pairs whose chain
     covers less than ``min_chain_coverage`` of the child are rejected -- this is
     what distinguishes "the parent actually contains this chapter" from "the
     parent is merely about the same topic".
  6. The winning window is scored with the symmetric mean Bimax for reporting,
     then padded by ``span_context_chunks`` on the parent side for export
     (Vecalign); scores are never recomputed after padding.

Unlike the previous coverage-window design, no candidate-window enumeration,
hit-coverage ranking, child-length ratio caps, or boundary extend/trim passes
are used: the monotone chain both locates the window and determines its length,
so the parent/child chunk-length ratio (which varies freely between Classical
Chinese and Vietnamese) never needs to be assumed.

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

    Score span: the chain-derived window; all scores are computed on this.
    Export span: score span padded by ``span_context_chunks`` on both ends (parent
    side only). ``source_start_chunk``/``target_start_chunk`` hold the export span;
    the score span is preserved in the ``*_score_*`` fields for debug.
    """
    # Export span (padded) — written to TSV and passed to Vecalign
    source_start_chunk: int
    source_end_chunk: int
    target_start_chunk: int
    target_end_chunk: int
    # Score span (chain window, before context padding)
    src_score_start_chunk: int
    src_score_end_chunk: int
    tar_score_start_chunk: int
    tar_score_end_chunk: int
    span_context_chunks: int     # chunks padded on each side of the parent window
    # Scores (computed on score span, never recomputed after padding)
    span_score: float            # 0.5 * (child_to_parent + parent_to_child)
    child_side: str              # "src" or "tar"
    child_to_parent: float       # mean best-match of child chunks into the window
    parent_to_child: float       # mean best-match of window chunks into the child
    chain_coverage: float        # fraction of child chunks on the anchor chain
    chain_sim: float             # mean anchor similarity along the chain
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


def _score_window(
    child_embs: np.ndarray,
    parent_embs: np.ndarray,
    start: int,
    end_excl: int,
    *,
    normalize: bool,
    device: str,
) -> Tuple[float, float]:
    """
    Bimax(child, parent[start:end]) directional components, for reporting.

    Returns (child_to_window, window_to_child). The span score is their average.
    """
    import torch
    from lib.aligner.retrieval import compute_bidirectional_scores

    C = torch.from_numpy(np.ascontiguousarray(child_embs)).float().to(device)
    W = torch.from_numpy(np.ascontiguousarray(parent_embs[start:end_excl])).float().to(device)
    a, b = compute_bidirectional_scores(C, W, normalize=normalize, trim_ratio=1.0)
    return float(a), float(b)


# =============================================================================
# Anchor construction + monotone chain DP
# =============================================================================

def build_anchors(
    child_embs: np.ndarray,
    parent_embs: np.ndarray,
    *,
    top_k: int,
    min_similarity: Optional[float],
    normalize: bool,
) -> List[Tuple[int, int, float]]:
    """
    For each child chunk, its top-k most similar parent chunks (a single matmul;
    no FAISS, no corpus index).

    Returns anchors as (child_chunk_idx, parent_chunk_idx, similarity), sorted by
    (child_idx, parent_idx).
    """
    nc = int(child_embs.shape[0])
    npar = int(parent_embs.shape[0])
    if nc == 0 or npar == 0:
        return []

    C = np.ascontiguousarray(child_embs, dtype=np.float32)
    P = np.ascontiguousarray(parent_embs, dtype=np.float32)
    if normalize:
        C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)

    sims = C @ P.T  # [child_len, parent_len]
    k = min(top_k, npar)
    idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]

    anchors: List[Tuple[int, int, float]] = []
    for ci in range(nc):
        for pj in idx[ci]:
            s = float(sims[ci, pj])
            if min_similarity is not None and s < min_similarity:
                continue
            anchors.append((ci, int(pj), s))
    anchors.sort(key=lambda a: (a[0], a[1]))
    return anchors


def best_monotone_chain(
    anchors: Sequence[Tuple[int, int, float]],
    *,
    anchor_base_sim: float,
    slope_cap: float,
    gap_penalty: float,
) -> List[Tuple[int, int, float]]:
    """
    Max-weight chain of anchors with strictly increasing child index and
    non-decreasing parent index.

    Weight per anchor is ``sim - anchor_base_sim`` (weak anchors are net-negative,
    so the chain does not stretch through low-similarity regions just to add
    length). A parent jump larger than ``slope_cap`` chunks per child step is
    charged ``gap_penalty`` per excess parent chunk, which stops the chain from
    teleporting across the parent to a lexically similar but unrelated region.

    O(n^2) DP with numpy-vectorized inner loop; n is at most top_k * child_len.
    Returns the winning chain in child order (possibly a single anchor).
    """
    n = len(anchors)
    if n == 0:
        return []

    cs = np.fromiter((a[0] for a in anchors), dtype=np.int64, count=n)
    ps = np.fromiter((a[1] for a in anchors), dtype=np.int64, count=n)
    w = np.fromiter((a[2] - anchor_base_sim for a in anchors), dtype=np.float64, count=n)

    best = w.copy()
    prev = np.full(n, -1, dtype=np.int64)
    for j in range(1, n):
        mask = (cs[:j] < cs[j]) & (ps[:j] <= ps[j])
        if not mask.any():
            continue
        dc = cs[j] - cs[:j]
        dp_ = ps[j] - ps[:j]
        excess = np.maximum(0.0, dp_ - slope_cap * dc)
        cand = best[:j] + w[j] - gap_penalty * excess
        cand[~mask] = -np.inf
        i = int(np.argmax(cand))
        if cand[i] > best[j]:
            best[j] = cand[i]
            prev[j] = i

    j = int(np.argmax(best))
    chain: List[Tuple[int, int, float]] = []
    while j != -1:
        chain.append(anchors[j])
        j = int(prev[j])
    chain.reverse()
    return chain


def chain_window_bounds(
    chain: Sequence[Tuple[int, int, float]],
    child_len: int,
    parent_len: int,
) -> Tuple[int, int]:
    """
    Parent window ``[start, end]`` (inclusive) for a chain: the chain's parent
    extent, extrapolated to child chunk 0 and child chunk C-1 using the local
    slope over up to one third of the chain at each end (a document's head and
    tail can expand at different rates than its bulk; a third keeps a single bad
    endpoint from tilting the estimate).
    """
    cs = [a[0] for a in chain]
    ps = [a[1] for a in chain]

    m = max(2, len(chain) // 3)

    def _slope(xs: Sequence[int], ys: Sequence[int]) -> float:
        if len(xs) < 2 or xs[-1] == xs[0]:
            return 1.5  # neutral expansion fallback
        r = float(np.polyfit(xs, ys, 1)[0])
        return min(4.0, max(0.3, r))

    r_head = _slope(cs[:m], ps[:m])
    r_tail = _slope(cs[-m:], ps[-m:])

    start = int(round(ps[0] - r_head * cs[0]))
    end = int(round(ps[-1] + r_tail * (child_len - 1 - cs[-1])))
    start = max(0, min(start, ps[0]))
    end = min(parent_len - 1, max(end, ps[-1]))
    return start, end


# =============================================================================
# Sentence-boundary snapping for exported char offsets
# =============================================================================

# Sentence terminators for both Vietnamese (Latin) and SinoNom/Chinese (CJK),
# plus newline. The chunk char offsets come from fixed token windows, so a span's
# raw [char_start, char_end) almost always lands mid-sentence; snapping to these
# boundaries makes the exported text read as whole sentences.
_SENT_END = set(".!?…。！？\n")
_SNAP_WS = set(" \t\n\r")


def _snap_span_to_sentence(
    text: str,
    start: int,
    end: int,
    *,
    max_extend: int = 400,
) -> Tuple[int, int]:
    """
    Expand a raw character span ``[start, end)`` outward to the nearest sentence
    boundaries, without crossing more than ``max_extend`` characters on either
    side.

    - start: if it lands inside a sentence, move back to just after the previous
      terminator so the span opens on a whole sentence.
    - end: if it lands inside a sentence, move forward to include the rest of the
      current sentence (through its terminator).

    Already-clean boundaries are left untouched. Returns the snapped ``(start, end)``;
    falls back to the original bounds when no terminator is found within range.
    """
    n = len(text)
    if not text or start < 0 or end <= start:
        return start, end
    start = max(0, min(start, n))
    end = max(0, min(end, n))

    # --- START: only move if currently mid-sentence ---
    k = start - 1
    while k >= 0 and text[k] in (" \t"):
        k -= 1
    if k >= 0 and text[k] not in _SENT_END:
        lo = max(0, start - max_extend)
        i = start - 1
        while i >= lo:
            if text[i] in _SENT_END:
                new_s = i + 1
                while new_s < start and text[new_s] in _SNAP_WS:
                    new_s += 1
                start = new_s
                break
            i -= 1

    # --- END: only move if currently mid-sentence ---
    j = end - 1
    while j >= start and text[j] in _SNAP_WS:
        j -= 1
    if j >= start and text[j] not in _SENT_END:
        hi = min(n, end + max_extend)
        k = end
        while k < hi:
            if text[k] in _SENT_END:
                end = k + 1
                break
            k += 1

    return start, end


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
        "anchor_count": 0,
        "chain_len": 0,
        "chain_coverage": 0.0,
        "chain_sim": 0.0,
        "chain_parent_start": -1,
        "chain_parent_end": -1,
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
    anchor_base_sim: float = 0.45,
    slope_cap: float = 6.0,
    gap_penalty: float = 0.05,
    min_chain_coverage: float = 0.35,
    context_chunks: int = 2,
    normalize: bool = True,
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[Optional[LocalizedSpan], Dict[str, Any]]:
    """
    Localize the best parent window for one (source, target) document pair.

    Directional document-level Bimax picks the child (more fully covered) and the
    parent. Top-k anchors per child chunk are chained by a monotone max-weight DP
    (see module docstring); the chain's parent extent, extrapolated to the child's
    ends, is the window. Pairs whose chain covers less than ``min_chain_coverage``
    of the child are rejected: a chain that anchors only a small fraction of the
    child means the parent does not actually contain this content, merely similar
    material.

    Returns ``(LocalizedSpan, debug)`` on success, or ``(None, debug)`` when the
    pair is skipped (debug always carries a ``reason``).
    """
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

    if verbose:
        print(
            f"    [pair] cover(src|tar)={cover_src_by_tar:.3f} "
            f"cover(tar|src)={cover_tar_by_src:.3f} -> child={child_side}"
        )

    # 2. Anchors: top-k parent hits per child chunk.
    anchors = build_anchors(
        child_embs, parent_embs, top_k=top_k_hits,
        min_similarity=hit_min_similarity, normalize=normalize,
    )
    debug["anchor_count"] = len(anchors)
    if not anchors:
        debug["reason"] = "no_anchors"
        return None, debug

    # 3. Max-weight monotone chain.
    chain = best_monotone_chain(
        anchors,
        anchor_base_sim=anchor_base_sim,
        slope_cap=slope_cap,
        gap_penalty=gap_penalty,
    )
    if not chain:
        debug["reason"] = "no_chain"
        return None, debug

    chain_children = {a[0] for a in chain}
    chain_cov = len(chain_children) / max(1, child_len)
    chain_sim = float(np.mean([a[2] for a in chain]))
    debug["chain_len"] = len(chain)
    debug["chain_coverage"] = round(chain_cov, 4)
    debug["chain_sim"] = round(chain_sim, 4)
    debug["chain_parent_start"] = int(chain[0][1])
    debug["chain_parent_end"] = int(chain[-1][1])

    if chain_cov < min_chain_coverage:
        debug["reason"] = "low_chain_coverage"
        if verbose:
            print(
                f"    [reject] chain covers {chain_cov:.2f} of child "
                f"(< {min_chain_coverage}) -- parent does not contain this content"
            )
        return None, debug

    # 4. Window = chain extent extrapolated to the child's ends.
    best_s, best_e_incl = chain_window_bounds(chain, child_len, parent_len)
    best_e = best_e_incl + 1  # exclusive

    debug["best_score_start_chunk"] = best_s
    debug["best_score_end_chunk"] = best_e - 1
    debug["best_window_length"] = best_e - best_s
    debug["reason"] = "ok"

    # 5. Symmetric mean Bimax on the window, for reporting only.
    c2w, w2c = _score_window(
        child_embs, parent_embs, best_s, best_e, normalize=normalize, device=device
    )
    best_score = 0.5 * (c2w + w2c)

    if verbose:
        print(
            f"    [chain] child={child_len} parent={parent_len} anchors={len(anchors)} "
            f"chain={len(chain)} cov={chain_cov:.2f} sim={chain_sim:.3f}"
        )
        print(
            f"      -> window [{best_s}..{best_e - 1}] len={best_e - best_s} "
            f"score={best_score:.4f} c2p={c2w:.3f} p2c={w2c:.3f}"
        )

    # 6. Score span -> export span (pad parent only; child stays whole).
    export_start = max(0, best_s - context_chunks)
    export_end = min(parent_len, best_e + context_chunks)

    if src_is_child:
        # src = child (whole), tar = parent
        src_score_s, src_score_e = 0, child_len - 1
        tar_score_s, tar_score_e = best_s, best_e - 1
        src_start_chunk, src_end_chunk = 0, child_len - 1
        tar_start_chunk, tar_end_chunk = export_start, export_end - 1
    else:
        # tar = child (whole), src = parent
        tar_score_s, tar_score_e = 0, child_len - 1
        src_score_s, src_score_e = best_s, best_e - 1
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
        chain_coverage=float(round(chain_cov, 4)),
        chain_sim=float(round(chain_sim, 4)),
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

    Anchoring is local per pair (child vs that parent only) -- no corpus FAISS
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
    snap_chars = getattr(args, "span_snap_max_chars", 400)

    # Cache raw document text per path so sentence-snapping reads each file once.
    _text_cache: Dict[str, str] = {}

    def _doc_text(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        if path not in _text_cache:
            try:
                _text_cache[path] = AlignerIO.load_document_text(path)
            except Exception:
                _text_cache[path] = None
        return _text_cache[path]

    def _snap(path: Optional[str], cs: int, ce: int) -> Tuple[int, int]:
        if snap_chars <= 0 or cs < 0 or ce < 0:
            return cs, ce
        txt = _doc_text(path)
        if not txt:
            return cs, ce
        return _snap_span_to_sentence(txt, cs, ce, max_extend=snap_chars)

    # Group candidate targets per source document, keep top-k by document score.
    by_src: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for s_idx, t_idx, score in pairs:
        by_src[s_idx].append((t_idx, score))

    rows: List[Dict[str, Any]] = []
    debug: List[Dict[str, Any]] = []
    n_localized = 0
    # Parent embeddings per file path, kept for the joint overlap-trim pass
    # (only parents that actually received a span are retained).
    parent_embs_cache: Dict[str, np.ndarray] = {}

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
                anchor_base_sim=getattr(args, "span_anchor_base_sim", 0.45),
                slope_cap=getattr(args, "span_slope_cap", 6.0),
                gap_penalty=getattr(args, "span_gap_penalty", 0.05),
                min_chain_coverage=getattr(args, "span_min_chain_coverage", 0.35),
                context_chunks=getattr(args, "span_context_chunks", 2),
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

            # Snap the chunk-derived char offsets to sentence boundaries so the
            # exported spans (and TSV) don't start/end mid-sentence. Chunk indices
            # and scores are unaffected.
            src_cs, src_ce = _snap(s_path, span.source_start_char, span.source_end_char)
            tar_cs, tar_ce = _snap(t_path, span.target_start_char, span.target_end_char)

            rows.append({
                # TSV columns
                "src_doc": s_name,
                "tar_doc": t_name,
                "document_score": float(doc_score),
                "span_score": span.span_score,
                "child_side": span.child_side,
                "child_to_parent": span.child_to_parent,
                "parent_to_child": span.parent_to_child,
                "chain_coverage": span.chain_coverage,
                "chain_sim": span.chain_sim,
                "span_context_chunks": span.span_context_chunks,
                "src_total_chunks": int(src_embs.shape[0]),
                # score span (chain window)
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
                "src_start_char": src_cs,
                "src_end_char": src_ce,
                "tar_start_char": tar_cs,
                "tar_end_char": tar_ce,
                # Internal fields for span export (not written to TSV)
                "_src_file_path": s_path,
                "_tar_file_path": t_path,
                "_parent_records": tar_records if span.child_side == "src" else src_records,
                "_child_embs": src_embs if span.child_side == "src" else tar_embs,
            })
            ppath = t_path if span.child_side == "src" else s_path
            parent_embs_cache[str(ppath)] = tar_embs if span.child_side == "src" else src_embs
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

    # ── Joint gap absorption over spans sharing a parent document ──────────
    # When several children localize onto the same parent (book chapters vs the
    # full translation), the small gaps BETWEEN adjacent parent windows are
    # exactly the boundary zones the per-pair chain cannot resolve: chapter
    # headings and opening paragraphs whose chunks straddle two chapters, embed
    # poorly, and never win top-k anchors (measured: 1-4 chunk head misses).
    # Absorb each small gap into BOTH neighbours' export windows -- the extra
    # slack is null-aligned by Vecalign downstream, whereas a missed chapter
    # opening cannot be recovered at all.
    gap_absorb = getattr(args, "span_gap_absorb", 6)
    context = getattr(args, "span_context_chunks", 2)
    if rows:
        by_parent: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            pside = "tar" if row["child_side"] == "src" else "src"
            ppath = row["_tar_file_path"] if pside == "tar" else row["_src_file_path"]
            by_parent[(pside, str(ppath))].append(row)

        n_absorbed = n_trimmed = 0
        for (pside, ppath_str), grp in by_parent.items():
            if len(grp) < 2:
                continue
            grp.sort(key=lambda r: r[f"{pside}_score_start_chunk"])
            changed = set()

            # -- overlap trimming: two chains claiming the same parent chunks --
            # A chain may start early inside the previous chapter when the two
            # are thematically continuous (measured: a 15-chunk backward
            # overshoot). Split the disputed range at the point that maximises
            # per-chunk similarity to the RIGHT child: each disputed parent
            # chunk goes to whichever child it resembles more.
            P = parent_embs_cache.get(ppath_str)
            if P is not None:
                Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
                for a, b in zip(grp, grp[1:]):
                    ov_s = b[f"{pside}_score_start_chunk"]
                    ov_e = a[f"{pside}_score_end_chunk"]
                    if ov_e < ov_s:
                        continue
                    seg = Pn[ov_s:ov_e + 1]

                    def _msim(row_):
                        C = row_["_child_embs"]
                        Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
                        return (seg @ Cn.T).max(axis=1)

                    diff = _msim(a) - _msim(b)
                    # split k = number of disputed chunks kept by a (prefix)
                    prefix = np.concatenate(([0.0], np.cumsum(diff)))
                    k = int(np.argmax(prefix))
                    new_a_end = ov_s + k - 1
                    new_b_start = ov_s + k
                    if new_a_end < a[f"{pside}_score_start_chunk"]:
                        new_a_end = a[f"{pside}_score_start_chunk"]
                        new_b_start = new_a_end + 1
                    a[f"{pside}_score_end_chunk"] = new_a_end
                    b[f"{pside}_score_start_chunk"] = min(new_b_start,
                                                          b[f"{pside}_score_end_chunk"])
                    # re-derive exports from the trimmed score windows
                    a[f"{pside}_end_chunk"] = new_a_end + context
                    b[f"{pside}_start_chunk"] = max(0, b[f"{pside}_score_start_chunk"] - context)
                    changed.update((id(a), id(b)))
                    n_trimmed += 1

            # -- gap absorption: boundary zones neither chain could anchor --
            if gap_absorb > 0:
                for a, b in zip(grp, grp[1:]):
                    a_end = a[f"{pside}_score_end_chunk"]
                    b_start = b[f"{pside}_score_start_chunk"]
                    gap = b_start - a_end - 1
                    if 0 < gap <= gap_absorb:
                        a[f"{pside}_end_chunk"] = max(a[f"{pside}_end_chunk"], b_start - 1)
                        b[f"{pside}_start_chunk"] = min(b[f"{pside}_start_chunk"], a_end + 1)
                        changed.update((id(a), id(b)))
                        n_absorbed += 1

            for row in grp:
                if id(row) not in changed:
                    continue
                ppath = row["_tar_file_path"] if pside == "tar" else row["_src_file_path"]
                cs, ce = _char_span(
                    row.get("_parent_records"),
                    row[f"{pside}_start_chunk"], row[f"{pside}_end_chunk"],
                )
                cs, ce = _snap(ppath, cs, ce)
                row[f"{pside}_start_char"], row[f"{pside}_end_char"] = cs, ce
        if n_trimmed:
            print(f"[*] Overlap trimming: split {n_trimmed} disputed range(s) "
                  f"between adjacent spans by per-chunk similarity.")
        if n_absorbed:
            print(f"[*] Gap absorption: widened export windows across "
                  f"{n_absorbed} small gap(s) between adjacent spans.")

    # ── Span-level rerank ──────────────────────────────────────────────────
    # Document-level score is topical (a related monograph can outscore the
    # actual translation); chain coverage x similarity measures how much of
    # the child is REALLY contained. Rank candidates per source doc by that.
    for row in rows:
        row["span_rank_score"] = round(row["chain_coverage"] * row["chain_sim"], 4)
    by_src_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_src_rows[row["src_doc"]].append(row)
    for grp in by_src_rows.values():
        grp.sort(key=lambda r: r["span_rank_score"], reverse=True)
        for rank, row in enumerate(grp, start=1):
            row["span_rank"] = rank

    # drop internal ndarray refs before returning
    for row in rows:
        row.pop("_child_embs", None)

    print(f"[*] Localized spans for {n_localized} document pair(s).")
    return rows, debug


# =============================================================================
# Span text export
# =============================================================================

_SPAN_COLS = [
    "src_doc", "tar_doc", "document_score", "span_rank", "span_rank_score",
    "span_score", "child_side",
    "child_to_parent", "parent_to_child", "chain_coverage", "chain_sim",
    "span_context_chunks",
    "src_total_chunks",
    "src_score_start_chunk", "src_score_end_chunk",  # chain window
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
                1_{tar_stem}.txt   # rank 1 by span_rank_score (highest)
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
        # Rank by span-level containment evidence (chain coverage x sim); the
        # document-level score is only a fallback for rows without a chain.
        group = sorted(
            group,
            key=lambda x: (x.get("span_rank_score", 0.0), x["document_score"]),
            reverse=True,
        )

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
                f"chain_cov={r['chain_coverage']:.3f}  chain_sim={r['chain_sim']:.3f}  "
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

    # Anchors + monotone chain
    parser.add_argument("--span_top_k_hits", type=int, default=3,
                        help="Top parent chunks kept per child chunk as chain anchors.")
    parser.add_argument("--span_hit_min_similarity", type=float, default=None,
                        help="Drop anchors below this cosine similarity (default: no floor).")
    parser.add_argument("--span_anchor_base_sim", type=float, default=0.45,
                        help="Anchor weight baseline: chain weight per anchor is sim minus this "
                             "value, so anchors weaker than it are net-negative and cannot pay "
                             "for stretching the chain. (default: 0.45)")
    parser.add_argument("--span_slope_cap", type=float, default=6.0,
                        help="Allowed parent chunks per child step before the chain gap penalty "
                             "applies. Bounds how fast the chain may advance through the parent; "
                             "generous by default so real expansion-ratio variation is free. "
                             "(default: 6.0)")
    parser.add_argument("--span_gap_penalty", type=float, default=0.05,
                        help="Penalty per parent chunk beyond the slope cap when chaining two "
                             "anchors. Stops the chain from teleporting across the parent to a "
                             "lexically similar but unrelated region. (default: 0.05)")
    parser.add_argument("--span_min_chain_coverage", type=float, default=0.35,
                        help="Reject a pair when the chain anchors fewer than this fraction of "
                             "child chunks: the parent is merely about the same topic and does "
                             "not actually contain the child's content. Measured on DVSKTT: "
                             "true spans >= 0.40, topical-only pairs <= 0.33. (default: 0.35)")
    parser.add_argument("--span_context_chunks", type=int, default=2,
                        help="Chunks to pad on each side of the score window for the "
                             "export span passed to Vecalign. Set 0 to disable padding. "
                             "(default: 2 -- chapter-boundary chunks embed poorly and the "
                             "chain start is typically 1-2 chunks late)")
    parser.add_argument("--span_gap_absorb", type=int, default=6,
                        help="When several spans localize onto the same parent document, "
                             "absorb gaps of at most this many chunks between adjacent "
                             "windows into BOTH neighbours (chapter headings/openings that "
                             "anchor poorly live in these gaps; Vecalign null-aligns the "
                             "slack). 0 disables. (default: 6)")
    parser.add_argument("--span_snap_max_chars", type=int, default=400,
                        help="Snap exported span char offsets to the nearest sentence boundary, "
                             "extending at most this many characters per side (token-chunk offsets "
                             "otherwise cut mid-sentence). Set 0 to disable snapping. (default: 400)")

    # Deprecated knobs from the coverage-window design; accepted and ignored so
    # existing run commands don't break.
    for dep in ("--span_min_child_length_ratio", "--span_max_child_length_ratio",
                "--span_trim_min_sim", "--span_extend_min_sim",
                "--span_extend_cap_ratio"):
        parser.add_argument(dep, type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--span_top_windows", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--span_extend_margin", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--span_no_extend", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--device", type=str, default=None,
                        help="torch device for the Bimax scoring (default: cuda if available else cpu).")

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

    # Ensure --output_path is empty before writing this run's outputs, so stale
    # files from a previous run/config (old spans/, index TSV, debug jsonl)
    # never linger alongside the new ones. Safe here: everything needed from
    # disk (alignment TSV, embeddings, metadata) has already been read above.
    if out_dir.exists():
        import shutil
        for child in out_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
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
