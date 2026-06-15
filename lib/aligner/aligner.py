import argparse
import sys
from typing import List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from lib.utils import get_filename_only, AlignerIO, cuda_available
from lib.aligner.retrieval import (
    build_retrieval_matrix,
    rerank_bimax,
    compute_csls,
    VotingParentRetriever,
)
from lib.aligner.eval import eval, plot_confusion_matrix

# Default score floor used for 1-1 mode when --edge_threshold is left unset.
_DEFAULT_1TO1_FLOOR = 0.08


# =============================================================================
# Config / paths
# =============================================================================

def build_config_tag(args) -> str:
    """Build the directory/config tag from split-mode arguments."""
    if args.split_mode == "sentence":
        return f"{args.split_mode}_n{args.num_of_sent}_o{args.overlap_sent}"
    return f"{args.split_mode}_s{args.chunk_size}_r{args.overlap_rate}"


def setup_paths(args) -> Tuple[Path, Path, str]:
    """Resolve (source_emb_path, target_emb_path, config_tag) from args."""
    base_path = Path(args.emb_base_path)
    config_tag = build_config_tag(args)
    src_emb_path = base_path / config_tag / args.src_lang
    trg_emb_path = base_path / config_tag / args.tar_lang
    return src_emb_path, trg_emb_path, config_tag


# =============================================================================
# Edge selection
# =============================================================================

def extract_edges_from_faiss(D: np.ndarray, I: np.ndarray, threshold: float = 0.75) -> List[Tuple[int, int, float]]:
    """
    Extract edges (matches) from FAISS-style result matrices using a score floor.
    """
    edges = []
    num_queries, k_neighbors = I.shape

    for src_idx in range(num_queries):
        for j in range(k_neighbors):
            trg_idx = I[src_idx, j]
            score = D[src_idx, j]
            if trg_idx == -1:
                continue
            if score >= threshold:
                edges.append((int(src_idx), int(trg_idx), float(score)))
    return edges


def extract_topk_per_source(
    D: np.ndarray,
    I: np.ndarray,
    top_k: int,
    floor: float = None,
) -> List[Tuple[int, int, float]]:
    """
    Keep up to ``top_k`` candidates per source row, ranked by score.

    Rows of ``D``/``I`` are assumed already sorted descending per source (as
    produced by ``compute_csls``). An optional ``floor`` drops candidates below
    a minimum score; because each row is sorted, the scan can stop at the first
    candidate under the floor.
    """
    edges: List[Tuple[int, int, float]] = []
    num_queries, k_neighbors = I.shape
    cap = min(top_k, k_neighbors) if (top_k and top_k > 0) else k_neighbors

    for src_idx in range(num_queries):
        kept = 0
        for j in range(k_neighbors):
            trg_idx = int(I[src_idx, j])
            score = float(D[src_idx, j])
            if trg_idx == -1:
                continue
            if floor is not None and score < floor:
                break  # sorted descending -> remaining are below floor too
            edges.append((src_idx, trg_idx, score))
            kept += 1
            if kept >= cap:
                break
    return edges


def filter_one_to_one(pairs: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
    """
    Greedy competitive matching: enforce a 1-1 mapping by claiming the highest
    scoring pair first, then skipping any pair whose source or target is taken.
    """
    sorted_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
    src_used = set()
    trg_used = set()
    final_pairs = []
    for s_idx, t_idx, score in sorted_pairs:
        if s_idx not in src_used and t_idx not in trg_used:
            final_pairs.append((s_idx, t_idx, score))
            src_used.add(s_idx)
            trg_used.add(t_idx)
    return final_pairs


def select_pairs(D_margin: np.ndarray, I_margin: np.ndarray, args) -> List[Tuple[int, int, float]]:
    """
    Turn the CSLS score/index matrices into a final edge list.

    - 1-1 mode: score floor (``--edge_threshold``, default 0.08) then greedy 1-1.
    - m-m mode: top-k candidates per source (``--top_k_pairs``), with an optional
      ``--edge_threshold`` floor (no floor by default).
    """
    if args.align_mode == "1-1":
        floor = args.edge_threshold if args.edge_threshold is not None else _DEFAULT_1TO1_FLOOR
        pairs = extract_edges_from_faiss(D_margin, I_margin, threshold=floor)
        print(f"[*] 1-1 mode: {len(pairs)} edges above floor {floor:.4f}; applying greedy 1-1...")
        pairs = filter_one_to_one(pairs)
    else:
        pairs = extract_topk_per_source(
            D_margin, I_margin, top_k=args.top_k_pairs, floor=args.edge_threshold
        )
        floor_txt = "none" if args.edge_threshold is None else f"{args.edge_threshold:.4f}"
        print(f"[*] m-m mode: top-{args.top_k_pairs} per source (floor={floor_txt}) -> {len(pairs)} edges.")
    return pairs


# =============================================================================
# Pipeline stages
# =============================================================================

def retrieve_and_rerank(src_emb_path: Path, trg_emb_path: Path, args, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run retrieval -> Bimax rerank -> CSLS margin, returning (D_margin, I_margin).

    Both FAISS indices are built once and reused for the two retrieval
    directions feeding CSLS.
    """
    # --- STAGE 1: INITIAL RETRIEVAL ---
    print("\n[1/3] Building Initial Retrieval Matrices...")
    retriever_target = VotingParentRetriever()   # target corpus indexed; queried by source
    retriever_target.load_target_corpus(trg_emb_path)
    retriever_source = VotingParentRetriever()   # source corpus indexed; queried by target
    retriever_source.load_target_corpus(src_emb_path)

    _, I_trg = build_retrieval_matrix(
        src_emb_path, trg_emb_path,
        top_k_chunks=args.top_k_chunks, top_k_docs=args.top_k_docs, retriever=retriever_target,
    )
    _, I_src = build_retrieval_matrix(
        trg_emb_path, src_emb_path,
        top_k_chunks=args.top_k_chunks, top_k_docs=args.top_k_docs, retriever=retriever_source,
    )

    # --- STAGE 2: RERANKING (BIMAX) ---
    # 'max' favours partial/containment matches (m-m); 'avg' verifies global
    # similarity for unique mappings (1-1).
    bimax_aggregation = "max" if args.align_mode == "m-m" else "avg"
    print("[2/3] Performing Bimax Reranking...")
    D_trg, I_trg_reranked = rerank_bimax(
        I_trg, src_emb_path, trg_emb_path,
        normalize=False, trim_ratio=args.bimax_trim_ratio, device=device, aggregation=bimax_aggregation,
    )
    D_src, I_src_reranked = rerank_bimax(
        I_src, trg_emb_path, src_emb_path,
        normalize=False, trim_ratio=args.bimax_trim_ratio, device=device, aggregation=bimax_aggregation,
    )

    # --- STAGE 3: CSLS MARGIN ---
    print("[3/3] Computing CSLS Margin...")
    D_margin, I_margin = compute_csls(
        D_trg, I_trg_reranked, D_src, I_src_reranked,
        knn_k=args.csls_k, top_k_out=args.csls_top_k_out,
    )
    return D_margin, I_margin


def align_documents(args, device: str):
    """
    Full document-level alignment.

    Returns (pairs, src_meta_df, trg_meta_df, src_emb_path, trg_emb_path, config_tag).
    ``pairs`` is a list of (src_doc_idx, tar_doc_idx, score).
    """
    src_emb_path, trg_emb_path, config_tag = setup_paths(args)

    if not src_emb_path.exists() or not trg_emb_path.exists():
        print(f"\n[!] Error: Embeddings directory not found for '{config_tag}'.")
        if not src_emb_path.exists():
            print(f"    -> Missing Source: {src_emb_path}")
        if not trg_emb_path.exists():
            print(f"    -> Missing Target: {trg_emb_path}")
        print("\n[!] Please execute the embedding generation script first to create the data!")
        sys.exit(1)

    print(f"[*] Source Path: {src_emb_path}")
    print(f"[*] Target Path: {trg_emb_path}")

    D_margin, I_margin = retrieve_and_rerank(src_emb_path, trg_emb_path, args, device)
    pairs = select_pairs(D_margin, I_margin, args)

    src_meta_df = AlignerIO.load_metadata(src_emb_path / "metadata").set_index("doc_idx")
    trg_meta_df = AlignerIO.load_metadata(trg_emb_path / "metadata").set_index("doc_idx")

    return pairs, src_meta_df, trg_meta_df, src_emb_path, trg_emb_path, config_tag


# =============================================================================
# Output
# =============================================================================

def to_named_pairs(pairs, src_meta_df, trg_meta_df):
    """Resolve (s_idx, t_idx, score) -> (s_idx, t_idx, s_name, t_name, score)."""
    named = []
    for s_idx, t_idx, score in pairs:
        s_path = AlignerIO.get_path_by_idx(src_meta_df, s_idx)
        t_path = AlignerIO.get_path_by_idx(trg_meta_df, t_idx)
        if s_path and t_path:
            named.append((s_idx, t_idx, get_filename_only(s_path), get_filename_only(t_path), float(score)))
    return named


def save_alignment_tsv(named_pairs, out_path: Path):
    """
    Write the alignment TSV. Both numeric ``doc_idx`` and display name are stored
    so the downstream span-localization step can join on indices robustly.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("src_doc_idx\ttar_doc_idx\tsrc_doc\ttar_doc\tscore\n")
        for s_idx, t_idx, s_name, t_name, score in named_pairs:
            f.write(f"{s_idx}\t{t_idx}\t{s_name}\t{t_name}\t{score:.4f}\n")


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cross-lingual document alignment pipeline.")

    # 1. Paths and Languages
    parser.add_argument("--emb_base_path", type=str, required=True, help="Base path containing generated embeddings")
    parser.add_argument("--src_lang", type=str, default="vi", help="Source language code (default: vi)")
    parser.add_argument("--tar_lang", type=str, default="zh", help="Target language code (default: zh)")

    # 2. Alignment mode
    parser.add_argument("--align_mode", type=str, choices=["1-1", "m-m"], default="m-m",
                        help="'1-1' unique mapping (greedy), 'm-m' top-k candidates per source")

    # 3. Config tag
    parser.add_argument("--split_mode", type=str, choices=["sentence", "chunk"], default="sentence")
    parser.add_argument("--num_of_sent", type=int, default=1)
    parser.add_argument("--overlap_sent", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--overlap_rate", type=float, default=0.5)

    # 4. Algorithm hyperparameters
    parser.add_argument("--top_k_chunks", type=int, default=5, help="Top K chunks to retrieve initially")
    parser.add_argument("--top_k_docs", type=int, default=10, help="Top K documents to retrieve initially")
    parser.add_argument("--bimax_trim_ratio", type=float, default=0.7, help="Trim ratio for Bimax reranking")
    parser.add_argument("--csls_k", type=int, default=10, help="K nearest neighbors for CSLS margin")
    parser.add_argument("--csls_top_k_out", type=int, default=10, help="Top K output edges per document in CSLS")

    # 4b. Edge selection
    parser.add_argument("--top_k_pairs", type=int, default=5,
                        help="m-m mode: number of target candidates kept per source (ranked by CSLS). "
                             "Capped by --csls_top_k_out.")
    parser.add_argument("--edge_threshold", type=float, default=None,
                        help="Optional score floor. 1-1 mode defaults to 0.08 when unset; "
                             "m-m mode applies no floor unless set.")

    # 5. Execution flags & outputs
    parser.add_argument("--save_results", action="store_true", help="Dump alignment pairs to a TSV file")
    parser.add_argument("--output_path", type=str, default=None, help="Directory to save alignment outputs")
    parser.add_argument("--eval", action="store_true", help="Run evaluation using ground truth")
    parser.add_argument("--gt_path", type=str, default=None, help="Path to ground truth TSV file (required if --eval)")
    parser.add_argument("--viz", action="store_true", help="Generate and save confusion matrix plot")
    return parser


def main():
    args = build_parser().parse_args()

    is_gpu = cuda_available(verbose=False)
    device = "cuda" if is_gpu else "cpu"

    config_tag = build_config_tag(args)
    print(f"[*] Processing alignment for config: {config_tag}")
    print(f"[*] Source language: {args.src_lang} | Target language: {args.tar_lang}")

    pairs, src_meta_df, trg_meta_df, src_emb_path, trg_emb_path, config_tag = align_documents(args, device)
    print(f"[*] Final alignment pairs: {len(pairs)}")

    named_pairs = to_named_pairs(pairs, src_meta_df, trg_meta_df)

    if args.save_results:
        out_dir = Path(args.output_path) if args.output_path else Path.cwd()
        out_path = out_dir / f"alignment_{config_tag}.tsv"
        save_alignment_tsv(named_pairs, out_path)
        print(f"\n[+] Results dumped to: {out_path}")
    else:
        print("\n[*] --- ALIGNMENT PAIRS (Source -> Target | Score) ---")
        if not named_pairs:
            print("    (No pairs found)")
        for _, _, s_name, t_name, score in named_pairs:
            print(f"    {s_name:3} -> {t_name:3} | Score: {score:.4f}")
        print("[*] --------------------------------------------------")

    # --- OPTIONAL: EVALUATION & VISUALIZATION ---
    if args.eval:
        if not args.gt_path:
            raise ValueError("[!] Error: --gt_path must be provided when --eval is used.")

        print(f"\n[*] Running Evaluation against Ground Truth: {args.gt_path}")
        result = eval(
            pairs, src_emb_path / "metadata", trg_emb_path / "metadata",
            Path(args.gt_path), normalize_fn=get_filename_only,
        )

        if args.viz:
            out_dir = Path(args.output_path) if args.output_path else Path.cwd()
            output_viz_path = out_dir / f"confusion_matrix_{config_tag}.png"
            output_viz_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[*] Plotting and saving Confusion Matrix to: {output_viz_path}")
            plot_confusion_matrix(result, save_path=output_viz_path)

    print("\n[+] Alignment process completed successfully!")


if __name__ == "__main__":
    main()
