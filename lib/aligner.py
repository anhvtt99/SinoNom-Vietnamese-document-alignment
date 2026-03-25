import argparse
import sys
import numpy as np
from typing import List, Tuple
from pathlib import Path

from .utils import get_filename_only
from .retrieval import build_retrieval_matrix, rerank_bimax, compute_csls
from .eval import eval, plot_confusion_matrix

def extract_edges_from_faiss(D: np.ndarray, I: np.ndarray, threshold: float = 0.75) -> List[Tuple[int, int, float]]:
    """
    Extracts a list of edges (matches) from FAISS search results based on a similarity threshold.
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

def main():
    parser = argparse.ArgumentParser(description="Run cross-lingual document alignment pipeline.")
    
    # 1. Paths configurations
    parser.add_argument("--emb_base_path", type=str, required=True, help="Base path containing generated embeddings")
    
    # 2. Config tag variables
    parser.add_argument("--split_mode", type=str, choices=["sentence", "chunk"], default="sentence")
    parser.add_argument("--num_of_sent", type=int, default=1)
    parser.add_argument("--overlap_sent", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--overlap_rate", type=float, default=0.5)
    
    # 3. Algorithm Hyperparameters
    parser.add_argument("--top_k_chunks", type=int, default=5, help="Top K chunks to retrieve initially")
    parser.add_argument("--top_k_docs", type=int, default=10, help="Top K documents to retrieve initially")
    parser.add_argument("--bimax_trim_ratio", type=float, default=0.7, help="Trim ratio for Bimax reranking")
    parser.add_argument("--csls_k", type=int, default=10, help="K nearest neighbors for CSLS margin")
    parser.add_argument("--csls_top_k_out", type=int, default=10, help="Top K output edges per document in CSLS")
    parser.add_argument("--edge_threshold", type=float, default=0.08, help="Threshold to filter final edges")
    
    # 4. Optional Execution Flags
    parser.add_argument("--eval", action="store_true", help="Run evaluation using ground truth")
    parser.add_argument("--gt_path", type=str, default=None, help="Path to ground truth TSV file (required if --eval is set)")
    parser.add_argument("--viz", action="store_true", help="Generate and save confusion matrix plot")
    parser.add_argument("--viz_path", type=str, default=None, help="Path to save confusion matrix plot")

    args = parser.parse_args()

    # --- SETUP PATHS ---
    base_path = Path(args.emb_base_path)
    
    if args.split_mode == "sentence":
        config_tag = f'{args.split_mode}_n{args.num_of_sent}_o{args.overlap_sent}'
    else:
        config_tag = f'{args.split_mode}_s{args.chunk_size}_r{args.overlap_rate}'

    vi_emb_path = base_path / config_tag / "vi"
    zh_emb_path = base_path / config_tag / "zh"

    print(f"[*] Processing alignment for config: {config_tag}")

    if not vi_emb_path.exists() or not zh_emb_path.exists():
        print(f"\n[!] Error: Embeddings directory not found for '{config_tag}'.")
        if not vi_emb_path.exists():
            print(f"    -> Missing: {vi_emb_path}")
        if not zh_emb_path.exists():
            print(f"    -> Missing: {zh_emb_path}")
            
        print("\n[!] Please execute the 'gen_embedding.sh' script (or the vector generation script) to generate data before running the Aligner!")
        sys.exit(1)

    print(f"[*] VI Path: {vi_emb_path}")
    print(f"[*] ZH Path: {zh_emb_path}")

    # --- STAGE 1: INITIAL RETRIEVAL ---
    print("\n[1/3] Building Initial Retrieval Matrices...")
    _, I_zh = build_retrieval_matrix(vi_emb_path, zh_emb_path, top_k_chunks=args.top_k_chunks, top_k_docs=args.top_k_docs)
    _, I_vi = build_retrieval_matrix(zh_emb_path, vi_emb_path, top_k_chunks=args.top_k_chunks, top_k_docs=args.top_k_docs)

    # --- STAGE 2: RERANKING (BIMAX) ---
    print("[2/3] Performing Bimax Reranking...")
    D_zh, I_zh_reranked = rerank_bimax(I_zh, vi_emb_path, zh_emb_path, normalize=False, trim_ratio=args.bimax_trim_ratio, aggregation="max")
    D_vi, I_vi_reranked = rerank_bimax(I_vi, zh_emb_path, vi_emb_path, normalize=False, trim_ratio=args.bimax_trim_ratio, aggregation="max")

    # --- STAGE 3: CSLS MARGIN & EDGE EXTRACTION ---
    print("[3/3] Computing CSLS Margin and Extracting Edges...")
    D_margin, I_margin = compute_csls(D_zh, I_zh_reranked, D_vi, I_vi_reranked, knn_k=args.csls_k, top_k_out=args.csls_top_k_out)
    pairs = extract_edges_from_faiss(D_margin, I_margin, threshold=args.edge_threshold)
    
    print(f"[*] Extracted {len(pairs)} alignment pairs.")

    # --- OPTIONAL: EVALUATION & VISUALIZATION ---
    if args.eval:
        if not args.gt_path:
            raise ValueError("[!] Error: --gt_path must be provided when --eval is used.")
        
        print(f"\n[*] Running Evaluation against Ground Truth: {args.gt_path}")
        gt_path = Path(args.gt_path)
        
        result = eval(pairs, vi_emb_path / "metadata", zh_emb_path / "metadata", gt_path, normalize_fn=get_filename_only)
        
        if args.viz:
            if args.viz_path:
                output_viz_path = Path(args.viz_path)
            else:
                output_viz_path = Path.cwd() / f"confusion_matrix_{config_tag}.png"
            
            output_viz_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"[*] Plotting and saving Confusion Matrix to: {output_viz_path}")
            plot_confusion_matrix(result, save_path=output_viz_path)
            
    print("\n[+] Alignment process completed successfully!")

if __name__ == "__main__":
    main()