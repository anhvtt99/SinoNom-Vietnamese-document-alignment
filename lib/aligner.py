import argparse
import sys
import numpy as np
from typing import List, Tuple
from pathlib import Path

from utils import get_filename_only, AlignerIO, cuda_available
from retrieval import build_retrieval_matrix, rerank_bimax, compute_csls
from eval import eval, plot_confusion_matrix

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
    
    # 1. Paths and Languages configurations
    parser.add_argument("--emb_base_path", type=str, required=True, help="Base path containing generated embeddings")
    parser.add_argument("--src_lang", type=str, default="vi", help="Source language code (default: vi)")
    parser.add_argument("--tar_lang", type=str, default="zh", help="Target language code (default: zh)")

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
    
    # 4. Optional Execution Flags & Outputs
    parser.add_argument("--save_results", action="store_true", help="Dump alignment pairs to a TSV file")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save alignment pairs (TSV)")
    parser.add_argument("--eval", action="store_true", help="Run evaluation using ground truth")
    parser.add_argument("--gt_path", type=str, default=None, help="Path to ground truth TSV file (required if --eval is set)")
    parser.add_argument("--viz", action="store_true", help="Generate and save confusion matrix plot")

    args = parser.parse_args()

    # --- SETUP PATHS ---
    base_path = Path(args.emb_base_path)
    is_gpu = cuda_available(verbose=False)
    device = "cuda" if is_gpu else "cpu"
    if args.split_mode == "sentence":
        config_tag = f'{args.split_mode}_n{args.num_of_sent}_o{args.overlap_sent}'
    else:
        config_tag = f'{args.split_mode}_s{args.chunk_size}_r{args.overlap_rate}'

    src_emb_path = base_path / config_tag / args.src_lang
    trg_emb_path = base_path / config_tag / args.tar_lang

    print(f"[*] Processing alignment for config: {config_tag}")
    print(f"[*] Source language: {args.src_lang} | Target language: {args.tar_lang}")

    # Check existence
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

    # --- STAGE 1: INITIAL RETRIEVAL ---
    print("\n[1/3] Building Initial Retrieval Matrices...")
    # I_trg: Indices of target docs for each source query
    # I_src: Indices of source docs for each target query
    _, I_trg = build_retrieval_matrix(src_emb_path, trg_emb_path, top_k_chunks=args.top_k_chunks, top_k_docs=args.top_k_docs)
    _, I_src = build_retrieval_matrix(trg_emb_path, src_emb_path, top_k_chunks=args.top_k_chunks, top_k_docs=args.top_k_docs)

    # --- STAGE 2: RERANKING (BIMAX) ---
    print("[2/3] Performing Bimax Reranking...")
    D_trg, I_trg_reranked = rerank_bimax(I_trg, src_emb_path, trg_emb_path, normalize=False, trim_ratio=args.bimax_trim_ratio, device=device, aggregation="max")
    D_src, I_src_reranked = rerank_bimax(I_src, trg_emb_path, src_emb_path, normalize=False, trim_ratio=args.bimax_trim_ratio, device=device,  aggregation="max")

    # --- STAGE 3: CSLS MARGIN & EDGE EXTRACTION ---
    print("[3/3] Computing CSLS Margin and Extracting Edges...")
    D_margin, I_margin = compute_csls(D_trg, I_trg_reranked, D_src, I_src_reranked, knn_k=args.csls_k, top_k_out=args.csls_top_k_out)
    pairs = extract_edges_from_faiss(D_margin, I_margin, threshold=args.edge_threshold)
    
    print(f"[*] Extracted {len(pairs)} alignment pairs.")
    # Load metadata df (set index 'doc_idx' để lookup O(1) như trong hàm eval của bạn)
    src_meta_df = AlignerIO.load_metadata(src_emb_path / "metadata").set_index('doc_idx')
    trg_meta_df = AlignerIO.load_metadata(trg_emb_path / "metadata").set_index('doc_idx')

    named_pairs = []
    for s_idx, t_idx, score in pairs:
        s_name = AlignerIO.get_path_by_idx(src_meta_df, s_idx)
        t_name = AlignerIO.get_path_by_idx(trg_meta_df, t_idx)

        if s_name and t_name:
            s_display = get_filename_only(s_name)
            t_display = get_filename_only(t_name)
            named_pairs.append((s_display, t_display, score))
    if args.save_results:
        final_out_path = Path(args.output_path) / f"alignment_{config_tag}.tsv" \
                         if args.output_path else Path.cwd() / f"alignment_{config_tag}.tsv"
        
        final_out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(final_out_path, "w", encoding="utf-8") as f:
            f.write("src_idx\ttar_idx\tscore\n")
            for s, t, sc in named_pairs:
                f.write(f"{s}\t{t}\t{sc:.4f}\n")
        print(f"\n[+] Results dumped to: {final_out_path}")
    else:
        # Nếu không dump thì in trực tiếp ra console
        print("\n[*] --- ALIGNMENT PAIRS (Source -> Target | Score) ---")
        if not named_pairs:
            print("    (No pairs found matching the threshold)")
        for s, t, sc in named_pairs:
            print(f"    {s:3} -> {t:3} | Score: {sc:.4f}")
        print("[*] --------------------------------------------------")
    # --- OPTIONAL: EVALUATION & VISUALIZATION ---
    if args.eval:
        if not args.gt_path:
            raise ValueError("[!] Error: --gt_path must be provided when --eval is used.")
        
        print(f"\n[*] Running Evaluation against Ground Truth: {args.gt_path}")
        gt_path = Path(args.gt_path)
        
        result = eval(pairs, src_emb_path / "metadata", trg_emb_path / "metadata", gt_path, normalize_fn=get_filename_only)
        
        if args.viz:
            if args.output_path:
                output_viz_path = Path(args.output_path) / f"confusion_matrix_{config_tag}.png"
            else:
                output_viz_path = Path.cwd() / f"confusion_matrix_{config_tag}.png"
            
            output_viz_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"[*] Plotting and saving Confusion Matrix to: {output_viz_path}")
            plot_confusion_matrix(result, save_path=output_viz_path)
            
    print("\n[+] Alignment process completed successfully!")

if __name__ == "__main__":
    main()