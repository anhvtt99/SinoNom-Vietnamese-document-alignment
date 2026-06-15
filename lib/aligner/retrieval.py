from typing import Literal, Optional, List, Union, Tuple
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import faiss

from lib.utils import AlignerIO


ScoreMode = Literal["csls", "margin1", "cosine"]

# -----------------------
# Retrieval
# -----------------------
class VotingParentRetriever:
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.chunk2doc: Optional[np.ndarray] = None    # FAISS row -> target doc_idx
        self.chunk2local: Optional[np.ndarray] = None  # FAISS row -> local chunk idx within its doc
        self.meta_df: Optional[pd.DataFrame] = None
        self.dim: int = 0

    def load_target_corpus(self, lang_path: Union[str, Path]):
        """
        Phase 1: Load the entire Target Corpus into RAM and build the FAISS Index.

        Builds two row->id maps:
          - chunk2doc:   FAISS global chunk row -> target doc_idx
          - chunk2local: FAISS global chunk row -> local chunk idx inside that doc

        Lengths are derived from the actual embedding matrices (shape[0]) rather
        than trusting metadata's ``n_chunks``, then validated against the index.

        Args:
            lang_path: Path to the target language folder.
        """
        lang_path = Path(lang_path)
        embeddings_dir = lang_path / "embeddings"
        meta_dir = lang_path / "metadata"

        # Load metadata containing document paths and chunk counts
        self.meta_df = AlignerIO.load_metadata(meta_dir)

        all_chunks = []
        chunk2doc_list: List[int] = []
        chunk2local_list: List[int] = []

        for _, row in self.meta_df.iterrows():
            emb_file = embeddings_dir / row['emb_file']
            if not emb_file.exists():
                print(f"[Warning] Missing file: {emb_file}")
                continue

            # Load the embedding matrix of the document. Shape: [n_chunks, dim]
            emb_matrix = AlignerIO.load_doc_embedding(embeddings_dir, row['emb_file'])
            n_rows = int(emb_matrix.shape[0])
            if n_rows == 0:
                continue

            all_chunks.append(emb_matrix)
            doc_idx = int(row['doc_idx'])
            chunk2doc_list.extend([doc_idx] * n_rows)      # [doc, doc, ...] (n_rows)
            chunk2local_list.extend(range(n_rows))         # [0, 1, ..., n_rows-1]

        if not all_chunks:
            raise RuntimeError(f"No embeddings found under {embeddings_dir}")

        # Flatten the list of 2D matrices into a single massive 2D matrix
        X = np.vstack(all_chunks)

        self.chunk2doc = np.array(chunk2doc_list, dtype=np.int64)
        self.chunk2local = np.array(chunk2local_list, dtype=np.int64)
        self.dim = X.shape[1]

        # Validate that every mapping lines up with the FAISS matrix length.
        if not (len(self.chunk2doc) == len(self.chunk2local) == X.shape[0]):
            raise ValueError(
                f"Chunk map length mismatch: chunk2doc={len(self.chunk2doc)}, "
                f"chunk2local={len(self.chunk2local)}, X={X.shape[0]}"
            )

        print(f"[*] Building FAISS IndexFlatIP for {X.shape[0]} chunks...")

        # Inner Product (IP) on L2-normalized vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(X)

        if self.index.ntotal != X.shape[0]:
            raise ValueError(
                f"FAISS index size {self.index.ntotal} != matrix rows {X.shape[0]}"
            )

        print(f"[+] Done! Index is ready. Total target documents: {len(self.meta_df)}")

    def retrieve(self, query_chunks: np.ndarray, top_k_chunks: int = 5, top_k_docs: int = 10) -> List[Tuple[int, int]]:
        """
        Phase 2 & 3: Multi-Query Search and Plurality Voting (Hit Count).

        Each source chunk contributes AT MOST ONE vote per target document, even
        if several of its top-k neighbors fall in the same target doc. This stops
        a single chunk from dominating the vote via repeated near-duplicates.

        Args:
            query_chunks: Matrix [num_chunks_in_source, dim] of the source document.
            top_k_chunks: Number of nearest neighbors to retrieve for EACH source chunk.
            top_k_docs: Number of top parent documents to return based on vote count.

        Returns:
            A list of tuples: [(target_doc_idx, total_votes), ...]
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call load_target_corpus() first.")

        if query_chunks.shape[0] == 0:
            return []

        # FAISS strictly requires float32
        query_chunks = query_chunks.astype('float32')

        # SEARCH all source chunks simultaneously. I: [num_src_chunks, top_k_chunks]
        _, I = self.index.search(query_chunks, k=top_k_chunks)

        # Per source chunk, dedup target docs (vote once per doc), then aggregate.
        vote_counter: Counter = Counter()
        for row in I:
            docs_in_row = {
                int(self.chunk2doc[cid])
                for cid in row
                if cid != -1
            }
            vote_counter.update(docs_in_row)

        return vote_counter.most_common(top_k_docs)

    def search_chunk_hits(
        self,
        query_chunks: np.ndarray,
        top_k_chunks: int = 5,
    ) -> List[dict]:
        """
        Forward chunk-level nearest-neighbor search for span localization.

        Returns one hit per (source chunk, retained neighbor) with both the
        global FAISS id and the resolved (target doc, local chunk) coordinates.

        Returns:
            List of dicts with keys:
              source_chunk_idx, target_global_chunk_idx, target_doc_idx,
              target_chunk_idx, faiss_rank, faiss_score
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call load_target_corpus() first.")

        if query_chunks.shape[0] == 0:
            return []

        query_chunks = query_chunks.astype('float32')
        scores, I = self.index.search(query_chunks, k=top_k_chunks)

        hits: List[dict] = []
        n_src, k = I.shape
        for s in range(n_src):
            for rank in range(k):
                gid = int(I[s, rank])
                if gid == -1:
                    continue
                hits.append({
                    "source_chunk_idx": s,
                    "target_global_chunk_idx": gid,
                    "target_doc_idx": int(self.chunk2doc[gid]),
                    "target_chunk_idx": int(self.chunk2local[gid]),
                    "faiss_rank": rank,
                    "faiss_score": float(scores[s, rank]),
                })
        return hits

    def get_doc_info(self, doc_idx: int) -> dict:
        """Utility function to retrieve metadata for a specific document index."""
        row = self.meta_df[self.meta_df['doc_idx'] == doc_idx].iloc[0]
        return row.to_dict()


def build_retrieval_matrix(
    source_lang_path: Union[str, Path],
    target_lang_path: Union[str, Path],
    top_k_chunks: int = 5,
    top_k_docs: int = 10,
    retriever: Optional[VotingParentRetriever] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper for the Retriever to process all documents in the corpus.
    Returns matrices matching the FAISS API format: (Scores/Votes, Indices).

    Args:
        retriever: Optional pre-built VotingParentRetriever already loaded with
                   the target corpus. Reused to avoid rebuilding the FAISS index.
    """
    source_lang_path = Path(source_lang_path)
    source_meta_path = source_lang_path / "metadata"
    source_emb_dir = source_lang_path / "embeddings"

    # Load target index (reuse a prebuilt retriever when provided)
    if retriever is None:
        retriever = VotingParentRetriever()
        retriever.load_target_corpus(target_lang_path)

    # Find the maximum document index to set the matrix size.
    # Use max() + 1 to prevent errors if some doc_idx are missing in the middle.
    source_meta_df = AlignerIO.load_metadata(source_meta_path)
    N_source = source_meta_df['doc_idx'].max() + 1
    
    # Initialize empty matrices.
    # I_matrix stores Target Document IDs (-1 means no target found).
    # V_matrix stores the Vote Counts (0 means no votes).
    I_matrix = np.full((N_source, top_k_docs), -1, dtype=np.int64)
    V_matrix = np.full((N_source, top_k_docs), 0, dtype=np.float32)
    
    total_docs = len(source_meta_df)
    print(f"[*] Building retrieval matrix for {total_docs} documents...")
    
    # Loop through each document in the source metadata
    for i, (_, row) in enumerate(source_meta_df.iterrows()):
        src_idx = row['doc_idx']
        
        # Print a simple progress update every 100 documents
        if (i + 1) % 100 == 0:
            print(f"    Processing {i + 1}/{total_docs}...")
        
        # 1. Load the chunk embeddings for the current source document
        src_emb_matrix = AlignerIO.load_doc_embedding(source_emb_dir, row['emb_file'])
        
        # 2. Query the Retriever
        # Returns a list of tuples: [(target_idx_1, votes_1), ...]
        candidates = retriever.retrieve(
            src_emb_matrix, 
            top_k_chunks=top_k_chunks, 
            top_k_docs=top_k_docs
        )
        
        # 3. Fill the results into the pre-allocated matrices
        for rank, (target_idx, votes) in enumerate(candidates):
            if rank >= top_k_docs:
                break
            I_matrix[src_idx, rank] = target_idx
            V_matrix[src_idx, rank] = votes
            
    print("[+] Retrieval matrix completed!")
    
    # Return V_matrix (Scores) and I_matrix (Indices) just like FAISS (D, I)
    return V_matrix, I_matrix

# -----------------------
# Rerank
# -----------------------
# Bimax part
from pathlib import Path
from typing import Union, Tuple
import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_bidirectional_scores(
    seg_embs_1: torch.Tensor,  # [n, d]
    seg_embs_2: torch.Tensor,  # [m, d]
    *,
    normalize: bool = True,
    trim_ratio: float = 1.0,   # Keep top K% matches
    batch_size: int = 2048,
) -> Tuple[float, float]:
    """
    Computes bidirectional similarity scores between two sets of segment embeddings.
    
    This function calculates the coverage of seg_embs_1 by seg_embs_2 and vice versa,
    returning raw scores for both directions without aggregation.

    Args:
        seg_embs_1: Tensor of shape [n, d], representing segments of Document 1.
        seg_embs_2: Tensor of shape [m, d], representing segments of Document 2.
        normalize: Whether to L2-normalize embeddings before dot product.
        trim_ratio: Float (0.0 < ratio <= 1.0). If < 1.0, only the top K% best matching 
                    segments are used to compute the mean score. This helps reduce noise 
                    from irrelevant segments in long documents.
        batch_size: Batch size for matrix multiplication to avoid OOM on GPUs.

    Returns:
        A tuple (score_1_to_2, score_2_to_1):
        - score_1_to_2: Represents how well Document 1 is covered by Document 2.
        - score_2_to_1: Represents how well Document 2 is covered by Document 1.
    """
    # 1. Handle empty inputs
    if seg_embs_1.numel() == 0 or seg_embs_2.numel() == 0:
        return 0.0, 0.0

    X = seg_embs_1
    Y = seg_embs_2
    
    # 2. Normalize embeddings
    if normalize:
        X = F.normalize(X, p=2, dim=1)
        Y = F.normalize(Y, p=2, dim=1)

    n = X.shape[0]
    m = Y.shape[0]

    # Helper function to compute trimmed mean
    def get_trimmed_mean(vals: torch.Tensor, ratio: float) -> float:
        if ratio >= 1.0:
            return vals.mean().item()
        # Ensure at least 1 segment is kept
        k = max(1, int(len(vals) * ratio))
        return torch.topk(vals, k).values.mean().item()

    # 3. Compute X -> Y (How well X is covered by Y)
    # For each segment in X, find the best matching segment in Y.
    row_max_vals = []
    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        # sim_chunk shape: [batch_size, m]
        sim_chunk = torch.matmul(X[i:end_i], Y.T) 
        # Max over dim=1 (columns/Y)
        max_val, _ = torch.max(sim_chunk, dim=1)
        row_max_vals.append(max_val)
    
    row_max = torch.cat(row_max_vals, dim=0)
    s1 = get_trimmed_mean(row_max, trim_ratio)

    # 4. Compute Y -> X (How well Y is covered by X)
    # For each segment in Y, find the best matching segment in X.
    col_max_vals = []
    for j in range(0, m, batch_size):
        end_j = min(j + batch_size, m)
        # sim_chunk shape: [batch_size, n]
        sim_chunk = torch.matmul(Y[j:end_j], X.T)
        # Max over dim=1 (columns/X)
        max_val, _ = torch.max(sim_chunk, dim=1)
        col_max_vals.append(max_val)
    
    col_max = torch.cat(col_max_vals, dim=0)
    s2 = get_trimmed_mean(col_max, trim_ratio)

    return s1, s2


def bimax_score(
    seg_embs_1: torch.Tensor,
    seg_embs_2: torch.Tensor,
    *,
    normalize: bool = True,
    trim_ratio: float = 1.0,
    batch_size: int = 2048,
    aggregation: str = 'avg' # Options: 'avg', 'max', 'min'
) -> float:
    """
    Wrapper for compute_bidirectional_scores that applies an aggregation strategy.
    
    Args:
        aggregation: Strategy to combine the two directional scores.
                     - 'avg': Standard BiMax (symmetric). Good for 1-1 alignment.
                     - 'max': Returns the better of the two directions. Good for containment (1-N/N-1).
                     - 'min': Strict matching. Both directions must be good.
    
    Returns:
        A single scalar float score.
    """
    s1, s2 = compute_bidirectional_scores(
        seg_embs_1, 
        seg_embs_2, 
        normalize=normalize, 
        trim_ratio=trim_ratio, 
        batch_size=batch_size
    )
    
    if aggregation == 'max':
        return max(s1, s2)
    elif aggregation == 'min':
        return min(s1, s2)
    else:
        # Default to average
        return 0.5 * (s1 + s2)

def rerank_bimax(
    I: np.ndarray,  # Matrix from FAISS [N, k]
    source_lang_path: Union[str, Path],
    target_lang_path: Union[str, Path],
    *,
    device: str = "cpu",
    normalize: bool = False,
    batch_size: int = 2048,
    trim_ratio: float = 1.0,
    aggregation: str = 'avg',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rerank the candidates retrieved by FAISS using detailed MaxSim/Bimax scoring.
    Loads data dynamically from the standard Aligner file system.
    
    Returns:
        D_sorted, I_sorted: The updated score matrix and indices, sorted descending.
    """
    I = np.asarray(I, dtype=np.int64)
    if I.ndim != 2:
        raise ValueError(f"I must be 2D (N,k), got {I.shape}")
        
    N, k = I.shape
    if N == 0 or k == 0:
        return np.zeros((N, k), dtype=np.float32), I

    # 1. Setup paths
    source_lang_path = Path(source_lang_path)
    target_lang_path = Path(target_lang_path)
    
    src_emb_dir = source_lang_path / "embeddings"
    tar_emb_dir = target_lang_path / "embeddings"

    # 2. Load metadata and set 'doc_idx' as the index
    src_meta_df = AlignerIO.load_metadata(source_lang_path / "metadata").set_index('doc_idx')
    tar_meta_df = AlignerIO.load_metadata(target_lang_path / "metadata").set_index('doc_idx')

    # Initialize score matrix with a very low number (-1e9) for invalid/missing pairs
    scores = np.full((N, k), -1e9, dtype=np.float32)

    print(f"[*] Starting Bimax Reranking for {N} documents...")

    # 3. Process each query document
    for src_idx in range(N):
        if (src_idx + 1) % 100 == 0:
            print(f"    Reranking {src_idx + 1}/{N}...")

        candidate_indices = I[src_idx]
        
        # Filter out invalid indices (-1 from FAISS)
        valid_cands = [idx for idx in candidate_indices if idx != -1]
        if not valid_cands:
            continue

        # --- LOAD SOURCE ---
        try:
            src_file = src_meta_df.loc[src_idx, 'emb_file']
            A_np = AlignerIO.load_doc_embedding(src_emb_dir, src_file)
        except KeyError:
            # Source doc_idx doesn't exist in metadata (e.g., missing file)
            continue
            
        A = torch.from_numpy(A_np).to(device)

        # --- LOAD TARGETS (BATCH CACHE) ---
        # Fetch all candidate embeddings for this query at once to minimize I/O overhead
        tar_embs_dict = AlignerIO.get_embs_by_indices(tar_meta_df, tar_emb_dir, valid_cands)

        # --- CALCULATE SCORES ---
        for col_idx, tar_idx in enumerate(candidate_indices):
            if tar_idx == -1 or tar_idx not in tar_embs_dict:
                continue

            B_np = tar_embs_dict[tar_idx]
            B = torch.from_numpy(B_np).to(device)

            # Core scoring logic
            s = bimax_score(
                A, B, 
                normalize=normalize, 
                trim_ratio=trim_ratio, 
                batch_size=batch_size, 
                aggregation=aggregation
            )
            scores[src_idx, col_idx] = float(s)

    print("[+] Reranking completed!")

    # 4. Sort the results per query (Row-wise descending sort)
    order = np.argsort(-scores, axis=1)
    rows = np.arange(N)[:, None]
    
    D_sorted = scores[rows, order].astype(np.float32)
    I_sorted = I[rows, order].astype(np.int64)

    return D_sorted, I_sorted


# Compute margin score 
def compute_csls(
    D_A: np.ndarray, I_A: np.ndarray,
    D_B: np.ndarray, I_B: np.ndarray,
    knn_k: int = 10,
    top_k_out: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate bi-directional CSLS (Margin) scores by combining candidates from both directions.
    Formula: CSLS(x, y) = 2 * Cosine(x, y) - margin(x) - margin(y)

    Args:
        D_A, I_A: Scores and Indices from Source -> Target (Shape: [N_A, K])
        D_B, I_B: Scores and Indices from Target -> Source (Shape: [N_B, K])
        knn_k: Number of neighbors used to calculate the margin penalty.
        top_k_out: Number of final candidates to return for each Source document.
    """
    N_A = D_A.shape[0]
    N_B = D_B.shape[0]

    # --- STEP 1: Calculate margin penalties ---
    # margin(x) and margin(y) are the average scores of the top-k neighbors
    k_A = min(knn_k, D_A.shape[1])
    k_B = min(knn_k, D_B.shape[1])

    valid_mask_A = I_A[:, :k_A] != -1
    D_A_masked = np.where(valid_mask_A, D_A[:, :k_A], np.nan)
    r_A = np.nanmean(D_A_masked, axis=1)
    r_A = np.nan_to_num(r_A, nan=0.0)

    valid_mask_B = I_B[:, :k_B] != -1
    D_B_masked = np.where(valid_mask_B, D_B[:, :k_B], np.nan)
    r_B = np.nanmean(D_B_masked, axis=1)
    r_B = np.nan_to_num(r_B, nan=0.0)

    # --- STEP 2: Extract all valid edges from Direction A -> B ---
    valid_A = I_A != -1
    x_A = np.repeat(np.arange(N_A), D_A.shape[1])[valid_A.flatten()]
    y_A = I_A[valid_A]
    scores_A = D_A[valid_A]

    # --- STEP 3: Extract all valid edges from Direction B -> A ---
    valid_B = I_B != -1
    y_B = np.repeat(np.arange(N_B), D_B.shape[1])[valid_B.flatten()]
    x_B = I_B[valid_B]
    scores_B = D_B[valid_B]

    # --- STEP 4: Merge pairs from both directions (Union of Edges) ---
    x_all = np.concatenate([x_A, x_B])
    y_all = np.concatenate([y_A, y_B])
    scores_all = np.concatenate([scores_A, scores_B])

    # --- STEP 5: Collapse duplicate pairs, keeping the MAX directional score ---
    # The same (x, y) pair can appear from both directions (A->B and B->A) with
    # different base scores. Taking the max (instead of np.unique's first-seen)
    # uses the strongest directional evidence before applying CSLS.
    pair_keys = x_all * N_B + y_all
    pair_df = pd.DataFrame({
        "key": pair_keys,
        "x": x_all,
        "y": y_all,
        "score": scores_all,
    })
    agg = (
        pair_df.groupby("key", sort=False)
        .agg(x=("x", "first"), y=("y", "first"), score=("score", "max"))
        .reset_index(drop=True)
    )

    x_uniq = agg["x"].to_numpy()
    y_uniq = agg["y"].to_numpy()
    base_scores = agg["score"].to_numpy()

    # --- STEP 6: Apply the CSLS Formula ---
    # Penalize "Hub" documents that are too close to everything
    csls_scores = 2.0 * base_scores - r_A[x_uniq] - r_B[y_uniq]

    # --- STEP 7: Rebuild and sort the output matrices [N_A, top_k_out] ---
    # We use Pandas here because after merging, some 'x' might have 5 candidates, others 15.
    # Pandas makes it extremely easy to group, sort, and slice jagged arrays.
    df = pd.DataFrame({'x': x_uniq, 'y': y_uniq, 'score': csls_scores})

    # Sort by 'x' (ascending) and then by 'score' (descending)
    df_sorted = df.sort_values(by=['x', 'score'], ascending=[True, False])

    # Keep only the absolute best 'top_k_out' candidates per source document
    top_k_df = df_sorted.groupby('x').head(top_k_out)

    # Initialize empty output matrices (-1e9 for safe sorting later, -1 for missing IDs)
    D_out = np.full((N_A, top_k_out), -1e9, dtype=np.float32)
    I_out = np.full((N_A, top_k_out), -1, dtype=np.int64)

    # Fill the top results back into the matrices
    for x_val, group in top_k_df.groupby('x'):
        n_cands = len(group)
        D_out[x_val, :n_cands] = group['score'].values
        I_out[x_val, :n_cands] = group['y'].values
    return D_out, I_out