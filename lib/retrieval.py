from typing import Literal, Optional, List, Union, Tuple
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import faiss

from .utils import AlignerIO


ScoreMode = Literal["csls", "margin1", "cosine"]

# -----------------------
# Retrieval
# -----------------------
class VotingParentRetriever:
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.chunk2doc: Optional[np.ndarray] = None
        self.meta_df: Optional[pd.DataFrame] = None
        self.dim: int = 0

    def load_target_corpus(self, lang_path: Union[str, Path]):
        """
        Phase 1: Load the entire Target Corpus into RAM and build the FAISS Index.
        
        Args:
            lang_path: Path to the target language folder.
        """
        lang_path = Path(lang_path)
        embeddings_dir = lang_path / "embeddings"
        meta_dir = lang_path / "metadata" 
        
        # Load metadata containing document paths and chunk counts
        self.meta_df = AlignerIO.load_metadata(meta_dir)
        
        all_chunks = []
        chunk2doc_list = []
        
        for _, row in self.meta_df.iterrows():
            emb_file = embeddings_dir / row['emb_file']
            if not emb_file.exists():
                print(f"[Warning] Missing file: {emb_file}")
                continue
                
            # Load the embedding matrix of the document. Shape: [n_chunks, dim]
            emb_matrix = AlignerIO.load_doc_embedding(embeddings_dir, row['emb_file'])
            all_chunks.append(emb_matrix)
            
            # Create the mapping: Repeat the 'doc_idx' for the number of chunks it has.
            # E.g., if doc 0 has 3 chunks, we append [0, 0, 0]
            chunk2doc_list.extend([row['doc_idx']] * int(row['n_chunks']))
            
        # Flatten the list of 2D matrices into a single massive 2D matrix
        X = np.vstack(all_chunks)
        
        # Convert the mapping list to a Numpy array for O(1) lookups later
        self.chunk2doc = np.array(chunk2doc_list, dtype=np.int64)
        self.dim = X.shape[1]
        
        print(f"[*] Building FAISS IndexFlatIP for {X.shape[0]} chunks...")
        
        # Use Inner Product (IP) since vectors are already L2 Normalized (equivalent to Cosine Similarity)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(X)
        
        print(f"[+] Done! Index is ready. Total target documents: {len(self.meta_df)}")

    def retrieve(self, query_chunks: np.ndarray, top_k_chunks: int = 5, top_k_docs: int = 10) -> List[Tuple[int, int]]:
        """
        Phase 2 & 3: Multi-Query Search and Plurality Voting (Hit Count).
        
        Args:
            query_chunks: Matrix [num_chunks_in_source, dim] of the source document.
            top_k_chunks: Number of nearest neighbors to retrieve for EACH source chunk.
            top_k_docs: Number of top parent documents to return based on vote count.
            
        Returns:
            A list of tuples: [(target_doc_idx, total_votes), ...]
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call load_target_corpus() first.")
            
        # FAISS strictly requires float32
        query_chunks = query_chunks.astype('float32')
        
        # 1. SEARCH: Query all source chunks simultaneously (Parallel Search)
        # 'I' is the Indices matrix. Shape: [num_chunks_in_source, top_k_chunks]
        # It contains the FAISS internal IDs of the most similar target chunks.
        scores, I = self.index.search(query_chunks, k=top_k_chunks)
        
        # 2. FILTER & FLATTEN: Remove missing neighbors (-1) and flatten the matrix
        valid_I = I[I != -1]
        
        # 3. MAP-BACK: Translate FAISS internal Chunk IDs to Parent Document IDs
        # This uses Numpy Advanced Indexing to map thousands of IDs instantly
        hit_docs = self.chunk2doc[valid_I]
        
        # 4. AGGREGATE (Hit Count): Count how many times each parent document was "hit"
        vote_counter = Counter(hit_docs.tolist())
        
        # Return the most frequently hit documents
        return vote_counter.most_common(top_k_docs)

    def get_doc_info(self, doc_idx: int) -> dict:
        """Utility function to retrieve metadata for a specific document index."""
        row = self.meta_df[self.meta_df['doc_idx'] == doc_idx].iloc[0]
        return row.to_dict()


def build_retrieval_matrix(
    source_lang_path: Union[str, Path],
    target_lang_path: Union[str, Path],
    top_k_chunks: int = 5,
    top_k_docs: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper for the Retriever to process all documents in the corpus.
    Returns matrices matching the FAISS API format: (Scores/Votes, Indices).
    """
    source_lang_path = Path(source_lang_path)
    source_meta_path = source_lang_path / "metadata"
    source_emb_dir = source_lang_path / "embeddings"

    # Load target index
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

from .utils import iter_npy_stream, load_needed_from_stream


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
