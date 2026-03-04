from typing import Literal
import numpy as np
import faiss

ScoreMode = Literal["csls", "margin1", "cosine"]

# -----------------------
# Retrieval
# -----------------------
def faiss_margin_search(
    query: np.ndarray,
    database: np.ndarray,
    *,
    k: int = 10,                    # final top-k returned (after rerank)
    retrieve_k: int | None = None,  # cosine candidates to consider (>=k). If None -> k
    knn_k: int | None = None,       # kNN size to compute baselines r_q and/or r_d (recommend 10 or 20)
    mode: ScoreMode = "csls",       # "cosine" | "margin1" | "csls"
    batch_size: int = 4096,
):
    """
    Returns (D, I) like faiss.Index.search but with different scoring:

    - mode="cosine": D = cosine/IP
    - mode="margin1": D = cos(q,y) - r_q(q)        (one-sided margin)
    - mode="csls":   D = 2*cos(q,y) - r_q(q) - r_db(y) (two-sided CSLS)
    """
    Q = np.asarray(query, dtype=np.float32)
    Y = np.asarray(database, dtype=np.float32)

    if Y.ndim != 2:
        raise ValueError(f"database must be 2D (m,d), got {Y.shape}")
    if Q.ndim == 1:
        Q = Q[None, :]
    elif Q.ndim != 2:
        raise ValueError(f"query must be 1D or 2D, got {Q.shape}")
    if Q.shape[1] != Y.shape[1]:
        raise ValueError(f"dim mismatch: query d={Q.shape[1]} vs database d={Y.shape[1]}")
    if Y.shape[0] == 0:
        return np.zeros((Q.shape[0], 0), np.float32), np.zeros((Q.shape[0], 0), np.int64)
    if k <= 0:
        return np.zeros((Q.shape[0], 0), np.float32), np.zeros((Q.shape[0], 0), np.int64)

    if retrieve_k is None or mode == "cosine":
        retrieve_k = k
    retrieve_k = min(int(retrieve_k), Y.shape[0])
    k = min(int(k), retrieve_k)

    if knn_k is None:
        knn_k = retrieve_k
    knn_k = min(int(knn_k), Y.shape[0])

    # normalize for cosine=IP
    Qn = np.ascontiguousarray(Q.copy())
    Yn = np.ascontiguousarray(Y.copy())
    faiss.normalize_L2(Qn)
    faiss.normalize_L2(Yn)

    d = Yn.shape[1]
    indexY = faiss.IndexFlatIP(d)
    indexY.add(Yn)

    # cosine candidates
    D_cos = np.empty((Qn.shape[0], retrieve_k), dtype=np.float32)
    I_cos = np.empty((Qn.shape[0], retrieve_k), dtype=np.int64)
    for s in range(0, Qn.shape[0], batch_size):
        e = min(s + batch_size, Qn.shape[0])
        D, I = indexY.search(Qn[s:e], retrieve_k)
        D_cos[s:e] = D
        I_cos[s:e] = I

    if mode == "cosine":
        return D_cos[:, :k].astype(np.float32), I_cos[:, :k].astype(np.int64)

    # r_q(q): mean top-knn_k cosine from q to database
    r_q = np.empty((Qn.shape[0],), dtype=np.float32)
    for s in range(0, Qn.shape[0], batch_size):
        e = min(s + batch_size, Qn.shape[0])
        D, _ = indexY.search(Qn[s:e], knn_k)
        r_q[s:e] = D.mean(axis=1)

    if mode == "margin1":
        D_score = D_cos - r_q[:, None]
    elif mode == "csls":
        # need r_db(y): mean top-knn_k cosine from y to queries
        indexQ = faiss.IndexFlatIP(d)
        indexQ.add(Qn)
        knn_k_y = min(int(knn_k), Qn.shape[0])

        r_db = np.empty((Yn.shape[0],), dtype=np.float32)
        for s in range(0, Yn.shape[0], batch_size):
            e = min(s + batch_size, Yn.shape[0])
            D, _ = indexQ.search(Yn[s:e], knn_k_y)
            r_db[s:e] = D.mean(axis=1)

        D_score = 2.0 * D_cos - r_q[:, None] - r_db[I_cos]
    else:
        raise ValueError("mode must be one of: 'cosine', 'margin1', 'csls'")

    order = np.argsort(-D_score, axis=1)
    rows = np.arange(Qn.shape[0])[:, None]
    I_sorted = I_cos[rows, order][:, :k]
    D_sorted = D_score[rows, order][:, :k].astype(np.float32)
    return D_sorted, I_sorted

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
                     - 'max': Returns the better of the two directions. Good for containment (M-N).
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
    I: np.ndarray,  # [N, k]
    *,
    src_stream_path: Union[str, Path],
    tar_stream_path: Union[str, Path],
    device: str = "cpu",
    normalize: bool = True,
    batch_size: int = 2048,
    trim_ratio: float = 1.0,
    aggregation: str = 'avg',
) -> Tuple[np.ndarray, np.ndarray]:
    I = np.asarray(I, dtype=np.int64)
    if I.ndim != 2:
        raise ValueError(f"I must be 2D (N,k), got {I.shape}")
    N, k = I.shape
    if N == 0 or k == 0:
        return np.zeros((N, k), dtype=np.float32), I
    
    # Preload target embedding
    cand_idx = set(int(x) for x in I.reshape(-1).tolist())
    tar_map = load_needed_from_stream(tar_stream_path, cand_idx)
    
    scores = np.zeros((N, k), dtype=np.float32)
    count_src = 0
    
    # stream through src docs; assume src stream order aligns with query row index
    for qi, src_arr in enumerate(iter_npy_stream(src_stream_path)):
        if qi >= N:
            raise ValueError(f"index is missmatch")
        count_src += 1
        
        A_np = np.asarray(src_arr, dtype=np.float32)
        if A_np.ndim != 2:
            raise ValueError(f"src doc #{qi} chunks must be 2D (n,d), got {A_np.shape}")
        
        A = torch.from_numpy(A_np).to(device)
        
        # Get set of cand from I[qi]
        for cj in range(k):
            ti = int(I[qi, cj])
            B_np = tar_map.get(ti)
            if B_np is None:
                raise ValueError(f"Target idx {ti} not found in tar_stream (needed by query {qi})")
            
            B = torch.from_numpy(B_np).to(device)
            
            s = bimax_score(A, B, 
                normalize=normalize, 
                trim_ratio=trim_ratio, 
                batch_size=batch_size, 
                aggregation=aggregation
            )
            scores[qi, cj] = float(s)

    if count_src < N:
        raise ValueError(f"src_stream has only {count_src} docs but I has N={N}")
    
    # Rerank per query
    order = np.argsort(-scores, axis=1)
    rows = np.arange(N)[:, None]
    
    return scores[rows, order].astype(np.float32), I[rows, order].astype(np.int64)
