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
def bimax_score(
    seg_embs_1: torch.Tensor,  # [n, d]
    seg_embs_2: torch.Tensor,  # [m, d]
    *,
    normalize: bool = True,
    row_bs: int = 2048, # Row batch_size for mul
    col_bs: int = 2048, # Col batch_size for mul
) -> torch.Tensor:
    """
    BiMax = 0.5*( mean_i max_j cos(x_i,y_j) + mean_j max_i cos(y_j,x_i) )
    """
    if seg_embs_1.numel() == 0 or seg_embs_2.numel() == 0:
        return seg_embs_1.new_tensor(0.0)

    X = seg_embs_1
    Y = seg_embs_2
    if normalize:
        X = F.normalize(X, p=2, dim=1)
        Y = F.normalize(Y, p=2, dim=1)

    n = X.shape[0]
    m = Y.shape[0]

    # X -> Y
    row_max_parts = []
    for i in range(0, n, row_bs):
        sim = X[i:i + row_bs] @ Y.T
        row_max_parts.append(sim.max(dim=1).values)
    row_max = torch.cat(row_max_parts, dim=0)
    s1 = row_max.mean()

    # Y -> X
    col_max_parts = []
    for j in range(0, m, col_bs):
        sim = Y[j:j + col_bs] @ X.T
        col_max_parts.append(sim.max(dim=1).values)
    col_max = torch.cat(col_max_parts, dim=0)
    s2 = col_max.mean()

    return 0.5 * (s1 + s2)

def rerank_bimax(
    I: np.ndarray,  # [N, k]
    *,
    src_stream_path: Union[str, Path],
    tar_stream_path: Union[str, Path],
    device: str = "cpu",
    normalize: bool = True,
    row_bs: int = 2048,
    col_bs: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rerank FAISS candidates using 2-way BiMax.

    Returns:
      scores_sorted: [N, k] float32 (desc per row)
      I_sorted:      [N, k] int64   (candidates reordered per row)
    """
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
        A = torch.from_numpy(np.asarray(src_arr, dtype=np.float32)).to(device)
        # Get set of cand from I[qi]
        for cj in range(k):
            ti = int(I[qi, cj])
            B_np = tar_map.get(ti)
            if B_np is None:
                raise ValueError(f"Target idx {ti} not found in tar_stream (needed by query {qi})")
            B = torch.from_numpy(B_np).to(device)

            s = bimax_score(A, B, normalize=normalize, row_bs=row_bs, col_bs=col_bs)
            scores[qi, cj] = float(s.item())
    if count_src < N:
        raise ValueError(f"src_stream has only {count_src} docs but I has N={N}")
    # Rerank per query
    order = np.argsort(-scores, axis=1)
    rows = np.arange(N)[:, None]
    return scores[rows, order].astype(np.float32), I[rows, order].astype(np.int64)
