import numpy as np
import faiss

# -----------------------
# Retrival
# -----------------------
def faiss_cos_search(
    query: np.ndarray,
    database: np.ndarray,
    *,
    k: int = 5,
):
    query = np.asarray(query, dtype=np.float32)
    database = np.asarray(database, dtype=np.float32)

    if database.ndim != 2:
        raise ValueError(f"database must be 2D (m,d), got {database.shape}")
    if query.ndim == 1:
        query = query[None, :]
    elif query.ndim != 2:
        raise ValueError(f"query must be 1D or 2D, got {query.shape}")

    d = database.shape[1]
    if query.shape[1] != d:
        raise ValueError(f"dim mismatch: query d={query.shape[1]} vs database d={d}")

    # cosine = inner product of L2-normalized vectors
    Xn = query.copy()
    Yn = database.copy()
    faiss.normalize_L2(Xn)
    faiss.normalize_L2(Yn)

    index = faiss.IndexFlatIP(d)
    index.add(Yn)

    k_eff = min(int(k), Yn.shape[0])
    D, I = index.search(Xn, k_eff)
    return D, I


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