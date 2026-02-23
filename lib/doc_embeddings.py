from pathlib import Path
from typing import Dict, List, Optional, Union, Literal
import numpy as np
import torch
from transformers import AutoTokenizer

from .utils import iter_npy_stream, load_doc2idx_tsv
from .generate_embeddings import DFCollector, SplitMode
from .doc_split import sent_split_tkn, chunk_split

import logging
logger = logging.getLogger(__name__)

Doc = Union[str, Path]


# -----------------------
# 1) Weight strategies
# -----------------------
def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = w.astype(np.float32, copy=False)
    if w.size == 0:
        return w
    s = float(w.sum())
    if s <= 0:
        return (np.ones_like(w, dtype=np.float32) / float(w.size)).astype(np.float32)
    return (w / s).astype(np.float32)

def cal_sl(
    input_ids: torch.Tensor,                          # [B, L] or [L]
    dfc: DFCollector,
    *,
    attention_mask: torch.Tensor | None = None,       # [B, L] or [L]
    mode: str = "len",                                # "len" | "sqrt" | "log,                                        # L, sprt(L), log(L)  
    is_normalize: bool = False,
) -> np.ndarray:
    """
    Sentence-length weights per row (B,) for units/chunks in ONE document.
    Length is computed over real tokens (attention_mask==1).
    If is_normalize=True: normalize to sum=1 (fallback uniform).
    """
    skip = set(getattr(dfc, "skip_id", set()))
    if input_ids.numel() == 0:
        return np.zeros((0,), dtype=np.float32)

    # allow [L] -> [1, L]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

    B = int(input_ids.shape[0])
    w = np.zeros((B,), dtype=np.float32)

    for i in range(B):
        ids_row = input_ids[i]
        if attention_mask is not None:
            ids_row = ids_row[attention_mask[i].bool()]

        ids = [int(t) for t in ids_row.tolist() if int(t) not in skip]
        L = len(ids)

        if mode == "len":
            val = float(L)
        elif mode == "sqrt":
            val = float(np.sqrt(L))
        elif mode == "log":
            val = float(np.log1p(L))  # log(1+L)
        else:
            raise ValueError("mode must be 'len', 'sqrt', or 'log'")

        w[i] = val

    return _normalize_weights(w) if is_normalize else w.astype(np.float32)



def _idf(tid: int, N: int, df: Dict[int, int]) -> float:
    # Smooth TF-IDF style (always > 0)
    dfi = df.get(int(tid), 0)
    return 1.0 + np.log((N + 1.0) / (dfi + 1.0))

def cal_idf(
    input_ids: torch.Tensor,                          # [B, L] or [L]
    dfc: DFCollector,
    *,
    attention_mask: torch.Tensor | None = None,        # [B, L] or [L]
    is_normalize: bool = False,
) -> np.ndarray:
    """
    Weight per row (B,) for units/chunks in ONE document.
    Each row weight = mean(IDF(tokens_in_row)), excluding df.skip_id and padding via attention_mask.
    If is_normalize=True: normalize to sum=1 (fallback uniform).
    """
    N = int(dfc.N_docs)
    df_map = dfc.df
    skip = set(getattr(dfc, "skip_id", set()))

    if input_ids.numel() == 0:
        return np.zeros((0,), dtype=np.float32)

    # allow [L] -> [1, L]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

    B = int(input_ids.shape[0])
    w = np.zeros((B,), dtype=np.float32)

    for i in range(B):
        ids_row = input_ids[i]
        if attention_mask is not None:
            ids_row = ids_row[attention_mask[i].bool()]  # keep only real tokens

        ids = [int(t) for t in ids_row.tolist() if int(t) not in skip]
        if not ids:
            w[i] = 0.0
            continue

        vals = [_idf(t, N, df_map) for t in ids]
        w[i] = float(np.mean(vals))

    return _normalize_weights(w) if is_normalize else w.astype(np.float32)

def weights_lp(input_ids: torch.Tensor, dfc: DFCollector , attention_mask: torch.Tensor) -> np.ndarray:
    """
    Length pooling: w_i proportional to sentence length in chars.
    Normalized to sum=1 within each doc.
    """
    if input_ids.numel() == 0:
        return np.zeros((0,), dtype=np.float32)
    return cal_sl(input_ids, dfc, attention_mask = attention_mask, mode="len", is_normalize=True)

def weights_token_idf(input_ids: torch.Tensor, dfc: DFCollector , attention_mask: torch.Tensor) -> np.ndarray:
    """
    Sentence weights from token-level IDF using LaBSE tokenizer:
      w_sent = mean_t idf(t)  , then normalize to sum=1 in doc
    """
    if input_ids.numel() == 0:
        return np.zeros((0,), dtype=np.float32)
    return cal_idf(input_ids, dfc, attention_mask = attention_mask, is_normalize=True)


def weights_token_lidf(input_ids: torch.Tensor, dfc: DFCollector , attention_mask: torch.Tensor) -> np.ndarray:
    """
    LIDF variant:
      w_sent = mean_t idf(t) * len(tokens)
    then normalize to sum=1 in doc.
    """
    if input_ids.numel() == 0:
        return np.zeros((0,), dtype=np.float32)

    sl = cal_sl(input_ids, dfc, attention_mask=attention_mask, mode="len", is_normalize=False)
    token_idf = cal_idf(input_ids, dfc, attention_mask=attention_mask, is_normalize=False)
    w = sl * token_idf
    return _normalize_weights(w)

def apply_weights(E: np.ndarray, w: Optional[np.ndarray]) -> np.ndarray:
    """
    Repo-style: multiply each sentence vector by its scalar weight.
    Applies to ALL merge strategies.
    """
    if w is None:
        return E
    if E.shape[0] == 0:
        return E
    if w.shape[0] != E.shape[0]:
        raise ValueError(f"weights/E mismatch: {w.shape[0]} vs {E.shape[0]}")
    return (E * w[:, None]).astype(np.float32, copy=False)


# -----------------------
# 2) Merge strategies
# -----------------------
def merge_mean(E: np.ndarray) -> np.ndarray:
    if E.shape[0] == 0:
        return np.zeros((E.shape[1],), dtype=np.float32)
    return E.mean(axis=0, dtype=np.float32)

def merge_median(E: np.ndarray) -> np.ndarray:
    if E.shape[0] == 0:
        return np.zeros((E.shape[1],), dtype=np.float32)
    return np.median(E, axis=0).astype(np.float32)

def merge_max(E: np.ndarray) -> np.ndarray:
    if E.shape[0] == 0:
        return np.zeros((E.shape[1],), dtype=np.float32)
    return np.max(E, axis=0).astype(np.float32)

def merge_split3_max_concat(E: np.ndarray) -> np.ndarray:
    """
    split doc into 3 consecutive parts, max-pool each,
    then CONCAT => (3*dim,). (order-ish)
    """
    d = E.shape[1]
    if E.shape[0] == 0:
        return np.zeros((3 * d,), dtype=np.float32)
    n = E.shape[0]
    if n >= 3:
        a = n // 3
        b = 2 * n // 3
        parts = [E[:a], E[a:b], E[b:]]
    elif n == 2:
        parts = [E[:1], E[1:2], E]
    else:
        parts = [E, E, E]
    return np.concatenate([merge_max(p) for p in parts], axis=0).astype(np.float32)

def merge_iterative_mean(E: np.ndarray) -> np.ndarray:
    """
    iterative mean, order-ish.
    """
    if E.shape[0] == 0:
        return np.zeros((E.shape[1],), dtype=np.float32)
    r = E[0].astype(np.float32)
    for i in range(1, E.shape[0]):
        r = ((r + E[i]) / 2.0).astype(np.float32)
    return r

def merge_topk_mean(E: np.ndarray, k: int = 16) -> np.ndarray:
    """
    Order-insensitive: take top-k sentences by L2 norm, then mean.
    Often stronger than plain mean.
    """
    if E.shape[0] == 0:
        return np.zeros((E.shape[1],), dtype=np.float32)
    k = int(k)
    if k <= 0:
        return merge_mean(E)
    k = min(k, E.shape[0])
    norms = np.linalg.norm(E, axis=1)  # (n,)
    idx = np.argpartition(norms, -k)[-k:]
    return E[idx].mean(axis=0, dtype=np.float32)


# -----------------------
# 3) Main API
# -----------------------

WeightsStrategy = Literal["none", "lp", "idf", "lidf"]
MergeStrategy = Literal["mean", "median", "max", "split3-max", "iter-mean", "topk-mean"]

def compute_document_vectors_or_chunks(
    sent_emb_stream_path,
    doc2idx_path,
    df_path,
    split_mode: SplitMode,
    *,
    weights_strategy: WeightsStrategy = "none",
    merging_strategy: MergeStrategy = "mean",
    strict_mismatch: bool = True,
    # topk-mean parmas
    topk: int = 15,
    # sentence spliter params
    langs,
    max_sent_len: Optional[int] = 10000,
    # chunk spliter params
    chunk_size: int = 100,
    overlap_rate: int = 0.5,
    max_tokens: Optional[int] = None,                     # None => use model max_seq_length
    # tokens params      
    tok_name: str = "sentence-transformers/LaBSE",

):
    docs = load_doc2idx_tsv(doc2idx_path)
    if split_mode == "sentence":
        if len(docs) != len(langs):
            raise ValueError("langs length must equal docs length from doc2idx")

    dfc = DFCollector.from_npz(df_path)
    tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if max_tokens is None:
        ml = getattr(tok, "model_max_length", 512)
        max_tokens = int(ml if isinstance(ml, int) and ml < 10000 else 512)
    outputs: List[np.ndarray] = []

    for i, E in enumerate(iter_npy_stream(sent_emb_stream_path)):
        E = np.asarray(E, dtype=np.float32)
        if E.ndim != 2:
            raise ValueError(f"Bad embedding array at doc #{i}: shape={E.shape}")

        # get split
        if split_mode == "sentence":
            features = sent_split_tkn(docs[i], tok, langs[i], max_len=max_sent_len, max_tokens=max_tokens)
        else:
            overlap_size = int(chunk_size * overlap_rate)
            features = chunk_split(docs[i], tok, chunk_size, overlap_size, max_tokens=max_tokens)

        # align count
        n_units = int(features["input_ids"].shape[0])
        if n_units != E.shape[0]:
            msg = f"Unit count mismatch at doc #{i}: split={n_units} emb={E.shape[0]} doc={docs[i]}"
            if strict_mismatch:
                raise ValueError(msg)
            m = min(n_units, E.shape[0])
            # slice embeddings
            E = E[:m]
            # slice features
            features = {k: (v[:m] if hasattr(v, "shape") and v.shape[0] == n_units else v) for k, v in features.items()}

        # compute weights for ALL units
        w = None
        if weights_strategy == "none":
            w = None
        elif weights_strategy == "lp":
            w = weights_lp(features["input_ids"], dfc, attention_mask=features.get("attention_mask"))
        elif weights_strategy == "idf":
            w = weights_token_idf(features["input_ids"], dfc, attention_mask=features.get("attention_mask"))
        elif weights_strategy == "lidf":
            w = weights_token_lidf(features["input_ids"], dfc, attention_mask=features.get("attention_mask"))
        else:
            raise ValueError("weights_strategy must be \"none\", \"lp\", \"idf\", \"lidf\"")

        if w is not None:
            E = apply_weights(E, w)
        
        if merging_strategy == "mean":
            if w is not None:
                doc_vec = E.sum(axis=0, dtype=np.float32)   # weight is already normalize
            else:
                doc_vec = merge_mean(E)
        elif merging_strategy == "median":
            doc_vec = merge_median(E)
        elif merging_strategy == "max":
            doc_vec = merge_max(E)
        elif merging_strategy == "split3-max":
            doc_vec = merge_split3_max_concat(E)
        elif merging_strategy == "iter-mean":
            doc_vec = merge_iterative_mean(E)
        elif merging_strategy == "topk-mean":
            doc_vec = merge_topk_mean(E, k=topk)
        else:
            raise ValueError("merging_strategy must be \"mean\", \"median\", \"max\", \"split3-max\", \"iter-mean\", \"topk-mean\"")

        outputs.append(doc_vec.astype(np.float32))

    return outputs
