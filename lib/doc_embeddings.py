from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, Literal
import math
import numpy as np
import re
from collections import Counter

from transformers import AutoTokenizer  # pip: transformers

from .doc_split import split as split_sents
import logging, time
logger = logging.getLogger(__name__)

Doc = Union[str, Path]

# -----------------------
# 0) Helpers: stream I/O
# -----------------------
def iter_npy_stream(path: Union[str, Path]) -> Iterable[np.ndarray]:
    """
    Yield arrays saved sequentially by repeated np.save(fd, arr).
    """
    with open(path, "rb") as f:
        while True:
            try:
                arr = np.load(f, allow_pickle=False)
            except EOFError:
                break
            except ValueError:
                break
            yield arr


def load_doc2idx_tsv(path: Union[str, Path]) -> List[str]:
    """
    The file format is: idx<TAB>doc_path
    Returns doc paths in embedding order.
    """
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            _, p = line.split("\t", 1)
            out.append(p)
    return out


# -----------------------
# 1) Weight strategies
# -----------------------
def weights_lp(sents: List[str]) -> np.ndarray:
    """
    Length pooling: w_i proportional to sentence length in chars.
    Normalized to sum=1 within each doc.
    """
    if not sents:
        return np.zeros((0,), dtype=np.float32)
    lens = np.asarray([max(len(s), 1) for s in sents], dtype=np.float32)
    s = float(lens.sum())
    if s <= 0:
        return np.ones((len(sents),), dtype=np.float32)
    return (lens / s).astype(np.float32)


def build_df_labse_subwords(
    docs, langs, *, model_name="sentence-transformers/LaBSE", max_sent_len=10000
):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    df = Counter()
    N = len(docs)

    for p, lg in zip(docs, langs):
        sents = split_sents(str(p), lang=lg, max_len=max_sent_len)
        if not sents:
            continue

        doc_token_ids = set()
        
        for s in sents:
            enc = tok(
                s,
                add_special_tokens=False,
                truncation=True,
                max_length=tok.model_max_length,
            )
            doc_token_ids.update(enc["input_ids"])

        for tid in doc_token_ids:
            df[tid] += 1

    return N, dict(df), tok


def _idf(tid: int, N: int, df: Dict[int, int]) -> float:
    dfi = df.get(tid, 0)
    if dfi <= 0:
        return 0.0
    return 1.0 + math.log(N / float(dfi))


def weights_token_idf_labse(
    sents: List[str],
    *,
    tokenizer: object,
    N: int,
    df: Dict[int, int],
) -> np.ndarray:
    """
    Sentence weights from token-level IDF using LaBSE tokenizer:
      w_sent = mean_t idf(t)  , then normalize to sum=1 in doc
    """
    if not sents:
        return np.zeros((0,), dtype=np.float32)

    w = np.empty((len(sents),), dtype=np.float32)
    for i, s in enumerate(sents):
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        if not ids:
            w[i] = 0.0
            continue
        vals = [_idf(tid, N, df) for tid in ids]
        w[i] = float(np.mean(vals))
    # Nomalize
    sw = float(w.sum())
    if sw <= 0:
        return np.ones((len(sents),), dtype=np.float32)
    return (w / sw).astype(np.float32)


def weights_token_lidf_labse(
    sents: List[str],
    *,
    tokenizer: object,
    N: int,
    df: Dict[int, int],
) -> np.ndarray:
    """
    LIDF variant:
      w_sent = mean_t idf(t) * len(tokens)
    then normalize to sum=1 in doc.
    """
    if not sents:
        return np.zeros((0,), dtype=np.float32)

    w = np.empty((len(sents),), dtype=np.float32)
    for i, s in enumerate(sents):
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        if not ids:
            w[i] = 0.0
            continue
        idf_mean = float(np.mean([_idf(tid, N, df) for tid in ids]))
        w[i] = idf_mean * float(len(ids))
    # Nomalize
    sw = float(w.sum())
    if sw <= 0:
        return np.ones((len(sents),), dtype=np.float32)
    return (w / sw).astype(np.float32)


def apply_weights_repo_style(E: np.ndarray, w: Optional[np.ndarray]) -> np.ndarray:
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


ChunkMode = Literal["nonoverlap", "sliding"]
SelectMode = Literal["head", "uniform", "topk_weight"]

def _chunk_indices(n: int, k: int, *, mode: ChunkMode, stride: Optional[int]) -> List[np.ndarray]:
    if k <= 0:
        raise ValueError("sent_limit must be positive")
    if n == 0:
        return []

    if mode == "nonoverlap":
        # chunks: [0:k], [k:2k], ...
        return [np.arange(i, min(i + k, n)) for i in range(0, n, k)]

    # sliding
    if stride is None:
        stride = k  # default no overlap if not specified
    if stride <= 0:
        raise ValueError("stride must be positive")

    out = []
    i = 0
    while i < n:
        j = min(i + k, n)
        out.append(np.arange(i, j))
        if j == n:
            break
        i += stride
    return out

# -----------------------
# 3) Main API
# -----------------------
def compute_document_vectors_or_chunks(
    sent_emb_stream_path,
    doc2idx_path,
    *,
    langs,
    max_sent_len: Optional[int] = 10000,

    weights_strategy: int = 0,              # 0 none, 1 LP, 2 token-IDF, 3 token-LIDF
    labse_tokenizer_name: str = "sentence-transformers/LaBSE",

    merging_strategy: int = 3,              # 1 mean,2 median,3 max,4 split3-max,5 iterative,6 topk-mean
    topk: int = 16,

    sent_limit: Optional[int] = None,       # None => per-doc vector; int => chunk vectors per doc
    chunk_mode: ChunkMode = "nonoverlap",   # nonoverlap or sliding
    stride: Optional[int] = None,           # only for sliding

    # (optional) if you want to pick only some sentences BEFORE chunking (rarely needed)
    # select_limit: Optional[int] = None,
    # select_mode: SelectMode = "uniform",

    strict_mismatch: bool = True,
):
    docs = load_doc2idx_tsv(doc2idx_path)
    if len(docs) != len(langs):
        raise ValueError("langs length must equal docs length from doc2idx")

    # Precompute DF once if needed
    N = 0
    df = {}
    tokenizer = None
    if weights_strategy in (2, 3):
        N, df, tokenizer = build_df_labse_subwords(
            docs, langs, model_name=labse_tokenizer_name, max_sent_len=max_sent_len
        )

    outputs: List[np.ndarray] = []

    for i, E in enumerate(iter_npy_stream(sent_emb_stream_path)):
        E = np.asarray(E, dtype=np.float32)
        if E.ndim != 2:
            raise ValueError(f"Bad embedding array at doc #{i}: shape={E.shape}")

        sents = split_sents(str(docs[i]), lang=langs[i], max_len=max_sent_len)

        # align count
        if len(sents) != E.shape[0]:
            msg = f"Sentence count mismatch at doc #{i}: split={len(sents)} emb={E.shape[0]} doc={docs[i]}"
            if strict_mismatch:
                raise ValueError(msg)
            m = min(len(sents), E.shape[0])
            sents = sents[:m]
            E = E[:m]

        # compute weights for ALL sentences
        w = None
        if weights_strategy == 0:
            w = None
        elif weights_strategy == 1:
            w = weights_lp(sents)
        elif weights_strategy == 2:
            assert tokenizer is not None
            w = weights_token_idf_labse(sents, tokenizer=tokenizer, N=N, df=df)
        elif weights_strategy == 3:
            assert tokenizer is not None
            w = weights_token_lidf_labse(sents, tokenizer=tokenizer, N=N, df=df)
        else:
            raise ValueError("weights_strategy must be 0..3")

        # helper: merge one chunk
        def merge_one(E_chunk: np.ndarray, w_chunk: Optional[np.ndarray]) -> np.ndarray:
            if w_chunk is not None:
                if len(w_chunk) != E_chunk.shape[0]:
                    raise ValueError("weights/E mismatch inside chunk")
                E_chunk = apply_weights_repo_style(E_chunk, w_chunk)

            if merging_strategy == 1:
                return merge_mean(E_chunk)
            elif merging_strategy == 2:
                return merge_median(E_chunk)
            elif merging_strategy == 3:
                return merge_max(E_chunk)
            elif merging_strategy == 4:
                return merge_split3_max_concat(E_chunk)
            elif merging_strategy == 5:
                return merge_iterative_mean(E_chunk)
            elif merging_strategy == 6:
                return merge_topk_mean(E_chunk, k=topk)
            else:
                raise ValueError("merging_strategy must be 1..6")

        # Case A: per-doc
        if sent_limit is None:
            v = merge_one(E, w)
            outputs.append(v.astype(np.float32, copy=False))
            continue

        # Case B: per-doc chunks => (n_chunks, dim_or_3dim)
        idx_chunks = _chunk_indices(E.shape[0], int(sent_limit), mode=chunk_mode, stride=stride)
        if not idx_chunks:
            # empty doc => 1 empty chunk vector (so shape consistent)
            v0 = merge_one(E, w)
            outputs.append(v0[None, :].astype(np.float32))
            continue

        chunk_vecs = []
        for idxs in idx_chunks:
            Ec = E[idxs]
            wc = w[idxs] if w is not None else None
            vc = merge_one(Ec, wc)
            chunk_vecs.append(vc.astype(np.float32, copy=False))

        outputs.append(np.stack(chunk_vecs, axis=0).astype(np.float32))

    return outputs
