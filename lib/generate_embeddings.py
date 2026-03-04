from pathlib import Path
from typing import List, Optional, Sequence, Union, Iterable, Literal, Dict, Set
from collections import Counter
import os
import numpy as np

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from .doc_split import sent_split_tkn, chunk_split

import logging, time
logger = logging.getLogger(__name__)

Doc = Union[str, Path]
Mode = Literal["per_doc", "rolling"]
SplitMode = Literal["sentence", "chunk"]

class DFCollector:
    """
    Collect document frequency over tokenizer subword ids.
    df[token_id] = number of docs where token_id appears at least once.
    """
    def __init__(self, N_docs: int = 0, df: Counter | None = None, skip_id: Set[int] | None = None):
        self.N_docs = int(N_docs)
        self.df = df if df is not None else Counter()
        self.skip_id: Set[int] = set(skip_id) if skip_id is not None else set()

    def set_skip_id(self, skip_id: Set[int]) -> None:
        self.skip_id = set(skip_id)

    def update_from_input_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        """Update DF using token IDs from ONE document."""
        self.N_docs += 1

        if input_ids.numel() == 0:
            return

        if attention_mask is None:
            ids = input_ids.reshape(-1).tolist()
        else:
            ids = input_ids[attention_mask.bool()].tolist()

        for tid in set(ids):
            tid = int(tid)
            if tid not in self.skip_id:
                self.df[tid] += 1

    def save_npz(self, out_path: Union[str, Path]) -> None:
        """
        Save DF to a compressed NPZ:
          - N_docs: int64
          - token_ids: int32 array
          - df: int32 array (same length as token_ids)
          - skip_id: int32 array
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        token_ids = np.fromiter(self.df.keys(), dtype=np.int32, count=len(self.df))
        dfs = np.fromiter((self.df[k] for k in self.df.keys()), dtype=np.int32, count=len(self.df))
        skip = np.array(sorted(self.skip_id), dtype=np.int32)

        np.savez_compressed(out_path, N_docs=np.int64(self.N_docs), token_ids=token_ids, df=dfs, skip_id=skip)

        logger.info("df: saved | N_docs=%d | uniq_tokens=%d | skip=%d | path=%s",
                    self.N_docs, len(token_ids), skip.shape[0], str(out_path))

    @classmethod
    def from_npz(cls, path: Union[str, Path]) -> "DFCollector":
        """Load DFCollector from npz file."""
        path = Path(path)
        z = np.load(path)

        N = int(z["N_docs"])
        token_ids = z["token_ids"].astype(np.int32)
        dfs = z["df"].astype(np.int32)

        # backward-compatible: old files might not have skip_id
        if "skip_id" in z.files:
            skip_id = set(map(int, z["skip_id"].astype(np.int32).tolist()))
        else:
            skip_id = set()

        if token_ids.shape[0] != dfs.shape[0]:
            raise ValueError(f"Bad df file: token_ids and df length mismatch: {token_ids.shape[0]} vs {dfs.shape[0]}")

        c = Counter({int(t): int(v) for t, v in zip(token_ids, dfs)})
        obj = cls(N_docs=N, df=c, skip_id=skip_id)

        logger.info("df: loaded | N_docs=%d | uniq_tokens=%d | skip=%d | path=%s",
                    obj.N_docs, len(obj.df), len(obj.skip_id), str(path))
        return obj

def st_encode_features(
    model,
    features: Dict[str, torch.Tensor],
    *,
    batch_size: int = 32,
    convert_to_numpy: bool = True,
    normalize_embeddings: bool = False,
) -> np.ndarray:
    """
    features:
      - input_ids: LongTensor [N, L]
      - attention_mask: LongTensor [N, L]
      - (optional) token_type_ids: LongTensor [N, L]
    return: np.ndarray [N, D] float32 (or torch.Tensor if convert_to_numpy=False)
    """
    input_ids = features["input_ids"]
    N = int(input_ids.shape[0])

    if N == 0:
        dim = model.get_sentence_embedding_dimension()
        empty = torch.empty((0, dim), dtype=torch.float32)
        return empty.numpy() if convert_to_numpy else empty

    device = model.device
    outs = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            batch = {k: v[start:start+batch_size].to(device) for k, v in features.items()}
            out = model(batch)["sentence_embedding"]  # [b, D]
            if normalize_embeddings:
                out = torch.nn.functional.normalize(out, p=2, dim=1)
            outs.append(out.detach().cpu())

    emb = torch.cat(outs, dim=0).to(torch.float32)
    return emb.numpy() if convert_to_numpy else emb


def _get_file_size_bytes(p: Doc) -> int:
    try:
        return os.path.getsize(str(p))
    except OSError:
        return 0


def _iter_doc_batches_by_size(
    docs: Sequence[Doc],
    max_mbytes_per_batch: Optional[float],
) -> Iterable[List[Doc]]:
    """
    Yield batches of docs based on total file size (best-effort).
    """
    if max_mbytes_per_batch is None or max_mbytes_per_batch < 0:
        yield list(docs)
        return

    max_bytes = int(max_mbytes_per_batch * 1024 * 1024)
    batch: List[Doc] = []
    acc = 0

    for d in docs:
        sz = _get_file_size_bytes(d)
        if batch and acc + sz > max_bytes:
            yield batch
            batch = []
            acc = 0
        batch.append(d)
        acc += sz

    if batch:
        yield batch

# [TBD]
# def _process_rolling(
#     docs: Sequence[Doc],
#     langs: Sequence[str],
#     embeddings_output: Union[str, Path],
#     model_name: str,
#     *,
#     split_mode: SplitMode = "sentence",
#     batch_size: int = 32,                             # minibatch inside model.encode
#     sentence_splitting: bool = True,
#     max_sent_len: Optional[int] = 10000,              # None => unlimited
#     max_mbytes_per_batch: Optional[float] = 200.0,    # None/-1 => disable
#     max_nolines_per_batch: Optional[int] = 200000,    # None/-1 => disable
#     doc2idx_path: Optional[Union[str, Path]] = None,
#     # chunk split params (only when split_mode="chunk")
#     chunk_size: int = 384,
#     overlap_size: int = 96,
#     max_tokens: int = 510,
#     # DF
#     collect_df: bool = False,
#     df_out_path: Optional[Union[str, Path]] = None,
#     df_tokenizer_name: Optional[str] = None,          # None => use model_name tokenizer 
# ) -> None:
#     """
#     Streaming embeddings generation:
#       - writes a stream of .npy arrays (one per doc) into embeddings_output
#       - each saved array has shape (n_sentences_doc, dim) float32

#     Notes:
#       - This function batches docs by total input size, then further splits if total sentences exceed max_nolines_per_batch.
#       - It DOES NOT keep all corpus sentences in RAM, only per-batch.
#     """
#     if len(docs) != len(langs):
#         raise ValueError("docs and langs must have the same length")
#     if not docs:
#         raise ValueError("docs is empty")
#     if batch_size <= 0:
#         raise ValueError("batch_size must be positive")

#     if max_nolines_per_batch is not None and max_nolines_per_batch < 0:
#         max_nolines_per_batch = None
#     if max_mbytes_per_batch is not None and max_mbytes_per_batch < 0:
#         max_mbytes_per_batch = None

#     model = SentenceTransformer(model_name)
#     dim = model.get_sentence_embedding_dimension()

#     tok = None
#     if split_mode == "chunk" or collect_df:
#         tok_name = df_tokenizer_name or model_name
#         tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)

#     dfc = None
#     if collect_df: # Collect document frequency enable
#         logger.info("Collect document frequency enable")        
#         dfc = DFCollector()
    
#     t0 = time.time()
#     logger.info(
#         "rolling: start | docs=%d | model=%s | dim=%d | batch_size=%d | split_mode=%s",
#         len(docs), model_name, dim, batch_size, split_mode
#     )

#     out_path = Path(embeddings_output)
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     if doc2idx_path is None:
#         doc2idx_path = out_path.with_suffix(out_path.suffix + ".doc2idx.tsv")
#     doc2idx_path = Path(doc2idx_path)
#     doc2idx_path.parent.mkdir(parents=True, exist_ok=True)

#     doc_index = 0

#     with open(out_path, "wb") as out_fd, open(doc2idx_path, "w", encoding="utf-8") as map_fd:
#         # outer batching by bytes
#         start = 0
#         batch_no = 0
#         processed_docs = 0
#         for docs_batch in _iter_doc_batches_by_size(docs, max_mbytes_per_batch):
#             batch_no += 1
#             logger.info("rolling: outer_batch #%d | batch_docs=%d | processed=%d/%d",
#                         batch_no, len(docs_batch), processed_docs, len(docs))
#             # We may need to further sub-batch this docs_batch by sentence count (max_nolines_per_batch)
#             # To do that, we build a rolling sub-batch.
#             rolling_docs: List[Doc] = []
#             rolling_langs: List[str] = []
#             rolling_counts: List[int] = []
#             rolling_segments: List[str] = []
#             rolling_total = 0

#             # figure corresponding langs slice (preserve original order)
#             batch_len = len(docs_batch)
#             langs_batch = langs[start:start + batch_len]
#             start += batch_len

#             def flush_rolling():
#                 nonlocal doc_index, rolling_docs, rolling_langs, rolling_counts, rolling_segments, rolling_total, processed_docs, docs
#                 if not rolling_docs:
#                     return

#                 logger.info(
#                     "rolling: flush | docs=%d | total_sents=%d | doc_index=%d",
#                     len(rolling_docs), len(rolling_segments), doc_index
#                 )

#                 if logger.isEnabledFor(logging.DEBUG):
#                     logger.debug(
#                         "rolling: flush detail | min/max sent per doc = %d/%d",
#                         min(rolling_counts) if rolling_counts else 0,
#                         max(rolling_counts) if rolling_counts else 0,
#                     )

#                 # encode all sentences in this rolling batch
#                 if rolling_segments:
#                     embs = model.encode(
#                         rolling_segments,
#                         batch_size=batch_size,
#                         convert_to_numpy=True,
#                         normalize_embeddings=False,
#                         show_progress_bar=False,
#                     )

#                     if embs.dtype != np.float32:
#                         embs = embs.astype(np.float32)
                    
#                     expected = sum(rolling_counts)
#                     if embs.shape[0] != expected:
#                         logger.warning("rolling: encode shape mismatch | embs=%s | expected_rows=%d",
#                                     str(embs.shape), expected)
#                 else:
#                     embs = np.zeros((0, dim), dtype=np.float32)

#                 # split back per doc and save
#                 prev = 0
#                 for d, n in zip(rolling_docs, rolling_counts):
#                     emb_doc = embs[prev:prev + n]
#                     prev += n
#                     np.save(out_fd, emb_doc)
#                     map_fd.write(f"{doc_index}\t{str(d)}\n")
#                     doc_index += 1
                
#                 processed_docs += len(rolling_docs)
#                 logger.info("rolling: saved | processed=%d/%d | last_doc_index=%d | elapsed=%.1fs",
#                             processed_docs, len(docs), doc_index, time.time()-t0)
                
#                 # reset rolling
#                 rolling_docs = []
#                 rolling_langs = []
#                 rolling_counts = []
#                 rolling_segments = []
#                 rolling_total = 0

#             for d, lg in zip(docs_batch, langs_batch):
#                 if sentence_splitting:
#                     sents = sent_split(d, lang=lg, max_len=max_sent_len)  # List[str]
#                 else:
#                     # no split: treat each line as a sentence
#                     text = Path(d).read_text(encoding="utf-8", errors="replace")
#                     sents = [ln.strip() for ln in text.splitlines() if ln.strip()]
#                     if max_sent_len is not None:
#                         sents = [s[:max_sent_len] for s in sents]

#                 # clean
#                 sents = [s.strip() for s in sents if s and s.strip()]
#                 n = len(sents)
#                 if n == 0:
#                     logger.warning("rolling: empty doc after split | doc=%s lang=%s", str(d), lg)

#                 # debug memory
#                 if max_nolines_per_batch is not None and n > max_nolines_per_batch:
#                     logger.warning(
#                         "rolling: single doc exceeds max_nolines | n=%d > max=%d | doc=%s",
#                         n, max_nolines_per_batch, str(d)
#                     )

#                 # DEBUG
#                 if logger.isEnabledFor(logging.DEBUG) and doc_index < 3:
#                     logger.debug("rolling: sample doc | doc=%s lang=%s n_sent=%d", str(d), lg, n)
                
#                 # If adding this doc exceeds max_nolines_per_batch, flush rolling first
#                 if max_nolines_per_batch is not None and rolling_docs and (rolling_total + n > max_nolines_per_batch):
#                     logger.info("rolling: flush because max_nolines reached | rolling_total=%d add=%d max=%d",
#                                 rolling_total, n, max_nolines_per_batch)
#                     flush_rolling()

#                 # Add doc to rolling
#                 rolling_docs.append(d)
#                 rolling_langs.append(lg)
#                 rolling_counts.append(n)
#                 rolling_segments.extend(sents)
#                 rolling_total += n

#                 # If a single doc already exceeds max_nolines_per_batch, we flush immediately
#                 # (still keeps memory bounded to this single doc)
#                 if max_nolines_per_batch is not None and rolling_total >= max_nolines_per_batch:
#                     logger.info("rolling: flush because rolling_total >= max_nolines | rolling_total=%d max=%d",
#                                 rolling_total, max_nolines_per_batch)
#                     flush_rolling()

#             # flush any remaining docs in this outer batch
#             flush_rolling()
#             logger.info("rolling: done | total_docs=%d | elapsed=%.1fs | out=%s",
#                         doc_index, time.time()-t0, str(out_path))

def _process_per_doc(
    docs: Sequence[Doc],
    langs: Sequence[str],
    embeddings_output: Union[str, Path],
    model_name: str,
    *,
    doc2idx_path: Optional[Union[str, Path]] = None,
    split_mode: SplitMode = "sentence",
    batch_size: int = 32,                 # minibatch inside model.encode
    # sentence split params (only when split_mode="sentence")
    num_of_sent: int = 1,
    max_sent_len: Optional[int] = 10000,  # None => unlimited
    # chunk split params (only when split_mode="chunk")
    chunk_size: int = 100,
    overlap_rate: int = 0.5,
    max_tokens: Optional[int] = None,                     # None => use model max_seq_length  
    # DF
    collect_df: bool = False,
    df_out_path: Optional[Union[str, Path]] = None,
    df_tokenizer_name: Optional[str] = None,          # None => use model_name tokenizer 
) -> None:
    if len(docs) != len(langs):
        raise ValueError("docs and langs must have the same length")
    if not docs:
        raise ValueError("docs is empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if split_mode ==  "chunk" and (overlap_rate < 0 or overlap_rate >= 1):
        raise ValueError("overlap_rate is invalid for chunk split")  

    model = SentenceTransformer(model_name)
    max_tokens = max_tokens if max_tokens is not None else model.max_seq_length

    out_path = Path(embeddings_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if doc2idx_path is None:
        doc2idx_path = out_path.with_suffix(out_path.suffix + ".doc2idx.tsv")
    doc2idx_path = Path(doc2idx_path)
    doc2idx_path.parent.mkdir(parents=True, exist_ok=True)

    tok_name = df_tokenizer_name or model_name
    tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)

    dfc = None
    if collect_df:
        logger.info("Collect document frequency enable")        
        dfc = DFCollector()
    skip_id = []
    for x in (tok.pad_token_id, tok.cls_token_id, tok.sep_token_id, tok.bos_token_id, tok.eos_token_id):
        if x is not None:
            skip_id.append(int(x))
    dfc.set_skip_id(set(skip_id))

    with open(out_path, "wb") as out_fd, open(doc2idx_path, "w", encoding="utf-8") as map_fd:
        for idx, (doc, lang) in enumerate(zip(docs, langs)):
            doc = str(doc)

            # get split
            if split_mode == "sentence":
                features = sent_split_tkn(doc, tok, lang, max_len=max_sent_len, max_tokens=max_tokens, num_of_sent=num_of_sent)
            else:
                overlap_size = int(chunk_size * overlap_rate)
                features = chunk_split(doc, tok, chunk_size, overlap_size, max_tokens=max_tokens)

            # collect DF
            if dfc is not None:
                dfc.update_from_input_ids(
                    features["input_ids"],
                    attention_mask=features.get("attention_mask")
                )

            # encode -> (n_sent, dim)
            emb_doc = st_encode_features(
                model,
                features,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )

            # store one array per doc
            np.save(out_fd, emb_doc)
            map_fd.write(f"{idx}\t{doc}\n")

    # Save DFCollector
    if dfc is not None:
        if df_out_path is None:
            df_out_path = out_path.with_suffix(out_path.suffix + ".df.npz")
        dfc.save_npz(df_out_path)

def process(
    docs: Sequence[Doc],
    langs: Sequence[str],
    embeddings_output: Union[str, Path],
    model_name: str,
    *,
    split_mode: SplitMode = "sentence",
    mode: Mode = "per_doc",
    batch_size: int = 32,
    max_sent_len: Optional[int] = 10000,
    num_of_sent: int = 1,
    max_mbytes_per_batch: Optional[float] = 200.0,
    max_nolines_per_batch: Optional[int] = 200000,
    doc2idx_path: Optional[Union[str, Path]] = None,
    chunk_size: int = 100,
    overlap_rate: int = 0.5,
    max_tokens: Optional[int] = None,
    collect_df: bool = False,
    df_out_path: Optional[Union[str, Path]] = None,
    df_tokenizer_name: Optional[str] = None,
) -> None:
    """
    Public API: choose implementation by mode, but keep the same output format.
    """
    if mode == "per_doc":
        return _process_per_doc(
            docs, langs, embeddings_output, 
            split_mode=split_mode,
            model_name=model_name,
            batch_size=batch_size,
            max_sent_len=max_sent_len,
            num_of_sent=num_of_sent,
            doc2idx_path=doc2idx_path,
            chunk_size=chunk_size, overlap_rate= overlap_rate,
            max_tokens = max_tokens,
            collect_df = collect_df, df_out_path = df_out_path, df_tokenizer_name = df_tokenizer_name
        )
    # [TBD]
    # elif mode == "rolling":
    #     return _process_rolling(
    #         docs, langs, embeddings_output,
    #         model_name=model_name,
    #         batch_size=batch_size,
    #         max_sent_len=max_sent_len,
    #         max_mbytes_per_batch=max_mbytes_per_batch,
    #         max_nolines_per_batch=max_nolines_per_batch,
    #         doc2idx_path=doc2idx_path,
    #     )
    else:
        raise ValueError("mode must be 'per_doc' or 'rolling'")
