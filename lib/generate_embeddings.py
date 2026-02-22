from pathlib import Path
from typing import List, Optional, Sequence, Union, Iterable, Literal
import os
import numpy as np

from sentence_transformers import SentenceTransformer
from .doc_split import split as split_sents

import logging, time
logger = logging.getLogger(__name__)

Doc = Union[str, Path]
Mode = Literal["per_doc", "rolling"]

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

def _process_rolling(
    docs: Sequence[Doc],
    langs: Sequence[str],
    embeddings_output: Union[str, Path],
    model_name: str,
    *,
    batch_size: int = 32,                             # minibatch inside model.encode
    sentence_splitting: bool = True,
    max_sent_len: Optional[int] = 10000,              # None => unlimited
    max_mbytes_per_batch: Optional[float] = 200.0,    # None/-1 => disable
    max_nolines_per_batch: Optional[int] = 200000,    # None/-1 => disable
    doc2idx_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Streaming embeddings generation:
      - writes a stream of .npy arrays (one per doc) into embeddings_output
      - each saved array has shape (n_sentences_doc, dim) float32

    Notes:
      - This function batches docs by total input size, then further splits if total sentences exceed max_nolines_per_batch.
      - It DOES NOT keep all corpus sentences in RAM, only per-batch.
    """
    if len(docs) != len(langs):
        raise ValueError("docs and langs must have the same length")
    if not docs:
        raise ValueError("docs is empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if max_nolines_per_batch is not None and max_nolines_per_batch < 0:
        max_nolines_per_batch = None
    if max_mbytes_per_batch is not None and max_mbytes_per_batch < 0:
        max_mbytes_per_batch = None

    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()

    t0 = time.time()
    logger.info(
        "rolling: start | docs=%d | model=%s | dim=%d | batch_size=%d | split=%s | max_sent_len=%s | max_mbytes=%s | max_nolines=%s",
        len(docs), model_name, dim, batch_size, sentence_splitting, str(max_sent_len),
        str(max_mbytes_per_batch), str(max_nolines_per_batch)
    )

    out_path = Path(embeddings_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if doc2idx_path is None:
        doc2idx_path = out_path.with_suffix(out_path.suffix + ".doc2idx.tsv")
    doc2idx_path = Path(doc2idx_path)
    doc2idx_path.parent.mkdir(parents=True, exist_ok=True)

    doc_index = 0

    with open(out_path, "wb") as out_fd, open(doc2idx_path, "w", encoding="utf-8") as map_fd:
        # outer batching by bytes
        start = 0
        batch_no = 0
        processed_docs = 0
        for docs_batch in _iter_doc_batches_by_size(docs, max_mbytes_per_batch):
            batch_no += 1
            logger.info("rolling: outer_batch #%d | batch_docs=%d | processed=%d/%d",
                        batch_no, len(docs_batch), processed_docs, len(docs))
            # We may need to further sub-batch this docs_batch by sentence count (max_nolines_per_batch)
            # To do that, we build a rolling sub-batch.
            rolling_docs: List[Doc] = []
            rolling_langs: List[str] = []
            rolling_counts: List[int] = []
            rolling_sents: List[str] = []
            rolling_total = 0

            # figure corresponding langs slice (preserve original order)
            batch_len = len(docs_batch)
            langs_batch = langs[start:start + batch_len]
            start += batch_len

            def flush_rolling():
                nonlocal doc_index, rolling_docs, rolling_langs, rolling_counts, rolling_sents, rolling_total, processed_docs, docs
                if not rolling_docs:
                    return

                logger.info(
                    "rolling: flush | docs=%d | total_sents=%d | doc_index=%d",
                    len(rolling_docs), len(rolling_sents), doc_index
                )

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "rolling: flush detail | min/max sent per doc = %d/%d",
                        min(rolling_counts) if rolling_counts else 0,
                        max(rolling_counts) if rolling_counts else 0,
                    )

                # encode all sentences in this rolling batch
                if rolling_sents:
                    embs = model.encode(
                        rolling_sents,
                        batch_size=batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=False,
                        show_progress_bar=False,
                    )

                    if embs.dtype != np.float32:
                        embs = embs.astype(np.float32)
                    
                    expected = sum(rolling_counts)
                    if embs.shape[0] != expected:
                        logger.warning("rolling: encode shape mismatch | embs=%s | expected_rows=%d",
                                    str(embs.shape), expected)
                else:
                    embs = np.zeros((0, dim), dtype=np.float32)

                # split back per doc and save
                prev = 0
                for d, n in zip(rolling_docs, rolling_counts):
                    emb_doc = embs[prev:prev + n]
                    prev += n
                    np.save(out_fd, emb_doc)
                    map_fd.write(f"{doc_index}\t{str(d)}\n")
                    doc_index += 1
                
                processed_docs += len(rolling_docs)
                logger.info("rolling: saved | processed=%d/%d | last_doc_index=%d | elapsed=%.1fs",
                            processed_docs, len(docs), doc_index, time.time()-t0)
                
                # reset rolling
                rolling_docs = []
                rolling_langs = []
                rolling_counts = []
                rolling_sents = []
                rolling_total = 0

            for d, lg in zip(docs_batch, langs_batch):
                if sentence_splitting:
                    sents = split_sents(d, lang=lg, max_len=max_sent_len)  # List[str]
                else:
                    # no split: treat each line as a sentence
                    text = Path(d).read_text(encoding="utf-8", errors="replace")
                    sents = [ln.strip() for ln in text.splitlines() if ln.strip()]
                    if max_sent_len is not None:
                        sents = [s[:max_sent_len] for s in sents]

                # clean
                sents = [s.strip() for s in sents if s and s.strip()]
                n = len(sents)
                if n == 0:
                    logger.warning("rolling: empty doc after split | doc=%s lang=%s", str(d), lg)

                # debug memory
                if max_nolines_per_batch is not None and n > max_nolines_per_batch:
                    logger.warning(
                        "rolling: single doc exceeds max_nolines | n=%d > max=%d | doc=%s",
                        n, max_nolines_per_batch, str(d)
                    )

                # DEBUG
                if logger.isEnabledFor(logging.DEBUG) and doc_index < 3:
                    logger.debug("rolling: sample doc | doc=%s lang=%s n_sent=%d", str(d), lg, n)
                
                # If adding this doc exceeds max_nolines_per_batch, flush rolling first
                if max_nolines_per_batch is not None and rolling_docs and (rolling_total + n > max_nolines_per_batch):
                    logger.info("rolling: flush because max_nolines reached | rolling_total=%d add=%d max=%d",
                                rolling_total, n, max_nolines_per_batch)
                    flush_rolling()

                # Add doc to rolling
                rolling_docs.append(d)
                rolling_langs.append(lg)
                rolling_counts.append(n)
                rolling_sents.extend(sents)
                rolling_total += n

                # If a single doc already exceeds max_nolines_per_batch, we flush immediately
                # (still keeps memory bounded to this single doc)
                if max_nolines_per_batch is not None and rolling_total >= max_nolines_per_batch:
                    logger.info("rolling: flush because rolling_total >= max_nolines | rolling_total=%d max=%d",
                                rolling_total, max_nolines_per_batch)
                    flush_rolling()

            # flush any remaining docs in this outer batch
            flush_rolling()
            logger.info("rolling: done | total_docs=%d | elapsed=%.1fs | out=%s",
                        doc_index, time.time()-t0, str(out_path))

def _process_per_doc(
    docs: Sequence[Doc],
    langs: Sequence[str],
    embeddings_output: Union[str, Path],
    model_name: str,
    *,
    batch_size: int = 32,                 # minibatch inside model.encode
    sentence_splitting: bool = True,
    max_sent_len: Optional[int] = 10000,  # None => unlimited
    doc2idx_path: Optional[Union[str, Path]] = None,
) -> None:
    if len(docs) != len(langs):
        raise ValueError("docs and langs must have the same length")
    if not docs:
        raise ValueError("docs is empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()

    out_path = Path(embeddings_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if doc2idx_path is None:
        doc2idx_path = out_path.with_suffix(out_path.suffix + ".doc2idx.tsv")
    doc2idx_path = Path(doc2idx_path)
    doc2idx_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as out_fd, open(doc2idx_path, "w", encoding="utf-8") as map_fd:
        for idx, (doc, lang) in enumerate(zip(docs, langs)):
            doc = str(doc)

            # 1) get sentences
            if sentence_splitting:
                sents = split_sents(doc, lang=lang, max_len=max_sent_len)  # expects List[str]
            else:
                text = Path(doc).read_text(encoding="utf-8", errors="replace")
                sents = [ln.strip() for ln in text.splitlines() if ln.strip()]
                if max_sent_len is not None:
                    sents = [s[:max_sent_len] for s in sents]

            sents = [s.strip() for s in sents if s and s.strip()]

            # 2) encode -> (n_sent, dim)
            if not sents:
                emb_doc = np.zeros((0, dim), dtype=np.float32)
            else:
                emb_doc = model.encode(
                    sents,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )
                if emb_doc.dtype != np.float32:
                    emb_doc = emb_doc.astype(np.float32)

            # 3) store one array per doc
            np.save(out_fd, emb_doc)
            map_fd.write(f"{idx}\t{doc}\n")

def process(
    docs: Sequence[Doc],
    langs: Sequence[str],
    embeddings_output: Union[str, Path],
    model_name: str,
    *,
    mode: Mode = "per_doc",
    batch_size: int = 32,
    sentence_splitting: bool = True,
    max_sent_len: Optional[int] = 10000,
    max_mbytes_per_batch: Optional[float] = 200.0,
    max_nolines_per_batch: Optional[int] = 200000,
    doc2idx_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Public API: choose implementation by mode, but keep the same output format.
    """
    if mode == "per_doc":
        return _process_per_doc(
            docs, langs, embeddings_output,
            model_name=model_name,
            batch_size=batch_size,
            sentence_splitting=sentence_splitting,
            max_sent_len=max_sent_len,
            doc2idx_path=doc2idx_path,
        )
    elif mode == "rolling":
        return _process_rolling(
            docs, langs, embeddings_output,
            model_name=model_name,
            batch_size=batch_size,
            sentence_splitting=sentence_splitting,
            max_sent_len=max_sent_len,
            max_mbytes_per_batch=max_mbytes_per_batch,
            max_nolines_per_batch=max_nolines_per_batch,
            doc2idx_path=doc2idx_path,
        )
    else:
        raise ValueError("mode must be 'per_doc' or 'rolling'")