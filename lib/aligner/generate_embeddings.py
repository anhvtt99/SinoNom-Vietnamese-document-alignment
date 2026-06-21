import argparse
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union, Literal, Dict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from lib.utils import AlignerIO, cuda_available
from lib.aligner.doc_split import sent_split_tkn, chunk_split


Doc = Union[str, Path]
Mode = Literal["per_doc", "rolling"]
SplitMode = Literal["sentence", "chunk"]

def st_encode_features(
    model: SentenceTransformer,
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
      - optional token_type_ids: LongTensor [N, L]

    return:
      np.ndarray [N, D] float32 if convert_to_numpy=True
      torch.Tensor [N, D] float32 if convert_to_numpy=False
    """
    input_ids = features["input_ids"]
    N = int(input_ids.shape[0])

    if N == 0:
        dim = model.get_sentence_embedding_dimension()
        empty = torch.empty((0, dim), dtype=torch.float32)
        return empty.numpy() if convert_to_numpy else empty

    device = model.device
    outs = []

    with torch.inference_mode():
        for start in range(0, N, batch_size):
            batch = {
                k: v[start:start + batch_size].to(device, non_blocking=True)
                for k, v in features.items()
            }

            out = model(batch)["sentence_embedding"]  # [b, D]

            if normalize_embeddings:
                out = torch.nn.functional.normalize(out, p=2, dim=1)

            # Keep outputs on GPU; copy to CPU only once after all mini-batches.
            outs.append(out.detach())

    # Concat on GPU.
    emb = torch.cat(outs, dim=0).to(torch.float32)

    if convert_to_numpy:
        # One GPU -> CPU copy per encode group.
        emb = emb.cpu()
        return emb.numpy()

    return emb

def concat_feature_batches(
    features_list: Sequence[Dict[str, torch.Tensor]],
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Concatenate tokenized feature dicts from multiple docs.

    ``pad_token_id`` should be the tokenizer's real pad id (0 for BERT/LaBSE,
    1 for XLM-R/M3); callers pass it so both architectures pad correctly. Padded
    positions are masked out via attention_mask regardless, so this only keeps
    the input_ids tensor faithful.

    Handles different sequence lengths by padding all tensors to the same L.

    Example:
      doc1 input_ids: [N1, 160]
      doc2 input_ids: [N2, 94]

    After padding:
      doc1 input_ids: [N1, 160]
      doc2 input_ids: [N2, 160]

    Then concat:
      merged input_ids: [N1 + N2, 160]
    """
    if not features_list:
        raise ValueError("features_list is empty")

    keys = list(features_list[0].keys())

    max_len = max(
        int(features["input_ids"].shape[1])
        for features in features_list
    )

    padded_features = {k: [] for k in keys}

    for features in features_list:
        cur_len = int(features["input_ids"].shape[1])
        pad_len = max_len - cur_len

        for k in keys:
            x = features[k]

            if pad_len > 0:
                if k == "input_ids":
                    pad_value = pad_token_id
                elif k == "attention_mask":
                    pad_value = 0
                elif k == "token_type_ids":
                    pad_value = 0
                else:
                    pad_value = 0

                pad = torch.full(
                    (x.shape[0], pad_len),
                    pad_value,
                    dtype=x.dtype,
                    device=x.device,
                )

                x = torch.cat([x, pad], dim=1)

            padded_features[k].append(x)

    return {
        k: torch.cat(v, dim=0)
        for k, v in padded_features.items()
    }


def _encode_and_save_feature_group(
    *,
    group_features: Sequence[Dict[str, torch.Tensor]],
    group_meta: Sequence[Dict],
    model: SentenceTransformer,
    embeddings_dir: Path,
    chunk_meta_dir: Path,
    doc2idx_data: list,
    batch_size: int,
    normalize_embeddings: bool,
) -> None:
    """
    Encode a group of already-tokenized docs, then split embeddings back per doc.
    """
    if not group_features:
        return

    # Use the model's real pad id (BERT/LaBSE=0, XLM-R/M3=1). Padded positions are
    # masked out by attention_mask, but keeping the correct id avoids surprises.
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    merged_features = concat_feature_batches(group_features, pad_token_id=pad_id)

    emb_all = st_encode_features(
        model,
        merged_features,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )

    start = 0

    for meta in group_meta:
        n_chunks = meta["n_chunks"]
        doc = meta["doc"]
        doc_idx = meta["doc_idx"]
        records = meta["records"]

        emb_doc = emb_all[start:start + n_chunks]
        start += n_chunks

        original_stem = doc.stem
        emb_file_name = f"{original_stem}.npy"

        np.save(embeddings_dir / emb_file_name, emb_doc)

        # One chunk metadata record per embedding row.
        assert len(records) == emb_doc.shape[0], (
            f"Chunk metadata mismatch for {doc}: {len(records)} records "
            f"vs {emb_doc.shape[0]} embedding rows"
        )
        meta_file_name = f"{original_stem}.jsonl"
        AlignerIO.save_chunk_metadata(chunk_meta_dir, meta_file_name, records)

        doc2idx_data.append({
            "doc_idx": doc_idx,
            "file_path": str(doc),
            "emb_file": emb_file_name,
            "n_chunks": emb_doc.shape[0],
            "chunk_meta_file": meta_file_name,
        })

    assert start == emb_all.shape[0], (
        f"Embedding split mismatch: consumed {start}, total {emb_all.shape[0]}"
    )

def _process_per_doc(
    docs: Sequence[Doc],
    embeddings_output: Union[str, Path],
    lang: str,
    model: SentenceTransformer,
    *,
    normalize_embeddings: bool = True,
    split_mode: SplitMode = "sentence",
    batch_size: int = 32,                 # minibatch inside model.encode
    # sentence split params (only when split_mode="sentence")
    num_of_sent: int = 1,
    overlap_sent: int = 0,
    max_sent_len: Optional[int] = 10000,  # None => unlimited
    # chunk split params (only when split_mode="chunk")
    chunk_size: int = 100,
    overlap_rate: int = 0.5,
) -> None:
    if not docs:
        raise ValueError("docs is empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if split_mode ==  "chunk" and (overlap_rate < 0 or overlap_rate >= 1):
        raise ValueError("overlap_rate is invalid for chunk split")  

    tok = model.tokenizer
    model_max_len = model.max_seq_length  # model's hard sequence-length limit (auto)

    if split_mode == "chunk" and chunk_size > model_max_len - 2:
        effective = model_max_len - 2
        warnings.warn(
            f"chunk_size={chunk_size} exceeds model max_seq_length-2={model_max_len - 2}; "
            f"clamping to {effective}. config.json and output directory will reflect the "
            f"effective value.",
            stacklevel=2,
        )
        chunk_size = effective

    if split_mode == "sentence":
        config_tag = f'{split_mode}_n{num_of_sent}_o{overlap_sent}'
    else:
        config_tag = f'{split_mode}_s{chunk_size}_r{overlap_rate}'

    # Setup Path
    print(f"[{config_tag}] Constructing embedding output file system...")
    lang_path = Path(embeddings_output) / config_tag / lang
    embeddings_dir = lang_path / "embeddings"
    meta_dir = lang_path / "metadata"
    chunk_meta_dir = lang_path / "chunk_metadata"

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    chunk_meta_dir.mkdir(parents=True, exist_ok=True)

    doc2idx_data = []

    print(f"[{config_tag}] Processing {len(docs)} documents for lang: {lang}...")
    for idx, (doc) in enumerate(docs):
        # get split (with one metadata record per embedding row)
        if split_mode == "sentence":
            features, records = sent_split_tkn(doc, tok, lang, max_len=max_sent_len, max_tokens=model_max_len, num_of_sent=num_of_sent, overlap_sent=overlap_sent, return_metadata=True)
        else:
            overlap_size = int(chunk_size * overlap_rate)
            features, records = chunk_split(doc, tok, chunk_size, overlap_size, max_tokens=model_max_len, return_metadata=True)

        emb_doc = st_encode_features(
            model,
            features,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )
        original_stem = doc.stem
        emb_file_name = f"{original_stem}.npy"
        np.save(embeddings_dir / emb_file_name, emb_doc)

        # One chunk metadata record per embedding row.
        assert len(records) == emb_doc.shape[0], (
            f"Chunk metadata mismatch for {doc}: {len(records)} records "
            f"vs {emb_doc.shape[0]} embedding rows"
        )
        meta_file_name = f"{original_stem}.jsonl"
        AlignerIO.save_chunk_metadata(chunk_meta_dir, meta_file_name, records)

        doc2idx_data.append({
            "doc_idx": idx,
            "file_path": str(doc),
            "emb_file": emb_file_name,
            "n_chunks": emb_doc.shape[0],
            "chunk_meta_file": meta_file_name,
        })
    # Save doc2idx.tsv
    AlignerIO.save_metadata(meta_dir, doc2idx_data)
    
    # Save config.json
    if split_mode == "sentence":
        config_data = {
                "model_name": model.model_card_data.base_model,
                "normalize": normalize_embeddings,
                "split_mode": split_mode,
                "num_of_sent": num_of_sent,
                "overlap_sent": overlap_sent,
                "max_sent_len": max_sent_len,
        }
    else:
        config_data = {
                "model_name": model.model_card_data.base_model,
                "normalize": normalize_embeddings,
                "split_mode": split_mode,
                "chunk_size": chunk_size,
                "overlap_rate": overlap_rate,
                "model_max_len": model_max_len,
        }
    AlignerIO.save_config(lang_path, config_data)

    print(f"Done! Embeddings saved to {embeddings_dir}")

def _process_per_batch(
    docs: Sequence[Doc],
    embeddings_output: Union[str, Path],
    lang: str,
    model: SentenceTransformer,
    *,
    normalize_embeddings: bool = True,
    split_mode: SplitMode = "sentence",
    batch_size: int = 64,
    encode_group_size: int = 2048,
    # sentence split params
    num_of_sent: int = 1,
    overlap_sent: int = 0,
    max_sent_len: Optional[int] = 10000,
    # chunk split params
    chunk_size: int = 100,
    overlap_rate: float = 0.5,
) -> None:
    """
    Faster embedding generation by batching chunks across multiple documents.

    This keeps output format identical to _process_per_doc:
      output/config_tag/lang/embeddings/<doc_stem>.npy
      output/config_tag/lang/metadata/doc2idx.tsv
      output/config_tag/lang/metadata/config.json

    Main difference:
      - split/tokenize docs one by one
      - accumulate chunks from many docs
      - encode a large group
      - split embeddings back per document
    """
    if not docs:
        raise ValueError("docs is empty")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if encode_group_size <= 0:
        raise ValueError("encode_group_size must be positive")

    if split_mode == "chunk" and (overlap_rate < 0 or overlap_rate >= 1):
        raise ValueError("overlap_rate is invalid for chunk split")

    tok = model.tokenizer
    model_max_len = model.max_seq_length  # model's hard sequence-length limit (auto)

    if split_mode == "chunk" and chunk_size > model_max_len - 2:
        effective = model_max_len - 2
        warnings.warn(
            f"chunk_size={chunk_size} exceeds model max_seq_length-2={model_max_len - 2}; "
            f"clamping to {effective}. config.json and output directory will reflect the "
            f"effective value.",
            stacklevel=2,
        )
        chunk_size = effective

    if split_mode == "sentence":
        config_tag = f"{split_mode}_n{num_of_sent}_o{overlap_sent}"
    else:
        config_tag = f"{split_mode}_s{chunk_size}_r{overlap_rate}"

    print(f"[{config_tag}] Constructing embedding output file system...")
    lang_path = Path(embeddings_output) / config_tag / lang
    embeddings_dir = lang_path / "embeddings"
    meta_dir = lang_path / "metadata"
    chunk_meta_dir = lang_path / "chunk_metadata"

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    chunk_meta_dir.mkdir(parents=True, exist_ok=True)

    doc2idx_data = []

    group_features = []
    group_meta = []
    group_n_chunks = 0

    print(
        f"[{config_tag}] Processing {len(docs)} documents for lang: {lang} "
        f"with grouped batching..."
    )

    for idx, doc in enumerate(docs):
        if split_mode == "sentence":
            features, records = sent_split_tkn(
                doc,
                tok,
                lang,
                max_len=max_sent_len,
                max_tokens=model_max_len,
                num_of_sent=num_of_sent,
                overlap_sent=overlap_sent,
                return_metadata=True,
            )
        else:
            overlap_size = int(chunk_size * overlap_rate)
            features, records = chunk_split(
                doc,
                tok,
                chunk_size,
                overlap_size,
                max_tokens=model_max_len,
                return_metadata=True,
            )

        n_chunks = int(features["input_ids"].shape[0])

        if n_chunks == 0:
            emb_file_name = f"{doc.stem}.npy"
            dim = model.get_sentence_embedding_dimension()
            np.save(
                embeddings_dir / emb_file_name,
                np.empty((0, dim), dtype=np.float32),
            )

            # Still write an (empty) metadata file so every doc has one.
            meta_file_name = f"{doc.stem}.jsonl"
            AlignerIO.save_chunk_metadata(chunk_meta_dir, meta_file_name, [])

            doc2idx_data.append({
                "doc_idx": idx,
                "file_path": str(doc),
                "emb_file": emb_file_name,
                "n_chunks": 0,
                "chunk_meta_file": meta_file_name,
            })
            continue

        group_features.append(features)
        group_meta.append({
            "doc_idx": idx,
            "doc": doc,
            "n_chunks": n_chunks,
            "records": records,
        })
        group_n_chunks += n_chunks

        if group_n_chunks >= encode_group_size:
            print(
                f"[batch] Encoding {len(group_meta)} docs, "
                f"{group_n_chunks} chunks..."
            )

            _encode_and_save_feature_group(
                group_features=group_features,
                group_meta=group_meta,
                model=model,
                embeddings_dir=embeddings_dir,
                chunk_meta_dir=chunk_meta_dir,
                doc2idx_data=doc2idx_data,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
            )

            group_features = []
            group_meta = []
            group_n_chunks = 0

    # Flush final group
    if group_features:
        print(
            f"[batch] Encoding final {len(group_meta)} docs, "
            f"{group_n_chunks} chunks..."
        )

        _encode_and_save_feature_group(
            group_features=group_features,
            group_meta=group_meta,
            model=model,
            embeddings_dir=embeddings_dir,
            chunk_meta_dir=chunk_meta_dir,
            doc2idx_data=doc2idx_data,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
        )

    # Save doc2idx.tsv
    AlignerIO.save_metadata(meta_dir, doc2idx_data)

    # Save config.json
    if split_mode == "sentence":
        config_data = {
            "model_name": model.model_card_data.base_model,
            "normalize": normalize_embeddings,
            "process_mode": "per_batch",
            "split_mode": split_mode,
            "num_of_sent": num_of_sent,
            "overlap_sent": overlap_sent,
            "max_sent_len": max_sent_len,
            "batch_size": batch_size,
            "encode_group_size": encode_group_size,
        }
    else:
        config_data = {
            "model_name": model.model_card_data.base_model,
            "normalize": normalize_embeddings,
            "process_mode": "per_batch",
            "split_mode": split_mode,
            "chunk_size": chunk_size,
            "overlap_rate": overlap_rate,
            "model_max_len": model_max_len,
            "batch_size": batch_size,
            "encode_group_size": encode_group_size,
        }

    AlignerIO.save_config(lang_path, config_data)

    print(f"Done! Embeddings saved to {embeddings_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for a directory of documents.")
    
    # Core arguments
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory containing input text files.")
    parser.add_argument("--file_ext", type=str, default=".txt", help="Extension of the files to process (default: .txt).")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the embeddings output.")
    parser.add_argument("--lang", type=str, required=True, help="Language code (e.g., 'vi', 'zh').")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HuggingFace model name or local path.")
    
    # Configuration arguments
    parser.add_argument("--split_mode", type=str, choices=["sentence", "chunk"], default="sentence", help="Mode to split documents.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for model inference.")
    parser.add_argument("--no_normalize", action="store_true", help="Pass this flag to disable L2 normalization of embeddings.")
    
    # Sentence split specific
    parser.add_argument("--num_of_sent", type=int, default=1, help="Number of sentences per split (if split_mode=sentence).")
    parser.add_argument("--overlap_sent", type=int, default=0, help="Number of overlapping sentences (if split_mode=sentence).")
    parser.add_argument("--max_sent_len", type=int, default=10000, help="Maximum sentence length in characters.")
    
    # Chunk split specific
    parser.add_argument("--chunk_size", type=int, default=100,
                        help="Chunk size in tokens (if split_mode=chunk). The model's sequence-length "
                             "limit is applied automatically; chunks exceeding it are clamped with a warning.")
    parser.add_argument("--overlap_rate", type=float, default=0.5, help="Overlap rate for chunks (0.0 to 1.0).")

    # Process mode config
    parser.add_argument(
        "--process_mode",
        type=str,
        choices=["per_doc", "per_batch"],
        default="per_batch",
        help="per_doc = encode each document separately; per_batch = batch chunks across documents.",
    )
    parser.add_argument(
        "--encode_group_size",
        type=int,
        default=2048,
        help="Number of chunks to accumulate before encoding when process_mode=per_batch.",
    )

    args = parser.parse_args()

    # 1. Gather documents
    input_path = Path(args.input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Input directory {args.input_dir} does not exist or is not a directory.")
        return

    docs = sorted(list(input_path.glob(f"*{args.file_ext}")))
    if not docs:
        print(f"Warning: No files ending with {args.file_ext} found in {args.input_dir}.")
        return

    print("\n[INFO] Checking hardware...")
    
    is_gpu = cuda_available(verbose=True)
    
    device = "cuda" if is_gpu else "cpu"
    print("------------------------------------------\n")

    # 2. Load the model
    print(f"Loading model: {args.model_name_or_path}...")
    model = SentenceTransformer(args.model_name_or_path, device=device)

    # 3. Process
    normalize_flag = not args.no_normalize
    
    process_fn = _process_per_batch if args.process_mode == "per_batch" else _process_per_doc

    if args.process_mode == "per_batch":
        process_fn(
            docs=docs,
            embeddings_output=args.output_dir,
            lang=args.lang,
            model=model,
            normalize_embeddings=normalize_flag,
            split_mode=args.split_mode,
            batch_size=args.batch_size,
            encode_group_size=args.encode_group_size,
            num_of_sent=args.num_of_sent,
            overlap_sent=args.overlap_sent,
            max_sent_len=args.max_sent_len,
            chunk_size=args.chunk_size,
            overlap_rate=args.overlap_rate,
        )
    else:
        process_fn(
            docs=docs,
            embeddings_output=args.output_dir,
            lang=args.lang,
            model=model,
            normalize_embeddings=normalize_flag,
            split_mode=args.split_mode,
            batch_size=args.batch_size,
            num_of_sent=args.num_of_sent,
            overlap_sent=args.overlap_sent,
            max_sent_len=args.max_sent_len,
            chunk_size=args.chunk_size,
            overlap_rate=args.overlap_rate,
        )

def run_embedding_generation(
    model: "SentenceTransformer",
    input_dir: str,
    output_dir: str,
    lang: str,
    split_mode: str = "sentence",
    batch_size: int = 128,
    normalize_embeddings: bool = True,
    process_mode: str = "per_batch",
    encode_group_size: int = 2048,
    num_of_sent: int = 1,
    overlap_sent: int = 0,
    max_sent_len: int = 10000,
    chunk_size: int = 100,
    overlap_rate: float = 0.5,
    file_ext: str = ".txt",
) -> None:
    """
    In-process entry point — model passed in pre-loaded.
    Same logic as main() without the model-loading block.
    Designed to be called from Streamlit using @st.cache_resource models.
    """
    input_path = Path(input_dir)
    docs = sorted(list(input_path.glob(f"*{file_ext}")))

    if not docs:
        print(f"Warning: No {file_ext} files found in {input_dir}")
        return

    process_fn = _process_per_batch if process_mode == "per_batch" else _process_per_doc

    if process_mode == "per_batch":
        process_fn(
            docs=docs,
            embeddings_output=output_dir,
            lang=lang,
            model=model,
            normalize_embeddings=normalize_embeddings,
            split_mode=split_mode,
            batch_size=batch_size,
            encode_group_size=encode_group_size,
            num_of_sent=num_of_sent,
            overlap_sent=overlap_sent,
            max_sent_len=max_sent_len,
            chunk_size=chunk_size,
            overlap_rate=overlap_rate,
        )
    else:
        process_fn(
            docs=docs,
            embeddings_output=output_dir,
            lang=lang,
            model=model,
            normalize_embeddings=normalize_embeddings,
            split_mode=split_mode,
            batch_size=batch_size,
            num_of_sent=num_of_sent,
            overlap_sent=overlap_sent,
            max_sent_len=max_sent_len,
            chunk_size=chunk_size,
            overlap_rate=overlap_rate,
        )


if __name__ == "__main__":
    main()