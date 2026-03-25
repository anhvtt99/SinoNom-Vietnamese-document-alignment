import argparse
from pathlib import Path
from typing import Optional, Sequence, Union, Literal, Dict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .utils import AlignerIO, cuda_available
from .doc_split import sent_split_tkn, chunk_split


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
    max_tokens: Optional[int] = None,                     # None => use model max_seq_length  
) -> None:
    if not docs:
        raise ValueError("docs is empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if split_mode ==  "chunk" and (overlap_rate < 0 or overlap_rate >= 1):
        raise ValueError("overlap_rate is invalid for chunk split")  

    if split_mode == "sentence":
        config_tag = f'{split_mode}_n{num_of_sent}_o{overlap_sent}'
    else:
        config_tag = f'{split_mode}_s{chunk_size}_r{overlap_rate}'

    tok = model.tokenizer
    max_tokens = max_tokens if max_tokens is not None else model.max_seq_length


    # Setup Path
    print(f"[{config_tag}] Constructing embedding output file system...")
    lang_path = Path(embeddings_output) / config_tag / lang
    embeddings_dir = lang_path / "embeddings"
    meta_dir = lang_path / "metadata"
    
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    doc2idx_data = []
    
    print(f"[{config_tag}] Processing {len(docs)} documents for lang: {lang}...")
    for idx, (doc) in enumerate(docs):
        # get split
        if split_mode == "sentence":
            features = sent_split_tkn(doc, tok, lang, max_len=max_sent_len, max_tokens=max_tokens, num_of_sent=num_of_sent, overlap_sent=overlap_sent)
        else:
            overlap_size = int(chunk_size * overlap_rate)
            features = chunk_split(doc, tok, chunk_size, overlap_size, max_tokens=max_tokens)

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

        doc2idx_data.append({
            "doc_idx": idx,
            "file_path": str(doc),
            "emb_file": emb_file_name,
            "n_chunks": emb_doc.shape[0]
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
                "max_tokens": max_tokens,
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
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for model inference.")
    parser.add_argument("--no_normalize", action="store_true", help="Pass this flag to disable L2 normalization of embeddings.")
    
    # Sentence split specific
    parser.add_argument("--num_of_sent", type=int, default=1, help="Number of sentences per split (if split_mode=sentence).")
    parser.add_argument("--overlap_sent", type=int, default=0, help="Number of overlapping sentences (if split_mode=sentence).")
    parser.add_argument("--max_sent_len", type=int, default=10000, help="Maximum sentence length in characters.")
    
    # Chunk split specific
    parser.add_argument("--chunk_size", type=int, default=100, help="Chunk size in tokens (if split_mode=chunk).")
    parser.add_argument("--overlap_rate", type=float, default=0.5, help="Overlap rate for chunks (0.0 to 1.0).")
    parser.add_argument("--max_tokens", type=int, default=None, help="Force maximum tokens (defaults to model's max_seq_length).")

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
    
    _process_per_doc(
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
        max_tokens=args.max_tokens,
    )

if __name__ == "__main__":
    main()