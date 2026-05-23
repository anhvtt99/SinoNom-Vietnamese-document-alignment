import os
import random
from typing import Union, Tuple, List, Dict, Any
from pathlib import Path
import json

import pandas as pd
import numpy as np
import torch

# -----------------------
# Helpers: stream I/O
# -----------------------
class AlignerIO:
    """Support read/write file system for Aligner."""
    
    @staticmethod
    def save_config(path: Union[str, Path], config_data: Dict[str, Any]):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def load_config(path: Union[str, Path]) -> Dict[str, Any]:
        with open(Path(path) / "config.json", "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_metadata(path: Union[str, Path], doc2idx_data: List[Dict[str, Any]]):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(doc2idx_data)
        df.to_csv(path / "doc2idx.tsv", sep='\t', index=False)

    @staticmethod
    def load_metadata(path: Union[str, Path]) -> pd.DataFrame:
        return pd.read_csv(Path(path) / "doc2idx.tsv", sep='\t')

    @staticmethod
    def load_doc_embedding(embeddings_dir: Union[str, Path], emb_file_name: str) -> np.ndarray:
        """Load an embedding from given embedding file name."""
        return np.load(Path(embeddings_dir) / emb_file_name).astype('float32')

    @staticmethod
    def load_all_embeddings(embeddings_dir: Union[str, Path], df_meta: pd.DataFrame) -> List[np.ndarray]:
        """Load all embedding base on metadata."""
        embeddings_dir = Path(embeddings_dir)
        return [np.load(embeddings_dir / f).astype('float32') for f in df_meta['emb_file']]

    @staticmethod
    def get_path_by_idx(meta_df: pd.DataFrame, idx: int) -> str:
        """Get path of doc base on idx."""
        try:
            return meta_df.loc[idx, "file_path"]
        except KeyError:
            return None

    @staticmethod
    def get_emb_by_idx(meta_df: pd.DataFrame, embeddings_dir: Union[str, Path], idx: int) -> np.ndarray:
        """Load embedding base on doc_idx."""
        try:
            emb_file = meta_df.loc[idx, "emb_file"]
            return np.load(Path(embeddings_dir) / emb_file).astype('float32')
        except KeyError:
            return None

    @staticmethod
    def get_info_by_idx(meta_df: pd.DataFrame, idx: int) -> Dict[str, Any]:
        """Load all info (path, filename, n_chunks) base on idx."""
        try:
            return meta_df.loc[idx].to_dict()
        except KeyError:
            return None
    
    @staticmethod
    def get_embs_by_indices(
        meta_df: pd.DataFrame, 
        embeddings_dir: Union[str, Path], 
        indices: List[int]
    ) -> Dict[int, np.ndarray]:
        """
        Load many embedding base on given list of index.
        """
        embeddings_dir = Path(embeddings_dir)
        results = {}
        
        unique_indices = list(set(indices))
        
        valid_indices = meta_df.index.intersection(unique_indices)
        needed_meta = meta_df.loc[valid_indices]

        for idx, row in needed_meta.iterrows():
            emb_file_path = embeddings_dir / row['emb_file']
            if emb_file_path.exists():
                results[idx] = np.load(emb_file_path).astype('float32')
            else:
                print(f"Warning: File {emb_file_path} not found.")
        
        return results


# -----------------------
# Helpers: Check if CUDA is available
# -----------------------
def cuda_available(verbose: bool = False) -> bool:
    """
    Returns True if CUDA is available (PyTorch sees at least 1 GPU).
    If verbose=True, prints basic GPU info.
    """
    ok = torch.cuda.is_available()
    if verbose:
        if ok:
            dev = torch.device("cuda")
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            cap = torch.cuda.get_device_capability(idx)
            mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
            print(f"CUDA available: True | device={idx} | name={name} | capability={cap} | VRAM={mem_gb:.2f} GB")
        else:
            print("CUDA available: False (using CPU)")
    return ok


def get_torch_device(prefer_cuda: bool = True) -> torch.device:
    """
    Returns torch.device('cuda') if prefer_cuda and available, else 'cpu'.
    """
    return torch.device("cuda") if (prefer_cuda and torch.cuda.is_available()) else torch.device("cpu")

# -----------------------
# Helpers: function for normalization path
# -----------------------
def get_filename_only(path_str: str) -> str:
    """Returns 'document.txt' from '/path/to/data/document.txt'"""
    return os.path.basename(path_str)

# -----------------------
# Helpers: function for normalization vietnamese word
# -----------------------
def normalize_vietnamese_phrase(text: str) -> str:
    """
    Normalize keyword text for stable comparison.
    Notes:
        - lowercased
        - underscores converted to spaces
        - repeated spaces collapsed
    """
    text = text.lower().replace("_", " ")
    return " ".join(text.split())

# -----------------------
# Helpers: sleep timing
# -----------------------
def random_sleep_seconds(seconds_range: Tuple[float, float]) -> float:
    """
    Return a random number of seconds from a (min, max) range.

    Examples:
        random_sleep_seconds((0.5, 1.0)) -> 0.73
        random_sleep_seconds((0.0, 0.0)) -> 0.0
    """
    lo, hi = seconds_range

    lo = max(0.0, float(lo))
    hi = max(0.0, float(hi))

    if hi <= 0:
        return 0.0

    if hi < lo:
        lo, hi = hi, lo

    return random.uniform(lo, hi)
