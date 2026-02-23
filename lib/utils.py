from typing import Union, Iterable, List, Dict
from pathlib import Path
import numpy as np
import torch

# -----------------------
# Helpers: stream I/O
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


def load_needed_from_stream(
    stream_path: Union[str, Path],
    need_indices: Iterable[int],
) -> Dict[int, np.ndarray]:
    """
    Scan stream once and load only arrays whose index is in need_indices.
    Each stream item is expected to be (n_chunks, dim) float32.
    """
    need = set(int(x) for x in need_indices)
    out: Dict[int, np.ndarray] = {}
    if not need:
        return out

    for i, arr in enumerate(iter_npy_stream(stream_path)):
        if i in need:
            out[i] = np.asarray(arr, dtype=np.float32)
            if len(out) == len(need):
                break

    missing = need - set(out.keys())
    if missing:
        raise ValueError(f"Missing indices in chunk stream: {sorted(missing)[:10]} ...")
    return out

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