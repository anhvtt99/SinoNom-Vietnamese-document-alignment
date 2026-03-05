import os
from pathlib import Path
from typing import List, Tuple, Union, Set, Dict, Optional, Callable, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import load_doc2idx_tsv

# -----------------------
# Metric
# -----------------------
def load_ground_truth(
    path: Union[str, Path], 
    normalize_fn: Optional[Callable[[str], str]] = None
) -> Set[Tuple[str, str]]:
    """
    Load Ground Truth file: src_path<TAB>tgt_path
    Returns a set of tuples: {(src_path, tgt_path), ...}
    """
    ground_truth = set()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            
            src_p, tgt_p = parts[0], parts[1]
            
            # Normalize paths if function provided (e.g., keep only filename)
            if normalize_fn:
                src_p = normalize_fn(src_p)
                tgt_p = normalize_fn(tgt_p)
                
            ground_truth.add((src_p, tgt_p))
    
    return ground_truth

def calculate_f1(
    pred_pairs_indices: List[Tuple[int, int]],
    src_map_path: Union[str, Path],
    tgt_map_path: Union[str, Path],
    gold_file_path: Union[str, Path],
    normalize_fn: Optional[Callable[[str], str]] = None
) -> Dict[str, Any]:
    """
    Calculate Precision, Recall, F1 and return detailed lists of TP, FP, FN.

    Returns:
        Dict with keys: 
        - Metrics: 'precision', 'recall', 'f1'
        - Counts: 'tp_count', 'fp_count', 'fn_count'
        - Lists: 'tp_list', 'fp_list', 'fn_list' (List of (src_path, tgt_path) tuples)
    """
    
    # 1. Load Mappings
    print("Loading mappings...")
    src_paths = load_doc2idx_tsv(src_map_path)
    tgt_paths = load_doc2idx_tsv(tgt_map_path)

    # 2. Load Ground Truth (Gold Standard)
    print("Loading ground truth...")
    gold_set = load_ground_truth(gold_file_path, normalize_fn=normalize_fn)
    
    # 3. Convert Predictions (Indices -> Paths)
    print("Converting predictions...")
    pred_set = set()
    
    for src_idx, tgt_idx in pred_pairs_indices:
        # Check bounds
        if src_idx < 0 or src_idx >= len(src_paths):
            continue
        if tgt_idx < 0 or tgt_idx >= len(tgt_paths):
            continue
            
        s_p = src_paths[src_idx]
        t_p = tgt_paths[tgt_idx]
        
        # Normalize paths if needed
        if normalize_fn:
            s_p = normalize_fn(s_p)
            t_p = normalize_fn(t_p)
            
        pred_set.add((s_p, t_p))

    # 4. Calculate Sets
    
    # True Positives: Pairs in BOTH Pred and Gold (Giao nhau)
    tp_set = pred_set.intersection(gold_set)
    
    # False Positives: Pairs in Pred BUT NOT in Gold
    fp_set = pred_set.difference(gold_set)
    
    # False Negatives: Pairs in Gold BUT NOT in Pred
    fn_set = gold_set.difference(pred_set)
    
    # 5. Calculate Metrics
    tp_count = len(tp_set)
    fp_count = len(fp_set)
    fn_count = len(fn_set)
    
    precision = tp_count / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = tp_count / len(gold_set) if len(gold_set) > 0 else 0.0
    
    f1 = 0.0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {
        # Metrics
        "precision": precision,
        "recall": recall,
        "f1": f1,
        
        # Counts
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "total_gold": len(gold_set),
        "total_pred": len(pred_set),
        
        # Detailed Lists (Convert sets back to sorted lists for readability)
        "tp_list": sorted(list(tp_set)),
        "fp_list": sorted(list(fp_set)),
        "fn_list": sorted(list(fn_set))
    }
 
# -----------------------
# Visualization
# -----------------------

# Venn Diagram
def plot_alignment_venn(metrics: Dict[str, float], figsize: Tuple[int, int] = (8, 6)):
    """
    Plot Venn diagram to visualize the connection between Ground Truth và Prediction.
    """
    try:
        from matplotlib_venn import venn2
    except ImportError:
        print("Please install: pip install matplotlib-venn")
        return

    tp = metrics.get('tp', 0)
    fp = metrics.get('fp', 0)
    fn = metrics.get('fn', 0)
    
    # Ground Truth = TP + FN
    # Prediction = TP + FP
    
    plt.figure(figsize=figsize)
    out = venn2(
        subsets=(fn, fp, tp), 
        set_labels=('Ground Truth', 'Prediction')
    )
    
    # Subset (1, 0): Only from Gold (False Negative)
    # Subset (0, 1): Only from Pred (False Positive)
    # Subset (1, 1): Intersection (True Positive)
    
    plt.title("Alignment Overlap (Venn Diagram)")
    plt.show()
    
 
def plot_confusion_matrix(metrics: dict, figsize: Tuple[int, int] = (7, 6)):
    tp = metrics.get('tp', 0)
    fp = metrics.get('fp', 0)
    fn = metrics.get('fn', 0)
    
    # Create 2x2 matrix, TN = 0 or NaN
    cm = np.array([
        [0, fp],   # Hàng 0: Actual Negative (TN ẩn, FP hiện)
        [fn, tp]   # Hàng 1: Actual Positive (FN hiện, TP hiện)
    ])
    

    labels = np.array([
        ["(Ignored)\nTrue Negative", f"False Positive\n(Wrong Prediction)\n{fp}"],
        [f"False Negative\n(Missed)\n{fn}", f"True Positive\n(Correct)\n{tp}"]
    ])
    
    plt.figure(figsize=figsize)
    
    # Mask
    mask = np.array([[True, False], [False, False]])
    
    sns.heatmap(
        cm, 
        annot=labels, 
        fmt='', 
        cmap='YlGnBu', 
        mask=mask,  
        cbar=False,
        linewidths=1,
        linecolor='black',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive']
    )
    
    plt.title("Alignment Matrix (Ignoring True Negatives)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()