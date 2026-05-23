from pathlib import Path
from typing import List, Tuple, Union, Set, Dict, Optional, Callable, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lib.utils import AlignerIO

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

def eval(
    pred_pairs_indices: List[Union[Tuple[int, int], Tuple[int, int, float]]],
    src_meta_path: Union[str, Path],
    tgt_meta_path: Union[str, Path],
    gold_file_path: Union[str, Path],
    normalize_fn: Optional[Callable[[str], str]] = None
) -> Dict[str, Any]:
    """
    Calculate Precision, Recall, F1-score and return detailed lists of TP, FP, FN.
    Uses AlignerIO to dynamically map indices to actual file paths.
    """
    
    # 1. Load Mappings (Metadata)
    print("[*] Loading metadata mappings...")
    # set_index('doc_idx') allows ultra-fast O(1) lookups for paths later
    src_meta_df = AlignerIO.load_metadata(src_meta_path).set_index('doc_idx')
    tgt_meta_df = AlignerIO.load_metadata(tgt_meta_path).set_index('doc_idx')

    # 2. Load Ground Truth (Gold Standard)
    print("[*] Loading ground truth...")
    # Assumes load_ground_truth returns a set of (src_path, tgt_path) tuples
    gold_set = load_ground_truth(gold_file_path, normalize_fn=normalize_fn)
    
    # 3. Convert Predictions (Indices -> Paths) and Store Scores
    print("[*] Converting predicted indices to file paths...")
    pred_set = set()
    pred_scores = {}
    
    for item in pred_pairs_indices:
        src_idx = item[0]
        tgt_idx = item[1]
        score = item[2] if len(item) > 2 else None
        
        # Fast lookup using AlignerIO. Returns None if idx doesn't exist.
        s_p = AlignerIO.get_path_by_idx(src_meta_df, src_idx)
        t_p = AlignerIO.get_path_by_idx(tgt_meta_df, tgt_idx)
        
        # Skip this pair if either source or target index is invalid/missing
        if s_p is None or t_p is None:
            continue
            
        # Normalize paths if a function is provided (e.g., removing .txt extensions)
        if normalize_fn:
            s_p = normalize_fn(s_p)
            t_p = normalize_fn(t_p)
            
        pair = (s_p, t_p)
        pred_set.add(pair)
        
        # Keep track of the score (MaxSim/CSLS) for deeper error analysis
        if score is not None:
            pred_scores[pair] = score

    # 4. Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    # Using Python's built-in Set operations for maximum performance
    tp_set = pred_set.intersection(gold_set)
    fp_set = pred_set.difference(gold_set)
    fn_set = gold_set.difference(pred_set)
    
    # 5. Calculate Standard Metrics
    tp_count = len(tp_set)
    fp_count = len(fp_set)
    fn_count = len(fn_set)
    
    precision = tp_count / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = tp_count / len(gold_set) if len(gold_set) > 0 else 0.0
    
    f1 = 0.0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    # 6. Format Detailed Lists for Output
    def format_list_with_scores(pair_set: set) -> List:
        """Helper to attach scores back to the pairs for debugging."""
        sorted_pairs = sorted(list(pair_set))
        if not pred_scores:
            return sorted_pairs
        return [(s, t, pred_scores.get((s, t), None)) for s, t in sorted_pairs]

    print(f"[+] Evaluation completed: F1={f1:.4f} | Precision={precision:.4f} | Recall={recall:.4f}")

    return {
        # Core Metrics
        "precision": precision,
        "recall": recall,
        "f1": f1,
        
        # Raw Counts
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "total_gold": len(gold_set),
        "total_pred": len(pred_set),
        
        # Detailed Lists (Great for writing error analysis in your thesis)
        "tp_list": format_list_with_scores(tp_set),
        "fp_list": format_list_with_scores(fp_set),
        "fn_list": sorted(list(fn_set)) # FN implies we missed it, so we don't have a predicted score for it
    }
 
# -----------------------
# Visualization
# -----------------------

# Venn Diagram
def plot_alignment_venn(metrics: Dict[str, float], figsize: Tuple[int, int] = (8, 6), save_path=None):
    """
    Plot Venn diagram to visualize the connection between Ground Truth and Prediction.
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
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[eval] Plot saved to {save_path}")
    else:
        plt.show()
    
 
def plot_confusion_matrix(metrics: dict, figsize: Tuple[int, int] = (7, 6), save_path=None):
    tp = metrics.get('tp', 0)
    fp = metrics.get('fp', 0)
    fn = metrics.get('fn', 0)
    
    # Create 2x2 matrix, TN = 0 or NaN
    cm = np.array([
        [0, fp],  
        [fn, tp]  
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
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[eval] Plot saved to {save_path}")
    else:
        plt.show()