#!/bin/bash

# --- 1. CONFIGURATION  ---
EMB_BASE_PATH="../db_embeddings"
SRC_LANG="vi"
TAR_LANG="zh"
GT_PATH="../data/sample_data/ground_truth.tsv"
RESULTS_DIR="../results"           

# --- 2. CONFIG TAG (Must match with Embedding generate) ---
SPLIT_MODE="sentence"
NUM_SENT=1
OVERLAP_SENT=0
CHUNK_SIZE=128
OVERLAP_RATE=0.15

# --- 3. ALGORITHM HYPERPARAMETERS ---
CSLS_K=15
TOP_K_DOCS=10
TOP_K_CHUNKS=5
BIMAX_TRIM=0.7
THRESHOLD=0.08


mkdir -p "$RESULTS_DIR"
# --- 4. EXECUTION ---
echo "====================================================="
echo "STARTING DOCUMENT ALIGNMENT PIPELINE"
echo "Config: $SPLIT_MODE (n=$NUM_SENT)"
echo "Threshold=$THRESHOLD"
echo "====================================================="
python ../lib/aligner/aligner.py \
    --emb_base_path "$EMB_BASE_PATH" \
    --src_lang "$SRC_LANG" \
    --tar_lang "$TAR_LANG" \
    --split_mode "$SPLIT_MODE" \
    --num_of_sent "$NUM_SENT" \
    --overlap_sent "$OVERLAP_SENT" \
    --chunk_size "$CHUNK_SIZE" \
    --overlap_rate "$OVERLAP_RATE" \
    --top_k_docs "$TOP_K_DOCS" \
    --top_k_chunks "$TOP_K_CHUNKS" \
    --bimax_trim_ratio "$BIMAX_TRIM" \
    --csls_k "$CSLS_K" \
    --edge_threshold "$THRESHOLD" \
    --gt_path "$GT_PATH" \
    --output_path "$RESULTS_DIR" \
    --save_results \
    --eval \
    --viz