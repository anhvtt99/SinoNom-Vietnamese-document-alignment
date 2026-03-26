#!/bin/bash

# --- EMBEDDING CONFIGURATION ---
# Model
MODEL_PATH="sentence-transformers/LaBSE"

# Input/Output
INPUT_BASE_DIR="../data/sample_data"
EMB_BASE_PATH="../db_embeddings"

# Split (Choose "sentence" or "chunk")
SPLIT_MODE="sentence"

# For "sentence"
NUM_SENT=1
OVERLAP_SENT=0

# For "chunk"
CHUNK_SIZE=128
OVERLAP_RATE=0.15

BATCH_SIZE=64

# --- ALGINER CONFIGURATION ---
SRC_LANG="vi"
TAR_LANG="zh"

CSLS_K=15
TOP_K_DOCS=10
TOP_K_CHUNKS=5
BIMAX_TRIM=0.7
THRESHOLD=0.08

GT_PATH="../data/sample_data/ground_truth.tsv"
RESULTS_DIR="../results"

# --- EXECUTION ---
echo -e "\n Processing For Source Language: $SRC_LANG..."
python ../lib/generate_embeddings.py \
    --input_dir "$INPUT_BASE_DIR/$SRC_LANG" \
    --output_dir "$EMB_BASE_PATH" \
    --lang "$SRC_LANG" \
    --model_name_or_path "$MODEL_PATH" \
    --split_mode "$SPLIT_MODE" \
    --batch_size $BATCH_SIZE \
    --num_of_sent $NUM_SENT \
    --overlap_sent $OVERLAP_SENT \
    --chunk_size $CHUNK_SIZE \
    --overlap_rate $OVERLAP_RATE

echo -e "\n Processing For Target Language: $TAR_LANG..."
python ../lib/generate_embeddings.py \
    --input_dir "$INPUT_BASE_DIR/$TAR_LANG" \
    --output_dir "$EMB_BASE_PATH" \
    --lang "$TAR_LANG" \
    --model_name_or_path "$MODEL_PATH" \
    --split_mode "$SPLIT_MODE" \
    --batch_size $BATCH_SIZE \
    --num_of_sent $NUM_SENT \
    --overlap_sent $OVERLAP_SENT \
    --chunk_size $CHUNK_SIZE \
    --overlap_rate $OVERLAP_RATE

echo -e "\n Starting Document Alignment Pipeline..."
python ../lib/aligner.py \
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
