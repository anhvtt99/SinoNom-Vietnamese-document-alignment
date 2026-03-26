#!/bin/bash

# --- CONFIGURATION ---
# Lang
SRC_LANG="vi"
TAR_LANG="zh"

# Model
MODEL_PATH="sentence-transformers/LaBSE"

# Input/Output
INPUT_BASE_DIR="../data/sample_data/$LANG"
OUTPUT_BASE_DIR="../db_embeddings"

# Split (Choose "sentence" or "chunk")
SPLIT_MODE="sentence"

# For "sentence"
NUM_SENT=1
OVERLAP_SENT=0

# For "chunk"
CHUNK_SIZE=128
OVERLAP_RATE=0.15

BATCH_SIZE=64

# --- EXECUTION ---

echo "====================================================="
echo "START GENERATING EMBEDDINGS"
echo "Model: $MODEL_PATH"
echo "Mode:  $SPLIT_MODE"
echo "Lang:  $LANG"
echo "====================================================="

echo -e "\n Processing For Language: $LANG..."
python ../lib/generate_embeddings.py \
    --input_dir "$INPUT_BASE_DIR" \
    --output_dir "$OUTPUT_BASE_DIR" \
    --lang "$LANG" \
    --model_name_or_path "$MODEL_PATH" \
    --split_mode "$SPLIT_MODE" \
    --batch_size $BATCH_SIZE \
    --num_of_sent $NUM_SENT \
    --overlap_sent $OVERLAP_SENT \
    --chunk_size $CHUNK_SIZE \
    --overlap_rate $OVERLAP_RATE