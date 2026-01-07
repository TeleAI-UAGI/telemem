#!/bin/bash

# Full-context inference script
# Usage: ./run_test.sh [options]

# Default parameters
MODEL="qwen3-8b"
BASE_URL="http://localhost:4000/v1"
API_KEY="dummy-key"
DATASET="../data/zh4o/data.json"
OUTPUT_DIR="./logs"
RATIO=1.0

echo "==================================="
echo "Full-Context Inference"
echo "==================================="
echo "Model: $MODEL"
echo "Base URL: $BASE_URL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "Ratio: $RATIO"
echo "==================================="

# Run inference
python inference.py \
    --model "$MODEL" \
    --base_url "$BASE_URL" \
    --api_key "$API_KEY" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --ratio "$RATIO"
