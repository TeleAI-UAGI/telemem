#!/usr/bin/env bash
set -euo pipefail

# 配置参数
DATASET="../data/zh4o/data.json"
MODEL="qwen3-8b"
OUTPUT="logs/"         
RATIO=1.0                                 
BACKEND="openai"                           
RETRIEVE_K=10
WORKERS=28
EMBEDDING_URL="http://localhost:8082/v1/embeddings"
EMBEDDING_MODEL="qwen3-8b-embedding"
LLM_BASE_URL="http://localhost:4000/v1"    
LLM_API_KEY="dummy-key"                    

# 运行
OPENAI_BASE_URL="$LLM_BASE_URL" \
OPENAI_API_KEY="$LLM_API_KEY" \
python inference.py \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --output "$OUTPUT" \
  --ratio "$RATIO" \
  --backend "$BACKEND" \
  --retrieve_k "$RETRIEVE_K" \
  --workers "$WORKERS" \
  --embedding_url "$EMBEDDING_URL" \
  --embedding_model "$EMBEDDING_MODEL" \
  --llm_base_url "$LLM_BASE_URL" \
  --llm_api_key "$LLM_API_KEY"