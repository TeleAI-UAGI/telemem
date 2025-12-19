echo "===>  Start setting environment"

export OPENAI_BASE_URL=""
export MODEL=""
export EMBED_BASE_URL=""
export EMBEDDING_MODEL=""
export OPENAI_API_KEY="sk-xxx"

echo "===>  Setting Done!"

python run.py \
  --input ../../data/locomo/ZH-4O_locomo_format.json \
  --output ./rag_results.json \
  --chunk_size 500 --k 1