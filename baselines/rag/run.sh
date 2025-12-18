echo "===>  Start setting environment"

export OPENAI_BASE_URL="http://117.145.68.73:20091/v1"
export MODEL="qwen3-8b"
export EMBED_BASE_URL="http://117.145.68.73:20416/v1"
export EMBEDDING_MODEL="qwen3-8b-embedding"
export OPENAI_API_KEY="EMPTY"  #"sk-55091139dcfa4d21a2d55d5387f29b43"

echo "===>  Setting Done!"

python run.py \
  --input ../../data/locomo/ZH-4O_locomo_format.json \
  --output ./rag_results_ZH_qwen3-8b.json \
  --chunk_size 500 --k 1