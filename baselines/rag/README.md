## 环境配置
```bash
conda create -n rag_test python=3.10
conda activate rag_test

pip install Jinja2 numpy openai tiktoken
```


## 使用
```bash
# 在 rag 目录下通过sh文件一键设置环境变量 + 运行（推荐）
bash run.sh

# 在 rag 目录下直接运行（推荐）
python run.py \
  --input ../../data/locomo/locomo10.json \
  --output ./rag_results_500_k1.json \
  --chunk_size 500 --k 1

# 或从仓库根目录以模块方式运行
python -m baseline.rag.run \
  --input data/locomo/locomo10.json \
  --output example_data/rag_results_500_k1.json \
  --chunk_size 500 --k 1
```

### 参数：
- `--input`: Locomo 原始数据文件路径（默认 `../../data/locomo/locomo10.json`）。
- `--output`: RAG 评测结果输出路径（默认 `../../example_data/rag_results_500_k1.json`）。
- `--chunk_size`: 文档分块的 token 大小；`-1` 表示不分块（默认 `500`）。
- `--k`: 检索返回的 chunk 数（默认 `1`）。
- `--model`: Chat 模型名或路径（默认从环境变量 `MODEL` 获取）。
- `--embedding_model`: Embedding 模型名或路径（默认从环境变量 `EMBEDDING_MODEL` 获取）。
- `--openai_base_url`: Chat/completions 服务地址（默认从环境变量 `OPENAI_BASE_URL` 获取）。
- `--embed_base_url`: Embeddings 服务地址（默认从环境变量 `EMBED_BASE_URL` 获取，若未设置则与 `openai_base_url` 相同）。

### 可选环境变量：
如果不在命令行中指定，程序会回退到以下环境变量：

- `OPENAI_BASE_URL`: Chat/completions 服务地址（vLLM 兼容）。
- `EMBED_BASE_URL`: Embeddings 服务地址（不设则与 `OPENAI_BASE_URL` 相同）。
- `MODEL`: Chat 模型名或路径。
- `EMBEDDING_MODEL`: Embedding 模型名或路径。
