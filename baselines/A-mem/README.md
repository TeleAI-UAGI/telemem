# Run the experiments on ZH-4O Chinese dataset:
```bash
python test_advanced.py \
    --dataset ../../data/zh4o/data.json \
    --model qwen3-8b \
    --backend openai \
    --retrieve_k 10 \
    --workers 1 \
    --embedding_url http://your-embedding-server:port/v1/embeddings \
    --embedding_model qwen3-8b-embedding \
    --llm_base_url http://your-llm-server:port/v1 \
    --llm_api_key your-api-key
```

**Parameters:**
- `--dataset`: Path to the dataset file (default: `data/ZH-4O_locomo_format.json`)
- `--model`: LLM model name (default: `qwen3-8b`)
- `--backend`: LLM backend, either `openai` or `ollama` (default: `openai`)
- `--retrieve_k`: Number of memories to retrieve (default: `10`)
- `--workers`: Number of parallel workers (default: `10`, use `1` for sequential processing)
- `--embedding_url`: URL of the embedding API endpoint
- `--embedding_model`: Name of the embedding model
- `--llm_base_url`: Base URL for the LLM API (OpenAI compatible)
- `--llm_api_key`: API key for the LLM API (can be dummy if server doesn't validate)
- `--ratio`: Ratio of dataset to evaluate, 0.0-1.0 (default: `1.0`)
- `--output`: Path to save evaluation results (optional)


# Agentic Memory 🧠

A novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way.

> **Note:** This repository is specifically designed to reproduce the results presented in our paper. If you want to use the A-MEM system in building your agents, please refer to our official implementation at: [A-mem-sys](https://github.com/WujiangXu/A-mem-sys)

For more details, please refer to our paper: [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/pdf/2502.12110)

## Introduction 🌟

Large Language Model (LLM) agents have demonstrated remarkable capabilities in handling complex real-world tasks through external tool usage. However, to effectively leverage historical experiences, they require sophisticated memory systems. Traditional memory systems, while providing basic storage and retrieval functionality, often lack advanced memory organization capabilities.

Our project introduces an innovative **Agentic Memory** system that revolutionizes how LLM agents manage and utilize their memories:

<div align="center">
  <img src="Figure/intro-a.jpg" alt="Traditional Memory System" width="600"/>
  <img src="Figure/intro-b.jpg" alt="Our Proposed Agentic Memory" width="600"/>
  <br>
  <em>Comparison between traditional memory system (top) and our proposed agentic memory (bottom). Our system enables dynamic memory operations and flexible agent-memory interactions.</em>
</div>

## Key Features ✨

- 🔄 Dynamic memory organization based on Zettelkasten principles
- 🔍 Intelligent indexing and linking of memories
- 📝 Comprehensive note generation with structured attributes
- 🌐 Interconnected knowledge networks
- 🔄 Continuous memory evolution and refinement
- 🤖 Agent-driven decision making for adaptive memory management

## Framework 🏗️

<div align="center">
  <img src="Figure/framework.jpg" alt="Agentic Memory Framework" width="800"/>
  <br>
  <em>The framework of our Agentic Memory system showing the dynamic interaction between LLM agents and memory components.</em>
</div>

## How It Works 🛠️

When a new memory is added to the system:
1. Generates comprehensive notes with structured attributes
2. Creates contextual descriptions and tags
3. Analyzes historical memories for relevant connections
4. Establishes meaningful links based on similarities
5. Enables dynamic memory evolution and updates

## Results 📊

Empirical experiments conducted on six foundation models demonstrate superior performance compared to existing SOTA baselines.

## Getting Started 🚀

1. Clone the repository:
```bash
git clone https://github.com/WujiangXu/AgenticMemory.git
cd AgenticMemory
```

2. Install dependencies:
Option 1: Using venv (Python virtual environment)
```bash
# Create and activate virtual environment
python -m venv a-mem
source a-mem/bin/activate  # Linux/Mac
a-mem\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

Option 2: Using Conda
```bash
# Create and activate conda environment
conda create -n myenv python=3.9
conda activate myenv

# Install dependencies
pip install -r requirements.txt
```

3. Run the experiments in LoCoMo dataset:
```python
python test_advanced.py 
```

**Note:** To achieve the optimal performance reported in our paper, please adjust the hyperparameter k value accordingly. 

## Citation 📚

If you use this code in your research, please cite our work:

```bibtex
@article{xu2025mem,
  title={A-mem: Agentic memory for llm agents},
  author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao and Zhang, Yongfeng},
  journal={arXiv preprint arXiv:2502.12110},
  year={2025}
}
```

## License 📄

This project is licensed under the MIT License. See LICENSE for details.


