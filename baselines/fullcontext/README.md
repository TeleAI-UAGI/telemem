# Full-Context Evaluation

将所有对话历史作为完整上下文传入 LLM 进行问答评估。

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 1. 运行推理

```bash
# 使用脚本
./run_test.sh

# 或直接使用 Python
python inference.py \
    --model qwen3-8b \
    --base_url http://localhost:8000/v1 \
    --dataset ../zh4o/data.json \
    --output_dir ./logs \
    --ratio 1.0
```

### 2. 评测结果

```bash
# 评测指定文件夹
python evaluate.py --results_dir ./logs/results_qwen3-8b_ratio1.0

# 保存评测结果到文件
python evaluate.py --results_dir ./logs/results_qwen3-8b_ratio1.0 --output eval_results.json
```

### 参数说明

**inference.py:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | qwen3-8b | 模型名称 |
| `--base_url` | http://localhost:8000/v1 | OpenAI API 地址 |
| `--api_key` | dummy-key | API Key |
| `--dataset` | ../data/zh4o/data.json | 数据集路径 |
| `--output_dir` | ./logs | 输出目录 |
| `--ratio` | 1.0 | 评估数据比例 |

**evaluate.py:**

| 参数 | 说明 |
|------|------|
| `--results_dir` | 包含 results_sample_*.json 的文件夹 |
| `--output` | (可选) 保存评测结果的 JSON 文件 |

## 修改 Prompt

编辑 `prompts.py` 文件即可修改 prompt 模板：

```python
# System prompt
SYSTEM_PROMPT = """你是一个智能助手..."""

# User prompt 模板
USER_PROMPT_TEMPLATE = """阅读以下对话历史...
{conversation}
问题：{question}
..."""
```

## 输出格式

推理结果（`results_sample_*.json`）记录：
- `qa_id` - 问题ID
- `question` - 问题
- `ground_truth` - 标准答案
- `category` - 问题类别
- `input_prompt` - 完整输入prompt
- `response` - 模型输出
