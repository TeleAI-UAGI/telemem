## Installation

### Requirements
- Python 3.9 - 3.12

### Install Dependencies

```bash
pip install -r requirements.txt
```


## Usage



#### 1. Add Memories

```bash
python run.py \
  --method add \
  --memory_provider local \
  --local_vllm_api_base http://127.0.0.1:4000/v1 \
  --local_chat_model qwen3-8b \
  --local_embed_api_base http://127.0.0.1:8082/v1 \
  --local_embed_model qwen3-8b-embedding \
  --memory_name zh4o \
  --input ../../data/zh4o/data.json
```

#### 2. Search Memories

```bash
python run.py \
  --method search \
  --input ../data/zh4o/data.json \
  --memory_name zh4o \
  --output ./logs/results_qwen3-8b_ratio1.0 \
  --top_k 30 \
  --memory_provider local \
  --local_vllm_api_base http://127.0.0.1:4000/v1 \
  --local_embed_api_base http://127.0.0.1:8082/v1 \
  --local_chat_model qwen3-8b \
  --local_embed_model qwen3-8b-embedding
```
#### 3. Eval
```bash
python ../evaluate.py --results_dir ./logs/results_qwen3-8b_ratio1.0 --output eval_results.json
```
## Configuration

### Command Line Arguments

**Basic Parameters:**
- `--method`: Operation mode - `add` or `search` (default: `add`)
- `--input`: Path to input data file (default: `./data/input.json`)
- `--output`: Path to output results directory (default: `../../results/mem0_results/`)
- `--top_k`: Number of memories to return in search (default: `30`)
- `--filter_memories`: Enable memory filtering (default: `False`)
- `--is_graph`: Enable graph-based search (default: `False`)
- `--memory_name`: Name for the memory storage (default: `memories`)

**Provider Configuration:**
- `--memory_provider`: Choose between `mem0` or `local` (default: `mem0`)
- `--local_base_save_dir`: Base directory for local storage (default: `video_segments`)
- `--local_vllm_api_base`: vLLM chat/completions API endpoint (e.g., `http://127.0.0.1:8000/v1`)
- `--local_chat_model`: Chat model name or path (default: from `MODEL` env var or `your-chat-model`)
- `--local_embed_api_base`: Embedding service endpoint (can share with chat)
- `--local_embed_model`: Embedding model name or path (default: from `EMBEDDING_MODEL` env var or `your-embedding-model`)

### Memory Name Parameter

The `--memory_name` parameter allows you to create and manage independent memory stores:

**Simple Mode (LOCAL_MEM_SIMPLE=1):**
- File format: `simple_mem_store_{memory_name}.json`
- Example: `--memory_name experiment1` → `video_segments/simple_mem_store_experiment1.json`

**Full mem0 Mode:**
- Directory format: `{base_save_dir}/{memory_name}/`
- Example: `--memory_name experiment1` → `video_segments/experiment1/chroma_db/` and `video_segments/experiment1/history.db`

**Use Cases:**
- Multiple experiments: Create separate memory stores for different experiments
- Dataset separation: Maintain independent memories for training and test sets
- Version control: Create memory snapshots for different data versions

**Important:** Both `add` and `search` operations must use the same `--memory_name` to access the same memory store.

## Performance Logging

When running `add` operations, the system automatically records timing statistics.

### Log File Location
- Path format: `{base_save_dir}/{memory_name}/add_timing.log`
- Example: With `--memory_name experiment1`, logs are saved to `video_segments/experiment1/add_timing.log`

### Log Contents
The log file is in JSON format and includes:
- `total_time`: Total execution time (seconds)
- `total_conversations`: Total number of conversations processed
- `success_count`: Number of successful operations
- `error_count`: Number of failed operations
- `success_rate`: Success rate (percentage)
- `failed_indices`: List of failed conversation indices
- `conversation_times`: Processing time and status for each conversation
- `statistics`: Statistical summary (average, min, max times)
- `timestamp`: Log generation timestamp

### Example Output
```json
{
  "total_time": 123.45,
  "total_conversations": 10,
  "success_count": 9,
  "error_count": 1,
  "success_rate": 90.0,
  "failed_indices": [5],
  "conversation_times": [
    {"idx": 0, "time": 12.3, "status": "success"},
    {"idx": 1, "time": 15.6, "status": "success"}
  ],
  "statistics": {
    "avg_time": 13.95,
    "min_time": 12.3,
    "max_time": 15.6
  },
  "timestamp": "2025-10-22 10:30:45"
}
```

## Environment Variables

Optional environment variables for configuration:

- `OPENAI_BASE_URL`: Chat/completions service endpoint (vLLM compatible)
- `OPENAI_API_KEY`: API key for the service (use dummy value for local vLLM)
- `MODEL`: Chat model name or path
- `EMBED_BASE_URL`: Embeddings service endpoint (defaults to OPENAI_BASE_URL if not set)
- `EMBEDDING_MODEL`: Embedding model name or path
- `MEM0_API_KEY`: API key for mem0 cloud service

## Project Structure

```
.
├── run.py                      # Main entry point
├── mem0_manager.py             # Memory manager orchestrator
├── memory_manager_local.py     # Local memory provider implementation
├── local_adapter.py            # Local storage adapter
├── add.py                      # Memory addition logic
├── search.py                   # Memory search logic
├── prompts.py                  # Prompt templates
├── requirements.txt            # Python dependencies
├── .env.example               # Environment configuration template
└── README.md                   # This file
```

## Data Format

### Input Data Format
The input JSON file should contain an array of conversation objects:

```json
[
  {
    "conversation_id": "unique_id",
    "turns": [
      {
        "role": "user",
        "content": "User message"
      },
      {
        "role": "assistant",
        "content": "Assistant response"
      }
    ]
  }
]
```

### Output Data Format
Search results are saved in the specified output directory, with one JSON file per sample (e.g., `results_sample_0.json`).

Each result record contains:
- `qa_id`: Original question ID
- `question`: The question text
- `ground_truth`: The expected answer
- `category`: Question category
- `input_prompt`: The full prompt sent to the LLM (including retrieved memories)
- `response`: The model's response (following a specific format `<eoe>ANSWER`)
- `input_tokens` / `output_tokens`: Token usage statistics
- `speaker_1_memories` / `speaker_2_memories`: Memories retrieved for each speaker

### LLM Constraints
Following the evaluation specification:
- **Temperature**: All LLM calls use `temperature=0.7`.
- **Thinking Mode**: Reasoning/Thinking features are disabled to ensure direct evaluation of memory retrieval.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project uses [mem0](https://mem0.ai/) for memory management capabilities.


