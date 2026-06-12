# Quickstart

## Setup

With [uv](https://docs.astral.sh/uv/) (recommended, reproducible via the committed `uv.lock`):

```shell
git clone https://github.com/TeleAI-UAGI/telemem.git
cd telemem
uv sync --all-extras
```

Or with pip:

```shell
pip install -e .            # core (text memory)
pip install -e ".[all]"     # + MCP server + video pipeline
```

## First memory

```shell
export OPENAI_API_KEY=sk-...
```

```python
import telemem as mem0

memory = mem0.Memory()

messages = [
    {"role": "user", "content": "Jordan, did you take the subway to work again today?"},
    {"role": "assistant", "content": "Yes, James. The subway is much faster than driving."},
    {"role": "user", "content": "Which station is closest?"},
    {"role": "assistant", "content": "Line 2 to Civic Center Station, Exit A, 5 minutes on foot."},
]

memory.add(messages=messages, user_id="Jordan")
results = memory.search("How does Jordan get to work?", user_id="Jordan")
for hit in results["results"]:   # same result shape as mem0
    print(hit["memory"])
```

## Character isolation

Every `user_id` gets an independent memory profile; conversation-level events are stored
under the shared pseudo-user `"events"` and searched automatically. In a two-character
dialogue, `search(..., user_id="Jordan")` only sees Jordan's profile plus shared events —
never another character's private memories.

## Local, no-cloud setup

Use the repository's Qwen + FAISS config, or the fully local Ollama config:

```shell
TELEMEM_CONFIG=config/config.ollama.yaml python examples/quickstart.py
```

See [LLM Providers](providers.md) for all supported backends.

## Framework integrations

The integration pattern is two calls: `search()` before answering, `add()` after each exchange.

- LangChain: [`examples/langchain_memory.py`](https://github.com/TeleAI-UAGI/telemem/blob/main/examples/langchain_memory.py)
- LlamaIndex: [`examples/llamaindex_memory.py`](https://github.com/TeleAI-UAGI/telemem/blob/main/examples/llamaindex_memory.py)
- Claude Desktop / Cursor / any MCP client: [MCP Server](MCP.md)
