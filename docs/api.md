# API Reference

`telemem.Memory` (alias of `telemem.TeleMemory`) subclasses `mem0.Memory`, so the full
mem0 surface — `get`, `get_all`, `update`, `delete`, `delete_all`, `history`, `reset` —
is available. The methods below are TeleMem's optimized or extended ones.

## add()

```python
memory.add(messages, *, user_id=None, agent_id=None, run_id=None,
           metadata=None, infer=True, memory_type=None, prompt=None, batch=False)
```

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `messages` | `str` or `List[Dict]` | A statement, or dialogue messages with `role`/`content` |
| `user_id` | `Optional[str]` | Character/user to attribute the memory to; each `user_id` gets an **independent memory profile**. Omit to store shared conversation-event memories |
| `agent_id` / `run_id` | `Optional[str]` | Additional mem0-compatible scopes |
| `metadata` | `Optional[Dict]` | Arbitrary metadata stored with each memory |
| `infer` | `bool` | Extract salient facts with the LLM (default `True`); `False` stores raw text |
| `prompt` | `Optional[str]` | Custom summarization prompt |
| `batch` | `bool` | Route through the high-throughput batched pipeline (`add_batch`) |

**Returns** `{"results": [{"id", "memory", "event"}, ...]}` — the mem0-compatible shape.

Internally: messages are merged per speaker → summarized from the global view and each
character's perspective → embedded and matched against similar memories → buffered →
LLM-fused (semantic clustering) → dual-written to FAISS + JSON.

## add_batch()

```python
memory.add_batch(messages, *, user_id=None, agent_id=None, run_id=None, ...)
```

High-throughput ingestion: `messages` is a list of message-lists processed concurrently
(thread pool), with buffer-based batch flushing. `user_id` may be a single id or a list
of character ids. Returns the same `{"results": [...]}` shape.

## search()

```python
memory.search(query, *, user_id=None, agent_id=None, run_id=None,
              limit=100, filters=None, threshold=None, rerank=True)
```

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `query` | `str` | Natural-language query |
| `user_id` | `Optional[str]` | Character profile to search; shared `"events"` memories are always included |
| `limit` | `int` | Max results (default 100) |
| `threshold` | `Optional[float]` | Similarity threshold in [0, 1] |
| `rerank` | `bool` | Rerank results when a reranker is configured |

**Returns** `{"results": [{"id", "memory", "score", ...}, ...]}`.

## add_mm() / search_mm()

Video memory — see [Video Memory](video.md).

## Configuration

`TeleMemoryConfig` extends mem0's `MemoryConfig`:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `buffer_size` | `64` | Memories buffered before a batch flush |
| `similarity_threshold` | `0.95` | Cosine threshold for semantic clustering |
| `vlm` | `{...}` | Video pipeline settings (VLM endpoint, FPS, clip length, embedding dims) |

Load from YAML/JSON:

```python
from telemem.utils import load_config
config = load_config("config/config.yaml")   # returns TeleMemoryConfig
```
