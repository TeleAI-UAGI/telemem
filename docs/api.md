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
| `user_id` | `Optional[str]` | Character/user to attribute the memory to; each `user_id` gets an **independent memory profile**. Omit to store into the shared `"events"` scope |
| `agent_id` / `run_id` | `Optional[str]` | Additional mem0-compatible scopes |
| `metadata` | `Optional[Dict]` | Arbitrary metadata stored with each memory |
| `infer` | `bool` | Extract salient facts with the LLM (default `True`); `False` stores each non-system message's content **verbatim, with no LLM call** (the message `role` is recorded in metadata) |
| `memory_type` | `Optional[str]` | Pass `"procedural_memory"` to delegate to mem0's procedural-memory pipeline; any other non-None value raises |
| `prompt` | `Optional[str]` | Custom extraction prompt. Replaces TeleMem's built-in summarization prompts as the **system prompt**; the raw transcript (current turn last) is sent as the user message. Have it answer in the `这段内容的摘要是：[...]` format or as a JSON list of strings. Ignored when `infer=False` |
| `batch` | `bool` | Route through the high-throughput batched pipeline (`add_batch`) |

At least one of `user_id` / `agent_id` / `run_id` is required (mem0-compatible);
omitting all three raises a validation error.

**Returns** `{"results": [{"id", "memory", "event"}, ...]}` — the mem0-compatible shape.

Internally (when `infer=True`): the latest turn is summarized against the earlier turns as
context — from the global view (no `user_id`, stored in `"events"`) or the named character's
perspective — then embedded, matched against similar memories, LLM-fused, and dual-written
to FAISS + JSON. To write one conversation into **multiple character profiles plus the
shared `"events"` scope in a single call**, use `add_batch()` (or `batch=True`).

## add_batch()

```python
memory.add_batch(messages, *, user_id=None, agent_id=None, run_id=None, ...)
```

High-throughput ingestion: `messages` is a list of message-lists processed concurrently
(thread pool), with buffer-based batch flushing. `user_id` may be a single id or a list
of character ids; every listed character gets its own extraction pass, **plus** a shared
`"events"` pass. `infer=False` stores contents verbatim into every scope with no LLM
calls; `prompt` overrides the extraction prompt for all scopes; `memory_type` is not
supported here. Returns the same `{"results": [...]}` shape.

## search()

```python
memory.search(query, *, user_id=None, agent_id=None, run_id=None,
              limit=100, filters=None, threshold=None, rerank=True)
```

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `query` | `str` | Natural-language query |
| `user_id` | `Optional[str]` | Character profile to search; shared `"events"` memories are always included |
| `limit` | `int` | Max results (default 100) — enforced on the **merged** result set across all searched scopes, highest scores first |
| `threshold` | `Optional[float]` | Similarity threshold in [0, 1] |
| `rerank` | `bool` | Rerank results when a reranker is configured |

**Returns** `{"results": [{"id", "memory", "score", "source", ...}, ...]}` — `source` is
the scope each hit came from (the profile id or `"events"`).

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

## Telemetry

TeleMem collects no telemetry of its own, and it **disables** the anonymized
PostHog telemetry inherited from the `mem0ai` base library by default:
`import telemem` sets `MEM0_TELEMETRY=False` unless that environment variable
is already set. Set `MEM0_TELEMETRY=true` before importing to opt back in.
