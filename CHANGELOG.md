# Changelog

All notable changes to TeleMem are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.9.0] - 2026-08-06

### Changed
- **MCP server migrated to the official MCP Python SDK v2** (`mcp>=2.0.0,<3`),
  which implements the current MCP specification (**2026-07-28**) and still
  serves older `initialize`-handshake clients from the same server. The v1
  SDK's `FastMCP` import path was removed upstream, so the server is now built
  on `mcp.server.mcpserver.MCPServer` and declares server-level metadata
  (`title`, `version`, `website_url`, icon).
- Every MCP tool now carries the standard metadata introduced by the
  2025-11-25/2026-07-28 specs: a human-readable `title`, behavior annotations
  (`readOnlyHint`/`destructiveHint`/`idempotentHint`/`openWorldHint`), and a
  derived output schema; results are emitted as `structuredContent` (wrapped
  under a `"result"` key) alongside the unchanged JSON text content.
- `memory_history` now returns `{"memory_id": ..., "history": [...]}` instead
  of a bare JSON array, so its result fits the structured-output object shape.
- `telemem.mcp.create_server()` no longer takes `host`/`port`; pass them to
  `server.run(transport="streamable-http", host=..., port=...)` instead
  (SDK v2 moved transport settings out of the constructor).
- `--transport sse` is deprecated (the MCP spec deprecated HTTP+SSE); it still
  works but logs a warning. Docs and the Docker image now show
  `streamable-http` for HTTP deployments.
- `examples/mcp_client.py` now uses the SDK v2 high-level `mcp.client.Client`;
  the MCP test suite drives both the modern (2026-07-28, stateless) and the
  legacy initialize-handshake protocol paths.
- `server.json` registry manifest: schema bumped to `2025-12-11`, version
  synced to the package version.

## [1.8.0] - 2026-07-11

### Fixed
- **Character memory extraction**: the per-character prompt received the
  `parse_messages` helper function instead of the parsed current dialogue,
  so character profiles were built from a function repr rather than the
  conversation. Character extraction now sees the actual turn.
- `add()` without a `user_id` now stores memories in the shared `"events"`
  scope (matching `add_batch()`), so `search()` — which always includes
  `"events"` — can actually retrieve them.
- `search(limit=N)` now enforces the limit on the merged result set across
  scopes (highest score first); previously searching a profile plus
  `"events"` without a reranker could return up to `2N` results.

### Added
- `add(infer=False)` and `add_batch(infer=False)` are now honored: message
  contents are stored verbatim with **zero LLM calls**, and the message
  `role` is recorded in metadata.
- `add(prompt=...)` is now honored as a custom extraction prompt (system
  prompt override, raw transcript as the user message); also threaded
  through `add_batch()`.
- `add(memory_type="procedural_memory")` now delegates to mem0's
  procedural-memory pipeline; any other non-None `memory_type` raises a
  validation error instead of being silently ignored.
- `add()` validates that at least one of `user_id`/`agent_id`/`run_id` is
  provided, matching the documented (and mem0's) contract.
- Offline contract test suite (`tests/test_contract.py`): deterministic
  fake-LLM/fake-embedder tests pinning the public behavior of
  `add`/`add_batch`/`search`, character-scope separation, raw ingestion,
  prompt override, and result limits.
- Multi-NPC reference example (`examples/multi_npc.py`): five NPCs live
  through one scene; each gets a private memory profile plus the shared
  `"events"` world-state, then recalls the scene from their own
  perspective.

### Changed
- Inherited mem0/PostHog telemetry is now **disabled by default**
  (`import telemem` sets `MEM0_TELEMETRY=False` unless already set);
  documented in the READMEs and API reference.
- `mem0ai` dependency pinned to the tested `>=2.0,<2.1` range (TeleMem
  extends private mem0 internals, so an unbounded range was unsafe);
  `requirements.txt` regenerated from the tested environment.
- Test functions no longer return values (removes pytest warnings).

## [1.7.1] - 2026-06-12

### Fixed
- MCP registry name uses the canonical GitHub org casing
  (`io.github.TeleAI-UAGI/telemem`) so OIDC namespace authorization and
  the PyPI ownership marker match.

## [1.7.0] - 2026-06-12

### Added
- Published to the official [MCP registry](https://registry.modelcontextprotocol.io)
  as `io.github.teleai-uagi/telemem`: `server.json` manifest, PyPI ownership
  marker, and an OIDC-authenticated publish job in the release workflow.
- New `telemem` console script (alias of `telemem-mcp`) so `uvx telemem`
  runs the MCP server with zero install, per the registry convention.
- Evaluation charter (`docs/evaluation.md`) aligned with
  [*The Benchmark Theatre*](https://essays.bloo-mind.ai/posts/2026-05-20-mem-eval/):
  baselines before architecture, constant base model, deterministic scoring,
  audited judges, multi-seed reporting with Wilson intervals,
  cost/latency as first-class metrics, conflict-of-interest disclosure.
- LongMemEval harness v2 implementing the charter: `--system
  {telemem, full-context, grep}`, `--seeds N` (mean ± std), per-type Wilson
  95% intervals, and `--validate-judge` adversarial judge auditing.
  The charter applies to new runs; existing published results are unchanged
  pending team review.

### Changed
- The MCP SDK (`mcp>=1.6.0`) is now a core dependency — TeleMem is
  MCP-ready out of the box; `telemem[mcp]` remains as a no-op alias.

## [1.6.0] - 2026-06-12

First release published to PyPI: `pip install telemem`.

### Added
- Provider config examples with offline tests: **Ollama** (fully local stack),
  **DeepSeek**, and **Moonshot (Kimi)** — `config/config.<provider>.yaml` +
  `tests/test_providers.py`.
- Framework integration examples: `examples/langchain_memory.py` and
  `examples/llamaindex_memory.py`.
- Documentation site at https://teleai-uagi.github.io/telemem/ (mkdocs-material,
  auto-deployed via GitHub Pages).
- LongMemEval evaluation harness: `baselines/longmemeval/run_telemem.py`
  (experimental; results pending).

### Changed
- Moved the research/evaluation variant of `TeleMemory` out of the shipped
  package: `telemem/main.py` → `baselines/telemem/telemem_legacy.py`. It backs
  the ZH-4O benchmark harness (`online_query`, `offline_build_graph_json`) and
  needs `tenacity`/`pytz`, which are not core dependencies. Also fixed the
  broken `eval.py` import in the TeleMem baseline.

## [1.5.0] - 2026-06-12

### Added
- mem0-compatible return values: `Memory.add()` / `add_batch()` now return
  `{"results": [{"id", "memory", "event"}, ...]}` and `Memory.search()` returns
  `{"results": [...]}` (previously `add` returned `None` and `search` returned a
  pre-joined string), making `import telemem as mem0` a true drop-in replacement.
- New `telemem[video]` extra. The default install is now lightweight; the video
  pipeline dependencies (opencv, yt-dlp, nano-vectordb, azure-identity) moved out
  of the core requirements. `telemem[all]` installs everything.
- `telemem.mm_utils` now exposes lazy exports (`MMCoreAgent`, `process_video`,
  `init_single_video_db`, `decode_video_to_frames`, `extract_choice_from_msg`),
  with a clear install hint when the video extras are missing.
- GitHub Actions CI (pytest on Python 3.10–3.12 + package build check) and a
  tag-triggered PyPI release workflow using trusted publishing.
- `CONTRIBUTING.md`, `CITATION.cff`, issue and pull-request templates.

### Fixed
- `Mem0ValidationError` was raised but never imported (a `NameError` on invalid
  `messages` input).
- `_sync_memory_to_vector_store` returned an always-empty list due to a variable
  typo; stored memories are now reported back to the caller.
- `from telemem.mm_utils import MMCoreAgent` (as documented in the README) did
  not work because `mm_utils/__init__.py` was empty.
- Importing `telemem` no longer mutates `sys.path` or the root logging level.

## [1.4.0] - 2026-06-11

### Added
- Model Context Protocol (MCP) server: `telemem-mcp` with 8 tools
  (`add_memory`, `search_memories`, `get_memories`, `get_memory`,
  `update_memory`, `delete_memory`, `delete_all_memories`, `memory_history`)
  over stdio / SSE / streamable-http ([docs/MCP.md](docs/MCP.md)).
- uv-managed environment with committed `uv.lock`.

## [1.3.0] - 2026-01-28

- Stability improvements; TeleMem Tech Report updated to v4 on arXiv.

## [1.2.0] - 2026-01-09

- Core improvements; MiniMax provider support via OpenAI-compatible API.

## [1.1.0] - 2025-12-31

- Memobase and RAG baselines; evaluation harness improvements.

## [1.0.0] - 2025-12-05

- Initial public release: character-aware long-term memory, LLM-based semantic
  clustering and deduplication, FAISS + JSON dual storage, batch writing, and
  the multimodal video pipeline (`add_mm` / `search_mm`).
