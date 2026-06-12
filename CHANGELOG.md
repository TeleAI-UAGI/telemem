# Changelog

All notable changes to TeleMem are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
