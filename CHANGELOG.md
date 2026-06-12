# Changelog

All notable changes to TeleMem are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
