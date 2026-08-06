"""Model Context Protocol (MCP) server for TeleMem.

Exposes TeleMem's long-term memory operations as MCP tools so that any
MCP-compatible client (Claude Desktop, Claude Code, Cursor, ...) can store
and retrieve memories through a local TeleMem instance.

Built on the official MCP Python SDK v2 (``mcp.server.mcpserver.MCPServer``),
which implements MCP spec 2026-07-28 and transparently serves older
(2025-era, ``initialize``-handshake) clients as well.

Run it with::

    telemem-mcp                                      # stdio (default)
    telemem-mcp --transport streamable-http          # Streamable HTTP on :8421
    TELEMEM_CONFIG=config/config.yaml telemem-mcp    # custom TeleMem config

Environment variables:
    TELEMEM_CONFIG           Path to a TeleMem YAML/JSON config file.
                             Without it, TeleMem's default configuration is
                             used (OpenAI models, local vector store).
    TELEMEM_DEFAULT_USER_ID  Memory scope used when a tool call provides no
                             user_id/agent_id/run_id (default: "telemem-mcp").
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import threading
from typing import Annotated, Any, Callable, Dict, List, Optional

from pydantic import Field

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.types import Icon, ToolAnnotations
except ImportError as exc:  # pragma: no cover - exercised only without the dep
    raise ImportError(
        "The MCP Python SDK v2 is required to run the TeleMem MCP server. "
        'Install it with: pip install "mcp>=2" (or pip install "telemem[mcp]")'
    ) from exc

from telemem import __version__ as TELEMEM_VERSION

logger = logging.getLogger("telemem.mcp")

SERVER_NAME = "telemem"

SERVER_INSTRUCTIONS = (
    "TeleMem long-term memory. Store conversations or facts with add_memory and "
    "retrieve them with search_memories, which returns one consolidated text "
    "passage (TeleMem fuses related memories rather than returning ranked rows). "
    "Memories are scoped by user_id/agent_id/run_id; conversation-level event "
    'memories are kept under the pseudo-user "events" and searched automatically.'
)

SERVER_WEBSITE = "https://teleai-uagi.github.io/telemem/"

SERVER_ICON = Icon(
    src="https://raw.githubusercontent.com/TeleAI-UAGI/telemem/main/assets/TeleMem.png",
    mime_type="image/png",
)

# Local memory store: no external world interaction from any tool.
_READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

_memory: Any = None
_memory_lock = threading.Lock()


def _default_user_id() -> str:
    return os.getenv("TELEMEM_DEFAULT_USER_ID", "telemem-mcp")


def _get_memory() -> Any:
    """Build the TeleMem ``Memory`` singleton lazily on first tool call.

    Deferring the construction keeps server startup instant and surfaces
    configuration problems as structured tool errors instead of crashes.
    """
    global _memory
    with _memory_lock:
        if _memory is None:
            import telemem
            from telemem.utils import load_config

            config_path = os.getenv("TELEMEM_CONFIG")
            config = load_config(config_path) if config_path else None
            _memory = telemem.Memory(config=config) if config else telemem.Memory()
            logger.info("TeleMem initialized (config=%s)", config_path or "<default>")
    return _memory


def _jsonable(payload: Any) -> Any:
    """Normalize a backend result to plain JSON types (UUIDs, dates -> str)."""
    return json.loads(json.dumps(payload, ensure_ascii=False, default=str))


def _call(action: Callable[[Any], Any]) -> Dict[str, Any]:
    """Run ``action`` against the shared Memory and return its result.

    Results are returned as plain dicts so the SDK can emit them as
    ``structuredContent`` (plus the serialized-JSON text fallback). Errors
    come back as ``{"error": ..., "detail": ...}`` so agents can self-correct
    instead of failing opaquely.

    stdout is redirected to stderr for the duration of the call: stdout
    belongs to the MCP stdio transport and some LLM/vector-store backends
    are chatty on it.
    """
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = action(_get_memory())
    except Exception as exc:
        logger.exception("TeleMem MCP tool failed")
        return {"error": type(exc).__name__, "detail": str(exc)}
    return _jsonable(result)


def _resolve_scope(
    user_id: Optional[str], agent_id: Optional[str], run_id: Optional[str]
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Fall back to the server's default user when no scope is given."""
    if not any((user_id, agent_id, run_id)):
        user_id = _default_user_id()
    return user_id, agent_id, run_id


def _fuse_search_results(raw: Any) -> str:
    """Collapse a search result into one text passage for the MCP client.

    TeleMem's ``Memory.search`` returns the mem0-compatible
    ``{"results": [{"memory": ...}, ...]}`` shape; older versions returned a
    pre-joined string. Accept both.
    """
    if isinstance(raw, dict):
        texts = [item.get("memory", "") for item in raw.get("results", [])]
        return " ".join(text for text in texts if text and text.strip())
    return raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)


def _scope_dict(
    user_id: Optional[str], agent_id: Optional[str], run_id: Optional[str]
) -> Dict[str, str]:
    return {
        key: value
        for key, value in (("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id))
        if value
    }


def create_server() -> MCPServer:
    """Create the TeleMem MCP server (run it on stdio or streamable HTTP)."""

    server = MCPServer(
        SERVER_NAME,
        title="TeleMem",
        instructions=SERVER_INSTRUCTIONS,
        website_url=SERVER_WEBSITE,
        icons=[SERVER_ICON],
        version=TELEMEM_VERSION,
    )

    @server.tool(
        title="Add memory",
        description=(
            "Store a fact, preference, or conversation in TeleMem long-term memory. "
            "Provide `text` for a single statement or `messages` for conversation turns. "
            "Scoped by user_id/agent_id/run_id; defaults to the server's default user."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
        ),
        structured_output=True,
    )
    def add_memory(
        text: Annotated[
            Optional[str],
            Field(description="A single statement to remember. Use this or `messages`."),
        ] = None,
        messages: Annotated[
            Optional[List[Dict[str, str]]],
            Field(
                description=(
                    'Conversation turns as [{"role": "user"|"assistant", "content": "..."}]. '
                    "Takes precedence over `text`."
                )
            ),
        ] = None,
        user_id: Annotated[
            Optional[str], Field(description="User the memory belongs to.")
        ] = None,
        agent_id: Annotated[Optional[str], Field(description="Optional agent scope.")] = None,
        run_id: Annotated[Optional[str], Field(description="Optional run/session scope.")] = None,
        metadata: Annotated[
            Optional[Dict[str, Any]],
            Field(description="Arbitrary metadata to attach to the stored memories."),
        ] = None,
        infer: Annotated[
            bool,
            Field(
                description=(
                    "Extract salient facts with the LLM before storing (TeleMem's "
                    "default pipeline). Set false to store the raw text as-is."
                )
            ),
        ] = True,
    ) -> Dict[str, Any]:
        conversation = messages or ([{"role": "user", "content": text}] if text else None)
        if not conversation:
            return {"error": "invalid_arguments", "detail": "Provide either `text` or `messages`."}
        scope = _resolve_scope(user_id, agent_id, run_id)

        def action(memory: Any) -> Dict[str, Any]:
            memory.add(
                conversation,
                user_id=scope[0],
                agent_id=scope[1],
                run_id=scope[2],
                metadata=metadata,
                infer=infer,
            )
            return {"status": "stored", "scope": _scope_dict(*scope)}

        return _call(action)

    @server.tool(
        title="Search memories",
        description=(
            "Semantic search over TeleMem memories. Returns the matching memories as "
            "one consolidated text passage (TeleMem fuses related memories). Shared "
            'event memories (pseudo-user "events") are searched automatically.'
        ),
        annotations=_READ_ONLY,
        structured_output=True,
    )
    def search_memories(
        query: Annotated[str, Field(description="Natural-language description of what to find.")],
        user_id: Annotated[
            Optional[str], Field(description="User scope; defaults to the server's default user.")
        ] = None,
        agent_id: Annotated[Optional[str], Field(description="Optional agent scope.")] = None,
        run_id: Annotated[Optional[str], Field(description="Optional run/session scope.")] = None,
        limit: Annotated[int, Field(description="Maximum memories to consider.", ge=1)] = 20,
        threshold: Annotated[
            Optional[float],
            Field(description="Minimum similarity score in [0, 1] for a memory to match."),
        ] = None,
    ) -> Dict[str, Any]:
        scope = _resolve_scope(user_id, agent_id, run_id)
        return _call(
            lambda memory: {
                "query": query,
                "memories": _fuse_search_results(
                    memory.search(
                        query,
                        user_id=scope[0],
                        agent_id=scope[1],
                        run_id=scope[2],
                        limit=limit,
                        threshold=threshold,
                    )
                ),
            }
        )

    @server.tool(
        title="List memories",
        description=(
            "List stored memories for a user/agent/run with their memory_ids. "
            'Use user_id "events" to list shared conversation-event memories.'
        ),
        annotations=_READ_ONLY,
        structured_output=True,
    )
    def get_memories(
        user_id: Annotated[
            Optional[str], Field(description="User scope; defaults to the server's default user.")
        ] = None,
        agent_id: Annotated[Optional[str], Field(description="Optional agent scope.")] = None,
        run_id: Annotated[Optional[str], Field(description="Optional run/session scope.")] = None,
        limit: Annotated[int, Field(description="Maximum memories to return.", ge=1)] = 20,
    ) -> Dict[str, Any]:
        scope = _resolve_scope(user_id, agent_id, run_id)
        return _call(lambda memory: memory.get_all(filters=_scope_dict(*scope), top_k=limit))

    @server.tool(
        title="Get memory",
        description="Fetch a single memory by its memory_id.",
        annotations=_READ_ONLY,
        structured_output=True,
    )
    def get_memory(
        memory_id: Annotated[str, Field(description="Exact memory_id to fetch.")],
    ) -> Dict[str, Any]:
        return _call(
            lambda memory: memory.get(memory_id)
            or {"error": "not_found", "memory_id": memory_id}
        )

    @server.tool(
        title="Update memory",
        description="Overwrite the text of an existing memory by memory_id.",
        annotations=ToolAnnotations(
            readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False
        ),
        structured_output=True,
    )
    def update_memory(
        memory_id: Annotated[str, Field(description="Exact memory_id to update.")],
        text: Annotated[str, Field(description="Replacement text for the memory.")],
        metadata: Annotated[
            Optional[Dict[str, Any]], Field(description="Optional replacement metadata.")
        ] = None,
    ) -> Dict[str, Any]:
        return _call(
            lambda memory: memory.update(memory_id, text, metadata=metadata)
            or {"status": "updated", "memory_id": memory_id}
        )

    @server.tool(
        title="Delete memory",
        description="Delete a single memory by memory_id.",
        annotations=ToolAnnotations(
            readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False
        ),
        structured_output=True,
    )
    def delete_memory(
        memory_id: Annotated[str, Field(description="Exact memory_id to delete.")],
    ) -> Dict[str, Any]:
        return _call(
            lambda memory: memory.delete(memory_id)
            or {"status": "deleted", "memory_id": memory_id}
        )

    @server.tool(
        title="Delete all memories in scope",
        description=(
            "Delete every memory in the given scope. Destructive: requires an explicit "
            "user_id, agent_id, or run_id — the default user is never assumed."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False
        ),
        structured_output=True,
    )
    def delete_all_memories(
        user_id: Annotated[Optional[str], Field(description="User scope to wipe.")] = None,
        agent_id: Annotated[Optional[str], Field(description="Agent scope to wipe.")] = None,
        run_id: Annotated[Optional[str], Field(description="Run scope to wipe.")] = None,
    ) -> Dict[str, Any]:
        if not any((user_id, agent_id, run_id)):
            return {
                "error": "missing_scope",
                "detail": (
                    "Refusing to wipe memories without an explicit "
                    "user_id, agent_id, or run_id."
                ),
            }
        return _call(
            lambda memory: memory.delete_all(user_id=user_id, agent_id=agent_id, run_id=run_id)
            or {"status": "deleted", "scope": _scope_dict(user_id, agent_id, run_id)}
        )

    @server.tool(
        title="Memory history",
        description="Show the change history (ADD/UPDATE/DELETE events) of a memory.",
        annotations=_READ_ONLY,
        structured_output=True,
    )
    def memory_history(
        memory_id: Annotated[str, Field(description="Exact memory_id to inspect.")],
    ) -> Dict[str, Any]:
        return _call(
            lambda memory: {"memory_id": memory_id, "history": memory.history(memory_id)}
        )

    return server


def main(argv: Optional[List[str]] = None) -> None:
    """Console entry point: ``telemem-mcp`` / ``python -m telemem.mcp``."""
    parser = argparse.ArgumentParser(
        prog="telemem-mcp",
        description="Run the TeleMem MCP server.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="MCP transport to serve on (default: stdio; sse is deprecated)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for HTTP transports")
    parser.add_argument("--port", type=int, default=8421, help="Bind port for HTTP transports")
    parser.add_argument(
        "--config",
        help="Path to a TeleMem YAML/JSON config file (overrides TELEMEM_CONFIG)",
    )
    args = parser.parse_args(argv)

    if args.config:
        os.environ["TELEMEM_CONFIG"] = args.config

    # stdout belongs to the stdio transport; keep our own logs on stderr.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(levelname)s %(name)s | %(message)s",
    )

    server = create_server()
    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        if args.transport == "sse":
            logger.warning(
                "The HTTP+SSE transport is deprecated by the MCP spec (2026-07-28); "
                "use --transport streamable-http instead."
            )
        server.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
