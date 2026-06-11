"""Model Context Protocol (MCP) server for TeleMem.

Exposes TeleMem's long-term memory operations as MCP tools so that any
MCP-compatible client (Claude Desktop, Claude Code, Cursor, ...) can store
and retrieve memories through a local TeleMem instance.

Run it with::

    telemem-mcp                                      # stdio (default)
    telemem-mcp --transport sse --port 8421          # SSE over HTTP
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
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The MCP SDK is required to run the TeleMem MCP server. "
        'Install it with: pip install "telemem[mcp]"'
    ) from exc

logger = logging.getLogger("telemem.mcp")

SERVER_NAME = "telemem"

SERVER_INSTRUCTIONS = (
    "TeleMem long-term memory. Store conversations or facts with add_memory and "
    "retrieve them with search_memories, which returns one consolidated text "
    "passage (TeleMem fuses related memories rather than returning ranked rows). "
    "Memories are scoped by user_id/agent_id/run_id; conversation-level event "
    'memories are kept under the pseudo-user "events" and searched automatically.'
)

_memory: Any = None
_memory_lock = threading.Lock()


def _default_user_id() -> str:
    return os.getenv("TELEMEM_DEFAULT_USER_ID", "telemem-mcp")


def _get_memory() -> Any:
    """Build the TeleMem ``Memory`` singleton lazily on first tool call.

    Deferring the import keeps server startup instant and surfaces
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


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _call(action: Callable[[Any], Any]) -> str:
    """Run ``action`` against the shared Memory and return its result as JSON.

    stdout is redirected to stderr for the duration of the call: stdout
    belongs to the MCP stdio transport and some LLM/vector-store backends
    are chatty on it.
    """
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = action(_get_memory())
    except Exception as exc:
        logger.exception("TeleMem MCP tool failed")
        return _dumps({"error": type(exc).__name__, "detail": str(exc)})
    return _dumps(result)


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
    return raw if isinstance(raw, str) else _dumps(raw)


def _scope_dict(
    user_id: Optional[str], agent_id: Optional[str], run_id: Optional[str]
) -> Dict[str, str]:
    return {
        key: value
        for key, value in (("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id))
        if value
    }


def create_server(host: str = "127.0.0.1", port: int = 8421) -> FastMCP:
    """Create the TeleMem FastMCP server (stdio, SSE, or streamable HTTP)."""

    server = FastMCP(SERVER_NAME, instructions=SERVER_INSTRUCTIONS, host=host, port=port)

    @server.tool(
        description=(
            "Store a fact, preference, or conversation in TeleMem long-term memory. "
            "Provide `text` for a single statement or `messages` for conversation turns. "
            "Scoped by user_id/agent_id/run_id; defaults to the server's default user."
        )
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
    ) -> str:
        conversation = messages or ([{"role": "user", "content": text}] if text else None)
        if not conversation:
            return _dumps(
                {"error": "invalid_arguments", "detail": "Provide either `text` or `messages`."}
            )
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
        description=(
            "Semantic search over TeleMem memories. Returns the matching memories as "
            "one consolidated text passage (TeleMem fuses related memories). Shared "
            'event memories (pseudo-user "events") are searched automatically.'
        )
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
    ) -> str:
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
        description=(
            "List stored memories for a user/agent/run with their memory_ids. "
            'Use user_id "events" to list shared conversation-event memories.'
        )
    )
    def get_memories(
        user_id: Annotated[
            Optional[str], Field(description="User scope; defaults to the server's default user.")
        ] = None,
        agent_id: Annotated[Optional[str], Field(description="Optional agent scope.")] = None,
        run_id: Annotated[Optional[str], Field(description="Optional run/session scope.")] = None,
        limit: Annotated[int, Field(description="Maximum memories to return.", ge=1)] = 20,
    ) -> str:
        scope = _resolve_scope(user_id, agent_id, run_id)
        return _call(lambda memory: memory.get_all(filters=_scope_dict(*scope), top_k=limit))

    @server.tool(description="Fetch a single memory by its memory_id.")
    def get_memory(
        memory_id: Annotated[str, Field(description="Exact memory_id to fetch.")],
    ) -> str:
        return _call(
            lambda memory: memory.get(memory_id)
            or {"error": "not_found", "memory_id": memory_id}
        )

    @server.tool(description="Overwrite the text of an existing memory by memory_id.")
    def update_memory(
        memory_id: Annotated[str, Field(description="Exact memory_id to update.")],
        text: Annotated[str, Field(description="Replacement text for the memory.")],
        metadata: Annotated[
            Optional[Dict[str, Any]], Field(description="Optional replacement metadata.")
        ] = None,
    ) -> str:
        return _call(
            lambda memory: memory.update(memory_id, text, metadata=metadata)
            or {"status": "updated", "memory_id": memory_id}
        )

    @server.tool(description="Delete a single memory by memory_id.")
    def delete_memory(
        memory_id: Annotated[str, Field(description="Exact memory_id to delete.")],
    ) -> str:
        return _call(
            lambda memory: memory.delete(memory_id)
            or {"status": "deleted", "memory_id": memory_id}
        )

    @server.tool(
        description=(
            "Delete every memory in the given scope. Destructive: requires an explicit "
            "user_id, agent_id, or run_id — the default user is never assumed."
        )
    )
    def delete_all_memories(
        user_id: Annotated[Optional[str], Field(description="User scope to wipe.")] = None,
        agent_id: Annotated[Optional[str], Field(description="Agent scope to wipe.")] = None,
        run_id: Annotated[Optional[str], Field(description="Run scope to wipe.")] = None,
    ) -> str:
        if not any((user_id, agent_id, run_id)):
            return _dumps(
                {
                    "error": "missing_scope",
                    "detail": (
                        "Refusing to wipe memories without an explicit "
                        "user_id, agent_id, or run_id."
                    ),
                }
            )
        return _call(
            lambda memory: memory.delete_all(user_id=user_id, agent_id=agent_id, run_id=run_id)
            or {"status": "deleted", "scope": _scope_dict(user_id, agent_id, run_id)}
        )

    @server.tool(description="Show the change history (ADD/UPDATE/DELETE events) of a memory.")
    def memory_history(
        memory_id: Annotated[str, Field(description="Exact memory_id to inspect.")],
    ) -> str:
        return _call(lambda memory: memory.history(memory_id))

    return server


def main(argv: Optional[List[str]] = None) -> None:
    """Console entry point: ``telemem-mcp`` / ``python -m telemem.mcp``."""
    parser = argparse.ArgumentParser(
        prog="telemem-mcp",
        description="Run the TeleMem MCP server.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport to serve on (default: stdio)",
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

    create_server(host=args.host, port=args.port).run(transport=args.transport)


if __name__ == "__main__":
    main()
