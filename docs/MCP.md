# TeleMem MCP Server

TeleMem ships a [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server that
exposes its long-term memory operations as MCP tools. Any MCP-compatible client — Claude
Desktop, Claude Code, Cursor, custom agents — can store and retrieve memories through a
local TeleMem instance.

## Installation

```shell
pip install "telemem[mcp]"
```

This installs the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) on
top of TeleMem and provides the `telemem-mcp` console command.

## Running the server

```shell
telemem-mcp                                      # stdio (default)
telemem-mcp --transport sse --port 8421          # SSE over HTTP
telemem-mcp --transport streamable-http          # Streamable HTTP
TELEMEM_CONFIG=config/config.yaml telemem-mcp    # custom TeleMem config
python -m telemem.mcp                            # equivalent module form
```

| Option | Default | Description |
| --- | --- | --- |
| `--transport` | `stdio` | One of `stdio`, `sse`, `streamable-http` |
| `--host` | `127.0.0.1` | Bind host for the HTTP transports |
| `--port` | `8421` | Bind port for the HTTP transports |
| `--config` | – | TeleMem YAML/JSON config (overrides `TELEMEM_CONFIG`) |

### Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `TELEMEM_CONFIG` | – | Path to a TeleMem YAML/JSON config file. Without it, TeleMem's default configuration is used (OpenAI models, local vector store), which needs `OPENAI_API_KEY`. |
| `TELEMEM_DEFAULT_USER_ID` | `telemem-mcp` | Memory scope used when a tool call provides no `user_id`/`agent_id`/`run_id`. |

The TeleMem `Memory` instance is created lazily on the first tool call, so the server starts
instantly and configuration problems surface as structured tool errors instead of crashes.

## Tools

| Tool | Description |
| --- | --- |
| `add_memory` | Store a fact (`text`) or conversation (`messages`) in long-term memory. |
| `search_memories` | Semantic search; returns one consolidated text passage (TeleMem fuses related memories). |
| `get_memories` | List stored memories with their `memory_id`s for a user/agent/run. |
| `get_memory` | Fetch a single memory by `memory_id`. |
| `update_memory` | Overwrite the text of an existing memory. |
| `delete_memory` | Delete a single memory by `memory_id`. |
| `delete_all_memories` | Wipe a scope. Destructive — requires an explicit `user_id`, `agent_id`, or `run_id`. |
| `memory_history` | Show the ADD/UPDATE/DELETE history of a memory. |

Errors are returned as structured JSON (`{"error": ..., "detail": ...}`) so agents can
self-correct instead of failing opaquely.

### TeleMem semantics worth knowing

- `search_memories` returns a single fused text passage, not ranked rows — this is
  TeleMem's design (related memories are clustered and merged).
- Conversation-level event memories live under the pseudo-user `"events"` and are searched
  automatically alongside the requested user. Use `get_memories` with `user_id="events"`
  to list them.
- When no scope is given, reads and writes default to `TELEMEM_DEFAULT_USER_ID`;
  `delete_all_memories` deliberately never assumes a default.

## Client configuration

### Claude Desktop / Cursor

Add to `claude_desktop_config.json` (or Cursor's `mcp.json`) — see
[examples/mcp_config.json](../examples/mcp_config.json):

```json
{
  "mcpServers": {
    "telemem": {
      "command": "telemem-mcp",
      "args": [],
      "env": {
        "TELEMEM_CONFIG": "/absolute/path/to/config/config.yaml",
        "TELEMEM_DEFAULT_USER_ID": "your-handle",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### Claude Code

```shell
claude mcp add telemem -e TELEMEM_CONFIG=/absolute/path/to/config/config.yaml -- telemem-mcp
```

### Python client

[examples/mcp_client.py](../examples/mcp_client.py) drives the server programmatically over
stdio — the quickstart flow (add a conversation, search it) expressed as MCP tool calls:

```shell
export OPENAI_API_KEY=sk-...        # or TELEMEM_CONFIG=config/config.yaml
python examples/mcp_client.py
```

## Embedding in your own server

```python
from telemem.mcp import create_server

server = create_server(host="127.0.0.1", port=8421)
server.run(transport="stdio")  # or "sse" / "streamable-http"
```

## Testing

The MCP layer has an offline test suite (no API key required) covering the tool registry,
argument mapping, scoping rules, error surfacing, and a full client/server protocol
round-trip:

```shell
python tests/test_mcp.py
```
