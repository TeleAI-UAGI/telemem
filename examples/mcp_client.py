"""Talk to TeleMem through the Model Context Protocol (MCP).

Spawns the TeleMem MCP server over stdio, stores a short conversation,
then retrieves it with a semantic search — the same flow as
examples/quickstart.py, but through MCP tool calls.

Requires the MCP extra and a configured backend:

    pip install "telemem[mcp]"
    export OPENAI_API_KEY=sk-...                 # or:
    export TELEMEM_CONFIG=config/config.yaml     # custom LLM/embedder/vector store

    python examples/mcp_client.py
"""

import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MESSAGES = [
    {"role": "user", "content": "Jordan, did you take the subway to work again today?"},
    {
        "role": "assistant",
        "content": "Yes, James. The subway is much faster than driving. I leave at 7 o'clock and it's just not crowded.",
    },
    {
        "role": "user",
        "content": "Jordan, I want to try taking the subway too. Can you tell me which station is closest?",
    },
    {
        "role": "assistant",
        "content": "Of course, James. You take Line 2 to Civic Center Station, exit from Exit A, and walk 5 minutes to the company.",
    },
]


def show(label, result):
    print(f"\n=== {label} ===")
    print(json.dumps(json.loads(result.content[0].text), indent=2, ensure_ascii=False))


async def main():
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "telemem.mcp"],
        env=dict(os.environ),  # pass through TELEMEM_CONFIG / API keys
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Available tools:", ", ".join(t.name for t in tools.tools))

            added = await session.call_tool(
                "add_memory", {"messages": MESSAGES, "user_id": "Jordan"}
            )
            show("add_memory", added)

            found = await session.call_tool(
                "search_memories",
                {
                    "query": "What transportation did Jordan use to go to work today?",
                    "user_id": "Jordan",
                },
            )
            show("search_memories", found)


if __name__ == "__main__":
    asyncio.run(main())
