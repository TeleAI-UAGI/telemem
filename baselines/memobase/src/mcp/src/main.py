from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dotenv import load_dotenv
from memobase import AsyncMemoBaseClient, ChatBlob
from memobase.utils import string_to_uuid
import asyncio
import json
import os

from utils import get_memobase_client

load_dotenv()

# Default user ID for memory operations
DEFAULT_USER_ID = string_to_uuid("user")


# Create a dataclass for our application context
@dataclass
class MemobaseContext:
    """Context for the Memobase MCP server."""

    memobase_client: AsyncMemoBaseClient


@asynccontextmanager
async def memobase_lifespan(server: FastMCP) -> AsyncIterator[MemobaseContext]:
    """
    Manages the Memobase client lifecycle.

    Args:
        server: The FastMCP server instance

    Yields:
        MemobaseContext: The context containing the Memobase client
    """
    # Create and return the Memory client with the helper function in utils.py
    memobase_client = get_memobase_client()
    assert await memobase_client.ping(), "Failed to connect to Memobase"
    print("Memobase client connected")
    try:
        yield MemobaseContext(memobase_client=memobase_client)
    finally:
        # No explicit cleanup needed for the Memobase client
        pass


# Initialize FastMCP server with the Memobase client as context
mcp = FastMCP(
    "memobase-mcp",
    description="MCP server for long term memory storage and retrieval with Memobase",
    lifespan=memobase_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", 8050),
)


@mcp.tool()
async def save_memory(ctx: Context, text: str) -> str:
    """Save information to your long-term memory.

    This tool is designed to store any type of information that might be useful in the future.
    The content will be processed and indexed for later retrieval through semantic search.

    Args:
        ctx: The MCP server provided context which includes the Memobase client
        text: The content to store in memory, including any relevant details and context
    """
    try:
        memobase_client: AsyncMemoBaseClient = (
            ctx.request_context.lifespan_context.memobase_client
        )
        messages = [{"role": "user", "content": text}]
        u = await memobase_client.get_or_create_user(DEFAULT_USER_ID)
        await u.insert(ChatBlob(messages=messages))
        await u.flush()
        return f"Successfully saved memory: {text[:100]}..."
    except Exception as e:
        return f"Error saving memory: {str(e)}"


@mcp.tool()
async def get_user_profiles(ctx: Context) -> str:
    """Get full user profiles.

    Call this tool when user asks for a summary of complete image of itself.

    Args:
        ctx: The MCP server provided context which includes the Memobase client

    Returns:
        A list of user profiles with topic, subtopic and content.
    """
    try:
        memobase_client: AsyncMemoBaseClient = (
            ctx.request_context.lifespan_context.memobase_client
        )
        u = await memobase_client.get_or_create_user(DEFAULT_USER_ID)
        ps = await u.profile()
        return "\n".join([f"- {p.describe}" for p in ps])
    except Exception as e:
        return f"Error retrieving memories: {str(e)}"


@mcp.tool()
async def search_memories(ctx: Context, query: str, max_length: int = 1000) -> str:
    """Search user memories

    Call this tool when user ask for recall some personal information.

    Args:
        ctx: The MCP server provided context which includes the Memobase client
        query: Search query string describing what you're looking for. Can be natural language.
        max_length: Maximum content length of the returned context.
    """
    try:
        memobase_client: AsyncMemoBaseClient = (
            ctx.request_context.lifespan_context.memobase_client
        )
        u = await memobase_client.get_or_create_user(DEFAULT_USER_ID)
        ps = await u.context(
            chats=[{"role": "user", "content": query}], max_token_size=max_length
        )
        return ps
    except Exception as e:
        return f"Error searching memories: {str(e)}"


async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == "sse":
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
