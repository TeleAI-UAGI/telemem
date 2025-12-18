<h1 align="center">Memobase-MCP: Long-Term Memory for AI Agents</h1>

> This project is forked from [coleam00/mcp-mem0](https://github.com/coleam00/mcp-mem0)

A template implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server integrated with [Memobase](https://memobase.io) for providing AI agents with persistent memory capabilities.

Use this as a reference point to build your MCP servers yourself, or give this as an example to an AI coding assistant and tell it to follow this example for structure and code correctness!

To run this mcp, you need to have your own Memobase backend:

- You can [deploy](../server/readme.md) a local one
- Or use [free credits](https://www.memobase.io/en) of Memobase Cloud

You should have:

- A project url. (local: `http://localhost:8019` , cloud `https://api.memobase.dev`)
- A project token. (local: `secret` , cloud `sk-proj-xxxxxx`)

## Overview

This project demonstrates how to build an MCP server that enables AI agents to store, retrieve, and search memories using semantic search. It serves as a practical template for creating your own MCP servers, simply using Memobase and a practical example.

The implementation follows the best practices laid out by Anthropic for building MCP servers, allowing seamless integration with any MCP-compatible client.

## Features

The server provides three essential memory management tools:

1. **`save_memory`**: Store any information in long-term memory with semantic indexing
2. **`get_user_profiles`**: Retrieve complete user profiles
3. **`search_memories`**: Find relevant context for a given query

## Prerequisites
- Python 3.11+

## Installation

### Using uv

1. Install uv if you don't have it:
   ```bash
   pip install uv
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/memodb-io/memobase
   ```

3. Navigate to the project directory:
   ```bash
   cd memobase/src/mcp
   ```

3. Install dependencies:
   ```bash
   uv pip install -e .
   ```

4. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```

5. Configure your environment variables in the `.env` file (see Configuration section)

### Using Docker (Recommended)

1. Build the Docker image:
   ```bash
   docker build -t memobase-mcp --build-arg PORT=8050 .
   ```

2. Create a `.env` file based on `.env.example` and configure your environment variables

## Configuration

The following environment variables can be configured in your `.env` file:

| Variable | Description | Example |
|----------|-------------|----------|
| `TRANSPORT` | Transport protocol (sse or stdio) | `sse` |
| `HOST` | Host to bind to when using SSE transport | `0.0.0.0` |
| `PORT` | Port to listen on when using SSE transport | `8050` |
| `MEMOBASE_API_KEY` | Memobase API key | `secret` |
| `MEMOBASE_BASE_URL` | Memobase base URL | `http://localhost:8019` |

## Running the Server

### Using uv

#### SSE Transport

```bash
# Set TRANSPORT=sse in .env then:
uv run src/main.py
```

The MCP server will essentially be run as an API endpoint that you can then connect to with config shown below.

### Using Docker

#### SSE Transport

```bash
docker run --env-file .env -p:8050:8050 memobase-mcp
```

The MCP server will essentially be run as an API endpoint within the container that you can then connect to with config shown below.

## Integration with MCP Clients

### Cursor

Once you have the server running with SSE transport, you can connect to it using this configuration (edit this in `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "memobase": {
      "transport": "sse",
      "url": "http://localhost:8050/sse"
    }
  }
}
```

> **Note for Windsurf users**: Use `serverUrl` instead of `url` in your configuration:
> ```json
> {
>   "mcpServers": {
>     "memobase": {
>       "transport": "sse",
>       "serverUrl": "http://localhost:8050/sse"
>     }
>   }
> }
> ```

> **Note for n8n users**: Use host.docker.internal instead of localhost since n8n has to reach outside of it's own container to the host machine:
> 
> So the full URL in the MCP node would be: http://host.docker.internal:8050/sse

Make sure to update the port if you are using a value other than the default 8050.

### Python with Stdio Configuration

Add this server to your MCP configuration for Claude Desktop, Windsurf, or any other MCP client:

```json
{
  "mcpServers": {
    "memobase": {
      "command": "your/path/to/mcp/.venv/Scripts/python.exe",
      "args": ["your/path/to/mcp/src/main.py"],
      "env": {
        "TRANSPORT": "stdio",
        "MEMOBASE_API_KEY": "YOUR-API-KEY",
        "MEMOBASE_BASE_URL": "YOUR-MEMOBASE-URL",
      }
    }
  }
}
```

### Docker with Stdio Configuration

```json
{
  "mcpServers": {
    "memobase": {
      "command": "docker",
      "args": ["run", "--rm", "-i", 
               "-e", "TRANSPORT", 
               "-e", "MEMOBASE_API_KEY", 
               "-e", "MEMOBASE_BASE_URL", 
               "memobase-mcp"],
      "env": {
        "TRANSPORT": "stdio",
        "MEMOBASE_API_KEY": "YOUR-API-KEY",
        "MEMOBASE_BASE_URL": "https://api.memobase.io",
      }
    }
  }
}
```

## Building Your Own Server

This template provides a foundation for building more complex MCP servers. To build your own:

1. Add your own tools by creating methods with the `@mcp.tool()` decorator
2. Create your own lifespan function to add your own dependencies (clients, database connections, etc.)
3. Modify the `utils.py` file for any helper functions you need for your MCP server
4. Feel free to add prompts and resources as well  with `@mcp.resource()` and `@mcp.prompt()`
