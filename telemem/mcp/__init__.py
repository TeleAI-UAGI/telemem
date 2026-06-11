"""TeleMem Model Context Protocol (MCP) integration.

Run the server with ``telemem-mcp`` or ``python -m telemem.mcp``,
or embed it via :func:`create_server`.
"""

from telemem.mcp.server import create_server, main

__all__ = ["create_server", "main"]
