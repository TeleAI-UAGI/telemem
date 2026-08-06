# TeleMem MCP server image.
#
# The server starts without any API key (the TeleMem Memory instance is
# created lazily on the first tool call), so introspection works out of
# the box:
#
#   docker build -t telemem-mcp .
#   docker run -i --rm -e OPENAI_API_KEY telemem-mcp                 # stdio
#   docker run -p 8421:8421 --rm telemem-mcp --transport streamable-http --host 0.0.0.0

FROM python:3.12-slim

# Install from the local source tree so the image always matches the checkout.
WORKDIR /app
COPY pyproject.toml README.md ./
COPY telemem/ telemem/
RUN pip install --no-cache-dir .

ENTRYPOINT ["telemem-mcp"]
