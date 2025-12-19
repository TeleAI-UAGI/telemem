FROM python:3.12-slim

ARG PORT=8050

WORKDIR /app

# Install uv
RUN pip install uv

# Copy the MCP server files
COPY . .

# Install packages
RUN python -m venv .venv
RUN uv pip install -e .

EXPOSE ${PORT}

# Command to run the MCP server
CMD ["uv", "run", "src/main.py"]