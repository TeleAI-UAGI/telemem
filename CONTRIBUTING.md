# Contributing to TeleMem

Thanks for your interest in improving TeleMem! Issues and pull requests are
welcome — this guide gets you productive quickly.

## Development setup

We use [uv](https://docs.astral.sh/uv/) for a reproducible environment:

```shell
git clone https://github.com/TeleAI-UAGI/telemem.git
cd telemem
uv sync --all-extras   # installs TeleMem (editable) + video/MCP extras + dev tools
```

## Running the tests

The default suite runs fully offline (no API keys needed):

```shell
uv run pytest tests/ -q
```

Tests that exercise a live LLM/embedding endpoint are opt-in:

```shell
TELEMEM_RUN_API_TESTS=1 OPENAI_API_KEY=sk-... uv run pytest tests/ -q
```

MiniMax integration tests additionally need `MINIMAX_API_KEY`.

## Pull requests

1. Fork and create a feature branch from `main`.
2. Keep changes focused; add or update tests for behavior changes.
3. Make sure `uv run pytest tests/ -q` passes — CI runs the same suite on
   Python 3.10–3.12.
4. If you change the public API, update both `README.md` and `README-ZH.md`,
   and add a line to `CHANGELOG.md`.

## Reporting issues

Use the issue templates. For bugs, include your Python version, TeleMem
version (`python -c "import telemem; print(telemem.__version__)"`), config
(redact keys!), and a minimal reproduction.

## Good first contributions

- New LLM / embedder provider configs (with a `config/config.<provider>.yaml`
  example and tests, like the MiniMax one).
- Framework integrations (LangChain, LlamaIndex, AutoGen, CrewAI...).
- Benchmark reproductions and evaluation scripts under `baselines/`.
- Documentation and bilingual README improvements.
