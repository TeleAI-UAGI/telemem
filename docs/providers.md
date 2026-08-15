# LLM Providers

TeleMem works with **any OpenAI-compatible endpoint**: set `provider: openai` and point
`openai_base_url` at your service. Ready-to-use config examples ship in
[`config/`](https://github.com/TeleAI-UAGI/telemem/tree/main/config):

| Provider | Config file | LLM | Embeddings | Notes |
| -------- | ----------- | --- | ---------- | ----- |
| **Ollama** (fully local) | `config.ollama.yaml` | any local model (e.g. `qwen3:8b`) | `nomic-embed-text`, local | No API key, no cloud |
| **DeepSeek** | `config.deepseek.yaml` | `deepseek-chat` / `deepseek-reasoner` | external (e.g. OpenAI) | `DEEPSEEK_API_KEY` + `OPENAI_API_KEY` |
| **Moonshot (Kimi)** | `config.moonshot.yaml` | `kimi-k2-0905-preview` | external (e.g. OpenAI) | `.cn` and `.ai` endpoints |
| **MiniMax** | `config.minimax.yaml` | `MiniMax-M3` (1M context) | external (e.g. OpenAI) | global `api.minimax.io` and China `api.minimaxi.com` endpoints; temperature must be in (0.0, 1.0] |
| **Local Qwen (vLLM etc.)** | `config.yaml` | `qwen3-8b` | `qwen3-8b-embedding` | the configuration used in the Tech Report |
| **OpenAI** | *(default)* | `Memory()` with no config | OpenAI | needs `OPENAI_API_KEY` |

## Usage

```python
from telemem.utils import load_config
import telemem as mem0

config = load_config("config/config.ollama.yaml")
memory = mem0.Memory(config=config)
```

Or via the environment for the bundled examples:

```shell
export DEEPSEEK_API_KEY=sk-...
export OPENAI_API_KEY=sk-...  # embeddings
TELEMEM_CONFIG=config/config.deepseek.yaml python examples/quickstart.py
```

Config values may reference environment variables as `${NAME}`. Expansion happens after
YAML/JSON parsing (so a value cannot inject configuration structure), and loading fails with
the missing variable's name if a referenced value is unset. The DeepSeek example uses this
form for both API keys.

## Fully local (Ollama)

```shell
ollama pull qwen3:8b
ollama pull nomic-embed-text
TELEMEM_CONFIG=config/config.ollama.yaml python examples/quickstart.py
```

Everything — LLM, embeddings, FAISS vector store, JSON metadata — runs on your machine.

!!! tip "Adding a provider"
    Contributions of new provider configs are very welcome — copy an existing
    `config/config.<provider>.yaml`, add offline tests like `tests/test_providers.py`,
    and open a PR. See [issue #9](https://github.com/TeleAI-UAGI/telemem/issues/9).

## Dimension consistency

If your embedder's output dimension differs from the default, pin it in **both** places:

```yaml
embedder:
  config:
    embedding_dims: 768          # what the model outputs
vector_store:
  config:
    embedding_model_dims: 768    # what the FAISS index expects
```
