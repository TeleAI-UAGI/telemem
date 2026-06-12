# Demo GIF

`demo.tape` is a [VHS](https://github.com/charmbracelet/vhs) tape file that generates
`docs/assets/demo.gif` — the animated quickstart GIF used in the main README.

## Prerequisites

```shell
# Install VHS (macOS)
brew install vhs

# Install VHS (Linux)
# See https://github.com/charmbracelet/vhs#installation
```

## Recording

```shell
export OPENAI_API_KEY=sk-...   # needs a real key — the demo calls add() and search()
cd /path/to/telemem
vhs demo/demo.tape             # outputs docs/assets/demo.gif
```

The resulting GIF is ~1–2 MB and covers:

1. `pip install telemem`
2. Storing a two-character conversation with `memory.add()`
3. Retrieving memories with `memory.search()`
4. `uvx telemem --help` — the zero-install MCP server

## Customising

Edit `demo.tape` to change font size, resolution, theme, or the conversation content.
VHS docs: <https://github.com/charmbracelet/vhs>
