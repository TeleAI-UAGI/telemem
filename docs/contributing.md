# Contributing

Issues and pull requests are welcome! The full guide lives in
[CONTRIBUTING.md](https://github.com/TeleAI-UAGI/telemem/blob/main/CONTRIBUTING.md).

## TL;DR

```shell
git clone https://github.com/TeleAI-UAGI/telemem.git
cd telemem
uv sync --all-extras
uv run pytest tests/ -q       # offline suite, no API keys needed
```

CI runs the same suite on Python 3.10–3.12 for every PR.

## Where to start

- [Good first issues](https://github.com/TeleAI-UAGI/telemem/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
- [Help wanted](https://github.com/TeleAI-UAGI/telemem/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) — including
  [LoCoMo / LongMemEval evaluation](https://github.com/TeleAI-UAGI/telemem/issues/10),
  the highest-impact contribution right now
- New provider configs ([pattern](providers.md))
- Framework integrations (AutoGen, CrewAI, ...)

## Citation

If you use TeleMem in research, please cite the
[Tech Report](https://arxiv.org/abs/2601.06037) — see
[CITATION.cff](https://github.com/TeleAI-UAGI/telemem/blob/main/CITATION.cff).
