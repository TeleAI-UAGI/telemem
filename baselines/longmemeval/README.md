# TeleMem on LongMemEval

Harness for evaluating TeleMem on [LongMemEval](https://github.com/xiaowu0162/LongMemEval)
(ICLR'25), built to TeleMem's
[evaluation charter](https://teleai-uagi.github.io/telemem/evaluation/): the goal is not a
leaderboard number but a *defensible* one.

> **Status: experimental.** The harness runs end-to-end; published TeleMem numbers are
> pending — see [issue #10](https://github.com/TeleAI-UAGI/telemem/issues/10).

## What the harness enforces

| Charter principle | Mechanism |
| ----------------- | --------- |
| Baselines before architecture | `--system {telemem, full-context, grep}` — same answer model + prompt, different retrieval. If TeleMem doesn't beat both baselines, the table says so |
| Hold the base model constant | One `--answer-model` for every system; numbers from other vendors' papers are never merged in |
| Audited judge | Judge prompt published verbatim in `run_telemem.py`; `--validate-judge` feeds gold answers (must pass) and shuffled wrong-but-topical answers (must fail) and reports both acceptance rates |
| Multi-seed | `--seeds N` (≥5 for headline claims) → mean ± std |
| Noise-floor awareness | Per-type Wilson 95% intervals in every summary; the reporting note forbids claiming wins across overlapping intervals |
| Cost/latency first-class | Ingestion wall-clock, search latency, and token usage in every run |

## Setup

1. Download the dataset (e.g. `longmemeval_s.json`) per the
   [official instructions](https://github.com/xiaowu0162/LongMemEval#dataset).
2. `uv sync` at the repo root; have an OpenAI-compatible endpoint for the answer/judge model.

## Run

```shell
# 0. Audit the judge first — judged scores are untrustworthy without this
python run_telemem.py --data longmemeval_s.json --validate-judge \
    --answer-base-url http://localhost:8081/v1 --answer-model qwen3-8b

# 1. The two baselines every comparison needs
python run_telemem.py --data longmemeval_s.json --system full-context \
    --answer-base-url http://localhost:8081/v1 --answer-model qwen3-8b \
    --output results/full_context.json
python run_telemem.py --data longmemeval_s.json --system grep \
    --answer-base-url http://localhost:8081/v1 --answer-model qwen3-8b \
    --output results/grep.json

# 2. TeleMem, multi-seed
python run_telemem.py --data longmemeval_s.json --system telemem \
    --telemem-config ../../config/config.yaml \
    --answer-base-url http://localhost:8081/v1 --answer-model qwen3-8b \
    --seeds 5 --output results/telemem.json
```

## Reading the output

`summary.accuracy_mean ± accuracy_std` across seeds; `accuracy_by_type_pooled` with
Wilson 95% intervals and per-type `n`. Per the charter: single-seed numbers are
preliminary, and a gap between two systems whose intervals overlap is noise, not a win.

## Caveats

- The built-in judge is a simplified replication of LongMemEval's official evaluation.
  Output keeps `question_id` + `hypothesis`, so feed it to the official `evaluate_qa.py`
  for paper-grade numbers — and run `--validate-judge` for whichever judge you use.
- Token usage covers answer/judge calls; TeleMem's internal summarization/clustering
  calls go through your configured LLM endpoint — meter them there for full-cost accounting.
- The grep baseline is a simplified, non-agentic approximation of Letta's
  filesystem-and-grep result (keyword overlap scoring, no iterative search). It is a
  floor, not a faithful reproduction.
