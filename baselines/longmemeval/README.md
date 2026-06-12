# TeleMem on LongMemEval

Harness for evaluating TeleMem on [LongMemEval](https://github.com/xiaowu0162/LongMemEval)
(ICLR'25), the standard benchmark for chat-assistant long-term memory: 500 questions over
five abilities — information extraction, multi-session reasoning, temporal reasoning,
knowledge updates, and abstention.

> **Status: experimental.** This harness runs end-to-end but published TeleMem numbers
> are pending — see [issue #10](https://github.com/TeleAI-UAGI/telemem/issues/10).
> Contributions of runs/results are very welcome.

## Setup

1. Download the dataset (e.g. `longmemeval_s.json`) following the
   [official instructions](https://github.com/xiaowu0162/LongMemEval#dataset).
2. Make sure TeleMem is installed (`uv sync` at the repo root) and your LLM/embedder
   endpoints are reachable (any OpenAI-compatible API).

## Run

```shell
python run_telemem.py \
    --data longmemeval_s.json \
    --telemem-config ../../config/config.yaml \
    --answer-base-url http://localhost:8081/v1 --answer-model qwen3-8b \
    --limit 50 \
    --output results/telemem_lme_s.json
```

What it does per instance:

1. **Ingest** every haystack session into a fresh, isolated TeleMem scope (`memory.add`)
2. **Retrieve** with `memory.search(question)` (top-k memories)
3. **Answer** with the configured chat model over the retrieved memories
4. **Grade** with an LLM judge (gold answer vs. hypothesis)

The output JSON contains per-question hypotheses plus a summary: overall and per-type
accuracy, average ingestion time, average search latency, and answer/judge token usage.

## Caveats

- The built-in judge is a **simplified replication** of LongMemEval's evaluation. For
  paper-grade numbers, feed the generated hypotheses to the official
  `evaluate_qa.py` from the LongMemEval repo (the output format includes
  `question_id` + `hypothesis` to make this easy).
- Token usage is counted for the answer/judge calls; TeleMem's internal
  summarization/clustering calls go through your configured LLM endpoint — meter them
  there for full-cost accounting.
- Run baselines (e.g. Mem0) with the **same** answer model, judge, and endpoints for a
  fair comparison.
