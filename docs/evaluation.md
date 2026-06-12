# Evaluation Principles

How TeleMem measures itself — and how we ask you to read any number we publish.

Agent-memory benchmark scores are, as a field, unreliable: corrupted ground
truth, lenient LLM judges, corpora that fit inside context windows, vendors
evaluating their own systems, single-run point estimates, and static QA that
misses what memory is actually for. The essay
[*The Benchmark Theatre*](https://essays.bloo-mind.ai/posts/2026-05-20-mem-eval/)
documents these failures in detail. TeleMem's evaluation strategy is designed
to not commit them — and where we still fall short, to say so explicitly.

## The charter

Every TeleMem-published result must satisfy, or explicitly disclose deviation from:

1. **Baselines before architecture.** Every table includes a **full-context
   baseline** (entire history in the prompt) and a **keyword-grep baseline**
   (no memory system, simple lexical retrieval). If TeleMem cannot beat both
   by a clear margin on *accuracy*, the claim must shift to what is actually
   measured: cost, latency, or scale beyond the context window.
2. **Hold the base model constant.** All systems in one table use the same
   LLM, embedder, and answer prompt. We never copy numbers from other vendors'
   papers into our tables.
3. **Deterministic scoring where possible.** Multiple-choice exact match
   (as in ZH-4O) over LLM judges wherever the dataset allows.
4. **Judges are published and audited.** When an LLM judge is unavoidable,
   the prompt is published verbatim in the repo, and the harness ships an
   **adversarial judge validation mode**: gold answers must pass and
   shuffled (wrong-but-topical) answers must fail. We report both acceptance
   rates next to any judged score.
5. **Multi-seed or it didn't happen.** Headline comparisons require ≥ 5
   independent runs (10 preferred), reported as mean ± std with Wilson 95%
   intervals per category. Single-run numbers are labeled *preliminary*.
6. **Deltas under the noise floor are noise.** We do not claim a win when
   confidence intervals overlap, and we treat sub-10-point gaps on static
   QA benchmarks as weak evidence regardless.
7. **Cost and latency are first-class metrics.** Every run reports token
   usage and wall-clock for ingestion and query, not just accuracy.
8. **Reproducibility is the deliverable.** Harnesses, configs, prompts, and
   exact model versions live in [`baselines/`](https://github.com/TeleAI-UAGI/telemem/tree/main/baselines);
   anything not reproducible from the repo is marked *self-reported*.
9. **Conflict of interest is disclosed, not hidden.** We built TeleMem; we
   also ran the baselines. Treat our numbers the way the essay advises —
   as claims awaiting independent reproduction — and use our harnesses to
   check them. Reproductions (confirming or not) are welcome as issues/PRs.

## Scope and status

This charter governs **new evaluation runs** — everything produced with the
harnesses in [`baselines/`](https://github.com/TeleAI-UAGI/telemem/tree/main/baselines)
going forward. Existing published results (such as the README's ZH-4O table,
which uses deterministic multiple-choice scoring with the same Qwen3-8B stack
for every system) predate the charter; bringing them fully under it — re-runs,
added baselines, and expanded disclosures — is being coordinated with the
broader TeleAI team and tracked in
[issue #10](https://github.com/TeleAI-UAGI/telemem/issues/10).

## Roadmap

| Stage | What | Status |
| ----- | ---- | ------ |
| 1 | Full-context + grep baselines built into the LongMemEval harness | ✅ shipped (`baselines/longmemeval/`) |
| 2 | Multi-seed runs with Wilson CIs; adversarial judge validation | ✅ harness support shipped; published numbers pending ([#10](https://github.com/TeleAI-UAGI/telemem/issues/10)) |
| 3 | ZH-4O dataset audit — label-error rate and theoretical ceiling, published before anyone else audits it for us | planned |
| 4 | On-policy evaluation — AMemGym / MemoryBench-style interactive testing, where the agent's behavior shapes what gets stored | planned |
| 5 | Action-conditional evaluation (MEMTRACK-style) for the MCP server | exploratory |

## Choosing a memory system (including ours)

We endorse the essay's selection procedure even when it cuts against us:
build a full-context baseline and a grep baseline **on your own data** first.
If your task fits in a 200K context window and TeleMem doesn't beat those
baselines by a clear margin *or* save you significant cost, you may not need
TeleMem — or any memory system. Where TeleMem earns its keep: histories
beyond the context window, multi-character isolation, video memory, and
millisecond retrieval at a token cost that full-context cannot match.
