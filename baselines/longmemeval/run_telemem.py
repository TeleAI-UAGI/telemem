#!/usr/bin/env python3
"""Run TeleMem on the LongMemEval benchmark.

LongMemEval (https://github.com/xiaowu0162/LongMemEval) tests chat-assistant
long-term memory across five abilities: information extraction, multi-session
reasoning, temporal reasoning, knowledge updates, and abstention.

Pipeline per instance:
  1. ingest every haystack session into a fresh TeleMem scope (memory.add)
  2. retrieve with memory.search(question)
  3. answer with an OpenAI-compatible chat model over the retrieved memories
  4. grade with a simple LLM judge (hypothesis vs. gold answer)

Also records ingestion wall-clock, search latency, and answer/judge token usage.

Usage:
    # 1. download the dataset (longmemeval_s.json) from the official repo:
    #    https://github.com/xiaowu0162/LongMemEval
    # 2. run:
    python run_telemem.py \
        --data longmemeval_s.json \
        --telemem-config ../../config/config.yaml \
        --answer-base-url http://localhost:8081/v1 --answer-model qwen3-8b \
        --limit 50 --output results/telemem_lme_s.json

NOTE: the built-in judge is a simplified replication. For paper-grade numbers,
feed the generated hypotheses to LongMemEval's official evaluation scripts.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ANSWER_PROMPT = """You are a helpful assistant with access to memories of past conversations with the user.

Memories (most relevant first):
{memories}

Current date: {question_date}

Based only on these memories, answer the question below. If the memories contain
no relevant information, say you don't have that information.

Question: {question}
Answer:"""

JUDGE_PROMPT = """You are grading a question-answering system.

Question: {question}
Gold answer: {answer}
Model answer: {hypothesis}

Does the model answer convey the same information as the gold answer?
For abstention questions (gold answer indicates the information is unknown/unanswerable),
the model is correct only if it also abstains.

Reply with exactly one word: yes or no."""


def build_memory(config_path: str | None, workdir: str):
    """Fresh TeleMemory with an isolated FAISS store under `workdir`."""
    import telemem
    from telemem.utils import load_config
    from telemem.configs import TeleMemoryConfig

    if config_path:
        config = load_config(config_path)
        config.vector_store.config.path = os.path.join(workdir, "faiss_db")
        config.history_db_path = os.path.join(workdir, "history.db")
    else:
        config = TeleMemoryConfig(
            vector_store={
                "provider": "faiss",
                "config": {"collection_name": "lme", "path": os.path.join(workdir, "faiss_db")},
            },
            history_db_path=os.path.join(workdir, "history.db"),
        )
    return telemem.Memory(config=config)


def normalize_session(session):
    """LongMemEval turns -> TeleMem messages (role/content only, non-empty)."""
    messages = []
    for turn in session:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    return messages


def format_memories(search_result, top_k):
    hits = search_result.get("results", []) if isinstance(search_result, dict) else []
    lines = [f"- {hit['memory']}" for hit in hits[:top_k] if hit.get("memory")]
    return "\n".join(lines) if lines else "(no relevant memories found)"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="Path to longmemeval_s.json (or _m/_oracle)")
    parser.add_argument("--telemem-config", default=None, help="TeleMem YAML config (LLM/embedder for memory ops)")
    parser.add_argument("--answer-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--answer-model", default="gpt-4.1-mini")
    parser.add_argument("--answer-api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--judge-model", default=None, help="Defaults to the answer model")
    parser.add_argument("--top-k", type=int, default=10, help="Memories given to the answer model")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N instances (0 = all)")
    parser.add_argument("--output", default="results/telemem_longmemeval.json")
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        instances = json.load(f)
    if args.limit:
        instances = instances[: args.limit]

    client = OpenAI(base_url=args.answer_base_url, api_key=args.answer_api_key)
    judge_model = args.judge_model or args.answer_model

    results = []
    tokens = defaultdict(int)
    correct_by_type = defaultdict(lambda: [0, 0])  # type -> [correct, total]

    for i, inst in enumerate(instances):
        qid = inst["question_id"]
        qtype = inst.get("question_type", "unknown")
        workdir = tempfile.mkdtemp(prefix=f"lme_{i}_")
        try:
            memory = build_memory(args.telemem_config, workdir)
            scope = f"lme_user_{i}"

            # 1. ingest haystack sessions
            t0 = time.time()
            for session in inst["haystack_sessions"]:
                messages = normalize_session(session)
                if messages:
                    memory.add(messages, user_id=scope)
            ingest_secs = time.time() - t0

            # 2. retrieve
            t0 = time.time()
            search_result = memory.search(inst["question"], user_id=scope, limit=args.top_k)
            search_secs = time.time() - t0
            memories_block = format_memories(search_result, args.top_k)

            # 3. answer
            response = client.chat.completions.create(
                model=args.answer_model,
                messages=[{
                    "role": "user",
                    "content": ANSWER_PROMPT.format(
                        memories=memories_block,
                        question_date=inst.get("question_date", "unknown"),
                        question=inst["question"],
                    ),
                }],
                temperature=0.0,
            )
            hypothesis = response.choices[0].message.content.strip()
            if response.usage:
                tokens["answer_prompt"] += response.usage.prompt_tokens
                tokens["answer_completion"] += response.usage.completion_tokens

            # 4. judge (simplified; use the official scripts for paper-grade numbers)
            verdict = client.chat.completions.create(
                model=judge_model,
                messages=[{
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=inst["question"], answer=inst["answer"], hypothesis=hypothesis
                    ),
                }],
                temperature=0.0,
            )
            is_correct = verdict.choices[0].message.content.strip().lower().startswith("yes")
            if verdict.usage:
                tokens["judge"] += verdict.usage.total_tokens

            correct_by_type[qtype][0] += int(is_correct)
            correct_by_type[qtype][1] += 1
            results.append({
                "question_id": qid,
                "question_type": qtype,
                "question": inst["question"],
                "gold_answer": inst["answer"],
                "hypothesis": hypothesis,
                "correct": is_correct,
                "memories": memories_block,
                "ingest_secs": round(ingest_secs, 2),
                "search_secs": round(search_secs, 4),
            })
            done = sum(v[1] for v in correct_by_type.values())
            acc = sum(v[0] for v in correct_by_type.values()) / max(done, 1)
            print(f"[{done}/{len(instances)}] {qid} ({qtype}): "
                  f"{'✓' if is_correct else '✗'}  running acc={acc:.3f}")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    summary = {
        "dataset": os.path.basename(args.data),
        "n": len(results),
        "accuracy": sum(r["correct"] for r in results) / max(len(results), 1),
        "accuracy_by_type": {t: c / max(n, 1) for t, (c, n) in sorted(correct_by_type.items())},
        "avg_ingest_secs": sum(r["ingest_secs"] for r in results) / max(len(results), 1),
        "avg_search_secs": sum(r["search_secs"] for r in results) / max(len(results), 1),
        "answer_judge_tokens": dict(tokens),
        "answer_model": args.answer_model,
        "telemem_config": args.telemem_config,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
