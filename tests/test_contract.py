"""Contract tests for TeleMem's public API — deterministic, offline.

Every promise the docs make about `add`/`add_batch`/`search` is pinned here
with fake LLM/embedder backends, so regressions in the public contract fail
CI without needing an API key.
"""

import json
import os

import numpy as np
import pytest

from telemem import TeleMemory
from telemem.mem0 import Mem0ValidationError


DEFAULT_SUMMARY = "这段内容的摘要是：\n[小明说他每天坐地铁二号线上班。]"


class FakeLLM:
    """Records every call; returns scripted responses or sensible defaults."""

    def __init__(self, responses=None):
        self.calls = []
        self.responses = list(responses or [])

    def generate_response(self, messages, response_format=None, **kwargs):
        self.calls.append({"messages": messages, "response_format": response_format})
        if self.responses:
            return self.responses.pop(0)
        if response_format and response_format.get("type") == "json_object":
            # Memory-fusion shape expected by get_update_memory_prompt handlers
            return json.dumps(
                {"stored_memories": [{"summary": "小明每天坐地铁二号线上班"}]},
                ensure_ascii=False,
            )
        return DEFAULT_SUMMARY


class FakeEmbedder:
    """Deterministic (per-process) unit vectors keyed on the input text."""

    def __init__(self, dim=8):
        self.dim = dim

    def embed(self, text, memory_action=None):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.random(self.dim)
        return (v / np.linalg.norm(v)).tolist()


def make_memory(llm=None):
    """A TeleMemory whose storage backends are replaced by recorders."""
    mem = TeleMemory.__new__(TeleMemory)
    mem.buffer_size = 2
    mem.similarity_threshold = 0.95
    mem.memory_buffer = {}
    mem.buffer_locks = {}
    mem.reranker = None
    mem.llm = llm if llm is not None else FakeLLM()
    mem.embedding_model = FakeEmbedder()

    mem.created = []

    def _create_memory(data, existing_embeddings=None, metadata=None):
        mem.created.append({"data": data, "metadata": metadata})
        return f"id-{len(mem.created)}"

    def _search_vector_store(query, filters, limit, threshold=None):
        return []

    mem._create_memory = _create_memory
    mem._search_vector_store = _search_vector_store
    return mem


# ---------------------------------------------------------------- add()


def test_character_prompt_receives_parsed_dialogue():
    """The per-character extraction prompt must contain the actual dialogue,
    not the repr of a helper function (regression test)."""
    llm = FakeLLM()
    mem = make_memory(llm=llm)
    mem.add(
        [
            {"role": "user", "content": "Jordan: I take Line 2 to work every day."},
            {"role": "assistant", "content": "Nice, the subway is fast."},
        ],
        user_id="Jordan",
    )
    extraction_user_prompt = llm.calls[0]["messages"][1]["content"]
    assert "<function" not in extraction_user_prompt
    assert "Nice, the subway is fast." in extraction_user_prompt
    assert "Jordan" in extraction_user_prompt


def test_add_requires_a_scope_id():
    mem = make_memory()
    with pytest.raises(Mem0ValidationError):
        mem.add("hello")


def test_add_rejects_unknown_memory_type():
    mem = make_memory()
    with pytest.raises(Mem0ValidationError):
        mem.add("hello", user_id="u", memory_type="semantic_banana")


def test_add_without_user_id_stores_in_events_scope():
    """Memories added with no user profile must land in the shared "events"
    scope, where search() (which always includes "events") can find them."""
    mem = make_memory()
    result = mem.add("Team meeting moved to Friday", run_id="run-1")
    assert result["results"], "expected at least one stored memory"
    assert all(c["metadata"]["user_id"] == "events" for c in mem.created)
    assert all(c["metadata"]["run_id"] == "run-1" for c in mem.created)


def test_infer_false_stores_raw_and_never_calls_llm():
    llm = FakeLLM()
    mem = make_memory(llm=llm)
    result = mem.add(
        [
            {"role": "system", "content": "you are a bot"},
            {"role": "user", "content": "I moved to Berlin"},
            {"role": "assistant", "content": "Congrats!"},
        ],
        user_id="u1",
        infer=False,
    )
    assert llm.calls == []
    assert [r["memory"] for r in result["results"]] == ["I moved to Berlin", "Congrats!"]
    assert [r["event"] for r in result["results"]] == ["ADD", "ADD"]
    assert mem.created[0]["metadata"]["role"] == "user"
    assert mem.created[1]["metadata"]["role"] == "assistant"
    assert all(c["metadata"]["user_id"] == "u1" for c in mem.created)


def test_prompt_override_becomes_system_prompt():
    llm = FakeLLM()
    mem = make_memory(llm=llm)
    mem.add(
        [{"role": "user", "content": "hello there"}],
        user_id="u",
        prompt="EXTRACT FACTS AS A JSON LIST",
    )
    extraction = llm.calls[0]["messages"]
    assert extraction[0]["content"] == "EXTRACT FACTS AS A JSON LIST"
    assert "hello there" in extraction[1]["content"]


def test_procedural_memory_type_delegates_to_mem0():
    mem = make_memory()
    seen = {}

    def fake_procedural(messages, metadata=None, prompt=None):
        seen.update(messages=messages, metadata=metadata, prompt=prompt)
        return {"results": [{"id": "p1", "memory": "proc", "event": "ADD"}]}

    mem._create_procedural_memory = fake_procedural
    result = mem.add(
        [{"role": "user", "content": "step 1: open the valve"}],
        agent_id="agent-1",
        memory_type="procedural_memory",
        prompt="P",
    )
    assert result["results"][0]["id"] == "p1"
    assert seen["prompt"] == "P"
    assert seen["metadata"]["agent_id"] == "agent-1"


# ---------------------------------------------------------------- add_batch()


def test_add_batch_infer_false_stores_raw_per_scope():
    mem = make_memory()
    result = mem.add_batch(
        [[{"role": "user", "content": "fact one"}]],
        user_id=["A", "B"],
        infer=False,
    )
    scopes = {c["metadata"]["user_id"] for c in mem.created}
    assert scopes == {"A", "B", "events"}
    assert len(result["results"]) == 3
    assert all(r["memory"] == "fact one" for r in result["results"])


def test_add_batch_rejects_memory_type():
    mem = make_memory()
    with pytest.raises(Mem0ValidationError):
        mem.add_batch([[{"role": "user", "content": "x"}]], user_id="u", memory_type="procedural_memory")


def test_add_batch_creates_character_and_events_scopes():
    """A single user_id must produce both the character profile and the shared
    "events" scope — the documented dual-write."""
    mem = make_memory()
    result = mem.add_batch(
        [[
            {"role": "user", "content": "Alice: I adopted a cat named Miso."},
            {"role": "assistant", "content": "Miso sounds adorable."},
        ]],
        user_id="Alice",
        run_id="r1",
    )
    scopes = {c["metadata"]["user_id"] for c in mem.created}
    assert scopes == {"Alice", "events"}
    assert result["results"], "expected flushed memories from both scopes"


# ---------------------------------------------------------------- search()


def _scope_search_results(per_scope):
    def fake_search(query, filters, limit, threshold=None):
        uid = filters["user_id"]
        return [
            {"id": f"{uid}-{i}", "memory": f"memory {uid} {i}", "score": 0.5 + 0.01 * i}
            for i in range(min(per_scope, limit))
        ]

    return fake_search


def test_search_caps_merged_results_to_limit():
    """user profile + "events" are both searched; the merged result must still
    honor `limit` (highest scores win) even without a reranker."""
    mem = make_memory()
    mem._search_vector_store = _scope_search_results(per_scope=5)
    result = mem.search("q", user_id="alice", limit=5)
    assert len(result["results"]) == 5
    scores = [r["score"] for r in result["results"]]
    assert scores == sorted(scores, reverse=True)


def test_search_includes_events_scope():
    mem = make_memory()
    mem._search_vector_store = _scope_search_results(per_scope=1)
    result = mem.search("q", user_id="alice", limit=10)
    assert {r["source"] for r in result["results"]} == {"alice", "events"}


# ---------------------------------------------------------------- telemetry


def test_mem0_telemetry_is_opt_in():
    import telemem  # noqa: F401 - importing telemem sets the default

    assert "MEM0_TELEMETRY" in os.environ
    if os.environ["MEM0_TELEMETRY"].lower() in ("false", "0", "no"):
        from mem0.memory import telemetry

        assert telemetry.MEM0_TELEMETRY is False
