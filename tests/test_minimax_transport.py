"""Exercise the shipped MiniMax preset through the real SDK with offline HTTP."""

import copy
import json
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

from telemem import TeleMemory
from telemem.utils import load_config


CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.minimax.yaml"
REGIONAL_URLS = ["https://api.minimax.io/v1", "https://api.minimax.cn/v1"]


@pytest.fixture
def memory_factory(tmp_path, monkeypatch):
    requests = []
    replies = []
    clients = []

    def respond(request):
        requests.append((str(request.url), json.loads(request.content)))
        message = replies.pop(0) if replies else {"role": "assistant", "content": "OK"}
        return httpx.Response(200, json={
            "id": "offline-completion", "object": "chat.completion", "created": 0,
            "model": requests[-1][1]["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    def sdk_client(**kwargs):
        client = OpenAI(http_client=httpx.Client(transport=httpx.MockTransport(respond)), **kwargs)
        clients.append(client)
        return client

    # Replace only the transport, retaining mem0's real factory and SDK serialization.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("mem0.llms.openai.OpenAI", sdk_client)

    def make(**overrides):
        config = load_config(str(CONFIG_PATH))
        config.llm.config.update(overrides)
        config.llm.config["api_key"] = "offline-test-key"
        config.embedder.config["api_key"] = "offline-embedding-key"
        config.vector_store.config.path = str(tmp_path / "vectors")
        config.history_db_path = str(tmp_path / "history.db")
        return TeleMemory(config), requests, replies

    yield make
    for client in clients:
        client.close()


@pytest.mark.parametrize("base_url", REGIONAL_URLS)
def test_preset_default_reaches_chat_completions(memory_factory, base_url):
    memory, requests, _ = memory_factory(openai_base_url=base_url)
    messages = [{"role": "user", "content": "Remember my preference."}]
    assert memory.llm.generate_response(messages) == "OK"
    url, payload = requests[-1]
    assert url == base_url + "/chat/completions"
    assert payload["model"] == "MiniMax-M3.1-Flash"
    assert payload["messages"] == messages
    # Omission selects the API's adaptive thinking and max effort defaults.
    for field in ("thinking", "reasoning_effort", "service_tier", "reasoning_split",
                  "store", "max_tokens", "temperature"):
        assert field not in payload


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
@pytest.mark.parametrize("base_url", REGIONAL_URLS)
def test_configured_effort_is_serialized(memory_factory, effort, base_url):
    memory, requests, _ = memory_factory(openai_base_url=base_url, reasoning_effort=effort)
    memory.llm.generate_response([{"role": "user", "content": "Remember this."}])
    assert requests[-1][1]["reasoning_effort"] == effort


@pytest.mark.parametrize("model,base_url", [
    ("MiniMax-M3", REGIONAL_URLS[0]),
    ("MiniMax-M2.7", "https://api.minimaxi.com/v1"),
    ("MiniMax-M2.7-highspeed", REGIONAL_URLS[0]),
    ("custom-model", "https://custom.example/v1"),
    ("gpt-4.1-mini", "https://api.openai.com/v1"),
])
def test_explicit_model_and_endpoint_are_not_rewritten(memory_factory, model, base_url):
    memory, requests, _ = memory_factory(model=model, openai_base_url=base_url,
                                         is_reasoning_model=False, temperature=0.7)
    memory.llm.generate_response([{"role": "user", "content": "Remember this."}])
    url, payload = requests[-1]
    assert url == base_url + "/chat/completions"
    assert payload["model"] == model
    assert payload["temperature"] == 0.7


def test_tool_reasoning_history_and_media_survive_sdk_serialization(memory_factory):
    raw_messages = []

    def capture_response(llm, response, params):
        raw_messages.append(response.choices[0].message.model_dump(exclude_none=True))

    memory, requests, replies = memory_factory(response_callback=capture_response)
    assistant = {
        "role": "assistant", "content": "<think>Recall the preference.</think>",
        "reasoning_content": "Look up the saved preference.",
        "tool_calls": [{"id": "call_memory", "type": "function", "function": {
            "name": "lookup", "arguments": '{"query":"preference"}',
        }}],
    }
    replies.append(assistant)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "What did I save?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "video_url", "video_url": {"url": "mm_file://offline-video", "fps": 1}},
    ]}]
    tools = [{"type": "function", "function": {
        "name": "lookup", "description": "Look up a saved memory.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    }}]
    original = copy.deepcopy(messages)
    result = memory.llm.generate_response(messages, tools=tools, tool_choice="auto")
    assert result["tool_calls"][0]["name"] == "lookup"
    assert raw_messages[0]["reasoning_content"] == assistant["reasoning_content"]
    assert raw_messages[0]["tool_calls"] == assistant["tool_calls"]
    assert messages == original
    history = messages + [raw_messages[0], {
        "role": "tool", "tool_call_id": "call_memory", "content": "Tea",
    }]
    assert memory.llm.generate_response(history) == "OK"
    assert requests[0][1]["tools"] == tools
    assert requests[0][1]["tool_choice"] == "auto"
    assert requests[0][1]["messages"] == original
    assert requests[1][1]["messages"] == history
