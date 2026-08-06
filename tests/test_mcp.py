#!/usr/bin/env python3
"""
Tests for the TeleMem MCP (Model Context Protocol) server.

The MCP server exposes TeleMem memory operations as MCP tools
(see telemem/mcp/server.py). All tests run offline: TeleMem's
Memory is replaced with an in-process fake, so no API key,
vector store, or network access is required.

Covered:
  - Tool registry: names, titles, descriptions, annotations, required parameters
  - Argument mapping onto the TeleMem Memory API
  - Default-user scoping and destructive-operation guards
  - Structured JSON error surfacing and structuredContent emission
  - stdout protection (stdout belongs to the stdio transport)
  - Full client/server protocol round-trips, in-process, over both the
    modern (2026-07-28) and legacy (initialize-handshake) protocol paths
"""

import asyncio
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import telemem.mcp.server as mcp_server
from telemem.mcp.server import create_server

EXPECTED_TOOLS = [
    "add_memory",
    "search_memories",
    "get_memories",
    "get_memory",
    "update_memory",
    "delete_memory",
    "delete_all_memories",
    "memory_history",
]

DEFAULT_USER = "telemem-mcp"


class FakeMemory:
    """Records every call; mimics TeleMem's Memory return shapes."""

    def __init__(self):
        self.calls = []

    def _record(self, name, **kwargs):
        self.calls.append((name, kwargs))

    def add(self, messages, **kwargs):
        self._record("add", messages=messages, **kwargs)
        return None  # TeleMemory.add returns None (buffered write)

    def search(self, query, **kwargs):
        self._record("search", query=query, **kwargs)
        return "Jordan commutes by subway."  # TeleMem returns fused text

    def get_all(self, *, filters=None, top_k=20):
        self._record("get_all", filters=filters, top_k=top_k)
        return {"results": [{"id": "mem-1", "memory": "Jordan commutes by subway."}]}

    def get(self, memory_id):
        self._record("get", memory_id=memory_id)
        if memory_id == "missing":
            return None
        return {"id": memory_id, "memory": "Jordan commutes by subway."}

    def update(self, memory_id, data, metadata=None):
        self._record("update", memory_id=memory_id, data=data, metadata=metadata)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        self._record("delete", memory_id=memory_id)
        return None

    def delete_all(self, user_id=None, agent_id=None, run_id=None):
        self._record("delete_all", user_id=user_id, agent_id=agent_id, run_id=run_id)
        return None

    def history(self, memory_id):
        self._record("history", memory_id=memory_id)
        return [{"memory_id": memory_id, "event": "ADD"}]


class RaisingMemory:
    """Every operation fails, to exercise the error path."""

    def __getattr__(self, name):
        def boom(*args, **kwargs):
            raise ValueError(f"backend exploded in {name}")

        return boom


class MCPToolTestCase(unittest.TestCase):
    """Base: a server wired to a FakeMemory, plus a call_tool helper."""

    memory_cls = FakeMemory

    def setUp(self):
        self.server = create_server()
        self.memory = self.memory_cls()
        self._saved_memory = mcp_server._memory
        mcp_server._memory = self.memory

    def tearDown(self):
        mcp_server._memory = self._saved_memory

    def call_tool(self, name, arguments=None):
        """Invoke a tool and decode its JSON text response.

        Also checks the text content agrees with the structuredContent that
        SDK v2 emits alongside it (open-shaped dicts are wrapped under
        "result" by the SDK's derived output schema).
        """
        result = asyncio.run(self.server.call_tool(name, arguments or {}))
        self.assertFalse(result.is_error)
        self.assertEqual(len(result.content), 1)
        payload = json.loads(result.content[0].text)
        self.assertEqual(result.structured_content, {"result": payload})
        return payload

    def last_call(self, name):
        matches = [kwargs for called, kwargs in self.memory.calls if called == name]
        self.assertTrue(matches, f"{name} was never called on Memory")
        return matches[-1]


class TestToolRegistry(MCPToolTestCase):
    """The server must expose exactly the documented tool surface."""

    def test_all_tools_registered(self):
        tools = asyncio.run(self.server.list_tools())
        self.assertEqual([t.name for t in tools], EXPECTED_TOOLS)

    def test_every_tool_has_a_description(self):
        for tool in asyncio.run(self.server.list_tools()):
            with self.subTest(tool=tool.name):
                self.assertTrue(tool.description and tool.description.strip())

    def test_every_tool_declares_annotations(self):
        """Spec 2025-11-25+ metadata: titles, behavior hints, output schemas."""
        read_only = {"search_memories", "get_memories", "get_memory", "memory_history"}
        destructive = {"update_memory", "delete_memory", "delete_all_memories"}
        for tool in asyncio.run(self.server.list_tools()):
            with self.subTest(tool=tool.name):
                self.assertTrue(tool.title and tool.title.strip())
                self.assertIsNotNone(tool.annotations)
                self.assertIsNotNone(tool.output_schema)
                self.assertEqual(tool.annotations.read_only_hint, tool.name in read_only)
                self.assertFalse(tool.annotations.open_world_hint)
                if tool.name not in read_only:
                    self.assertEqual(tool.annotations.destructive_hint, tool.name in destructive)

    def test_required_parameters(self):
        required = {
            tool.name: tool.input_schema.get("required", [])
            for tool in asyncio.run(self.server.list_tools())
        }
        self.assertEqual(required["search_memories"], ["query"])
        self.assertEqual(required["get_memory"], ["memory_id"])
        self.assertEqual(required["update_memory"], ["memory_id", "text"])
        self.assertEqual(required["delete_memory"], ["memory_id"])
        self.assertEqual(required["memory_history"], ["memory_id"])
        # add/list/delete_all validate their scopes at call time instead
        self.assertEqual(required["add_memory"], [])
        self.assertEqual(required["get_memories"], [])
        self.assertEqual(required["delete_all_memories"], [])


class TestAddMemory(MCPToolTestCase):
    def test_add_with_text_wraps_into_messages(self):
        result = self.call_tool("add_memory", {"text": "Jordan takes the subway."})
        self.assertEqual(result["status"], "stored")
        call = self.last_call("add")
        self.assertEqual(call["messages"], [{"role": "user", "content": "Jordan takes the subway."}])

    def test_add_defaults_to_server_user(self):
        result = self.call_tool("add_memory", {"text": "hello"})
        self.assertEqual(result["scope"], {"user_id": DEFAULT_USER})
        self.assertEqual(self.last_call("add")["user_id"], DEFAULT_USER)

    def test_add_with_explicit_scope_keeps_it(self):
        self.call_tool("add_memory", {"text": "hi", "user_id": "Jordan", "run_id": "r1"})
        call = self.last_call("add")
        self.assertEqual(call["user_id"], "Jordan")
        self.assertEqual(call["run_id"], "r1")

    def test_add_with_messages_passes_them_through(self):
        messages = [
            {"role": "user", "content": "Did you take the subway?"},
            {"role": "assistant", "content": "Yes, it beats driving."},
        ]
        self.call_tool("add_memory", {"messages": messages, "user_id": "Jordan"})
        self.assertEqual(self.last_call("add")["messages"], messages)

    def test_add_forwards_metadata_and_infer(self):
        self.call_tool(
            "add_memory",
            {"text": "x", "metadata": {"topic": "commute"}, "infer": False},
        )
        call = self.last_call("add")
        self.assertEqual(call["metadata"], {"topic": "commute"})
        self.assertFalse(call["infer"])

    def test_add_without_text_or_messages_is_rejected(self):
        result = self.call_tool("add_memory", {})
        self.assertEqual(result["error"], "invalid_arguments")
        self.assertEqual(self.memory.calls, [])


class TestSearchMemories(MCPToolTestCase):
    def test_search_returns_fused_text(self):
        result = self.call_tool("search_memories", {"query": "how does Jordan commute?"})
        self.assertEqual(result["memories"], "Jordan commutes by subway.")
        self.assertEqual(result["query"], "how does Jordan commute?")

    def test_search_defaults_to_server_user(self):
        self.call_tool("search_memories", {"query": "q"})
        self.assertEqual(self.last_call("search")["user_id"], DEFAULT_USER)

    def test_search_forwards_limit_and_threshold(self):
        self.call_tool(
            "search_memories",
            {"query": "q", "user_id": "Jordan", "limit": 5, "threshold": 0.8},
        )
        call = self.last_call("search")
        self.assertEqual(call["limit"], 5)
        self.assertEqual(call["threshold"], 0.8)
        self.assertEqual(call["user_id"], "Jordan")


class TestReadTools(MCPToolTestCase):
    def test_get_memories_maps_scope_to_filters(self):
        result = self.call_tool("get_memories", {"user_id": "Jordan", "limit": 7})
        call = self.last_call("get_all")
        self.assertEqual(call["filters"], {"user_id": "Jordan"})
        self.assertEqual(call["top_k"], 7)
        self.assertEqual(result["results"][0]["id"], "mem-1")

    def test_get_memories_defaults_to_server_user(self):
        self.call_tool("get_memories", {})
        self.assertEqual(self.last_call("get_all")["filters"], {"user_id": DEFAULT_USER})

    def test_get_memory_by_id(self):
        result = self.call_tool("get_memory", {"memory_id": "mem-1"})
        self.assertEqual(result["id"], "mem-1")

    def test_get_memory_not_found(self):
        result = self.call_tool("get_memory", {"memory_id": "missing"})
        self.assertEqual(result["error"], "not_found")

    def test_memory_history(self):
        result = self.call_tool("memory_history", {"memory_id": "mem-1"})
        self.assertEqual(
            result,
            {"memory_id": "mem-1", "history": [{"memory_id": "mem-1", "event": "ADD"}]},
        )


class TestWriteTools(MCPToolTestCase):
    def test_update_memory(self):
        result = self.call_tool(
            "update_memory",
            {"memory_id": "mem-1", "text": "Jordan now cycles.", "metadata": {"v": 2}},
        )
        call = self.last_call("update")
        self.assertEqual(call["data"], "Jordan now cycles.")
        self.assertEqual(call["metadata"], {"v": 2})
        self.assertIn("message", result)

    def test_delete_memory_acks_when_backend_returns_none(self):
        result = self.call_tool("delete_memory", {"memory_id": "mem-1"})
        self.assertEqual(result, {"status": "deleted", "memory_id": "mem-1"})

    def test_delete_all_requires_explicit_scope(self):
        result = self.call_tool("delete_all_memories", {})
        self.assertEqual(result["error"], "missing_scope")
        self.assertEqual(self.memory.calls, [])

    def test_delete_all_with_scope(self):
        result = self.call_tool("delete_all_memories", {"user_id": "Jordan"})
        self.assertEqual(self.last_call("delete_all")["user_id"], "Jordan")
        self.assertEqual(result["scope"], {"user_id": "Jordan"})


class TestErrorSurface(MCPToolTestCase):
    memory_cls = RaisingMemory

    def test_backend_errors_become_structured_json(self):
        for name, args in [
            ("search_memories", {"query": "q"}),
            ("get_memories", {}),
            ("get_memory", {"memory_id": "m"}),
            ("delete_memory", {"memory_id": "m"}),
        ]:
            with self.subTest(tool=name):
                result = self.call_tool(name, args)
                self.assertEqual(result["error"], "ValueError")
                self.assertIn("backend exploded", result["detail"])


class TestStdoutProtection(MCPToolTestCase):
    """stdout carries the MCP stdio protocol; tool calls must not write to it."""

    class ChattyMemory(FakeMemory):
        def search(self, query, **kwargs):
            print("noisy backend writing to stdout")  # noqa: T201
            return super().search(query, **kwargs)

    memory_cls = ChattyMemory

    def test_backend_prints_are_kept_off_stdout(self):
        captured = io.StringIO()
        with redirect_stdout(captured):
            result = self.call_tool("search_memories", {"query": "q"})
        self.assertEqual(captured.getvalue(), "")
        self.assertEqual(result["memories"], "Jordan commutes by subway.")


class TestDefaultUserOverride(MCPToolTestCase):
    def test_env_var_changes_default_user(self):
        os.environ["TELEMEM_DEFAULT_USER_ID"] = "alice"
        try:
            self.call_tool("search_memories", {"query": "q"})
            self.assertEqual(self.last_call("search")["user_id"], "alice")
        finally:
            del os.environ["TELEMEM_DEFAULT_USER_ID"]


class TestProtocolRoundTrip(unittest.TestCase):
    """Drive the server through a real MCP client, in-process.

    SDK v2 serves both protocol eras from one server: the modern
    (2026-07-28, stateless) path and the legacy initialize-handshake
    path. Both are exercised here.
    """

    def setUp(self):
        self.memory = FakeMemory()
        self._saved_memory = mcp_server._memory
        mcp_server._memory = self.memory

    def tearDown(self):
        mcp_server._memory = self._saved_memory

    def _scenario(self, mode):
        from mcp.client import Client

        async def scenario():
            server = create_server()
            async with Client(server, mode=mode) as client:
                tools = await client.list_tools()
                self.assertEqual([t.name for t in tools.tools], EXPECTED_TOOLS)

                added = await client.call_tool(
                    "add_memory", {"text": "Jordan takes the subway.", "user_id": "Jordan"}
                )
                self.assertFalse(added.is_error)
                self.assertEqual(json.loads(added.content[0].text)["status"], "stored")

                found = await client.call_tool(
                    "search_memories", {"query": "commute", "user_id": "Jordan"}
                )
                self.assertFalse(found.is_error)
                payload = json.loads(found.content[0].text)
                self.assertEqual(payload["memories"], "Jordan commutes by subway.")
                self.assertEqual(found.structured_content, {"result": payload})

        asyncio.run(scenario())

    def test_modern_list_and_call(self):
        self._scenario("auto")

    def test_legacy_initialize_list_and_call(self):
        self._scenario("legacy")


def main():
    print("=" * 60)
    print("TeleMem MCP Server – Tests (offline, no API key required)")
    print("=" * 60)
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
