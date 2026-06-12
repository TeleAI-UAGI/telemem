"""TeleMem as long-term memory for a LlamaIndex chat LLM.

Same pattern as the LangChain example:
  1. `memory.search(...)` retrieves relevant facts before answering;
  2. `memory.add(...)` persists each exchange afterwards.

Requirements:
    pip install telemem llama-index-llms-openai
    export OPENAI_API_KEY=sk-...

Run:
    python examples/llamaindex_memory.py
"""

import os

try:
    from llama_index.core.llms import ChatMessage
    from llama_index.llms.openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This example needs LlamaIndex: pip install llama-index-llms-openai"
    ) from exc

import telemem
from telemem.utils import load_config

USER_ID = "Jordan"


def make_memory():
    config_path = os.getenv("TELEMEM_CONFIG")
    if config_path:
        return telemem.Memory(config=load_config(config_path))
    return telemem.Memory()


def recall(memory, query: str) -> str:
    results = memory.search(query, user_id=USER_ID, limit=5)
    return "\n".join(f"- {hit['memory']}" for hit in results["results"])


def chat(llm, memory, user_input: str) -> str:
    memories = recall(memory, user_input)
    system = (
        "You are a helpful assistant with long-term memory.\n"
        f"Relevant things you remember about {USER_ID}:\n{memories or '- (nothing yet)'}"
    )
    response = llm.chat(
        [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user_input),
        ]
    )
    answer = response.message.content

    memory.add(
        [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": answer},
        ],
        user_id=USER_ID,
    )
    return answer


def main():
    llm = OpenAI(model="gpt-4.1-mini")
    memory = make_memory()

    print(chat(llm, memory, "Hi! I commute by subway on Line 2 from Civic Center Station."))
    print(chat(llm, memory, "Remind me — how do I usually get to work?"))


if __name__ == "__main__":
    main()
