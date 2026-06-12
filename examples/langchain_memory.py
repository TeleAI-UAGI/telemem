"""TeleMem as long-term memory for a LangChain chat model.

The pattern is framework-agnostic and survives LangChain API churn:
  1. before answering, `memory.search(...)` retrieves relevant facts and
     they are injected into the system prompt;
  2. after each exchange, `memory.add(...)` stores the new turns.

Requirements:
    pip install telemem langchain-core langchain-openai
    export OPENAI_API_KEY=sk-...

Run:
    python examples/langchain_memory.py
"""

import os

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This example needs LangChain: pip install langchain-core langchain-openai"
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
    """Fetch relevant TeleMem memories as one context block."""
    results = memory.search(query, user_id=USER_ID, limit=5)
    return "\n".join(f"- {hit['memory']}" for hit in results["results"])


def chat(llm, memory, user_input: str) -> str:
    memories = recall(memory, user_input)
    system = (
        "You are a helpful assistant with long-term memory.\n"
        f"Relevant things you remember about {USER_ID}:\n{memories or '- (nothing yet)'}"
    )
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user_input)])

    # Persist the exchange so future sessions remember it.
    memory.add(
        [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response.content},
        ],
        user_id=USER_ID,
    )
    return response.content


def main():
    llm = ChatOpenAI(model="gpt-4.1-mini")
    memory = make_memory()

    print(chat(llm, memory, "Hi! I commute by subway on Line 2 from Civic Center Station."))
    print(chat(llm, memory, "Remind me — how do I usually get to work?"))


if __name__ == "__main__":
    main()
