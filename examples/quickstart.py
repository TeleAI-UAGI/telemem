import os

import telemem as mem0
from telemem.utils import load_config


def make_memory():
    config_path = os.getenv("TELEMEM_CONFIG")
    if config_path:
        return mem0.Memory(config=load_config(config_path))
    return mem0.Memory()


def main():
    memory = make_memory()

    messages = [
        {"role": "user", "content": "Jordan, did you take the subway to work again today?"},
        {
            "role": "assistant",
            "content": "Yes, James. The subway is much faster than driving. I leave at 7 o'clock and it's just not crowded.",
        },
        {
            "role": "user",
            "content": "Jordan, I want to try taking the subway too. Can you tell me which station is closest?",
        },
        {
            "role": "assistant",
            "content": "Of course, James. You take Line 2 to Civic Center Station, exit from Exit A, and walk 5 minutes to the company.",
        },
    ]

    memory.add(messages=messages, user_id="Jordan")
    results = memory.search(
        "What transportation did Jordan use to go to work today?",
        user_id="Jordan",
    )
    print(results)


if __name__ == "__main__":
    main()
