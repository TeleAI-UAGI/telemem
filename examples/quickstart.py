# quickstart.py
from vendor.TeleMem.TeleMemory import TeleMemory
from vendor.TeleMem.utils import load_config

# Load configuration and initialize memory system
config = load_config("data/config.yaml")
memory = TeleMemory.from_config(config)

# Simulate multi-turn dialogue data
messages = [
    {"role": "user", "content": "Jordan, did you take the subway to work again today?"},
    {"role": "assistant", "content": "Yes, James. The subway is much faster than driving. I leave at 7 o'clock and it's just not crowded."},
    {"role": "user", "content": "Jordan, I want to try taking the subway too. Can you tell me which station is closest?"},
    {"role": "assistant", "content": "Of course, James. You take Line 2 to Civic Center Station, exit from Exit A, and walk 5 minutes to the company."}
]

# Add conversation memory
memory.add(
    messages=messages,
    metadata={
        "sample_id": "session_001",
        "user": ["James", "Jordan"]
    }
)

# Retrieve relevant memories
query = "What transportation did Jordan use to go to work today?"
retrieved = memory.search(query=query, run_id="session_001", limit=3)

print("Retrieval results:")
print(retrieved)
