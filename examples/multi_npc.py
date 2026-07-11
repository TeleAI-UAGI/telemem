"""Multi-NPC memory: shared world events + private per-character memories.

Five tavern NPCs live through the same scene. TeleMem's add_batch() writes the
conversation once and automatically produces:

  - one PRIVATE memory profile per named character (user_id per NPC), and
  - one SHARED "events" profile holding what happened in the world.

Afterwards, each NPC recalls the scene from their own perspective — Mira knows
her ring was stolen, Toma knows he saw the thief, and the shared events scope
holds the plot — without cross-character leakage.

Run with any configured provider (uses your default config, or point
TELEMEM_CONFIG at a YAML like config/config.yaml):

    TELEMEM_CONFIG=config/config.yaml python examples/multi_npc.py
"""

import os

import telemem as mem0
from telemem.utils import load_config

NPCS = ["Mira", "Toma", "Serrin", "Old Hal", "Vex"]

# One in-world scene, written as ordinary role/content turns. Speaker names in
# the content let the per-character extractor attribute facts correctly.
SCENE = [
    [
        {"role": "user", "content": "Mira: My silver ring is gone! I left it on the bar while I washed the tankards."},
        {"role": "assistant", "content": "Old Hal: Calm down, girl. Nobody's left the tavern since the storm started."},
    ],
    [
        {"role": "user", "content": "Toma: I saw Vex near the bar right before Mira shouted. He slipped something into his coat."},
        {"role": "assistant", "content": "Serrin: That is a serious accusation, Toma. Vex, turn out your pockets."},
    ],
    [
        {"role": "user", "content": "Vex: Fine — it's a ring, but it's MY mother's ring. I pawned it here years ago and Hal kept it behind the bar."},
        {"role": "assistant", "content": "Old Hal: ...That's true. I forgot it was still in the cigar box. Mira's ring must have rolled behind the taps."},
    ],
]


def make_memory():
    config_path = os.getenv("TELEMEM_CONFIG")
    if config_path:
        return mem0.Memory(config=load_config(config_path))
    return mem0.Memory()


def main():
    memory = make_memory()

    # One call: every NPC gets a private profile pass + a shared "events" pass.
    memory.add_batch(SCENE, user_id=NPCS, run_id="tavern_night_012")

    print("=" * 70)
    print("What does each NPC privately remember about the missing ring?")
    print("=" * 70)
    for npc in NPCS:
        results = memory.search(
            "What do you know about the missing silver ring?",
            user_id=npc,
            run_id="tavern_night_012",
            limit=3,
        )
        print(f"\n--- {npc} ---")
        for hit in results["results"]:
            # `source` says which scope the memory came from: the NPC's own
            # private profile, or the shared "events" world-state.
            print(f"  [{hit['source']}] {hit['memory']}")

    print("\n" + "=" * 70)
    print("Shared world events (no user_id → the \"events\" scope only)")
    print("=" * 70)
    results = memory.search(
        "What happened in the tavern tonight?",
        run_id="tavern_night_012",
        limit=5,
    )
    for hit in results["results"]:
        print(f"  - {hit['memory']}")


if __name__ == "__main__":
    main()
