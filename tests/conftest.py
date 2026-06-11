"""Pytest configuration for the TeleMem test suite.

Tests that talk to a real LLM/embedding endpoint are skipped unless
TELEMEM_RUN_API_TESTS is set, so the default suite (and CI) runs offline:

    TELEMEM_RUN_API_TESTS=1 OPENAI_API_KEY=sk-... pytest tests/
"""

import os

collect_ignore = []

if not os.getenv("TELEMEM_RUN_API_TESTS"):
    # test_telemem.py exercises add/search against a live LLM endpoint;
    # test_minimax.py's integration classes need MINIMAX_API_KEY.
    collect_ignore.append("test_telemem.py")

# Memory() instantiation requires an API key to construct the OpenAI client,
# even though no network call is made. Provide a placeholder for offline runs.
os.environ.setdefault("OPENAI_API_KEY", "sk-telemem-offline-tests")
