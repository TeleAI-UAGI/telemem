"""TeleMem: a high-performance drop-in replacement for Mem0."""

import os as _os

# TeleMem is local-first: the anonymized PostHog telemetry inherited from the
# mem0 base library is DISABLED by default. Opt back in by setting
# MEM0_TELEMETRY=true in your environment before importing telemem.
# (setdefault: an explicit user choice always wins.)
_os.environ.setdefault("MEM0_TELEMETRY", "False")

from mem0 import *  # noqa: F401,F403,E402 - re-export the mem0 API surface
import mem0 as _mem0  # noqa: E402

from .configs import TeleMemoryConfig
from .mem0 import TeleMemory
from .mem0 import TeleMemory as Memory

__version__ = "1.8.0"

# Everything mem0 exposes, with TeleMem's classes layered on top.
__all__ = sorted(
    {name for name in dir(_mem0) if not name.startswith("_")}
    | {"Memory", "TeleMemory", "TeleMemoryConfig"}
)
