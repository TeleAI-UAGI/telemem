"""TeleMem: a high-performance drop-in replacement for Mem0."""

from mem0 import *  # noqa: F401,F403 - re-export the mem0 API surface
import mem0 as _mem0

from .configs import TeleMemoryConfig
from .mem0 import TeleMemory
from .mem0 import TeleMemory as Memory

__version__ = "1.6.0"

# Everything mem0 exposes, with TeleMem's classes layered on top.
__all__ = sorted(
    {name for name in dir(_mem0) if not name.startswith("_")}
    | {"Memory", "TeleMemory", "TeleMemoryConfig"}
)
