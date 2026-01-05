"""
TeleMem: Enhanced memory management for AI Agents.

A drop-in replacement for mem0 with character-aware memory,
multimodal support, and improved performance.
"""

__version__ = "1.1.0"

# Re-export mem0 for drop-in compatibility
from mem0 import *

# Export TeleMemory explicitly
from .memory import TeleMemory
from .config import TeleMemoryConfig

# For backward compatibility: TeleMemory as Memory
Memory = TeleMemory

__all__ = [
    "TeleMemory",
    "Memory",  # Alias for backward compatibility
    "TeleMemoryConfig",
]
