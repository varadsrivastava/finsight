"""
FinSight Shared Memory Module

Contains memory management capabilities for the FinSight system including:
- Vector database management with ChromaDB
- Shared JSON storage for agent communication
- Memory search and retrieval functions
"""

from .memory_manager import SharedMemoryManager

__all__ = [
    "SharedMemoryManager",
] 