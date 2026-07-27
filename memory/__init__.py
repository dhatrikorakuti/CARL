"""
CARL Memory System — Phase 17

Persistent semantic memory that gives CARL continuity across sessions.

Modules:
    memory_manager.py   — Legacy long-term memory (identity, preferences) [unchanged]
    config_manager.py   — API key and config storage [unchanged]
    models.py           — Data models (Memory, Project, ConversationTurn, etc.)
    importance.py       — Importance scoring, decay, and cleanup logic
    memory_store.py     — Unified structured store with importance scoring
    session_memory.py   — Session state persistence (project, conversation, tasks)
    context_engine.py   — Context retrieval for prompt injection
    summarizer.py       — Conversation summarization (every 25 turns)
    retriever.py        — Semantic search and smart recall
    project_manager.py  — Project lifecycle management and auto-detection
"""

from memory.models import (
    MemoryCategory,
    ImportanceLevel,
    ProjectStatus,
    Memory,
    Project,
    ConversationTurn,
    SearchResult,
    MemorySettings,
)
from memory.importance import ImportanceScorer
from memory.memory_store import MemoryStore, MemoryEntry, ProjectMemory, Importance
from memory.session_memory import SessionMemory
from memory.context_engine import ContextEngine
from memory.summarizer import ConversationSummarizer
from memory.retriever import MemoryRetriever
from memory.project_manager import ProjectManager

__all__ = [
    # Models
    "MemoryCategory",
    "ImportanceLevel",
    "ProjectStatus",
    "Memory",
    "Project",
    "ConversationTurn",
    "SearchResult",
    "MemorySettings",
    # Scoring
    "ImportanceScorer",
    # Store (legacy-compatible names)
    "MemoryStore",
    "MemoryEntry",
    "ProjectMemory",
    "Importance",
    # Session
    "SessionMemory",
    # Context
    "ContextEngine",
    # Summarizer
    "ConversationSummarizer",
    # Retriever
    "MemoryRetriever",
    # Project Manager
    "ProjectManager",
]
