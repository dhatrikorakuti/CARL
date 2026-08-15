"""
memory/models.py — Dedicated data models for CARL's memory system.

Centralizes all memory-related data structures in one place for clean
imports and consistent serialization across the package.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, List


# ── Memory Categories ────────────────────────────────────────────────────────

class MemoryCategory:
    """Standard memory categories."""
    IDENTITY = "identity"
    PREFERENCES = "preferences"
    PROJECTS = "projects"
    CODING = "coding"
    WORK = "work"
    HABITS = "habits"
    FAVORITE_APPS = "favorite_apps"
    FAVORITE_WEBSITES = "favorite_websites"
    FAVORITE_FOLDERS = "favorite_folders"
    REMINDERS = "reminders"
    CUSTOM_COMMANDS = "custom_commands"
    NOTES = "notes"
    CONVERSATION_SUMMARY = "conversation_summary"
    TASK = "task"
    FACT = "fact"
    FAVORITE = "favorite"

    ALL = [
        IDENTITY, PREFERENCES, PROJECTS, CODING, WORK, HABITS,
        FAVORITE_APPS, FAVORITE_WEBSITES, FAVORITE_FOLDERS,
        REMINDERS, CUSTOM_COMMANDS, NOTES, CONVERSATION_SUMMARY,
        TASK, FACT, FAVORITE,
    ]


# ── Importance Levels ────────────────────────────────────────────────────────

class ImportanceLevel(IntEnum):
    """Memory importance with associated expiration behavior."""
    TEMPORARY = 1    # Expires after 24 hours
    NORMAL = 2       # Expires after 7 days
    IMPORTANT = 3    # Never expires
    CRITICAL = 4     # Never expires, pinned


# ── Project Status ───────────────────────────────────────────────────────────

class ProjectStatus:
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


# ── Memory Entry Model ───────────────────────────────────────────────────────

@dataclass
class Memory:
    """A single memory entry with full metadata."""
    id: str
    category: str
    key: str
    content: str
    importance: int = ImportanceLevel.NORMAL
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    source: str = ""
    project: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    pinned: bool = False

    @property
    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600

    @property
    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "key": self.key,
            "content": self.content,
            "importance": self.importance,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "source": self.source,
            "project": self.project,
            "metadata": self.metadata,
            "pinned": self.pinned,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Memory":
        return cls(
            id=d.get("id", ""),
            category=d.get("category", ""),
            key=d.get("key", ""),
            content=d.get("content", ""),
            importance=d.get("importance", ImportanceLevel.NORMAL),
            tags=d.get("tags", []),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            accessed_at=d.get("accessed_at", time.time()),
            access_count=d.get("access_count", 0),
            source=d.get("source", ""),
            project=d.get("project", ""),
            metadata=d.get("metadata", {}),
            pinned=d.get("pinned", False),
        )


# ── Project Model ────────────────────────────────────────────────────────────

@dataclass
class Project:
    """Full project model with timeline tracking."""
    name: str
    description: str = ""
    status: str = ProjectStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    current_phase: str = ""
    next_task: str = ""
    completed_phases: list[str] = field(default_factory=list)
    pending_phases: list[str] = field(default_factory=list)
    recent_files: list[str] = field(default_factory=list)
    last_errors: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    mission_history: list[dict] = field(default_factory=list)
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    timeline: list[dict] = field(default_factory=list)
    # timeline entries: [{"event": str, "ts": float, "detail": str}]

    @property
    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    @property
    def last_active_days_ago(self) -> float:
        return (time.time() - self.updated_at) / 86400

    def add_timeline_event(self, event: str, detail: str = "") -> None:
        """Record a timeline event."""
        self.timeline.append({
            "event": event,
            "ts": time.time(),
            "detail": detail[:200],
        })
        # Keep last 50 timeline entries
        if len(self.timeline) > 50:
            self.timeline = self.timeline[-50:]
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current_phase": self.current_phase,
            "next_task": self.next_task,
            "completed_phases": self.completed_phases,
            "pending_phases": self.pending_phases,
            "recent_files": self.recent_files[-10:],
            "last_errors": self.last_errors[-5:],
            "goals": self.goals[-10:],
            "tools_used": list(set(self.tools_used))[:20],
            "mission_history": self.mission_history[-20:],
            "notes": self.notes[:500],
            "tags": self.tags,
            "timeline": self.timeline[-50:],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        return cls(
            name=d.get("name", ""),
            description=d.get("description", ""),
            status=d.get("status", ProjectStatus.ACTIVE),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            current_phase=d.get("current_phase", ""),
            next_task=d.get("next_task", ""),
            completed_phases=d.get("completed_phases", []),
            pending_phases=d.get("pending_phases", []),
            recent_files=d.get("recent_files", []),
            last_errors=d.get("last_errors", []),
            goals=d.get("goals", []),
            tools_used=d.get("tools_used", []),
            mission_history=d.get("mission_history", []),
            notes=d.get("notes", ""),
            tags=d.get("tags", []),
            timeline=d.get("timeline", []),
        )


# ── Conversation Turn ────────────────────────────────────────────────────────

@dataclass
class ConversationTurn:
    """A single conversation turn for tracking."""
    role: str           # "user" or "carl"
    text: str
    ts: float = field(default_factory=time.time)
    tool_used: str = ""
    project: str = ""

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "text": self.text[:500],
            "ts": self.ts,
            "tool_used": self.tool_used,
            "project": self.project,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConversationTurn":
        return cls(
            role=d.get("role", ""),
            text=d.get("text", ""),
            ts=d.get("ts", time.time()),
            tool_used=d.get("tool_used", ""),
            project=d.get("project", ""),
        )


# ── Memory Search Result ─────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A scored search result."""
    memory: Memory
    score: float
    match_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "memory": self.memory.to_dict(),
            "score": round(self.score, 3),
            "match_reason": self.match_reason,
        }


# ── Memory Settings ──────────────────────────────────────────────────────────

@dataclass
class MemorySettings:
    """Configurable memory features."""
    session_memory_enabled: bool = True
    long_term_memory_enabled: bool = True
    learning_enabled: bool = True
    conversation_summary_enabled: bool = True
    auto_cleanup_enabled: bool = True
    context_injection_enabled: bool = True
    max_context_tokens: int = 1500
    summary_interval: int = 25
    cleanup_interval_hours: int = 6

    def to_dict(self) -> dict:
        return {
            "session_memory_enabled": self.session_memory_enabled,
            "long_term_memory_enabled": self.long_term_memory_enabled,
            "learning_enabled": self.learning_enabled,
            "conversation_summary_enabled": self.conversation_summary_enabled,
            "auto_cleanup_enabled": self.auto_cleanup_enabled,
            "context_injection_enabled": self.context_injection_enabled,
            "max_context_tokens": self.max_context_tokens,
            "summary_interval": self.summary_interval,
            "cleanup_interval_hours": self.cleanup_interval_hours,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemorySettings":
        return cls(
            session_memory_enabled=d.get("session_memory_enabled", True),
            long_term_memory_enabled=d.get("long_term_memory_enabled", True),
            learning_enabled=d.get("learning_enabled", True),
            conversation_summary_enabled=d.get("conversation_summary_enabled", True),
            auto_cleanup_enabled=d.get("auto_cleanup_enabled", True),
            context_injection_enabled=d.get("context_injection_enabled", True),
            max_context_tokens=d.get("max_context_tokens", 1500),
            summary_interval=d.get("summary_interval", 25),
            cleanup_interval_hours=d.get("cleanup_interval_hours", 6),
        )
