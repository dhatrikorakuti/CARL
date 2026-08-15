"""
memory/memory_store.py — Unified structured memory store for CARL.

Extends the existing memory_manager.py (which handles basic identity/preferences)
with a richer memory system that supports:
  - Importance scoring (temporary → useful → important → critical)
  - Automatic expiration of low-importance memories
  - Project memory with timeline
  - Structured entries with metadata
  - Learning from usage patterns (favorites detection)

Storage: memory/semantic_store.json
Coexists with the existing memory/long_term.json (never modifies it).
"""
from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, List, Optional


def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


_STORE_PATH = _get_base_dir() / "memory" / "semantic_store.json"
_MAX_ENTRIES = 500
_lock = threading.Lock()


# ── Importance levels ────────────────────────────────────────────────────────

class Importance(IntEnum):
    TEMPORARY = 1    # Expires after 24 hours
    USEFUL = 2       # Expires after 7 days
    IMPORTANT = 3    # Expires after 90 days
    CRITICAL = 4     # Never expires


# Expiration windows in seconds
_EXPIRY_MAP = {
    Importance.TEMPORARY: 86400,       # 24 hours
    Importance.USEFUL: 604800,         # 7 days
    Importance.IMPORTANT: 7776000,     # 90 days
    Importance.CRITICAL: 0,            # never
}


# ── Memory Entry ─────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single memory entry with metadata."""
    id: str
    category: str                       # project, habit, conversation, fact, preference, task
    key: str                            # short identifier
    content: str                        # the actual memory content
    importance: int = Importance.USEFUL
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    source: str = ""                    # what triggered this memory (user, system, learning)
    project: str = ""                   # associated project name (if any)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        expiry = _EXPIRY_MAP.get(Importance(self.importance), 0)
        if expiry == 0:
            return False
        return (time.time() - self.updated_at) > expiry

    @property
    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    def touch(self) -> None:
        """Mark as accessed (extends useful life)."""
        self.accessed_at = time.time()
        self.access_count += 1

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
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(
            id=d.get("id", ""),
            category=d.get("category", ""),
            key=d.get("key", ""),
            content=d.get("content", ""),
            importance=d.get("importance", Importance.USEFUL),
            tags=d.get("tags", []),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            accessed_at=d.get("accessed_at", time.time()),
            access_count=d.get("access_count", 0),
            source=d.get("source", ""),
            project=d.get("project", ""),
            metadata=d.get("metadata", {}),
        )


# ── Project Memory ───────────────────────────────────────────────────────────

@dataclass
class ProjectMemory:
    """Structured memory for a specific project."""
    name: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    current_phase: str = ""
    next_task: str = ""
    last_edited_files: list[str] = field(default_factory=list)
    last_errors: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    completed_phases: list[str] = field(default_factory=list)
    pending_phases: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current_phase": self.current_phase,
            "next_task": self.next_task,
            "last_edited_files": self.last_edited_files[-10:],
            "last_errors": self.last_errors[-5:],
            "goals": self.goals[-10:],
            "completed_phases": self.completed_phases,
            "pending_phases": self.pending_phases,
            "tools_used": list(set(self.tools_used))[:20],
            "notes": self.notes[:500],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectMemory":
        return cls(
            name=d.get("name", ""),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            current_phase=d.get("current_phase", ""),
            next_task=d.get("next_task", ""),
            last_edited_files=d.get("last_edited_files", []),
            last_errors=d.get("last_errors", []),
            goals=d.get("goals", []),
            completed_phases=d.get("completed_phases", []),
            pending_phases=d.get("pending_phases", []),
            tools_used=d.get("tools_used", []),
            notes=d.get("notes", ""),
        )


# ── Memory Store ─────────────────────────────────────────────────────────────

class MemoryStore:
    """
    Unified memory store for CARL's semantic memory system.

    Manages:
      - General memories (facts, habits, preferences, conversations)
      - Project memories (per-project state and timeline)
      - Usage patterns (learning favorites)
      - Automatic expiration and cleanup

    Coexists with the legacy memory_manager.py — does NOT modify long_term.json.
    """

    def __init__(self, path: Path | None = None):
        self._path = path or _STORE_PATH
        self._entries: list[MemoryEntry] = []
        self._projects: dict[str, ProjectMemory] = {}
        self._usage_counts: dict[str, int] = {}  # app/folder → usage count
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    # ── Memory CRUD ──────────────────────────────────────────────────────────

    def store(
        self,
        category: str,
        key: str,
        content: str,
        importance: int = Importance.USEFUL,
        tags: list[str] | None = None,
        project: str = "",
        source: str = "system",
        metadata: dict | None = None,
    ) -> MemoryEntry:
        """
        Store or update a memory entry.

        If an entry with the same category+key exists, it's updated.
        Otherwise a new entry is created.
        """
        self._ensure_loaded()

        existing = self._find(category, key)
        if existing:
            existing.content = content
            existing.updated_at = time.time()
            existing.importance = max(existing.importance, importance)
            if tags:
                existing.tags = list(set(existing.tags + tags))
            if project:
                existing.project = project
            if metadata:
                existing.metadata.update(metadata)
            self._save()
            return existing

        entry = MemoryEntry(
            id=f"{category}_{key}_{int(time.time())}",
            category=category,
            key=key,
            content=content,
            importance=importance,
            tags=tags or [],
            source=source,
            project=project,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        self._save()
        return entry

    def recall(self, category: str = "", key: str = "", project: str = "") -> list[MemoryEntry]:
        """Retrieve memories matching filters. Empty filter = match all."""
        self._ensure_loaded()
        results = []
        for e in self._entries:
            if e.is_expired:
                continue
            if category and e.category != category:
                continue
            if key and e.key != key:
                continue
            if project and e.project != project:
                continue
            e.touch()
            results.append(e)
        return results

    def recall_recent(self, count: int = 10, category: str = "") -> list[MemoryEntry]:
        """Get most recently updated memories."""
        self._ensure_loaded()
        filtered = [e for e in self._entries if not e.is_expired]
        if category:
            filtered = [e for e in filtered if e.category == category]
        filtered.sort(key=lambda e: e.updated_at, reverse=True)
        return filtered[:count]

    def forget(self, category: str, key: str) -> bool:
        """Remove a specific memory."""
        self._ensure_loaded()
        before = len(self._entries)
        self._entries = [e for e in self._entries if not (e.category == category and e.key == key)]
        if len(self._entries) < before:
            self._save()
            return True
        return False

    # ── Project Memory ───────────────────────────────────────────────────────

    def get_project(self, name: str) -> ProjectMemory | None:
        """Get project memory by name (case-insensitive)."""
        self._ensure_loaded()
        key = name.lower().strip()
        return self._projects.get(key)

    def update_project(
        self,
        name: str,
        current_phase: str = "",
        next_task: str = "",
        edited_file: str = "",
        error: str = "",
        goal: str = "",
        completed_phase: str = "",
        notes: str = "",
    ) -> ProjectMemory:
        """Update or create a project memory."""
        self._ensure_loaded()
        key = name.lower().strip()

        if key not in self._projects:
            self._projects[key] = ProjectMemory(name=name)

        proj = self._projects[key]
        proj.updated_at = time.time()

        if current_phase:
            proj.current_phase = current_phase
        if next_task:
            proj.next_task = next_task
        if edited_file and edited_file not in proj.last_edited_files:
            proj.last_edited_files.append(edited_file)
            proj.last_edited_files = proj.last_edited_files[-10:]
        if error:
            proj.last_errors.append(error)
            proj.last_errors = proj.last_errors[-5:]
        if goal and goal not in proj.goals:
            proj.goals.append(goal)
        if completed_phase and completed_phase not in proj.completed_phases:
            proj.completed_phases.append(completed_phase)
            # Remove from pending if it was there
            proj.pending_phases = [p for p in proj.pending_phases if p != completed_phase]
        if notes:
            proj.notes = notes[:500]

        self._save()
        return proj

    def get_active_project(self) -> ProjectMemory | None:
        """Get the most recently updated project."""
        self._ensure_loaded()
        if not self._projects:
            return None
        return max(self._projects.values(), key=lambda p: p.updated_at)

    def list_projects(self) -> list[ProjectMemory]:
        """List all projects sorted by most recently updated."""
        self._ensure_loaded()
        return sorted(self._projects.values(), key=lambda p: p.updated_at, reverse=True)

    # ── Usage Learning ───────────────────────────────────────────────────────

    def record_usage(self, item: str, category: str = "app") -> None:
        """Record usage of an app/folder/tool for learning favorites."""
        self._ensure_loaded()
        key = f"{category}:{item.lower().strip()}"
        self._usage_counts[key] = self._usage_counts.get(key, 0) + 1
        # Auto-promote to favorite after 5 uses
        if self._usage_counts[key] == 5:
            self.store(
                category="favorite",
                key=item.lower().strip(),
                content=f"Frequently used {category}: {item}",
                importance=Importance.IMPORTANT,
                tags=[category, "learned"],
                source="learning",
            )
        self._save()

    def get_favorites(self, category: str = "") -> list[tuple[str, int]]:
        """Get frequently used items sorted by count."""
        self._ensure_loaded()
        items = []
        for key, count in self._usage_counts.items():
            cat, name = key.split(":", 1) if ":" in key else ("", key)
            if category and cat != category:
                continue
            if count >= 3:  # minimum threshold for "favorite"
                items.append((name, count))
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:20]

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def cleanup_expired(self) -> int:
        """Remove expired memories. Returns count of removed entries."""
        self._ensure_loaded()
        before = len(self._entries)
        self._entries = [e for e in self._entries if not e.is_expired]
        removed = before - len(self._entries)
        if removed > 0:
            self._save()
            print(f"[Memory] [C] Cleaned {removed} expired memories")
        return removed

    # ── Stats ────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return memory store statistics."""
        self._ensure_loaded()
        active = [e for e in self._entries if not e.is_expired]
        categories: dict[str, int] = {}
        for e in active:
            categories[e.category] = categories.get(e.category, 0) + 1
        return {
            "total_memories": len(active),
            "total_projects": len(self._projects),
            "categories": categories,
            "favorites_count": len(self.get_favorites()),
            "expired_pending": sum(1 for e in self._entries if e.is_expired),
        }

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load store from disk."""
        if not self._path.exists():
            return
        try:
            with _lock:
                data = json.loads(self._path.read_text(encoding="utf-8"))
            self._entries = [MemoryEntry.from_dict(d) for d in data.get("entries", [])]
            self._projects = {
                k: ProjectMemory.from_dict(v)
                for k, v in data.get("projects", {}).items()
            }
            self._usage_counts = data.get("usage_counts", {})
        except (json.JSONDecodeError, OSError) as e:
            print(f"[Memory] [!] Store load error: {e}")

    def _save(self) -> None:
        """Persist store to disk."""
        # Trim if over max
        if len(self._entries) > _MAX_ENTRIES:
            # Remove oldest expired first, then oldest low-importance
            self._entries = [e for e in self._entries if not e.is_expired]
            if len(self._entries) > _MAX_ENTRIES:
                self._entries.sort(key=lambda e: (e.importance, e.updated_at))
                self._entries = self._entries[-_MAX_ENTRIES:]

        data = {
            "entries": [e.to_dict() for e in self._entries],
            "projects": {k: v.to_dict() for k, v in self._projects.items()},
            "usage_counts": self._usage_counts,
        }

        with _lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError as e:
                print(f"[Memory] [!] Store save error: {e}")

    def _find(self, category: str, key: str) -> MemoryEntry | None:
        """Find an existing entry by category+key."""
        for e in self._entries:
            if e.category == category and e.key == key:
                return e
        return None
