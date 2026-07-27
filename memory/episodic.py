"""
memory/episodic.py — Episodic Memory Layer for CARL.

Stores important events and completed tasks as discrete episodes.
Each episode is a meaningful event (not raw chat messages).

Examples:
  "Completed Phase 16 Reasoning Engine"
  "Redesigned CARL HUD to holographic rings"
  "User prefers aqua blue theme"
  "Cleaned sales.xlsx — removed 92 duplicates"

Episodes are reinforced when accessed and decay when forgotten.
"""
from __future__ import annotations

import time
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

_STORE_PATH = Path(__file__).resolve().parent.parent / "memory" / "episodic_store.json"
_MAX_EPISODES = 300
_lock = threading.Lock()


@dataclass
class Episode:
    id: str
    content: str
    category: str  # task, milestone, decision, insight, error, interaction
    importance: int = 5  # 1-10
    confidence: float = 0.9
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    tags: list[str] = field(default_factory=list)
    source: str = ""  # user, system, reasoning, tool
    project: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    @property
    def decay_score(self) -> float:
        """Lower = more likely to be forgotten. Based on importance, access, recency."""
        recency = max(0, 1.0 - (time.time() - self.last_accessed) / (86400 * 30))
        return (self.importance / 10) * 0.4 + recency * 0.3 + min(self.access_count / 10, 1.0) * 0.3

    def touch(self):
        self.last_accessed = time.time()
        self.access_count += 1

    def to_dict(self) -> dict:
        return {
            "id": self.id, "content": self.content, "category": self.category,
            "importance": self.importance, "confidence": self.confidence,
            "created_at": self.created_at, "last_accessed": self.last_accessed,
            "access_count": self.access_count, "tags": self.tags,
            "source": self.source, "project": self.project, "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class EpisodicMemory:
    """Stores and retrieves meaningful life events/tasks."""

    def __init__(self):
        self._episodes: list[Episode] = []
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self._load()
            self._loaded = True

    def record(self, content: str, category: str = "task", importance: int = 5,
               tags: list[str] | None = None, source: str = "system", project: str = "") -> Episode:
        """Record a new episode."""
        self._ensure_loaded()
        ep = Episode(
            id=f"ep_{int(time.time())}_{len(self._episodes)}",
            content=content, category=category, importance=importance,
            tags=tags or [], source=source, project=project,
        )
        self._episodes.append(ep)
        self._save()
        return ep

    def recall(self, keywords: list[str] = None, category: str = "", limit: int = 10) -> list[Episode]:
        """Retrieve episodes matching keywords or category."""
        self._ensure_loaded()
        results = self._episodes[:]
        if category:
            results = [e for e in results if e.category == category]
        if keywords:
            scored = []
            for e in results:
                score = sum(1 for kw in keywords if kw.lower() in e.content.lower() or kw.lower() in ' '.join(e.tags))
                if score > 0:
                    scored.append((score, e))
            scored.sort(key=lambda x: (-x[0], -x[1].importance))
            results = [e for _, e in scored]
        else:
            results.sort(key=lambda e: -e.last_accessed)
        # Touch accessed episodes (reinforcement)
        for e in results[:limit]:
            e.touch()
        return results[:limit]

    def recent(self, count: int = 10) -> list[Episode]:
        """Get most recent episodes."""
        self._ensure_loaded()
        return sorted(self._episodes, key=lambda e: -e.created_at)[:count]

    def forget_trivial(self) -> int:
        """Remove low-importance episodes that haven't been accessed in 30+ days."""
        self._ensure_loaded()
        before = len(self._episodes)
        self._episodes = [e for e in self._episodes if e.decay_score > 0.15 or e.importance >= 7]
        removed = before - len(self._episodes)
        if removed:
            self._save()
        return removed

    def _load(self):
        if not _STORE_PATH.exists():
            return
        try:
            with _lock:
                data = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
            self._episodes = [Episode.from_dict(d) for d in data]
        except Exception as e:
            print(f"[Memory] Episodic load error: {e}")

    def _save(self):
        if len(self._episodes) > _MAX_EPISODES:
            self._episodes.sort(key=lambda e: e.decay_score)
            self._episodes = self._episodes[-_MAX_EPISODES:]
        with _lock:
            _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _STORE_PATH.write_text(
                json.dumps([e.to_dict() for e in self._episodes], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
