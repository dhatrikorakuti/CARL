"""
memory/retriever.py — Semantic search and smart recall for CARL.

Provides:
  - Fuzzy keyword search across all memories
  - Temporal queries ("what was I working on yesterday?")
  - Project timeline queries
  - Smart recall with natural language understanding

Uses a lightweight approach: TF-IDF-like keyword scoring with
synonym expansion and temporal parsing. No external embedding
model required — works fully offline.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import List, Optional

from memory.memory_store import MemoryStore, MemoryEntry, ProjectMemory, Importance


# ── Synonym map for semantic-like matching ───────────────────────────────────

_SYNONYMS = {
    "build": ["create", "make", "develop", "construct", "implement"],
    "fix": ["repair", "solve", "resolve", "debug", "patch"],
    "open": ["launch", "start", "run"],
    "close": ["shut", "exit", "quit", "stop"],
    "delete": ["remove", "erase", "destroy", "clean"],
    "search": ["find", "look", "locate", "query"],
    "write": ["code", "create", "author", "compose"],
    "project": ["app", "application", "program", "system"],
    "memory": ["remember", "recall", "store", "save"],
    "plan": ["reasoning", "planner", "strategy", "organize"],
    "voice": ["speech", "audio", "tts", "speak"],
    "browser": ["chrome", "edge", "firefox", "web"],
    "file": ["document", "folder", "directory"],
    "yesterday": ["last day", "previous day"],
    "last week": ["previous week", "past week"],
}

# Build reverse map for lookup
_REVERSE_SYNONYMS: dict[str, list[str]] = {}
for _key, _vals in _SYNONYMS.items():
    for _v in _vals:
        if _v not in _REVERSE_SYNONYMS:
            _REVERSE_SYNONYMS[_v] = []
        _REVERSE_SYNONYMS[_v].append(_key)
    if _key not in _REVERSE_SYNONYMS:
        _REVERSE_SYNONYMS[_key] = []
    _REVERSE_SYNONYMS[_key].extend(_vals)


# ── Temporal parsing ─────────────────────────────────────────────────────────

_TEMPORAL_PATTERNS = {
    "today": lambda: (datetime.now().replace(hour=0, minute=0, second=0), datetime.now()),
    "yesterday": lambda: (
        (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0),
        datetime.now().replace(hour=0, minute=0, second=0),
    ),
    "last week": lambda: (
        datetime.now() - timedelta(days=7),
        datetime.now(),
    ),
    "this week": lambda: (
        datetime.now() - timedelta(days=datetime.now().weekday()),
        datetime.now(),
    ),
    "last month": lambda: (
        datetime.now() - timedelta(days=30),
        datetime.now(),
    ),
    "recently": lambda: (
        datetime.now() - timedelta(days=3),
        datetime.now(),
    ),
}


class MemoryRetriever:
    """
    Smart memory retrieval with semantic-like search capabilities.

    Supports:
      - search(query) — fuzzy keyword search with synonym expansion
      - recall_temporal(query) — time-based queries ("what did I do yesterday?")
      - project_timeline(name) — full project history
      - what_was_i_working_on(when) — natural temporal recall
      - find_related(topic) — find memories related to a topic
    """

    def __init__(self, store: MemoryStore):
        self._store = store

    # ── Primary search ───────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """
        Semantic-like search across all memories.

        Uses keyword extraction + synonym expansion for fuzzy matching.
        Example: "the thing we built last week" matches "Reasoning Engine"
        if it was created last week and tagged with "build".
        """
        keywords = self._expand_keywords(query)
        time_range = self._detect_time_range(query)

        all_entries = self._store.recall_recent(100)
        scored: list[tuple[float, MemoryEntry]] = []

        for entry in all_entries:
            score = self._score_match(entry, keywords, time_range)
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    def recall_temporal(self, query: str) -> list[MemoryEntry]:
        """
        Answer time-based questions like:
          - "What was I working on yesterday?"
          - "What did we do last week?"
          - "Show me recent activity"
        """
        time_range = self._detect_time_range(query)
        if not time_range:
            # Default to last 24 hours
            time_range = (time.time() - 86400, time.time())

        start_ts, end_ts = time_range
        all_entries = self._store.recall_recent(100)

        matches = [
            e for e in all_entries
            if start_ts <= e.updated_at <= end_ts
        ]
        matches.sort(key=lambda e: e.updated_at, reverse=True)
        return matches[:20]

    def what_was_i_working_on(self, when: str = "yesterday") -> str:
        """
        Natural language answer to "what was I working on?"

        Returns a formatted summary string suitable for CARL to speak.
        """
        entries = self.recall_temporal(when)

        if not entries:
            return f"I don't have any records of activity for '{when}'."

        # Group by category
        projects_mentioned: set[str] = set()
        actions: list[str] = []
        tasks: list[str] = []

        for entry in entries[:15]:
            if entry.project:
                projects_mentioned.add(entry.project)
            if entry.category == "task":
                tasks.append(entry.content[:60])
            else:
                actions.append(entry.content[:60])

        parts: list[str] = []

        if projects_mentioned:
            parts.append(f"Projects: {', '.join(projects_mentioned)}")

        if tasks:
            parts.append(f"Tasks: {'; '.join(tasks[:3])}")
        elif actions:
            parts.append(f"Activity: {'; '.join(actions[:3])}")

        # Check project memory for more detail
        for proj_name in projects_mentioned:
            proj = self._store.get_project(proj_name)
            if proj and proj.current_phase:
                parts.append(f"{proj.name} was at phase: {proj.current_phase}")

        return ". ".join(parts) if parts else "Some activity recorded, but no specific details."

    # ── Project queries ──────────────────────────────────────────────────────

    def project_timeline(self, name: str) -> dict:
        """
        Get full timeline for a project.

        Returns:
            {
                "name": str,
                "created": float,
                "phases": [...],
                "current_phase": str,
                "next_task": str,
                "recent_activity": [...]
            }
        """
        proj = self._store.get_project(name)
        if not proj:
            # Try fuzzy match
            all_projects = self._store.list_projects()
            name_lower = name.lower()
            for p in all_projects:
                if name_lower in p.name.lower() or p.name.lower() in name_lower:
                    proj = p
                    break

        if not proj:
            return {"error": f"No project found matching '{name}'"}

        # Get related memories
        related = self._store.recall(project=proj.name)
        related.sort(key=lambda e: e.updated_at, reverse=True)

        return {
            "name": proj.name,
            "created": proj.created_at,
            "updated": proj.updated_at,
            "current_phase": proj.current_phase,
            "next_task": proj.next_task,
            "completed_phases": proj.completed_phases,
            "pending_phases": proj.pending_phases,
            "goals": proj.goals,
            "last_errors": proj.last_errors,
            "last_edited_files": proj.last_edited_files,
            "recent_activity": [
                {"content": e.content[:80], "when": e.updated_at}
                for e in related[:10]
            ],
        }

    def list_all_projects(self) -> list[dict]:
        """List all projects with basic info."""
        projects = self._store.list_projects()
        return [
            {
                "name": p.name,
                "phase": p.current_phase,
                "updated": p.updated_at,
                "next": p.next_task,
            }
            for p in projects
        ]

    # ── Related memory lookup ────────────────────────────────────────────────

    def find_related(self, topic: str, limit: int = 5) -> list[MemoryEntry]:
        """Find memories related to a topic (with synonym expansion)."""
        return self.search(topic, limit=limit)

    # ── Favorites / learned patterns ─────────────────────────────────────────

    def get_favorites(self, category: str = "") -> list[tuple[str, int]]:
        """Get learned favorites (apps, folders, tools used frequently)."""
        return self._store.get_favorites(category)

    # ── Internal scoring ─────────────────────────────────────────────────────

    def _score_match(
        self,
        entry: MemoryEntry,
        keywords: list[str],
        time_range: tuple[float, float] | None,
    ) -> float:
        """Score an entry against keywords and optional time range."""
        score = 0.0
        searchable = (
            f"{entry.content} {entry.key} {' '.join(entry.tags)} "
            f"{entry.category} {entry.project}"
        ).lower()

        # Keyword matching
        for kw in keywords:
            if kw in searchable:
                score += 1.0
            # Partial match (substring)
            elif any(kw in word for word in searchable.split()):
                score += 0.5

        if score == 0:
            return 0.0

        # Time range filter
        if time_range:
            start, end = time_range
            if not (start <= entry.updated_at <= end):
                score *= 0.3  # penalize out-of-range, don't exclude completely

        # Importance boost
        score *= (1.0 + entry.importance * 0.2)

        # Recency boost
        age_hours = (time.time() - entry.updated_at) / 3600
        if age_hours < 24:
            score *= 1.3
        elif age_hours < 168:  # 1 week
            score *= 1.1

        # Access frequency boost
        if entry.access_count > 3:
            score *= 1.1

        return score

    def _expand_keywords(self, query: str) -> list[str]:
        """Extract keywords and expand with synonyms."""
        # Basic tokenization
        tokens = re.findall(r'\b[a-zA-Z]{2,}\b', query.lower())

        # Filter stop words
        _stops = frozenset({
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "can", "to", "of", "in", "for", "on",
            "with", "at", "by", "from", "and", "but", "or", "not",
            "what", "when", "where", "who", "how", "which", "that",
            "this", "it", "my", "your", "we", "me", "you", "they",
        })
        keywords = [t for t in tokens if t not in _stops]

        # Expand with synonyms
        expanded = set(keywords)
        for kw in keywords:
            if kw in _REVERSE_SYNONYMS:
                for syn in _REVERSE_SYNONYMS[kw][:3]:
                    expanded.add(syn)
            if kw in _SYNONYMS:
                for syn in _SYNONYMS[kw][:3]:
                    expanded.add(syn)

        return list(expanded)

    def _detect_time_range(self, query: str) -> tuple[float, float] | None:
        """
        Detect temporal references in a query.
        Returns (start_timestamp, end_timestamp) or None.
        """
        query_lower = query.lower()

        for pattern, range_fn in _TEMPORAL_PATTERNS.items():
            if pattern in query_lower:
                start_dt, end_dt = range_fn()
                return (start_dt.timestamp(), end_dt.timestamp())

        # Try to detect "N days ago" pattern
        match = re.search(r'(\d+)\s*days?\s*ago', query_lower)
        if match:
            days = int(match.group(1))
            end = datetime.now()
            start = end - timedelta(days=days)
            return (start.timestamp(), end.timestamp())

        # "N hours ago"
        match = re.search(r'(\d+)\s*hours?\s*ago', query_lower)
        if match:
            hours = int(match.group(1))
            end = datetime.now()
            start = end - timedelta(hours=hours)
            return (start.timestamp(), end.timestamp())

        return None
