"""
memory/importance.py — Importance scoring and decay logic for CARL.

Determines and adjusts memory importance based on:
  - Content analysis (keywords suggesting high importance)
  - Access frequency (frequently accessed → promoted)
  - Age decay (unused memories gradually lose importance)
  - User signals (explicit "remember this" → CRITICAL)
  - Context (project-related → higher importance)

Also handles automatic expiration and cleanup decisions.
"""
from __future__ import annotations

import time
from typing import List

from memory.models import ImportanceLevel, Memory, MemoryCategory


# ── Expiration windows (seconds) ─────────────────────────────────────────────

EXPIRY_MAP = {
    ImportanceLevel.TEMPORARY: 86400,       # 24 hours
    ImportanceLevel.NORMAL: 604800,         # 7 days
    ImportanceLevel.IMPORTANT: 0,           # never
    ImportanceLevel.CRITICAL: 0,            # never
}


# ── Keywords that suggest higher importance ──────────────────────────────────

_CRITICAL_KEYWORDS = frozenset({
    "password", "api key", "secret", "credential", "token",
    "deadline", "emergency", "urgent", "critical", "never forget",
    "always remember", "important",
})

_IMPORTANT_KEYWORDS = frozenset({
    "project", "goal", "plan", "decision", "preference",
    "birthday", "name", "address", "phone", "email",
    "favorite", "habit", "routine", "workflow",
    "remember", "note", "pinned",
})

_TEMPORARY_KEYWORDS = frozenset({
    "weather", "time", "today", "right now", "current",
    "just", "quickly", "temporary", "for now",
})


# ── Categories with inherent importance ──────────────────────────────────────

_CATEGORY_IMPORTANCE = {
    MemoryCategory.IDENTITY: ImportanceLevel.IMPORTANT,
    MemoryCategory.PREFERENCES: ImportanceLevel.IMPORTANT,
    MemoryCategory.PROJECTS: ImportanceLevel.IMPORTANT,
    MemoryCategory.CUSTOM_COMMANDS: ImportanceLevel.IMPORTANT,
    MemoryCategory.REMINDERS: ImportanceLevel.NORMAL,
    MemoryCategory.CODING: ImportanceLevel.NORMAL,
    MemoryCategory.WORK: ImportanceLevel.NORMAL,
    MemoryCategory.HABITS: ImportanceLevel.IMPORTANT,
    MemoryCategory.FAVORITE_APPS: ImportanceLevel.IMPORTANT,
    MemoryCategory.FAVORITE_WEBSITES: ImportanceLevel.NORMAL,
    MemoryCategory.FAVORITE_FOLDERS: ImportanceLevel.NORMAL,
    MemoryCategory.NOTES: ImportanceLevel.NORMAL,
    MemoryCategory.CONVERSATION_SUMMARY: ImportanceLevel.NORMAL,
    MemoryCategory.TASK: ImportanceLevel.TEMPORARY,
    MemoryCategory.FACT: ImportanceLevel.NORMAL,
}


class ImportanceScorer:
    """
    Scores and manages memory importance levels.

    Usage:
        scorer = ImportanceScorer()
        level = scorer.score_new_memory(category, content, source)
        scorer.decay_check(memory)  # adjust based on age/access
    """

    def score_new_memory(
        self,
        category: str,
        content: str,
        source: str = "system",
        tags: list[str] | None = None,
    ) -> int:
        """
        Determine the initial importance level for a new memory.

        Factors:
          - Category inherent importance
          - Content keyword analysis
          - Source (user-explicit > system-inferred)
          - Tags
        """
        # Start with category default
        base = _CATEGORY_IMPORTANCE.get(category, ImportanceLevel.NORMAL)

        # Analyze content
        content_lower = content.lower()
        tag_str = " ".join(tags or []).lower()
        combined = f"{content_lower} {tag_str}"

        # Check for critical keywords
        if any(kw in combined for kw in _CRITICAL_KEYWORDS):
            return ImportanceLevel.CRITICAL

        # Check for important keywords
        if any(kw in combined for kw in _IMPORTANT_KEYWORDS):
            return max(base, ImportanceLevel.IMPORTANT)

        # Check for temporary keywords
        if any(kw in combined for kw in _TEMPORARY_KEYWORDS):
            return ImportanceLevel.TEMPORARY

        # User-explicit memories are at least IMPORTANT
        if source == "user":
            return max(base, ImportanceLevel.IMPORTANT)

        return base

    def should_promote(self, memory: Memory) -> bool:
        """
        Check if a memory should be promoted based on access patterns.

        Rules:
          - Accessed 5+ times → promote to IMPORTANT
          - Accessed 10+ times → promote to CRITICAL
          - Pinned memories are always CRITICAL
        """
        if memory.pinned:
            return memory.importance < ImportanceLevel.CRITICAL

        if memory.access_count >= 10 and memory.importance < ImportanceLevel.CRITICAL:
            return True
        if memory.access_count >= 5 and memory.importance < ImportanceLevel.IMPORTANT:
            return True

        return False

    def promote(self, memory: Memory) -> int:
        """
        Promote a memory's importance based on access patterns.
        Returns the new importance level.
        """
        if memory.pinned:
            memory.importance = ImportanceLevel.CRITICAL
        elif memory.access_count >= 10:
            memory.importance = ImportanceLevel.CRITICAL
        elif memory.access_count >= 5:
            memory.importance = ImportanceLevel.IMPORTANT

        return memory.importance

    def is_expired(self, memory: Memory) -> bool:
        """Check if a memory has expired based on its importance level."""
        if memory.pinned:
            return False

        expiry = EXPIRY_MAP.get(ImportanceLevel(memory.importance), 0)
        if expiry == 0:
            return False

        # Use updated_at (not created_at) — accessing resets the clock
        return (time.time() - memory.updated_at) > expiry

    def decay_check(self, memory: Memory) -> int | None:
        """
        Check if a memory should be decayed (demoted) due to age + no access.

        Returns the new importance level if decayed, or None if no change.

        Rules:
          - NORMAL memory not accessed in 30 days → TEMPORARY
          - IMPORTANT memory not accessed in 180 days → NORMAL (unless pinned)
          - CRITICAL never decays
          - Pinned never decays
        """
        if memory.pinned or memory.importance == ImportanceLevel.CRITICAL:
            return None

        age_since_access = time.time() - memory.accessed_at

        if memory.importance == ImportanceLevel.IMPORTANT:
            # 180 days without access → demote to NORMAL
            if age_since_access > 15552000:
                memory.importance = ImportanceLevel.NORMAL
                return ImportanceLevel.NORMAL

        elif memory.importance == ImportanceLevel.NORMAL:
            # 30 days without access → demote to TEMPORARY
            if age_since_access > 2592000:
                memory.importance = ImportanceLevel.TEMPORARY
                return ImportanceLevel.TEMPORARY

        return None

    def get_cleanup_candidates(self, memories: list[Memory], max_to_remove: int = 50) -> list[Memory]:
        """
        Identify memories that should be cleaned up.

        Priority for removal:
          1. Expired TEMPORARY memories
          2. Old NORMAL memories with zero access
          3. Duplicate content (same category+similar content)

        Never removes IMPORTANT or CRITICAL.
        Never removes pinned.
        """
        candidates: list[Memory] = []

        for mem in memories:
            if mem.pinned:
                continue
            if mem.importance >= ImportanceLevel.IMPORTANT:
                continue

            # Expired
            if self.is_expired(mem):
                candidates.append(mem)
                continue

            # Zero access + old
            if mem.access_count == 0 and mem.age_days > 14:
                candidates.append(mem)

        # Sort by importance (lowest first), then by age (oldest first)
        candidates.sort(key=lambda m: (m.importance, -m.age_days))
        return candidates[:max_to_remove]

    def find_duplicates(self, memories: list[Memory]) -> list[Memory]:
        """
        Find duplicate memories (same category + very similar content).
        Returns the older duplicates (keeping the newer one).
        """
        seen: dict[str, Memory] = {}  # content_hash → newest entry
        duplicates: list[Memory] = []

        for mem in sorted(memories, key=lambda m: m.updated_at, reverse=True):
            # Simple dedup: same category + first 50 chars of content
            key = f"{mem.category}:{mem.content[:50].lower().strip()}"
            if key in seen:
                duplicates.append(mem)
            else:
                seen[key] = mem

        return duplicates
