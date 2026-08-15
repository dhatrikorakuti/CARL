"""
memory/context_engine.py — Context retrieval engine for CARL.

Before Gemini receives a user request, the Context Engine assembles
relevant memories and injects them into the system prompt. This gives
CARL awareness of:
  - Active project and current task
  - Session continuity (unfinished work, recent conversation)
  - Relevant stored facts and preferences
  - Project timeline and history

The engine is lightweight and fast — no LLM calls. It uses keyword
matching and recency to select the most relevant context within a
character budget.
"""
from __future__ import annotations

import time
from typing import List

from memory.memory_store import MemoryStore, MemoryEntry, Importance
from memory.session_memory import SessionMemory


_CONTEXT_BUDGET = 1500  # max characters injected into the prompt


class ContextEngine:
    """
    Retrieves and assembles relevant context for each Gemini interaction.

    Usage:
        engine = ContextEngine(store, session)
        context_str = engine.build_context(user_message="Continue")
        # → inject context_str into system prompt before Gemini processes
    """

    def __init__(self, store: MemoryStore, session: SessionMemory):
        self._store = store
        self._session = session

    def build_context(self, user_message: str = "") -> str:
        """
        Assemble relevant context for the current interaction.

        Layers (in priority order):
          1. Session state (project, phase, unfinished work)
          2. Relevant memories (matched by keywords from user message)
          3. Project context (if an active project exists)
          4. Recent conversation summary

        Returns a formatted string for injection into the system prompt.
        Limited to _CONTEXT_BUDGET characters.
        """
        parts: list[str] = []
        budget = _CONTEXT_BUDGET

        # 1. Session state — always included (highest priority)
        session_ctx = self._session.format_for_prompt()
        if session_ctx:
            parts.append(session_ctx)
            budget -= len(session_ctx)

        # 2. Active project details
        project_ctx = self._get_project_context()
        if project_ctx and budget > 200:
            parts.append(project_ctx)
            budget -= len(project_ctx)

        # 3. Relevant memories based on user message
        if user_message and budget > 200:
            relevant = self._retrieve_relevant(user_message, budget)
            if relevant:
                parts.append(relevant)
                budget -= len(relevant)

        # 4. Continuity cues (what was happening recently)
        if budget > 100:
            continuity = self._get_continuity_cue(user_message)
            if continuity:
                parts.append(continuity)

        result = "\n".join(parts).strip()
        return result[:_CONTEXT_BUDGET] if result else ""

    def _get_project_context(self) -> str:
        """Get active project context from the memory store."""
        project_name = self._session.get_active_project()
        if not project_name:
            # Try getting most recently updated project from store
            proj = self._store.get_active_project()
            if proj:
                project_name = proj.name

        if not project_name:
            return ""

        proj = self._store.get_project(project_name)
        if not proj:
            return ""

        lines = [f"[PROJECT: {proj.name}]"]
        if proj.current_phase:
            lines.append(f"  Phase: {proj.current_phase}")
        if proj.next_task:
            lines.append(f"  Next: {proj.next_task}")
        if proj.goals:
            lines.append(f"  Goals: {', '.join(proj.goals[-3:])}")
        if proj.last_errors:
            lines.append(f"  Last error: {proj.last_errors[-1][:80]}")

        return "\n".join(lines)

    def _retrieve_relevant(self, user_message: str, budget: int) -> str:
        """
        Find memories relevant to the user's message using keyword matching.

        Fast heuristic approach (no embedding model needed):
          - Extract keywords from user message
          - Match against memory content, tags, and keys
          - Rank by relevance score (matches × importance)
        """
        keywords = self._extract_keywords(user_message)
        if not keywords:
            return ""

        # Score all non-expired memories
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self._store.recall_recent(50):
            score = self._score_relevance(entry, keywords)
            if score > 0:
                scored.append((score, entry))

        if not scored:
            return ""

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build output within budget
        lines = ["[RELEVANT MEMORIES]"]
        used = len(lines[0])
        for score, entry in scored[:5]:
            line = f"  • {entry.content[:100]}"
            if used + len(line) + 1 > budget:
                break
            lines.append(line)
            used += len(line) + 1

        return "\n".join(lines) if len(lines) > 1 else ""

    def _get_continuity_cue(self, user_message: str) -> str:
        """
        Detect continuity commands ("continue", "go on", "what's next")
        and inject the necessary context so CARL can resume.
        """
        msg_lower = user_message.lower().strip()
        continuity_triggers = {
            "continue", "go on", "keep going", "what's next",
            "resume", "where were we", "carry on", "next",
            "devam", "devam et",  # Turkish support
        }

        if not any(trigger in msg_lower for trigger in continuity_triggers):
            return ""

        # User wants to continue — inject what was happening
        parts: list[str] = []

        # Check unfinished tasks
        unfinished = self._session.get_unfinished_tasks()
        if unfinished:
            last = unfinished[-1]
            parts.append(f"[CONTINUITY] User wants to continue. Last unfinished: {last['description']}")

        # Check last conversation for context
        recent = self._session.get_recent_conversation(3)
        if recent and not parts:
            last_carl = ""
            for turn in reversed(recent):
                if turn["role"] == "carl":
                    last_carl = turn["text"][:150]
                    break
            if last_carl:
                parts.append(f"[CONTINUITY] Last CARL response: {last_carl}")

        # Check active project next task
        project_name = self._session.get_active_project()
        if project_name:
            proj = self._store.get_project(project_name)
            if proj and proj.next_task:
                parts.append(f"[CONTINUITY] Next task for {proj.name}: {proj.next_task}")

        return "\n".join(parts) if parts else ""

    def _extract_keywords(self, text: str) -> list[str]:
        """
        Extract meaningful keywords from text for memory matching.
        Filters out stop words and short tokens.
        """
        _STOP_WORDS = frozenset({
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "shall", "to",
            "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above",
            "below", "between", "and", "but", "or", "not", "no", "so",
            "if", "then", "than", "too", "very", "just", "about",
            "what", "when", "where", "who", "how", "which", "that",
            "this", "these", "those", "it", "its", "my", "your", "our",
            "me", "you", "he", "she", "we", "they", "i", "im",
        })

        # Tokenize and filter
        import re
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [t for t in tokens if t not in _STOP_WORDS]

        # Deduplicate while preserving order
        seen = set()
        result = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                result.append(kw)

        return result[:10]  # cap at 10 keywords

    def _score_relevance(self, entry: MemoryEntry, keywords: list[str]) -> float:
        """
        Score how relevant a memory entry is to the given keywords.

        Scoring factors:
          - Keyword matches in content, key, tags
          - Importance level (multiplier)
          - Recency bonus
        """
        score = 0.0
        searchable = f"{entry.content} {entry.key} {' '.join(entry.tags)}".lower()

        for kw in keywords:
            if kw in searchable:
                score += 1.0

        if score == 0:
            return 0.0

        # Importance multiplier
        importance_mult = {
            Importance.TEMPORARY: 0.5,
            Importance.USEFUL: 1.0,
            Importance.IMPORTANT: 1.5,
            Importance.CRITICAL: 2.0,
        }.get(Importance(entry.importance), 1.0)

        score *= importance_mult

        # Recency bonus (memories from last 24h get a boost)
        age_hours = (time.time() - entry.updated_at) / 3600
        if age_hours < 1:
            score *= 1.5
        elif age_hours < 24:
            score *= 1.2

        return score

    # ── Utility ──────────────────────────────────────────────────────────────

    def get_full_context_snapshot(self) -> dict:
        """
        Full context snapshot for diagnostics / UI display.
        Not used in the prompt — just for inspection.
        """
        return {
            "session": self._session.get_current_context(),
            "active_project": self._session.get_active_project(),
            "recent_memories": len(self._store.recall_recent(20)),
            "unfinished_tasks": len(self._session.get_unfinished_tasks()),
            "store_stats": self._store.get_stats(),
        }
