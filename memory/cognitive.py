"""
memory/cognitive.py — Unified Cognitive Memory Interface for CARL.

Single entry point that orchestrates all six memory layers:
  1. Working Memory (session_memory.py)
  2. Episodic Memory (episodic.py)
  3. Semantic Memory (memory_store.py)
  4. Preference Memory (preferences.py)
  5. Project Memory (project_manager.py)
  6. Dataset Memory (dataset_memory.py)

Provides:
  - Pre-response context assembly (inject relevant memories before Gemini)
  - Natural commands (remember/forget/recall)
  - Automatic recording of events
  - Memory decay and reinforcement
"""
from __future__ import annotations

import time
from typing import List

from memory.session_memory import SessionMemory
from memory.episodic import EpisodicMemory
from memory.memory_store import MemoryStore
from memory.preferences import PreferenceMemory
from memory.project_manager import ProjectManager
from memory.dataset_memory import DatasetMemory


_CONTEXT_BUDGET = 1200  # max chars injected into prompt


class CognitiveMemory:
    """
    Unified memory interface. Used by main.py before every Gemini request.

    Usage:
        brain = CognitiveMemory()
        context = brain.build_context("Continue the project")
        # → inject context into system prompt
    """

    def __init__(self):
        self._store = MemoryStore()
        self._session = SessionMemory()
        self._episodic = EpisodicMemory()
        self._prefs = PreferenceMemory()
        self._projects = ProjectManager(self._store)
        self._datasets = DatasetMemory()
        # Restore session on init
        self._session.restore()

    @property
    def session(self) -> SessionMemory:
        return self._session

    @property
    def episodic(self) -> EpisodicMemory:
        return self._episodic

    @property
    def preferences(self) -> PreferenceMemory:
        return self._prefs

    @property
    def projects(self) -> ProjectManager:
        return self._projects

    @property
    def datasets(self) -> DatasetMemory:
        return self._datasets

    @property
    def store(self) -> MemoryStore:
        return self._store

    # ── Pre-response context assembly ────────────────────────────────────────

    def build_context(self, user_message: str = "") -> str:
        """
        Assemble relevant memories for injection before Gemini processes.
        Returns formatted context string (max ~1200 chars).
        """
        parts: list[str] = []
        budget = _CONTEXT_BUDGET

        # 1. Session state (project, phase, task)
        sess_ctx = self._session.format_for_prompt()
        if sess_ctx and budget > 0:
            parts.append(sess_ctx)
            budget -= len(sess_ctx)

        # 2. Learned preferences
        pref_ctx = self._prefs.format_for_context()
        if pref_ctx and budget > 200:
            parts.append(pref_ctx)
            budget -= len(pref_ctx)

        # 3. Relevant episodic memories (keyword match)
        if user_message and budget > 200:
            keywords = [w for w in user_message.lower().split() if len(w) > 3][:5]
            if keywords:
                episodes = self._episodic.recall(keywords=keywords, limit=3)
                if episodes:
                    ep_lines = ["[RELEVANT MEMORIES]"]
                    for ep in episodes:
                        line = f"  • {ep.content[:80]}"
                        if len("\n".join(ep_lines)) + len(line) < budget:
                            ep_lines.append(line)
                    if len(ep_lines) > 1:
                        ep_text = "\n".join(ep_lines)
                        parts.append(ep_text)
                        budget -= len(ep_text)

        # 4. Active project context
        if budget > 150:
            proj = self._projects.get_active_project()
            if proj:
                proj_text = f"[PROJECT: {proj.name}] Phase: {proj.current_phase or 'N/A'}"
                if proj.next_task:
                    proj_text += f" | Next: {proj.next_task}"
                parts.append(proj_text)

        return "\n".join(parts) if parts else ""

    # ── Natural commands ─────────────────────────────────────────────────────

    def handle_command(self, text: str) -> str | None:
        """
        Detect and handle memory commands in user speech.
        Returns response string if it was a command, None otherwise.
        """
        lower = text.lower().strip()

        # "Remember this" / "Remember that X"
        if lower.startswith("remember"):
            content = text[8:].strip().lstrip("this").lstrip("that").strip()
            if not content:
                return None  # not a memory command, just the word
            self._episodic.record(content, category="explicit", importance=8, source="user")
            self._store.store("notes", content[:30].replace(" ", "_"), content, importance=3, source="user")
            return f"Remembered: {content[:60]}"

        # "Forget that" / "Forget X"
        if lower.startswith("forget"):
            target = text[6:].strip().lstrip("that").strip()
            if not target:
                return None
            # Search and remove matching memories
            results = self._episodic.recall(keywords=target.split(), limit=1)
            if results:
                self._episodic._episodes.remove(results[0])
                self._episodic._save()
                return f"Forgotten: {results[0].content[:60]}"
            return "I couldn't find that in my memory."

        # "What do you remember about me?"
        if "remember about me" in lower or "what do you know" in lower:
            episodes = self._episodic.recent(5)
            prefs = self._prefs.get_all_learned()
            parts = []
            if prefs:
                parts.append("Preferences: " + ", ".join(f"{p.key}={p.value}" for p in prefs[:5]))
            if episodes:
                parts.append("Recent: " + "; ".join(e.content[:40] for e in episodes[:3]))
            return " | ".join(parts) if parts else "I don't have much stored yet."

        # "Summarize what we've accomplished"
        if "summarize" in lower and ("accomplish" in lower or "done" in lower or "built" in lower):
            milestones = self._episodic.recall(category="milestone", limit=5)
            if milestones:
                return "Accomplishments: " + "; ".join(m.content[:50] for m in milestones)
            return "No major milestones recorded yet."

        # "Clear temporary memory"
        if "clear" in lower and ("temp" in lower or "working" in lower):
            self._session._state.conversation = []
            self._session.save()
            return "Working memory cleared."

        # "Continue" / "Continue our project"
        if lower in ("continue", "continue project", "continue our project", "where were we"):
            proj = self._projects.get_active_project()
            if proj and proj.next_task:
                return f"Continuing {proj.name}. Next: {proj.next_task}"
            recent = self._episodic.recent(1)
            if recent:
                return f"Last activity: {recent[0].content}"
            return None  # Not a memory command — pass through to Gemini

        return None  # Not a memory command

    # ── Automatic recording ──────────────────────────────────────────────────

    def record_task_completion(self, task: str, project: str = ""):
        """Automatically record when a task/tool completes."""
        self._episodic.record(task, category="task", importance=5,
                             source="system", project=project)

    def record_milestone(self, milestone: str, project: str = ""):
        """Record a significant achievement."""
        self._episodic.record(milestone, category="milestone", importance=8,
                             source="system", project=project)

    def record_preference(self, key: str, value: str, source: str = "observed"):
        """Record/reinforce a user preference."""
        self._prefs.observe(key, value, source)

    def record_conversation_turn(self, role: str, text: str):
        """Track conversation in working memory."""
        self._session.add_conversation_turn(role, text)

    # ── Maintenance ──────────────────────────────────────────────────────────

    def cleanup(self) -> int:
        """Run periodic cleanup: decay trivial memories, save state."""
        removed = self._episodic.forget_trivial()
        removed += self._store.cleanup_expired()
        self._session.auto_save_if_dirty()
        return removed
