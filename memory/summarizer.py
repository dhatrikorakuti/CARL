"""
memory/summarizer.py — Conversation summarization for CARL.

Every 25 messages, summarizes the conversation and stores the summary
in the memory store. This keeps memory efficient while preserving
important context across sessions.

Summarization uses the local LLM (lightweight call) or falls back
to extractive summarization if the LLM is unavailable.
"""
from __future__ import annotations

import time
from typing import List

from memory.memory_store import MemoryStore, Importance


_SUMMARY_INTERVAL = 25  # messages between summaries
_MAX_SUMMARIES = 50     # stored summaries before oldest are trimmed


class ConversationSummarizer:
    """
    Manages periodic conversation summarization.

    Usage:
        summarizer = ConversationSummarizer(store)
        summarizer.add_turn("user", "Open Chrome")
        summarizer.add_turn("carl", "Done. Chrome is open.")
        # ... after 25 turns, auto-summarizes
    """

    def __init__(self, store: MemoryStore):
        self._store = store
        self._buffer: list[dict] = []
        self._total_turns: int = 0
        self._summaries: list[dict] = []

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def add_turn(self, role: str, text: str) -> str | None:
        """
        Add a conversation turn to the buffer.

        Returns a summary string if summarization was triggered, else None.
        """
        if not text.strip():
            return None

        self._buffer.append({
            "role": role,
            "text": text[:300],
            "ts": time.time(),
        })
        self._total_turns += 1

        # Check if summarization threshold reached
        if len(self._buffer) >= _SUMMARY_INTERVAL:
            summary = self._summarize_buffer()
            return summary

        return None

    def force_summarize(self) -> str | None:
        """Force summarization of current buffer (e.g., on session end)."""
        if len(self._buffer) < 3:
            return None
        return self._summarize_buffer()

    def get_summaries(self, count: int = 5) -> list[dict]:
        """Get recent conversation summaries."""
        entries = self._store.recall(category="conversation_summary")
        entries.sort(key=lambda e: e.updated_at, reverse=True)
        return [
            {"text": e.content, "ts": e.updated_at, "metadata": e.metadata}
            for e in entries[:count]
        ]

    def get_full_history_summary(self) -> str:
        """Get a combined summary of all stored conversation summaries."""
        entries = self._store.recall(category="conversation_summary")
        if not entries:
            return ""
        entries.sort(key=lambda e: e.created_at)
        parts = [e.content for e in entries[-5:]]  # last 5 summaries
        return "\n---\n".join(parts)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _summarize_buffer(self) -> str:
        """Summarize the current buffer and store the result."""
        conversation_text = self._format_buffer()

        # Try LLM-based summarization first
        summary = self._llm_summarize(conversation_text)

        # Fallback to extractive if LLM unavailable
        if not summary:
            summary = self._extractive_summarize()

        if summary:
            self._store_summary(summary)

        # Clear buffer
        self._buffer = []
        return summary

    def _llm_summarize(self, text: str) -> str:
        """Use local LLM to summarize conversation. Returns empty on failure."""
        try:
            from core.llm_client import call_llm_text, ensure_ollama_running

            # Quick check — don't wait for Ollama if it's not running
            prompt = (
                "Summarize this conversation in 2-4 sentences. "
                "Focus on: what was accomplished, what decisions were made, "
                "and what's pending. Be concise.\n\n"
                f"Conversation:\n{text}"
            )

            result = call_llm_text(
                prompt=prompt,
                system="You are a conversation summarizer. Output only the summary, nothing else.",
                timeout=6,
            )
            return result.strip() if result else ""

        except Exception as e:
            print(f"[Memory] Summarizer LLM unavailable: {e}")
            return ""

    def _extractive_summarize(self) -> str:
        """
        Fallback: extract key information without an LLM.

        Strategy:
          - Identify user requests (user turns)
          - Identify CARL's key actions (tool mentions, completions)
          - Combine into a brief summary
        """
        if not self._buffer:
            return ""

        user_requests: list[str] = []
        carl_actions: list[str] = []

        for turn in self._buffer:
            text = turn["text"][:100]
            if turn["role"] == "user":
                # Skip very short or noise turns
                if len(text.strip()) > 5:
                    user_requests.append(text.strip())
            else:
                # Extract action-like responses from CARL
                if any(kw in text.lower() for kw in
                       ("done", "completed", "opened", "found", "created",
                        "error", "failed", "result", "finished")):
                    carl_actions.append(text.strip())

        parts: list[str] = []

        # Summarize user requests
        if user_requests:
            if len(user_requests) <= 3:
                parts.append(f"User asked: {'; '.join(user_requests)}")
            else:
                parts.append(f"User made {len(user_requests)} requests including: "
                           f"{'; '.join(user_requests[:3])}")

        # Summarize CARL actions
        if carl_actions:
            if len(carl_actions) <= 2:
                parts.append(f"Completed: {'; '.join(carl_actions)}")
            else:
                parts.append(f"Completed {len(carl_actions)} actions")

        # Timestamp range
        if self._buffer:
            start = self._buffer[0]["ts"]
            end = self._buffer[-1]["ts"]
            duration_min = int((end - start) / 60)
            if duration_min > 0:
                parts.append(f"({duration_min} min conversation)")

        return ". ".join(parts) if parts else "Brief conversation with no major actions."

    def _format_buffer(self) -> str:
        """Format buffer as readable conversation text for LLM input."""
        lines: list[str] = []
        for turn in self._buffer:
            role = "User" if turn["role"] == "user" else "CARL"
            text = turn["text"][:150]
            lines.append(f"{role}: {text}")
        # Cap total size for LLM input
        result = "\n".join(lines)
        return result[:2000]

    def _store_summary(self, summary: str) -> None:
        """Store a conversation summary in the memory store."""
        timestamp = time.time()
        key = f"summary_{int(timestamp)}"

        self._store.store(
            category="conversation_summary",
            key=key,
            content=summary,
            importance=Importance.USEFUL,
            tags=["conversation", "summary", "auto"],
            source="summarizer",
            metadata={
                "turns_summarized": _SUMMARY_INTERVAL,
                "total_turns_at_summary": self._total_turns,
            },
        )

        # Trim old summaries if over limit
        all_summaries = self._store.recall(category="conversation_summary")
        if len(all_summaries) > _MAX_SUMMARIES:
            # Remove oldest
            all_summaries.sort(key=lambda e: e.created_at)
            for old in all_summaries[:-_MAX_SUMMARIES]:
                self._store.forget("conversation_summary", old.key)

        print(f"[Memory] [N] Conversation summarized ({self._total_turns} total turns)")
