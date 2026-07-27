"""
memory/session_memory.py — Session state persistence for CARL.

Remembers the current working state across restarts:
  - Active project
  - Current task / phase
  - Open applications
  - Last conversation turns
  - Unfinished work
  - Session timestamps

On startup, CARL restores this state so it can resume naturally
without asking "what were we working on?"

Storage: memory/session_state.json (overwritten each save)
"""
from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List


def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


_SESSION_PATH = _get_base_dir() / "memory" / "session_state.json"
_lock = threading.Lock()
_MAX_CONVERSATION_TURNS = 30
_MAX_UNFINISHED = 10


# ── Session State ────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    """Snapshot of CARL's current working state."""

    # Active project context
    active_project: str = ""
    current_phase: str = ""
    current_task: str = ""

    # Recent conversation (last N turns for continuity)
    conversation: list[dict] = field(default_factory=list)
    # Format: [{"role": "user"|"carl", "text": "...", "ts": float}]

    # Unfinished work (things CARL was doing when interrupted/shut down)
    unfinished_tasks: list[dict] = field(default_factory=list)
    # Format: [{"description": "...", "tool": "...", "started_at": float}]

    # Open applications (tracked for context)
    open_apps: list[str] = field(default_factory=list)

    # Last active folders/files
    last_folders: list[str] = field(default_factory=list)
    last_files: list[str] = field(default_factory=list)

    # Timestamps
    session_started_at: float = field(default_factory=time.time)
    last_interaction_at: float = field(default_factory=time.time)
    last_saved_at: float = field(default_factory=time.time)

    # Session metadata
    total_commands: int = 0
    total_tool_calls: int = 0
    session_id: str = ""

    def to_dict(self) -> dict:
        return {
            "active_project": self.active_project,
            "current_phase": self.current_phase,
            "current_task": self.current_task,
            "conversation": self.conversation[-_MAX_CONVERSATION_TURNS:],
            "unfinished_tasks": self.unfinished_tasks[-_MAX_UNFINISHED:],
            "open_apps": self.open_apps[:20],
            "last_folders": self.last_folders[-5:],
            "last_files": self.last_files[-10:],
            "session_started_at": self.session_started_at,
            "last_interaction_at": self.last_interaction_at,
            "last_saved_at": time.time(),
            "total_commands": self.total_commands,
            "total_tool_calls": self.total_tool_calls,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionState":
        return cls(
            active_project=d.get("active_project", ""),
            current_phase=d.get("current_phase", ""),
            current_task=d.get("current_task", ""),
            conversation=d.get("conversation", []),
            unfinished_tasks=d.get("unfinished_tasks", []),
            open_apps=d.get("open_apps", []),
            last_folders=d.get("last_folders", []),
            last_files=d.get("last_files", []),
            session_started_at=d.get("session_started_at", time.time()),
            last_interaction_at=d.get("last_interaction_at", time.time()),
            last_saved_at=d.get("last_saved_at", time.time()),
            total_commands=d.get("total_commands", 0),
            total_tool_calls=d.get("total_tool_calls", 0),
            session_id=d.get("session_id", ""),
        )


# ── Session Memory Manager ───────────────────────────────────────────────────

class SessionMemory:
    """
    Manages session state persistence.

    Usage:
        sm = SessionMemory()
        sm.restore()  # Load previous session on startup
        sm.set_project("Mark XLVIII", phase="16.1")
        sm.add_conversation_turn("user", "Continue working on the memory engine")
        sm.save()     # Persist to disk
    """

    def __init__(self, path: Path | None = None):
        self._path = path or _SESSION_PATH
        self._state = SessionState()
        self._dirty = False

    @property
    def state(self) -> SessionState:
        return self._state

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def restore(self) -> bool:
        """
        Restore previous session state from disk.
        Returns True if a previous session was found and loaded.
        """
        if not self._path.exists():
            self._new_session()
            return False

        try:
            with _lock:
                data = json.loads(self._path.read_text(encoding="utf-8"))
            self._state = SessionState.from_dict(data)
            print(f"[Memory] [*] Session restored — project: {self._state.active_project or 'none'}")
            return True
        except (json.JSONDecodeError, OSError) as e:
            print(f"[Memory] [!] Session restore failed: {e}")
            self._new_session()
            return False

    def save(self) -> None:
        """Persist current session state to disk."""
        self._state.last_saved_at = time.time()
        data = self._state.to_dict()

        with _lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError as e:
                print(f"[Memory] [!] Session save failed: {e}")

        self._dirty = False

    def _new_session(self) -> None:
        """Initialize a fresh session."""
        import secrets
        self._state = SessionState(
            session_id=secrets.token_hex(8),
            session_started_at=time.time(),
        )

    # ── Project tracking ─────────────────────────────────────────────────────

    def set_project(self, name: str, phase: str = "", task: str = "") -> None:
        """Set the active project context."""
        self._state.active_project = name
        if phase:
            self._state.current_phase = phase
        if task:
            self._state.current_task = task
        self._state.last_interaction_at = time.time()
        self._dirty = True

    def get_active_project(self) -> str:
        return self._state.active_project

    def get_current_context(self) -> dict:
        """Get a snapshot of current context for injection into prompts."""
        return {
            "active_project": self._state.active_project,
            "current_phase": self._state.current_phase,
            "current_task": self._state.current_task,
            "last_interaction": self._state.last_interaction_at,
            "unfinished_count": len(self._state.unfinished_tasks),
        }

    # ── Conversation tracking ────────────────────────────────────────────────

    def add_conversation_turn(self, role: str, text: str) -> None:
        """Record a conversation turn (user or carl)."""
        if not text.strip():
            return
        self._state.conversation.append({
            "role": role,
            "text": text[:500],  # cap individual turn length
            "ts": time.time(),
        })
        # Trim to max
        if len(self._state.conversation) > _MAX_CONVERSATION_TURNS:
            self._state.conversation = self._state.conversation[-_MAX_CONVERSATION_TURNS:]
        self._state.last_interaction_at = time.time()
        self._state.total_commands += 1 if role == "user" else 0
        self._dirty = True

    def get_recent_conversation(self, count: int = 10) -> list[dict]:
        """Get the last N conversation turns."""
        return self._state.conversation[-count:]

    def get_last_user_message(self) -> str:
        """Get the most recent user message."""
        for turn in reversed(self._state.conversation):
            if turn["role"] == "user":
                return turn["text"]
        return ""

    # ── Unfinished work ──────────────────────────────────────────────────────

    def mark_task_started(self, description: str, tool: str = "") -> None:
        """Record that a task has been started (for resume-on-restart)."""
        self._state.unfinished_tasks.append({
            "description": description,
            "tool": tool,
            "started_at": time.time(),
        })
        if len(self._state.unfinished_tasks) > _MAX_UNFINISHED:
            self._state.unfinished_tasks = self._state.unfinished_tasks[-_MAX_UNFINISHED:]
        self._dirty = True

    def mark_task_completed(self, description: str) -> None:
        """Remove a task from unfinished list."""
        self._state.unfinished_tasks = [
            t for t in self._state.unfinished_tasks
            if t["description"] != description
        ]
        self._dirty = True

    def get_unfinished_tasks(self) -> list[dict]:
        """Get list of tasks that were started but not completed."""
        return self._state.unfinished_tasks

    # ── Application & file tracking ──────────────────────────────────────────

    def record_app_opened(self, app_name: str) -> None:
        """Track that an application was opened."""
        name = app_name.strip()
        if name and name not in self._state.open_apps:
            self._state.open_apps.append(name)
            self._state.open_apps = self._state.open_apps[-20:]
        self._dirty = True

    def record_file_accessed(self, file_path: str) -> None:
        """Track that a file was accessed."""
        if file_path and file_path not in self._state.last_files:
            self._state.last_files.append(file_path)
            self._state.last_files = self._state.last_files[-10:]
        self._dirty = True

    def record_folder_accessed(self, folder_path: str) -> None:
        """Track that a folder was accessed."""
        if folder_path and folder_path not in self._state.last_folders:
            self._state.last_folders.append(folder_path)
            self._state.last_folders = self._state.last_folders[-5:]
        self._dirty = True

    def record_tool_call(self, tool_name: str) -> None:
        """Increment tool call counter."""
        self._state.total_tool_calls += 1
        self._dirty = True

    # ── Context formatting ───────────────────────────────────────────────────

    def format_for_prompt(self) -> str:
        """
        Format session context as a string for injection into the system prompt.
        Returns empty string if no meaningful context exists.
        """
        parts: list[str] = []

        if self._state.active_project:
            parts.append(f"Active Project: {self._state.active_project}")
            if self._state.current_phase:
                parts.append(f"  Current Phase: {self._state.current_phase}")
            if self._state.current_task:
                parts.append(f"  Current Task: {self._state.current_task}")

        if self._state.unfinished_tasks:
            parts.append("Unfinished work:")
            for t in self._state.unfinished_tasks[-3:]:
                parts.append(f"  - {t['description']}")

        # Time since last interaction
        if self._state.last_interaction_at:
            gap = time.time() - self._state.last_interaction_at
            if gap > 3600:
                hours = int(gap / 3600)
                parts.append(f"Last interaction: {hours} hour{'s' if hours > 1 else ''} ago")

        if not parts:
            return ""

        return "[SESSION CONTEXT]\n" + "\n".join(parts) + "\n"

    # ── Auto-save ────────────────────────────────────────────────────────────

    def auto_save_if_dirty(self) -> None:
        """Save only if state has changed since last save."""
        if self._dirty:
            self.save()
