"""
memory/project_manager.py — Dedicated project lifecycle manager for CARL.

Manages project detection, creation, tracking, and timeline.
Builds on the ProjectMemory in memory_store.py but adds:
  - Auto-detection of projects from user conversations
  - Full lifecycle management (create → active → paused → completed)
  - Timeline with events (phase started, error fixed, mission completed)
  - Smart project switching
  - Project-level memory association

Storage: Uses MemoryStore's project system (memory/semantic_store.json)
"""
from __future__ import annotations

import time
from typing import List, Optional

from memory.memory_store import MemoryStore, ProjectMemory, Importance
from memory.models import Project, ProjectStatus


# ── Known project name patterns for auto-detection ───────────────────────────

_PROJECT_INDICATORS = frozenset({
    "project", "app", "application", "website", "system",
    "engine", "module", "package", "repo", "repository",
    "build", "working on", "developing", "creating",
})

# Common project names CARL has seen (populated from store on init)
_KNOWN_NAMES: set[str] = set()


class ProjectManager:
    """
    Manages the full lifecycle of projects.

    Features:
      - Auto-detect project references in conversation
      - Create/update/complete/archive projects
      - Track phases, tasks, files, errors
      - Record timeline events
      - Switch active project
      - Query project status and history
    """

    def __init__(self, store: MemoryStore):
        self._store = store
        self._active_project: str = ""
        self._load_known_names()

    def _load_known_names(self) -> None:
        """Load known project names from store for fast detection."""
        global _KNOWN_NAMES
        projects = self._store.list_projects()
        _KNOWN_NAMES = {p.name.lower() for p in projects}

    # ── Project CRUD ─────────────────────────────────────────────────────────

    def create_project(
        self,
        name: str,
        description: str = "",
        phases: list[str] | None = None,
        goals: list[str] | None = None,
    ) -> ProjectMemory:
        """Create a new project and set it as active."""
        proj = self._store.update_project(name)
        if description:
            proj.notes = description
        if phases:
            proj.pending_phases = phases
            if phases:
                proj.current_phase = phases[0]
        if goals:
            proj.goals = goals

        self._active_project = name
        _KNOWN_NAMES.add(name.lower())

        # Record timeline event
        self._add_event(name, "created", f"Project '{name}' created")

        # Store as important memory
        self._store.store(
            category="projects",
            key=name.lower().replace(" ", "_"),
            content=f"Project: {name}. {description}",
            importance=Importance.IMPORTANT,
            tags=["project", "active"],
            project=name,
            source="system",
        )

        print(f"[ProjectMgr] 📁 Created project: {name}")
        return proj

    def get_project(self, name: str) -> ProjectMemory | None:
        """Get a project by name (case-insensitive, fuzzy)."""
        # Exact match first
        proj = self._store.get_project(name)
        if proj:
            return proj

        # Fuzzy match
        name_lower = name.lower().strip()
        for p in self._store.list_projects():
            if name_lower in p.name.lower() or p.name.lower() in name_lower:
                return p

        return None

    def get_active_project(self) -> ProjectMemory | None:
        """Get the currently active project."""
        if self._active_project:
            return self.get_project(self._active_project)
        # Fall back to most recently updated
        return self._store.get_active_project()

    def set_active(self, name: str) -> str:
        """Switch the active project."""
        proj = self.get_project(name)
        if not proj:
            return f"No project found matching '{name}'."
        self._active_project = proj.name
        self._add_event(proj.name, "activated", "Set as active project")
        return f"Active project: {proj.name}"

    def list_projects(self, status: str = "") -> list[ProjectMemory]:
        """List all projects, optionally filtered by status."""
        projects = self._store.list_projects()
        if not status:
            return projects
        # Filter by checking notes/metadata for status (stored in memory_store)
        return projects  # memory_store doesn't track status separately yet

    # ── Phase & task management ──────────────────────────────────────────────

    def start_phase(self, phase: str, project_name: str = "") -> str:
        """Mark a phase as started."""
        name = project_name or self._active_project
        if not name:
            return "No active project."

        self._store.update_project(name, current_phase=phase)
        self._add_event(name, "phase_started", f"Started phase: {phase}")
        return f"Phase '{phase}' started for {name}."

    def complete_phase(self, phase: str, project_name: str = "") -> str:
        """Mark a phase as completed."""
        name = project_name or self._active_project
        if not name:
            return "No active project."

        self._store.update_project(name, completed_phase=phase)
        self._add_event(name, "phase_completed", f"Completed phase: {phase}")
        return f"Phase '{phase}' completed for {name}."

    def set_next_task(self, task: str, project_name: str = "") -> str:
        """Set the next task for a project."""
        name = project_name or self._active_project
        if not name:
            return "No active project."

        self._store.update_project(name, next_task=task)
        return f"Next task for {name}: {task}"

    def add_goal(self, goal: str, project_name: str = "") -> str:
        """Add a goal to a project."""
        name = project_name or self._active_project
        if not name:
            return "No active project."

        self._store.update_project(name, goal=goal)
        self._add_event(name, "goal_added", goal)
        return f"Goal added to {name}: {goal}"

    def record_error(self, error: str, project_name: str = "") -> None:
        """Record an error for the active project."""
        name = project_name or self._active_project
        if not name:
            return

        self._store.update_project(name, error=error)
        self._add_event(name, "error", error[:100])

    def record_file_edit(self, file_path: str, project_name: str = "") -> None:
        """Record a file edit for the active project."""
        name = project_name or self._active_project
        if not name:
            return

        self._store.update_project(name, edited_file=file_path)

    def record_mission(self, goal: str, success: bool, summary: str, project_name: str = "") -> None:
        """Record a completed mission/plan in the project history."""
        name = project_name or self._active_project
        if not name:
            return

        proj = self.get_project(name)
        if proj:
            # mission_history is tracked in the underlying ProjectMemory's metadata
            event_type = "mission_completed" if success else "mission_failed"
            self._add_event(name, event_type, f"{goal}: {summary[:100]}")

    # ── Auto-detection ───────────────────────────────────────────────────────

    def detect_project(self, text: str) -> str | None:
        """
        Detect a project reference in user text.

        Returns the project name if found, None otherwise.

        Strategy:
          1. Check against known project names
          2. Look for project indicator keywords + capitalized name
        """
        text_lower = text.lower()

        # Check known project names
        for known in _KNOWN_NAMES:
            if known in text_lower:
                # Find the original-case version
                for p in self._store.list_projects():
                    if p.name.lower() == known:
                        return p.name
                return known

        # Look for "working on X" / "project X" patterns
        import re
        patterns = [
            r'(?:project|working on|building|developing)\s+["\']?([A-Z][A-Za-z0-9 _-]{2,30})',
            r'(?:the|my)\s+([A-Z][A-Za-z0-9 _-]{2,20})\s+(?:project|app|system)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(1).strip()
                if len(candidate) > 2:
                    return candidate

        return None

    def auto_track(self, user_text: str, tool_name: str = "", tool_args: dict | None = None) -> None:
        """
        Automatically track project activity from user interactions.

        Called after each tool execution to keep project state current.
        """
        # Detect project mention
        detected = self.detect_project(user_text)
        if detected:
            self._active_project = detected
            # Ensure project exists
            if detected.lower() not in _KNOWN_NAMES:
                self.create_project(detected)
            else:
                self._store.update_project(detected)  # touch updated_at

        # Track file operations
        if tool_args and tool_name in ("file_controller", "code_helper"):
            path = tool_args.get("file_path", "") or tool_args.get("path", "")
            if path and self._active_project:
                self.record_file_edit(path)

    # ── Timeline ─────────────────────────────────────────────────────────────

    def get_timeline(self, project_name: str = "", limit: int = 20) -> list[dict]:
        """
        Get timeline events for a project.

        Returns list of {"event": str, "ts": float, "detail": str}
        """
        name = project_name or self._active_project
        if not name:
            return []

        # Retrieve from memory store entries tagged with this project
        entries = self._store.recall(project=name)
        entries.sort(key=lambda e: e.updated_at, reverse=True)

        timeline = []
        for e in entries[:limit]:
            timeline.append({
                "event": e.key,
                "ts": e.updated_at,
                "detail": e.content[:100],
            })

        return timeline

    def get_project_summary(self, project_name: str = "") -> str:
        """Get a natural language summary of a project's current state."""
        name = project_name or self._active_project
        if not name:
            return "No active project."

        proj = self.get_project(name)
        if not proj:
            return f"Project '{name}' not found."

        parts = [f"Project: {proj.name}"]
        if proj.current_phase:
            parts.append(f"Phase: {proj.current_phase}")
        if proj.next_task:
            parts.append(f"Next: {proj.next_task}")
        if proj.completed_phases:
            parts.append(f"Completed: {', '.join(proj.completed_phases[-5:])}")
        if proj.goals:
            parts.append(f"Goals: {', '.join(proj.goals[-3:])}")
        if proj.last_errors:
            parts.append(f"Last error: {proj.last_errors[-1][:60]}")

        return " | ".join(parts)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _add_event(self, project_name: str, event_type: str, detail: str) -> None:
        """Store a timeline event as a memory entry."""
        self._store.store(
            category="projects",
            key=f"{event_type}_{int(time.time())}",
            content=detail,
            importance=Importance.USEFUL,
            tags=["timeline", event_type],
            project=project_name,
            source="project_manager",
            metadata={"event_type": event_type},
        )
