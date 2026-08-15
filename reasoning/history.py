"""
reasoning/history.py — Mission history persistence for CARL.

Records every completed plan as a MissionRecord in a JSON log file.
Supports querying history for reporting and future learning phases.

Storage: memory/mission_history.json (append-only, capped at 200 entries)
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import List

from reasoning.models import MissionRecord, PlanResult


def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


_HISTORY_PATH = _get_base_dir() / "memory" / "mission_history.json"
_MAX_ENTRIES = 200
_lock = threading.Lock()


class MissionHistory:
    """
    Persistent log of completed plans.

    Records:
      - goal
      - timestamps (start, end)
      - duration
      - tools used
      - retries
      - success/failure
      - final summary
    """

    def __init__(self, path: Path | None = None):
        self._path = path or _HISTORY_PATH

    def record(self, result: PlanResult) -> None:
        """Save a completed plan result to the history log."""
        record = MissionRecord.from_plan_result(result)
        self._append(record)
        print(f"[Reasoning] [N] Mission recorded: {record.goal[:50]} "
              f"({'success' if record.success else 'partial'})")

    def record_failure(self, goal: str, failure_type: str, error: str) -> None:
        """
        Record a planning failure (timeout, offline, parse error).

        Creates a minimal MissionRecord with success=False and the
        failure reason in the summary field.
        """
        import time as _time
        now = _time.time()
        failure_record = {
            "goal": goal,
            "started_at": now,
            "completed_at": now,
            "duration_sec": 0.0,
            "success": False,
            "confidence": 0.0,
            "total_steps": 0,
            "completed_steps": 0,
            "failed_steps": 0,
            "tools_used": [],
            "total_retries": 0,
            "summary": f"Planning failed: {failure_type} — {error[:150]}",
            "failure_type": failure_type,
        }
        with _lock:
            data = self._load_raw()
            data.append(failure_record)
            if len(data) > _MAX_ENTRIES:
                data = data[-_MAX_ENTRIES:]
            self._save_raw(data)
        print(f"[Reasoning] [N] Failure recorded: {failure_type} — {goal[:40]}")

    def get_all(self) -> List[MissionRecord]:
        """Load all history entries."""
        data = self._load()
        return [self._dict_to_record(d) for d in data]

    def get_recent(self, count: int = 10) -> List[MissionRecord]:
        """Get the N most recent mission records."""
        data = self._load()
        return [self._dict_to_record(d) for d in data[-count:]]

    def get_stats(self) -> dict:
        """Return aggregate statistics about mission history."""
        data = self._load()
        if not data:
            return {
                "total_missions": 0,
                "success_rate": 0.0,
                "avg_duration_sec": 0.0,
                "total_steps_executed": 0,
                "most_used_tools": [],
            }

        total = len(data)
        successes = sum(1 for d in data if d.get("success", False))
        durations = [d.get("duration_sec", 0) for d in data]
        steps = sum(d.get("completed_steps", 0) for d in data)

        # Count tool usage
        tool_counts: dict[str, int] = {}
        for d in data:
            for tool in d.get("tools_used", []):
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
        top_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_missions": total,
            "success_rate": round((successes / total) * 100, 1) if total > 0 else 0.0,
            "avg_duration_sec": round(sum(durations) / total, 1) if total > 0 else 0.0,
            "total_steps_executed": steps,
            "most_used_tools": [{"tool": t, "count": c} for t, c in top_tools],
        }

    def search(self, keyword: str) -> List[MissionRecord]:
        """Search history by keyword in goal or summary."""
        data = self._load()
        keyword_lower = keyword.lower()
        matches = [
            d for d in data
            if keyword_lower in d.get("goal", "").lower()
            or keyword_lower in d.get("summary", "").lower()
        ]
        return [self._dict_to_record(d) for d in matches]

    def clear(self) -> None:
        """Clear all history (for testing/reset)."""
        with _lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text("[]", encoding="utf-8")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _append(self, record: MissionRecord) -> None:
        """Append a record to the history file, trimming if over max."""
        with _lock:
            data = self._load_raw()
            data.append(record.to_dict())

            # Trim oldest entries if over capacity
            if len(data) > _MAX_ENTRIES:
                data = data[-_MAX_ENTRIES:]

            self._save_raw(data)

    def _load(self) -> list[dict]:
        """Load history data (thread-safe read)."""
        with _lock:
            return self._load_raw()

    def _load_raw(self) -> list[dict]:
        """Load raw JSON data (caller must hold lock)."""
        if not self._path.exists():
            return []
        try:
            text = self._path.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, list):
                return data
            return []
        except (json.JSONDecodeError, OSError) as e:
            print(f"[Reasoning] [!] History load error: {e}")
            return []

    def _save_raw(self, data: list[dict]) -> None:
        """Save raw JSON data (caller must hold lock)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            print(f"[Reasoning] [!] History save error: {e}")

    @staticmethod
    def _dict_to_record(d: dict) -> MissionRecord:
        """Convert a dict back to a MissionRecord."""
        return MissionRecord(
            goal=d.get("goal", ""),
            started_at=d.get("started_at", 0),
            completed_at=d.get("completed_at", 0),
            duration_sec=d.get("duration_sec", 0),
            success=d.get("success", False),
            confidence=d.get("confidence", 0),
            total_steps=d.get("total_steps", 0),
            completed_steps=d.get("completed_steps", 0),
            failed_steps=d.get("failed_steps", 0),
            tools_used=d.get("tools_used", []),
            total_retries=d.get("total_retries", 0),
            summary=d.get("summary", ""),
        )
