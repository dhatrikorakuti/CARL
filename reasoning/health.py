"""
reasoning/health.py — Planner health monitoring for CARL.

Tracks planner availability, latency, errors, and status.
Used by the UI to show planner state and by the engine to decide
whether to attempt planning or fall back to direct execution.
"""
from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Optional


class PlannerStatus(Enum):
    READY = "ready"
    OFFLINE = "offline"
    PLANNING = "planning"
    EXECUTING = "executing"
    ERROR = "error"
    UNKNOWN = "unknown"


class PlannerHealth:
    """
    Singleton health monitor for the planner subsystem.

    Thread-safe — queried from UI thread, updated from async executor.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._available: bool = False
        self._status: PlannerStatus = PlannerStatus.UNKNOWN
        self._last_latency_ms: float = 0.0
        self._last_error: str = ""
        self._last_error_time: float = 0.0
        self._last_success_time: float = 0.0
        self._total_plans: int = 0
        self._total_failures: int = 0
        self._checked: bool = False

    # ── Queries ──────────────────────────────────────────────────────────────

    def planner_available(self) -> bool:
        """Returns True if the local LLM is reachable."""
        with self._lock:
            return self._available

    def planner_latency(self) -> float:
        """Returns last measured latency in milliseconds."""
        with self._lock:
            return self._last_latency_ms

    def planner_last_error(self) -> str:
        """Returns the last error message, or empty string."""
        with self._lock:
            return self._last_error

    def planner_status(self) -> PlannerStatus:
        """Returns the current planner status."""
        with self._lock:
            return self._status

    def is_checked(self) -> bool:
        """Returns True if startup check has been performed."""
        with self._lock:
            return self._checked

    def get_stats(self) -> dict:
        """Full health snapshot for diagnostics."""
        with self._lock:
            return {
                "available": self._available,
                "status": self._status.value,
                "latency_ms": round(self._last_latency_ms, 1),
                "last_error": self._last_error,
                "last_error_time": self._last_error_time,
                "last_success_time": self._last_success_time,
                "total_plans": self._total_plans,
                "total_failures": self._total_failures,
            }

    # ── Updates (called by planner/engine) ───────────────────────────────────

    def mark_available(self, latency_ms: float = 0.0) -> None:
        """Mark planner as available after successful check."""
        with self._lock:
            self._available = True
            self._status = PlannerStatus.READY
            self._last_latency_ms = latency_ms
            self._checked = True

    def mark_offline(self, error: str = "") -> None:
        """Mark planner as offline (LLM unreachable)."""
        with self._lock:
            self._available = False
            self._status = PlannerStatus.OFFLINE
            self._last_error = error or "Local LLM unreachable"
            self._last_error_time = time.time()
            self._checked = True

    def mark_planning(self) -> None:
        """Mark that a planning operation is in progress."""
        with self._lock:
            self._status = PlannerStatus.PLANNING

    def mark_executing(self) -> None:
        """Mark that plan execution is in progress."""
        with self._lock:
            self._status = PlannerStatus.EXECUTING

    def mark_success(self, latency_ms: float) -> None:
        """Record a successful plan generation."""
        with self._lock:
            self._status = PlannerStatus.READY
            self._last_latency_ms = latency_ms
            self._last_success_time = time.time()
            self._total_plans += 1
            self._available = True

    def mark_failure(self, error: str, latency_ms: float = 0.0) -> None:
        """Record a planning failure."""
        with self._lock:
            self._status = PlannerStatus.READY if self._available else PlannerStatus.ERROR
            self._last_error = error
            self._last_error_time = time.time()
            self._last_latency_ms = latency_ms
            self._total_failures += 1

    def mark_ready(self) -> None:
        """Return to ready state after planning/executing completes."""
        with self._lock:
            if self._available:
                self._status = PlannerStatus.READY

    # ── Startup check ────────────────────────────────────────────────────────

    def check_availability(self) -> bool:
        """
        Perform a quick health check of the local LLM.
        Returns True if available, False if offline.

        Non-blocking: uses a short timeout (3s).
        """
        start = time.monotonic()
        try:
            from core.llm_client import ensure_ollama_running, get_llm_provider

            provider = get_llm_provider()

            if provider == "ollama":
                import requests
                from core.llm_client import get_llm_settings
                url, _ = get_llm_settings()
                resp = requests.get(f"{url}/api/tags", timeout=3)
                if resp.status_code == 200:
                    latency = (time.monotonic() - start) * 1000
                    self.mark_available(latency)
                    return True
                else:
                    self.mark_offline(f"Ollama returned {resp.status_code}")
                    return False
            else:
                # OpenAI-compatible: check /v1/models
                import requests
                from core.llm_client import get_llm_settings
                url, _ = get_llm_settings()
                resp = requests.get(f"{url}/v1/models", timeout=3)
                if resp.status_code == 200:
                    latency = (time.monotonic() - start) * 1000
                    self.mark_available(latency)
                    return True
                else:
                    self.mark_offline(f"LLM server returned {resp.status_code}")
                    return False

        except ImportError:
            self.mark_offline("core.llm_client not available")
            return False
        except Exception as e:
            self.mark_offline(str(e)[:100])
            return False

    def status_text(self) -> str:
        """Human-readable status for UI/logging."""
        with self._lock:
            if self._status == PlannerStatus.READY:
                return "Planner Ready"
            elif self._status == PlannerStatus.OFFLINE:
                return "Planner Offline — Direct Command Mode"
            elif self._status == PlannerStatus.PLANNING:
                return "Planning..."
            elif self._status == PlannerStatus.EXECUTING:
                return "Executing Plan..."
            elif self._status == PlannerStatus.ERROR:
                return f"Planner Error: {self._last_error[:50]}"
            return "Planner: Unknown"
