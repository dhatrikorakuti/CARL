"""
reasoning/task_manager.py — State machine for active plan management.

Holds the current plan, tracks step statuses, and supports session commands:
  - pause   → halt after current step completes
  - resume  → continue from where paused
  - cancel  → abort remaining steps
  - retry   → re-attempt the last failed step
  - status  → snapshot of current progress

Only one plan can be active at a time (singleton per session).
"""
from __future__ import annotations

import threading
import time
from enum import Enum

from reasoning.models import Plan, Step, StepStatus


# ── Plan State ───────────────────────────────────────────────────────────────

class PlanState(Enum):
    IDLE = "idle"               # No active plan
    RUNNING = "running"         # Executing steps
    PAUSED = "paused"           # User requested pause
    AWAITING = "awaiting"       # Waiting for user confirmation
    COMPLETED = "completed"     # All steps done
    CANCELLED = "cancelled"     # User cancelled
    FAILED = "failed"           # Unrecoverable failure


# ── Task Manager ─────────────────────────────────────────────────────────────

class TaskManager:
    """
    Manages the lifecycle of an active plan.

    Thread-safe — can be queried from the UI thread while the executor
    runs on an asyncio executor thread.
    """

    def __init__(self):
        self._plan: Plan | None = None
        self._state: PlanState = PlanState.IDLE
        self._lock = threading.Lock()
        self._current_step_index: int = 0
        self._confirmation_pending: Step | None = None
        self._user_confirmed: bool | None = None  # None=waiting, True/False=answered

    # ── Plan lifecycle ───────────────────────────────────────────────────────

    def start_plan(self, plan: Plan) -> None:
        """Activate a new plan. Replaces any existing plan."""
        with self._lock:
            self._plan = plan
            self._state = PlanState.RUNNING
            self._current_step_index = 0
            self._confirmation_pending = None
            self._user_confirmed = None
        print(f"[Reasoning] [P] Plan started: {plan.goal} ({plan.total_steps} steps)")

    def get_plan(self) -> Plan | None:
        """Return the current plan (or None if idle)."""
        with self._lock:
            return self._plan

    @property
    def state(self) -> PlanState:
        with self._lock:
            return self._state

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._state in (PlanState.RUNNING, PlanState.PAUSED, PlanState.AWAITING)

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._state == PlanState.RUNNING

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._state == PlanState.PAUSED

    # ── Step advancement ─────────────────────────────────────────────────────

    def get_next_step(self) -> Step | None:
        """
        Return the next step to execute, or None if:
          - plan is paused/cancelled
          - all steps are done
          - a confirmation is pending
        """
        with self._lock:
            if self._state != PlanState.RUNNING:
                return None
            if self._plan is None:
                return None
            if self._confirmation_pending is not None:
                return None

            # Find next pending step whose dependencies are satisfied
            for step in self._plan.steps:
                if step.status != StepStatus.PENDING:
                    continue
                if self._dependencies_met(step):
                    return step
            return None

    def _dependencies_met(self, step: Step) -> bool:
        """Check if all dependencies of a step are completed."""
        if not step.dependencies:
            return True
        for dep_id in step.dependencies:
            dep_step = self._find_step(dep_id)
            if dep_step is None:
                continue  # unknown dep — skip
            if dep_step.status != StepStatus.COMPLETED:
                return False
        return True

    def _find_step(self, step_id: str) -> Step | None:
        if self._plan is None:
            return None
        for s in self._plan.steps:
            if s.id == step_id:
                return s
        return None

    def mark_step_running(self, step: Step) -> None:
        """Mark a step as currently executing."""
        with self._lock:
            step.status = StepStatus.RUNNING
            step.started_at = time.time()

    def mark_step_completed(self, step: Step, result: str = "") -> None:
        """Mark a step as successfully completed."""
        with self._lock:
            step.status = StepStatus.COMPLETED
            step.result = result
            step.completed_at = time.time()
            self._check_plan_completion()

    def mark_step_failed(self, step: Step, error: str = "") -> None:
        """Mark a step as failed."""
        with self._lock:
            step.status = StepStatus.FAILED
            step.error = error
            step.completed_at = time.time()
            self._check_plan_completion()

    def mark_step_retrying(self, step: Step) -> None:
        """Mark a step as retrying."""
        with self._lock:
            step.status = StepStatus.RETRYING
            step.retries_used += 1

    def mark_step_skipped(self, step: Step) -> None:
        """Mark a step as skipped (e.g. user declined confirmation)."""
        with self._lock:
            step.status = StepStatus.SKIPPED
            step.completed_at = time.time()
            self._check_plan_completion()

    # ── Confirmation flow ────────────────────────────────────────────────────

    def request_confirmation(self, step: Step) -> None:
        """Pause execution and wait for user confirmation on a step."""
        with self._lock:
            self._state = PlanState.AWAITING
            self._confirmation_pending = step
            self._user_confirmed = None
            step.status = StepStatus.AWAITING_CONFIRMATION

    def get_pending_confirmation(self) -> Step | None:
        """Return the step awaiting confirmation, or None."""
        with self._lock:
            return self._confirmation_pending

    def confirm(self, approved: bool) -> None:
        """User responds to confirmation request."""
        with self._lock:
            self._user_confirmed = approved
            if self._confirmation_pending:
                if approved:
                    self._confirmation_pending.status = StepStatus.PENDING
                else:
                    self._confirmation_pending.status = StepStatus.SKIPPED
                    self._confirmation_pending.completed_at = time.time()
                self._confirmation_pending = None
            self._state = PlanState.RUNNING

    def is_awaiting_confirmation(self) -> bool:
        with self._lock:
            return self._state == PlanState.AWAITING

    def get_user_confirmation(self) -> bool | None:
        """Return the user's confirmation answer (True/False/None=pending)."""
        with self._lock:
            return self._user_confirmed

    # ── Session commands ─────────────────────────────────────────────────────

    def pause(self) -> str:
        """Pause execution after current step completes."""
        with self._lock:
            if self._state == PlanState.RUNNING:
                self._state = PlanState.PAUSED
                return "Plan paused. Say 'continue' to resume."
            return "No active plan to pause."

    def resume(self) -> str:
        """Resume a paused plan."""
        with self._lock:
            if self._state == PlanState.PAUSED:
                self._state = PlanState.RUNNING
                return "Resuming plan."
            return "Plan is not paused."

    def cancel(self) -> str:
        """Cancel the active plan. Remaining steps are abandoned."""
        with self._lock:
            if self._state in (PlanState.RUNNING, PlanState.PAUSED, PlanState.AWAITING):
                self._state = PlanState.CANCELLED
                # Mark remaining pending steps as skipped
                if self._plan:
                    for step in self._plan.steps:
                        if step.status in (StepStatus.PENDING, StepStatus.AWAITING_CONFIRMATION):
                            step.status = StepStatus.SKIPPED
                self._confirmation_pending = None
                return "Plan cancelled."
            return "No active plan to cancel."

    def retry_last(self) -> str:
        """Re-attempt the most recently failed step."""
        with self._lock:
            if self._plan is None:
                return "No active plan."
            # Find last failed step
            for step in reversed(self._plan.steps):
                if step.status == StepStatus.FAILED:
                    step.status = StepStatus.PENDING
                    step.error = ""
                    self._state = PlanState.RUNNING
                    return f"Retrying: {step.description}"
            return "No failed steps to retry."

    # ── Progress snapshot ────────────────────────────────────────────────────

    def get_progress(self) -> dict:
        """Return a snapshot of current plan progress."""
        with self._lock:
            if self._plan is None:
                return {"active": False}

            current = self._plan.current_step
            return {
                "active": True,
                "state": self._state.value,
                "goal": self._plan.goal,
                "total_steps": self._plan.total_steps,
                "completed_steps": self._plan.completed_steps,
                "failed_steps": self._plan.failed_steps,
                "progress_pct": round(self._plan.progress_pct, 1),
                "current_step": current.description if current else None,
                "confidence": self._plan.confidence,
            }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _check_plan_completion(self) -> None:
        """Check if all steps are done and update plan state accordingly."""
        if self._plan and self._plan.is_complete:
            if self._plan.is_successful:
                self._state = PlanState.COMPLETED
            else:
                # Has failures but all steps attempted
                self._state = PlanState.COMPLETED

    def reset(self) -> None:
        """Clear the task manager for a new plan."""
        with self._lock:
            self._plan = None
            self._state = PlanState.IDLE
            self._current_step_index = 0
            self._confirmation_pending = None
            self._user_confirmed = None
