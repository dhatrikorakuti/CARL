"""
reasoning/progress.py — Natural language progress reporting for CARL.

Generates:
  - Voice-friendly step updates (spoken via Gemini session)
  - UI content cards (displayed in the HUD content panel)
  - Completion summaries

All output follows CARL's persona: calm, composed, British, no filler.
"""
from __future__ import annotations

from reasoning.models import Plan, Step, StepStatus, PlanResult


# ── Voice Updates ────────────────────────────────────────────────────────────

class ProgressReporter:
    """Generates natural-language progress messages for voice and UI."""

    def plan_started(self, plan: Plan) -> str:
        """Message when a plan begins execution."""
        n = plan.total_steps
        if n == 2:
            return f"I'll handle this in two steps."
        if n <= 4:
            return f"I'll take care of this in {n} steps."
        return f"This will require {n} steps. Beginning now."

    def step_started(self, step: Step, step_num: int, total: int) -> str:
        """Message when a step begins."""
        if step_num == 1:
            return f"First, {self._lower_first(step.description)}."
        if step_num == total:
            return f"Final step: {self._lower_first(step.description)}."
        return f"Next, {self._lower_first(step.description)}."

    def step_completed(self, step: Step, step_num: int, total: int) -> str:
        """Message when a step completes successfully."""
        remaining = total - step_num
        if remaining == 0:
            return "Done."
        if remaining == 1:
            return "One step remaining."
        # Don't narrate every step — only speak on milestones
        if step_num == 1 and total > 3:
            return f"First step complete. {remaining} to go."
        return ""  # Silent for mid-plan steps to avoid over-talking

    def step_failed(self, step: Step) -> str:
        """Message when a step fails."""
        short_err = step.error[:80] if step.error else "unknown error"
        return f"Step encountered an issue: {short_err}. Continuing where possible."

    def step_skipped(self, step: Step) -> str:
        """Message when a step is skipped (user declined or dependency failed)."""
        return f"Skipping: {self._lower_first(step.description)}."

    def plan_completed(self, result: PlanResult) -> str:
        """Final summary message after plan execution."""
        if result.success:
            dur = self._format_duration(result.duration_sec)
            return f"Everything is finished. Completed in {dur}."
        
        plan = result.plan
        completed = plan.completed_steps
        failed = plan.failed_steps
        
        if failed == 0:
            return f"Plan complete. {completed} of {plan.total_steps} steps executed."
        
        return f"Plan finished with {failed} issue{'s' if failed > 1 else ''}. {result.summary}"

    def plan_cancelled(self) -> str:
        """Message when user cancels the plan."""
        return "Plan cancelled."

    def confidence_warning(self, plan: Plan) -> str:
        """Message when confidence is moderate (0.4-0.69)."""
        return "I'm making some assumptions here. I'll proceed, but let me know if I've misunderstood."

    def clarification_needed(self, goal: str) -> str:
        """Message when confidence is too low to proceed."""
        return f"I'd like to clarify before proceeding. Could you be more specific about what you'd like me to do?"

    # ── UI Card Formatting ───────────────────────────────────────────────────

    def format_card(self, plan: Plan) -> str:
        """
        Format a compact progress card for the UI content panel.
        Shows current status of all steps.
        """
        lines: list[str] = []
        lines.append(f"TASK: {plan.goal}")
        lines.append(f"Progress: {plan.completed_steps}/{plan.total_steps} "
                     f"({plan.progress_pct:.0f}%)")
        lines.append("")

        for i, step in enumerate(plan.steps, 1):
            icon = self._status_icon(step.status)
            lines.append(f"  {icon} {i}. {step.description}")
            if step.status == StepStatus.FAILED and step.error:
                lines.append(f"       Error: {step.error[:60]}")
            if step.result and step.status == StepStatus.COMPLETED:
                short = step.result[:60]
                if len(step.result) > 60:
                    short += "..."
                lines.append(f"       Result: {short}")

        return "\n".join(lines)

    def format_result_card(self, result: PlanResult) -> str:
        """Format a final result card for the UI."""
        lines: list[str] = []
        status = "COMPLETED" if result.success else "PARTIAL"
        lines.append(f"TASK {status}: {result.plan.goal}")
        lines.append(f"Duration: {self._format_duration(result.duration_sec)}")
        lines.append(f"Steps: {result.plan.completed_steps}/{result.plan.total_steps} completed")
        if result.total_retries > 0:
            lines.append(f"Retries: {result.total_retries}")
        lines.append(f"Tools: {', '.join(result.tools_used)}")
        lines.append("")
        lines.append(f"Summary: {result.summary}")
        return "\n".join(lines)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _lower_first(text: str) -> str:
        """Lowercase the first character for mid-sentence embedding."""
        if not text:
            return text
        if text[0].isupper() and (len(text) < 2 or not text[1].isupper()):
            return text[0].lower() + text[1:]
        return text

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds into a natural duration string."""
        if seconds < 2:
            return "under a second"
        if seconds < 60:
            return f"{int(seconds)} seconds"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if secs == 0:
            return f"{minutes} minute{'s' if minutes > 1 else ''}"
        return f"{minutes}m {secs}s"

    @staticmethod
    def _status_icon(status: StepStatus) -> str:
        """Return a compact icon for step status."""
        return {
            StepStatus.PENDING: "[ ]",
            StepStatus.RUNNING: "▶",
            StepStatus.COMPLETED: "[OK]",
            StepStatus.FAILED: "[X]",
            StepStatus.RETRYING: "↻",
            StepStatus.SKIPPED: "—",
            StepStatus.AWAITING_CONFIRMATION: "?",
        }.get(status, "·")
