"""
reasoning/engine.py — The main entry point for CARL's Reasoning Engine.

ReasoningEngine is the single class that main.py interacts with.
It decides whether a command needs planning, executes plans, and
handles session commands (pause/resume/cancel).

The planner is an ENHANCEMENT, not a dependency. If planning fails
for any reason, CARL falls back to direct execution or asks for
clarification — commands are never blocked.

Flow:
    1. evaluate(tool_name, args) → ExecutionLevel
    2. If SINGLE → bypass, execute normally
    3. If PLAN → plan_and_execute(goal, ui, speak_fn) → result string
       - On planner failure → fallback to SINGLE or clarification
    4. If CLARIFY → return clarification question
    5. If MISSION → future (treated as PLAN for now)
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Callable

from reasoning.models import ExecutionLevel, PlanResult
from reasoning.planner import Planner, PlanningError, PlannerTimeout, PlannerOffline
from reasoning.tool_resolver import ToolResolver
from reasoning.safety import SafetyValidator
from reasoning.task_manager import TaskManager
from reasoning.executor import Executor
from reasoning.progress import ProgressReporter
from reasoning.history import MissionHistory
from reasoning.health import PlannerHealth, PlannerStatus


# ── Tools that are ALWAYS single-step (bypass planner completely) ────────────

_ALWAYS_SINGLE = frozenset({
    "open_app",
    "weather_report",
    "send_message",
    "reminder",
    "youtube_video",
    "screen_process",
    "close_camera",
    "computer_settings",
    "system_status",
    "save_memory",
    "shutdown_jarvis",
    "game_updater",
    "flight_finder",
    "file_processor",
    "web_search",
})

# ── Tools that CAN trigger planning depending on complexity ──────────────────

_PLANNABLE_TOOLS = frozenset({
    "file_controller",
    "browser_control",
    "desktop_control",
    "computer_control",
    "code_helper",
    "dev_agent",
})

# ── Argument patterns that suggest a complex multi-step goal ─────────────────

_COMPLEX_INDICATORS = {
    "file_controller": {
        "actions": {"organize_desktop"},
        "keywords": {"organize", "clean", "sort all", "categorize", "deduplicate",
                     "backup", "archive all", "restructure"},
    },
    "browser_control": {
        "actions": set(),
        "keywords": {"fill form", "complete form", "automate", "scrape",
                     "download all", "batch"},
    },
    "desktop_control": {
        "actions": {"organize", "clean"},
        "keywords": {"organize", "clean", "sort", "arrange"},
    },
    "computer_control": {
        "actions": set(),
        "keywords": {"automate", "sequence", "workflow", "fill form",
                     "complete registration"},
    },
    "code_helper": {
        "actions": {"build"},
        "keywords": {"project", "full app", "complete program", "refactor all"},
    },
    "dev_agent": {
        "actions": set(),
        "keywords": set(),
    },
}


# ── Session command detection ────────────────────────────────────────────────

_SESSION_COMMANDS = {
    "pause":    ["pause", "hold", "wait", "stop for now"],
    "resume":   ["continue", "resume", "go ahead", "proceed", "keep going"],
    "cancel":   ["cancel", "abort", "stop", "never mind", "forget it"],
    "retry":    ["retry", "try again", "repeat"],
    "status":   ["status", "progress", "how far", "where are you"],
}


# ── Fallback result indicators ───────────────────────────────────────────────

class FallbackResult:
    """Returned when planning fails and fallback to direct execution is needed."""
    DIRECT = "fallback_direct"      # Execute the original tool call directly
    CLARIFY = "fallback_clarify"    # Ask user for more specifics


class ReasoningEngine:
    """
    Main orchestrator for CARL's reasoning capabilities.

    The planner is an enhancement — if it's offline, times out, or fails,
    commands still execute via direct fallback. Never blocks.
    """

    def __init__(self):
        self._health = PlannerHealth()
        self._planner = Planner(health=self._health)
        self._resolver = ToolResolver()
        self._safety = SafetyValidator()
        self._task_manager = TaskManager()
        self._executor = Executor(self._task_manager, self._resolver, self._safety)
        self._progress = ProgressReporter()
        self._history = MissionHistory()
        self._context_engine = None  # Set via set_context_engine() from main.py

    def set_context_engine(self, context_engine) -> None:
        """
        Inject the context engine from main.py.
        Called after both ReasoningEngine and ContextEngine are instantiated.
        """
        self._context_engine = context_engine

    @property
    def health(self) -> PlannerHealth:
        """Access planner health monitor."""
        return self._health

    # ── Startup check ────────────────────────────────────────────────────────

    def startup_check(self) -> bool:
        """
        Perform a one-time planner availability check at startup.
        Returns True if planner is available, False if offline.
        Non-blocking, fast (3s timeout).
        """
        available = self._health.check_availability()
        if available:
            print(f"[Reasoning] [OK] Planner Ready (latency: {self._health.planner_latency():.0f}ms)")
        else:
            print(f"[Reasoning] [!] Planner Offline — Using Direct Command Mode")
        return available

    # ── Primary interface ────────────────────────────────────────────────────

    def evaluate(self, tool_name: str, args: dict) -> ExecutionLevel:
        """
        Decide how a tool call should be processed.

        If planner is offline, always returns SINGLE (direct execution).
        Commands are never blocked by planner unavailability.
        """
        # Always-single tools bypass completely
        if tool_name in _ALWAYS_SINGLE:
            return ExecutionLevel.SINGLE

        # dev_agent has its own internal planning
        if tool_name == "dev_agent":
            return ExecutionLevel.SINGLE

        # If planner is offline, fall back to direct execution
        if not self._health.planner_available():
            return ExecutionLevel.SINGLE

        # Check if this specific call is complex enough to warrant planning
        if tool_name in _PLANNABLE_TOOLS:
            if self._is_complex(tool_name, args):
                return ExecutionLevel.PLAN

        return ExecutionLevel.SINGLE

    def should_plan(self, tool_name: str, args: dict) -> bool:
        """Convenience wrapper: returns True if planning is needed."""
        return self.evaluate(tool_name, args) == ExecutionLevel.PLAN

    def plan_and_execute(
        self,
        goal: str,
        ui: Any = None,
        speak_fn: Callable[[str], None] | None = None,
        confirm_fn: Callable[[str], bool] | None = None,
        context: str = "",
    ) -> str:
        """
        Full pipeline: plan → validate → execute → record → return summary.

        FALLBACK STRATEGY:
          - PlannerTimeout → return FallbackResult.DIRECT
          - PlannerOffline → return FallbackResult.DIRECT
          - PlanningError (invalid JSON) → return FallbackResult.CLARIFY
          - Any other error → return FallbackResult.DIRECT

        When FallbackResult.DIRECT is returned, main.py executes the original
        tool call normally (as if planning was never attempted).

        Returns:
            A natural-language result string, OR a FallbackResult constant.
        """
        try:
            # 1. Enrich context via Context Engine (Phase 17 integration)
            enriched_context = context
            if self._context_engine:
                try:
                    memory_ctx = self._context_engine.build_context(goal)
                    if memory_ctx:
                        enriched_context = f"{memory_ctx}\n\n{context}" if context else memory_ctx
                except Exception as e:
                    print(f"[Reasoning] Context engine error (non-fatal): {e}")

            # 2. Generate plan
            print(f"[Reasoning] [B] Planning: {goal}")
            self._health.mark_planning()
            plan = self._planner.create_plan(goal, context=enriched_context)
            print(f"[Reasoning] [P] Plan ready: {plan.total_steps} steps, "
                  f"confidence={plan.confidence:.2f}")

            # 2. Check confidence
            if plan.confidence < 0.4:
                msg = self._progress.clarification_needed(goal)
                return msg

            # 3. Speak plan start
            start_msg = self._progress.plan_started(plan)
            if speak_fn:
                speak_fn(start_msg)

            # 4. Warn on moderate confidence
            if plan.confidence < 0.7:
                warn_msg = self._progress.confidence_warning(plan)
                if speak_fn:
                    speak_fn(warn_msg)

            # 5. Show plan in UI
            if ui:
                try:
                    ui.show_content("TASK PLAN", self._progress.format_card(plan))
                except Exception:
                    pass

            # 6. Execute
            self._health.mark_executing()

            def _on_step_complete(step, completed, total):
                if ui:
                    try:
                        ui.show_content("TASK PROGRESS", self._progress.format_card(plan))
                    except Exception:
                        pass
                if speak_fn:
                    msg = self._progress.step_completed(step, completed, total)
                    if msg:
                        speak_fn(msg)

            result = self._executor.run(
                plan=plan,
                ui=ui,
                speak_fn=speak_fn,
                confirm_fn=confirm_fn,
                on_step_complete=_on_step_complete,
            )

            # 7. Record to history
            self._history.record(result)

            # 8. Show final result in UI
            if ui:
                try:
                    ui.show_content("TASK COMPLETE", self._progress.format_result_card(result))
                except Exception:
                    pass

            self._health.mark_ready()

            # 9. Return summary
            return self._progress.plan_completed(result)

        except PlannerTimeout as e:
            self._log_failure(goal, "timeout", str(e))
            self._history.record_failure(goal, "Planner Timeout", str(e))
            print(f"[Reasoning] [T] Planner timeout — falling back to direct execution")
            return FallbackResult.DIRECT

        except PlannerOffline as e:
            self._log_failure(goal, "offline", str(e))
            self._history.record_failure(goal, "Planner Offline", str(e))
            print(f"[Reasoning] [S] Planner offline — falling back to direct execution")
            return FallbackResult.DIRECT

        except PlanningError as e:
            self._log_failure(goal, "parse_error", str(e))
            self._history.record_failure(goal, "Invalid JSON", str(e))
            print(f"[Reasoning] [!] Planning failed — asking for clarification")
            return FallbackResult.CLARIFY

        except Exception as e:
            self._log_failure(goal, "unexpected", str(e))
            self._history.record_failure(goal, "Unexpected Error", str(e))
            print(f"[Reasoning] [!] Unexpected error: {e}")
            traceback.print_exc()
            return FallbackResult.DIRECT

        finally:
            self._health.mark_ready()

    # ── Session commands ─────────────────────────────────────────────────────

    def handle_session_command(self, text: str) -> str | None:
        """
        Check if user text is a session command (pause/resume/cancel/status).
        Only active when a plan is running.
        """
        if not self._task_manager.is_active:
            return None

        text_lower = text.lower().strip()

        for cmd, triggers in _SESSION_COMMANDS.items():
            if any(t in text_lower for t in triggers):
                return self._execute_session_command(cmd)

        return None

    def _execute_session_command(self, cmd: str) -> str:
        """Execute a session command and return the response."""
        if cmd == "pause":
            return self._task_manager.pause()
        elif cmd == "resume":
            return self._task_manager.resume()
        elif cmd == "cancel":
            return self._task_manager.cancel()
        elif cmd == "retry":
            return self._task_manager.retry_last()
        elif cmd == "status":
            progress = self._task_manager.get_progress()
            if not progress.get("active"):
                return "No active task."
            pct = progress["progress_pct"]
            current = progress.get("current_step", "")
            return (f"Task is {pct:.0f}% complete. "
                    f"Currently: {current}" if current else f"Task is {pct:.0f}% complete.")
        return "Unknown command."

    @property
    def is_plan_active(self) -> bool:
        """Check if a plan is currently being executed."""
        return self._task_manager.is_active

    def get_progress(self) -> dict:
        """Get current plan progress snapshot."""
        return self._task_manager.get_progress()

    @property
    def history(self) -> MissionHistory:
        """Access mission history for querying."""
        return self._history

    # ── Complexity detection ─────────────────────────────────────────────────

    def _is_complex(self, tool_name: str, args: dict) -> bool:
        """
        Determine if a specific tool call represents a complex multi-step goal.
        Heuristic-based — no LLM call needed for this gate.
        """
        indicators = _COMPLEX_INDICATORS.get(tool_name)
        if not indicators:
            return False

        action = args.get("action", "").lower()
        if action in indicators.get("actions", set()):
            return True

        desc = (
            args.get("description", "") +
            " " + args.get("task", "") +
            " " + args.get("query", "")
        ).lower()

        keywords = indicators.get("keywords", set())
        if any(kw in desc for kw in keywords):
            return True

        return False

    # ── Goal extraction ──────────────────────────────────────────────────────

    def extract_goal(self, tool_name: str, args: dict) -> str:
        """Extract a natural-language goal from a tool call's arguments."""
        goal = args.get("description", "")
        if goal:
            return goal

        goal = args.get("task", "")
        if goal:
            return goal

        goal = args.get("query", "")
        if goal:
            return goal

        action = args.get("action", "")
        path = args.get("path", "")
        if action and path:
            return f"{action} {path}"
        if action:
            return f"{action} files"

        return f"Complete {tool_name} task"

    # ── Failure logging ──────────────────────────────────────────────────────

    def _log_failure(self, goal: str, error_type: str, error: str) -> None:
        """Log a planner failure with timestamp, goal, error, and latency."""
        latency = self._health.planner_latency()
        print(
            f"[Reasoning] FAILURE | type={error_type} | "
            f"goal={goal[:50]} | latency={latency:.0f}ms | "
            f"error={error[:80]}"
        )
