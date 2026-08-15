"""
reasoning/executor.py — Step-by-step plan execution with retry logic.

Iterates through a plan's steps via the TaskManager, resolves each step's
capability to a concrete tool call via the ToolResolver, executes it,
and handles retries and error recovery.

The executor does NOT modify existing action modules — it calls them
exactly as main.py's _execute_tool does, through their public interfaces.
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Callable

from reasoning.models import Plan, PlanResult, Step, StepStatus
from reasoning.task_manager import TaskManager, PlanState
from reasoning.tool_resolver import ToolResolver, ResolutionError
from reasoning.safety import SafetyValidator


class Executor:
    """
    Executes plan steps sequentially with retry logic and error recovery.

    Usage:
        executor = Executor(task_manager, tool_resolver, safety_validator)
        result = executor.run(plan, ui=player, speak_fn=speak, confirm_fn=confirm)
    """

    def __init__(
        self,
        task_manager: TaskManager,
        tool_resolver: ToolResolver,
        safety_validator: SafetyValidator,
    ):
        self._tm = task_manager
        self._resolver = tool_resolver
        self._safety = safety_validator

    def run(
        self,
        plan: Plan,
        ui: Any = None,
        speak_fn: Callable[[str], None] | None = None,
        confirm_fn: Callable[[str], bool] | None = None,
        on_step_complete: Callable[[Step, int, int], None] | None = None,
    ) -> PlanResult:
        """
        Execute a plan step by step.

        Args:
            plan: The validated Plan to execute.
            ui: The UI object (passed as 'player' to action functions).
            speak_fn: Optional callback to speak progress updates.
            confirm_fn: Callback that asks user for confirmation. Returns True/False.
                        If None, destructive steps are skipped.
            on_step_complete: Called after each step with (step, completed_count, total).

        Returns:
            PlanResult with success status, summary, and statistics.
        """
        start_time = time.time()
        tools_used: list[str] = []
        total_retries = 0

        # Validate safety before execution
        self._safety.validate_plan(plan)

        # Start the plan in the task manager
        self._tm.start_plan(plan)

        # Execute steps
        while self._tm.is_active:
            # Check for pause
            if self._tm.is_paused:
                time.sleep(0.5)
                continue

            # Get next step
            step = self._tm.get_next_step()
            if step is None:
                # No more steps available — plan may be complete or blocked
                if plan.is_complete:
                    break
                # Could be waiting on dependencies or confirmation
                time.sleep(0.2)
                # Safety: if nothing changes after a reasonable wait, break
                if self._tm.state in (PlanState.COMPLETED, PlanState.CANCELLED, PlanState.FAILED):
                    break
                continue

            # Handle confirmation requirement
            if step.requires_confirmation:
                if confirm_fn:
                    msg = self._safety.get_confirmation_message(step)
                    approved = confirm_fn(msg)
                    if not approved:
                        self._tm.mark_step_skipped(step)
                        print(f"[Reasoning] ⏭️  Skipped (user declined): {step.description}")
                        continue
                else:
                    # No confirm function — skip destructive steps for safety
                    self._tm.mark_step_skipped(step)
                    print(f"[Reasoning] ⏭️  Skipped (no confirm available): {step.description}")
                    continue

            # Execute the step
            success = self._execute_step(step, ui, speak_fn)

            # Track tools used
            tool_name = self._resolver.get_tool_name(step.capability)
            if tool_name not in tools_used:
                tools_used.append(tool_name)

            # Handle retry on failure
            if not success and step.retries_used < step.retry_policy.max_retries:
                total_retries += 1
                self._tm.mark_step_retrying(step)
                time.sleep(step.retry_policy.backoff_sec)

                # Retry
                success = self._execute_step(step, ui, speak_fn)
                if not success and step.retries_used < step.retry_policy.max_retries:
                    # One more retry
                    total_retries += 1
                    self._tm.mark_step_retrying(step)
                    time.sleep(step.retry_policy.backoff_sec * 2)
                    success = self._execute_step(step, ui, speak_fn)

            # Final status
            if success:
                self._tm.mark_step_completed(step, step.result)
            else:
                self._tm.mark_step_failed(step, step.error)
                print(f"[Reasoning] [!] Step failed: {step.description} — {step.error}")

                # Decide whether to continue or abort
                if not self._can_continue_after_failure(step, plan):
                    if speak_fn:
                        speak_fn(f"Step failed: {step.description}. Cannot safely continue.")
                    break

            # Notify progress
            if on_step_complete:
                on_step_complete(step, plan.completed_steps, plan.total_steps)

        # Build result
        duration = time.time() - start_time
        summary = self._build_summary(plan)

        result = PlanResult(
            success=plan.is_successful,
            summary=summary,
            plan=plan,
            duration_sec=duration,
            tools_used=tools_used,
            total_retries=total_retries,
        )

        self._tm.reset()
        return result

    def _execute_step(self, step: Step, ui: Any, speak_fn: Callable | None) -> bool:
        """
        Execute a single step. Returns True on success, False on failure.
        Updates step.result or step.error accordingly.
        """
        self._tm.mark_step_running(step)
        print(f"[Reasoning] [>]  Executing: {step.description}")

        try:
            # Resolve capability to concrete tool
            tool_func, mapped_args = self._resolver.resolve(step.capability, step.arguments)
        except ResolutionError as e:
            step.error = f"Cannot resolve capability: {e}"
            return False

        try:
            # Call the action function
            # Different action functions have different signatures.
            # Most accept (parameters=dict, player=ui) or similar.
            result = self._call_tool(tool_func, mapped_args, ui)
            step.result = str(result) if result else "Done."
            return True

        except Exception as e:
            step.error = f"{type(e).__name__}: {str(e)[:200]}"
            traceback.print_exc()
            return False

    def _call_tool(self, tool_func: Callable, args: dict, ui: Any) -> Any:
        """
        Call an action function with the appropriate signature.

        Existing action functions use various signatures:
          - func(parameters=dict, player=ui)
          - func(parameters=dict, response=None, player=ui)
          - func(parameters=dict, player=ui, speak=speak_fn)
          - func() — for simple ones like get_system_status

        We try the most common pattern first, then fall back.
        """
        import inspect

        sig = inspect.signature(tool_func)
        params = list(sig.parameters.keys())

        # get_system_status() takes no args
        if not params:
            return tool_func()

        # Try (parameters, player) pattern — most common
        kwargs: dict[str, Any] = {}

        if "parameters" in params:
            kwargs["parameters"] = args
        elif "params" in params:
            kwargs["params"] = args

        if "player" in params:
            kwargs["player"] = ui

        if "response" in params:
            kwargs["response"] = None

        if "session_memory" in params:
            kwargs["session_memory"] = None

        if "speak" in params:
            kwargs["speak"] = lambda text: None  # silent during plan execution

        # If we matched at least one kwarg, call with kwargs
        if kwargs:
            return tool_func(**kwargs)

        # Last resort: try positional
        return tool_func(args)

    def _can_continue_after_failure(self, failed_step: Step, plan: Plan) -> bool:
        """
        Decide if remaining steps can safely execute after a failure.

        If other steps depend on the failed step, they cannot run.
        If remaining steps are independent, we can continue.
        """
        failed_id = failed_step.id
        remaining = [s for s in plan.steps if s.status == StepStatus.PENDING]

        # If all remaining steps depend on the failed step, abort
        dependent = [s for s in remaining if failed_id in s.dependencies]
        independent = [s for s in remaining if failed_id not in s.dependencies]

        # If there are independent steps, continue
        if independent:
            # Skip dependent steps
            for s in dependent:
                self._tm.mark_step_skipped(s)
            return True

        return False

    def _build_summary(self, plan: Plan) -> str:
        """Generate a natural-language summary of the plan execution."""
        completed = plan.completed_steps
        total = plan.total_steps
        failed = plan.failed_steps
        skipped = sum(1 for s in plan.steps if s.status == StepStatus.SKIPPED)

        if plan.is_successful:
            if total == 1:
                return f"Done. {plan.steps[0].result}"
            return f"All {completed} steps completed successfully."

        parts: list[str] = []
        if completed > 0:
            parts.append(f"{completed} of {total} steps completed")
        if failed > 0:
            failed_descs = [s.description for s in plan.steps if s.status == StepStatus.FAILED]
            parts.append(f"{failed} failed: {', '.join(failed_descs[:2])}")
        if skipped > 0:
            parts.append(f"{skipped} skipped")

        return ". ".join(parts) + "." if parts else "Plan did not complete."
