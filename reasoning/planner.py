"""
reasoning/planner.py — LLM-based goal decomposition for CARL.

Takes a high-level goal and produces a structured Plan with capability-based
steps, confidence scoring, and estimated duration.

The planner outputs CAPABILITIES (e.g. "list_files", "categorize_files")
rather than concrete tool names. The Tool Resolver handles mapping later.
"""
from __future__ import annotations

import json
import re
from typing import Any

from reasoning.models import Plan, Step, RetryPolicy


# ── Available capabilities the planner can reference ─────────────────────────

CAPABILITY_CATALOG = """
AVAILABLE CAPABILITIES (use these exact names in your plan):

File operations:
  list_files          — List contents of a directory
  create_file         — Create a new file with content
  create_folder       — Create a new directory
  delete_files        — Delete files or folders (DESTRUCTIVE)
  move_files          — Move files to another location (DESTRUCTIVE)
  copy_files          — Copy files to another location
  rename_files        — Rename a file or folder (DESTRUCTIVE)
  read_file           — Read contents of a file
  write_file          — Write/overwrite file content (DESTRUCTIVE)
  find_files          — Search for files by name or extension
  file_info           — Get file metadata (size, dates)
  disk_usage          — Check disk space
  organize_files      — Auto-organize files by type

Desktop:
  set_wallpaper       — Change desktop wallpaper
  organize_desktop    — Organize desktop items
  desktop_stats       — Get desktop file statistics

Browser:
  open_url            — Navigate to a URL
  web_search          — Search the web
  click_element       — Click a web page element
  type_text_browser   — Type text into a web form
  browser_screenshot  — Take a screenshot of the browser
  close_browser       — Close the browser

Applications:
  open_application    — Launch an application
  close_application   — Close an application

System:
  adjust_volume       — Change system volume
  adjust_brightness   — Change screen brightness
  system_status       — Get CPU/RAM/GPU metrics
  lock_screen         — Lock the computer
  restart_system      — Restart the computer (DESTRUCTIVE)
  shutdown_system     — Shut down the computer (DESTRUCTIVE)

Communication:
  send_message        — Send a message via WhatsApp/Telegram
  set_reminder        — Create a timed reminder

Media:
  play_video          — Play a YouTube video
  search_video        — Search YouTube

Code:
  write_code          — Write a code file
  run_code            — Execute a code file
  edit_code           — Modify existing code
  explain_code        — Explain what code does

Computer control:
  type_text           — Type text with keyboard
  click_screen        — Click at screen coordinates
  press_key           — Press a keyboard key
  hotkey              — Press a key combination
  scroll              — Scroll the screen
  screenshot          — Take a screenshot
"""


# ── Planner prompt template ──────────────────────────────────────────────────

_PLANNER_SYSTEM = """You are a task planning engine for an AI operating system called CARL.

Your job: decompose a user's goal into a structured JSON plan.

Rules:
1. Output ONLY valid JSON. No markdown, no explanation, no commentary.
2. Use CAPABILITIES from the catalog — never invent tool names.
3. Each step must be atomic (one action).
4. Order steps logically. Use dependencies when a step needs a prior result.
5. Set requires_confirmation=true for any destructive action (delete, move, rename, overwrite, install, registry, shutdown, restart).
6. Estimate total duration in seconds.
7. Set confidence (0.0-1.0) based on how well you understand the goal:
   - 0.9+ : Crystal clear goal, obvious steps
   - 0.7-0.89 : Clear goal, some assumptions needed
   - 0.4-0.69 : Ambiguous, multiple interpretations possible
   - <0.4 : Too vague to plan safely
8. Keep plans between 2-8 steps. If more are needed, group related actions.
9. Arguments should be specific and actionable.

{capability_catalog}

Output format (strict JSON):
{{
  "goal": "<restated goal>",
  "confidence": <float 0.0-1.0>,
  "estimated_duration_sec": <int>,
  "steps": [
    {{
      "id": "step_1",
      "description": "<what this step does>",
      "capability": "<capability_name from catalog>",
      "arguments": {{"<key>": "<value>"}},
      "dependencies": [],
      "retry_policy": {{"max_retries": 2, "backoff_sec": 1}},
      "requires_confirmation": false
    }}
  ]
}}"""


_PLANNER_USER = """Goal: {goal}

Context:
{context}

Create a structured plan. Output ONLY the JSON object."""


# ── Planner class ────────────────────────────────────────────────────────────

class Planner:
    """Decomposes complex goals into structured Plans using the local LLM."""

    _TIMEOUT_SEC = 8  # Max time for a single LLM planning call

    def __init__(self, health=None):
        from reasoning.health import PlannerHealth
        self._health = health or PlannerHealth()

    @property
    def health(self):
        return self._health

    def create_plan(
        self,
        goal: str,
        context: str = "",
    ) -> Plan:
        """
        Generate a structured plan for the given goal.

        Includes:
          - 8-second timeout (cancels if LLM is too slow)
          - Offline detection (catches connection errors)
          - Retry on JSON parse failure (one retry)
          - Never crashes — raises PlannerTimeout, PlannerOffline, or PlanningError

        Args:
            goal: The user's high-level goal (natural language)
            context: Additional context (memory, current state, etc.)

        Returns:
            A Plan object with capability-based steps and confidence score.

        Raises:
            PlannerTimeout: LLM took longer than 8 seconds.
            PlannerOffline: Local LLM is unreachable.
            PlanningError: LLM returned invalid/unparseable output after retry.
        """
        import time as _time

        self._health.mark_planning()
        start = _time.monotonic()

        # Attempt 1
        raw = self._call_llm(goal, context)
        latency_ms = (_time.monotonic() - start) * 1000

        try:
            plan_data = self._parse_plan_json(raw)
            plan = self._build_plan(plan_data, goal)
            self._health.mark_success(latency_ms)
            return plan
        except PlanningError:
            pass  # Fall through to retry

        # Attempt 2 (retry once on parse failure)
        print(f"[Reasoning] [!] Plan parse failed — retrying...")
        start2 = _time.monotonic()
        try:
            raw2 = self._call_llm(goal, context)
            latency_ms = (_time.monotonic() - start) * 1000
            plan_data = self._parse_plan_json(raw2)
            plan = self._build_plan(plan_data, goal)
            self._health.mark_success(latency_ms)
            return plan
        except (PlannerTimeout, PlannerOffline):
            raise
        except Exception as e:
            latency_ms = (_time.monotonic() - start) * 1000
            self._health.mark_failure(f"Parse failed after retry: {e}", latency_ms)
            raise PlanningError(f"Invalid plan JSON after retry: {e}")

    def _call_llm(self, goal: str, context: str) -> str:
        """
        Call the local LLM with an 8-second timeout.

        Raises:
            PlannerTimeout: If the call exceeds 8 seconds.
            PlannerOffline: If the LLM is unreachable.
        """
        import time as _time
        import concurrent.futures

        system = _PLANNER_SYSTEM.format(capability_catalog=CAPABILITY_CATALOG)
        user_msg = _PLANNER_USER.format(goal=goal, context=context or "None provided.")

        def _do_call() -> str:
            from core.llm_client import call_llm_text
            return call_llm_text(
                prompt=user_msg,
                system=system,
                timeout=self._TIMEOUT_SEC,
            )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_do_call)
                result = future.result(timeout=self._TIMEOUT_SEC)
                return result

        except concurrent.futures.TimeoutError:
            latency_ms = self._TIMEOUT_SEC * 1000
            self._health.mark_failure("Timeout", latency_ms)
            raise PlannerTimeout(f"Planner timed out after {self._TIMEOUT_SEC}s")

        except RuntimeError as e:
            err = str(e)
            if "Cannot connect" in err or "Connection" in err:
                self._health.mark_offline(err[:100])
                raise PlannerOffline(err[:100])
            self._health.mark_failure(err[:100])
            raise PlanningError(f"LLM call failed: {err[:100]}")

        except Exception as e:
            err = str(e)
            # Detect offline conditions
            offline_keywords = ("Cannot connect", "ConnectionError", "Connection refused",
                                "ConnectTimeout", "not running", "refused")
            if any(kw.lower() in err.lower() for kw in offline_keywords):
                self._health.mark_offline(err[:100])
                raise PlannerOffline(err[:100])
            self._health.mark_failure(err[:100])
            raise PlanningError(f"LLM call failed: {err[:100]}")

    def _parse_plan_json(self, raw: str) -> dict:
        """Extract and parse JSON from LLM response."""
        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in the text
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise PlanningError(f"Failed to parse plan JSON from LLM response: {raw[:200]}")

    def _build_plan(self, data: dict, original_goal: str) -> Plan:
        """Convert parsed JSON dict into a Plan object with validation."""
        steps_data = data.get("steps", [])
        if not steps_data:
            raise PlanningError("Plan contains no steps.")

        steps: list[Step] = []
        for i, s in enumerate(steps_data):
            step_id = s.get("id", f"step_{i + 1}")
            capability = s.get("capability", "")
            if not capability:
                continue  # skip steps without a capability

            rp_data = s.get("retry_policy", {})
            retry_policy = RetryPolicy(
                max_retries=rp_data.get("max_retries", 2),
                backoff_sec=rp_data.get("backoff_sec", 1.0),
            )

            steps.append(Step(
                id=step_id,
                description=s.get("description", f"Step {i + 1}"),
                capability=capability,
                arguments=s.get("arguments", {}),
                dependencies=s.get("dependencies", []),
                retry_policy=retry_policy,
                requires_confirmation=s.get("requires_confirmation", False),
            ))

        if not steps:
            raise PlanningError("No valid steps could be extracted from plan.")

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return Plan(
            goal=data.get("goal", original_goal),
            confidence=confidence,
            steps=steps,
            estimated_duration_sec=float(data.get("estimated_duration_sec", 30)),
        )


# ── Exceptions ───────────────────────────────────────────────────────────────

class PlanningError(Exception):
    """Raised when the planner fails to produce a valid plan."""
    pass


class PlannerTimeout(PlanningError):
    """Raised when the planner exceeds the time limit."""
    pass


class PlannerOffline(PlanningError):
    """Raised when the local LLM is unreachable."""
    pass
