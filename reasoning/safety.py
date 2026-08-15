"""
reasoning/safety.py — Safety Validator for the CARL Reasoning Engine.

Sits between the Planner and Executor. Inspects every step in a plan and
automatically flags destructive actions for user confirmation before execution.

Destructive actions include:
  - File deletion, move, rename, overwrite
  - Registry edits
  - Software installations
  - Shell/system commands
  - System configuration changes (shutdown, restart, network)
  - Browser form submissions with sensitive data
"""
from __future__ import annotations

from reasoning.models import Plan, Step, StepStatus


# ── Destructive capability sets ──────────────────────────────────────────────

# Capabilities that ALWAYS require confirmation
_ALWAYS_DESTRUCTIVE = frozenset({
    "delete_files",
    "move_files",
    "rename_files",
    "write_file",
    "restart_system",
    "shutdown_system",
})

# Capabilities that are destructive depending on arguments
_CONDITIONALLY_DESTRUCTIVE = frozenset({
    "organize_files",       # moves files around
    "organize_desktop",     # rearranges desktop
    "close_application",    # force-closes apps (possible data loss)
    "run_code",             # arbitrary code execution
    "edit_code",            # modifies existing files
    "hotkey",              # could trigger destructive shortcuts
})

# Argument patterns that escalate a step to destructive
_DANGEROUS_ARG_PATTERNS = {
    "action": {"delete", "remove", "uninstall", "format", "wipe", "reset",
               "overwrite", "replace", "drop", "truncate"},
    "command": {"rm", "del", "rmdir", "format", "reg", "regedit",
                "shutdown", "restart", "kill", "taskkill"},
}

# Keywords in descriptions that suggest destructive intent
_DANGEROUS_DESCRIPTION_KEYWORDS = frozenset({
    "delete", "remove", "erase", "wipe", "format", "uninstall",
    "overwrite", "replace", "destroy", "clear all", "empty",
    "registry", "regedit", "system restore", "factory reset",
    "drop database", "truncate",
})


# ── Safety Validator ─────────────────────────────────────────────────────────

class SafetyValidator:
    """
    Validates plan steps for safety. Marks destructive steps as
    requires_confirmation=True so the Executor asks the user before proceeding.
    """

    def validate_plan(self, plan: Plan) -> Plan:
        """
        Inspect all steps in the plan and flag destructive ones.

        Modifies steps in-place (sets requires_confirmation=True).
        Returns the same Plan object for chaining.
        """
        for step in plan.steps:
            if self._is_destructive(step):
                step.requires_confirmation = True
        return plan

    def validate_step(self, step: Step) -> bool:
        """
        Check a single step. Returns True if the step requires confirmation.
        Also sets step.requires_confirmation accordingly.
        """
        destructive = self._is_destructive(step)
        step.requires_confirmation = destructive
        return destructive

    def _is_destructive(self, step: Step) -> bool:
        """Determine if a step involves a destructive action."""
        # Already flagged by the planner
        if step.requires_confirmation:
            return True

        # Always-destructive capabilities
        if step.capability in _ALWAYS_DESTRUCTIVE:
            return True

        # Conditionally destructive capabilities
        if step.capability in _CONDITIONALLY_DESTRUCTIVE:
            if self._has_dangerous_args(step):
                return True
            # For organize/run — flag by default to be safe
            if step.capability in ("run_code", "organize_files"):
                return True

        # Check arguments for dangerous patterns
        if self._has_dangerous_args(step):
            return True

        # Check description for dangerous keywords
        if self._has_dangerous_description(step):
            return True

        return False

    def _has_dangerous_args(self, step: Step) -> bool:
        """Check if step arguments contain dangerous values."""
        for arg_key, dangerous_values in _DANGEROUS_ARG_PATTERNS.items():
            val = str(step.arguments.get(arg_key, "")).lower().strip()
            if val in dangerous_values:
                return True
            # Check if any dangerous keyword is a substring
            for dv in dangerous_values:
                if dv in val:
                    return True
        return False

    def _has_dangerous_description(self, step: Step) -> bool:
        """Check if step description suggests destructive intent."""
        desc_lower = step.description.lower()
        return any(kw in desc_lower for kw in _DANGEROUS_DESCRIPTION_KEYWORDS)

    def get_confirmation_message(self, step: Step) -> str:
        """Generate a natural confirmation prompt for the user."""
        desc = step.description
        capability = step.capability

        if capability == "delete_files":
            path = step.arguments.get("path", "the specified files")
            return f"I need to delete {path}. Shall I proceed?"

        if capability == "move_files":
            src = step.arguments.get("path", "files")
            dst = step.arguments.get("destination", "another location")
            return f"I'll move {src} to {dst}. Is that alright?"

        if capability == "rename_files":
            name = step.arguments.get("new_name", "")
            return f"I'll rename to '{name}'. Confirm?"

        if capability == "write_file":
            path = step.arguments.get("path", "a file")
            return f"I'll write to {path}, which may overwrite existing content. Proceed?"

        if capability in ("restart_system", "shutdown_system"):
            action = "restart" if "restart" in capability else "shut down"
            return f"This will {action} the computer. Are you sure?"

        if capability == "run_code":
            return f"I'll execute code: {desc}. Shall I go ahead?"

        # Generic fallback
        return f"Step requires confirmation: {desc}. Proceed?"
