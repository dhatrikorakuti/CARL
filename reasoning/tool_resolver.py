"""
reasoning/tool_resolver.py — Maps capability names to existing action modules.

The Planner outputs abstract capabilities (e.g. "list_files", "open_url").
The Tool Resolver translates these into concrete tool calls against the
existing action modules without modifying them.

This indirection keeps the planner independent of implementation details
and allows action modules to evolve without breaking existing plans.
"""
from __future__ import annotations

from typing import Any, Callable


# ── Capability → (tool_function_module, action_name, argument_mapper) ────────
#
# Each entry maps:
#   capability_name → {
#       "module":  import path to the action function
#       "func":    function name to call
#       "map_args": optional callable that transforms planner args → tool args
#   }
#
# The executor imports and calls these at runtime.

def _identity(args: dict) -> dict:
    """Pass arguments through unchanged."""
    return args


def _file_action(action: str):
    """Create an argument mapper that injects a file_controller action."""
    def mapper(args: dict) -> dict:
        return {"action": action, **args}
    return mapper


def _browser_action(action: str):
    """Create an argument mapper that injects a browser_control action."""
    def mapper(args: dict) -> dict:
        return {"action": action, **args}
    return mapper


def _desktop_action(action: str):
    """Create an argument mapper that injects a desktop_control action."""
    def mapper(args: dict) -> dict:
        return {"action": action, **args}
    return mapper


def _computer_settings_action(action: str):
    """Create an argument mapper for computer_settings."""
    def mapper(args: dict) -> dict:
        return {"action": action, **args}
    return mapper


def _computer_control_action(action: str):
    """Create an argument mapper for computer_control."""
    def mapper(args: dict) -> dict:
        return {"action": action, **args}
    return mapper


def _code_action(action: str):
    """Create an argument mapper for code_helper."""
    def mapper(args: dict) -> dict:
        return {"action": action, **args}
    return mapper


# ── Capability Registry ──────────────────────────────────────────────────────

CAPABILITY_MAP: dict[str, dict[str, Any]] = {
    # ── File operations ──────────────────────────────────────────────────────
    "list_files": {
        "module": "actions.file_controller",
        "func": "file_controller",
        "map_args": _file_action("list"),
    },
    "create_file": {
        "module": "actions.file_controller",
        "func": "file_controller",
        "map_args": _file_action("create_file"),
    },
    "create_folder": {
        "module": "actions.file_controller",
        "func": "file_controller",
        "map_args": _file_action("create_folder"),
    },
    "delete_files": {
        "module": "actions.file_controller",
        "func": "file_controller",
        "map_args": _file_action("delete"),
    },
    "move_files": {
        "module": "actions.file_controller",
        "func": "file_controller",
        "map_args": _file_action("move"),
    },
    "copy_files": {
        "module": "actions.file_controller",
        "func": "file_controller",
        "map_args": _file_action("copy"),
    },
    "rename_files": {
        "module": "actions.file_controller",
        "func": "file_controller",
        "map_args": _file_action("rename"),
    },
    "read_file": {
        "module": "actions.file_controller",
        "func": "file_controller",
        "map_args": _file_action("read"),
    },
    "write_file": {
        "module": "actions.file_controller",
        "func": "file_controller",
        "map_args": _file_action("write"),
    },
    "find_files": {
        "module": "actions.file_controller",
        "func": "file_controller",
        "map_args": _file_action("find"),
    },
    "file_info": {
        "module": "actions.file_controller",
        "func": "file_controller",
        "map_args": _file_action("info"),
    },
    "disk_usage": {
        "module": "actions.file_controller",
        "func": "file_controller",
        "map_args": _file_action("disk_usage"),
    },
    "organize_files": {
        "module": "actions.file_controller",
        "func": "file_controller",
        "map_args": _file_action("organize_desktop"),
    },

    # ── Desktop ──────────────────────────────────────────────────────────────
    "set_wallpaper": {
        "module": "actions.desktop",
        "func": "desktop_control",
        "map_args": _desktop_action("wallpaper"),
    },
    "organize_desktop": {
        "module": "actions.desktop",
        "func": "desktop_control",
        "map_args": _desktop_action("organize"),
    },
    "desktop_stats": {
        "module": "actions.desktop",
        "func": "desktop_control",
        "map_args": _desktop_action("stats"),
    },

    # ── Browser ──────────────────────────────────────────────────────────────
    "open_url": {
        "module": "actions.browser_control",
        "func": "browser_control",
        "map_args": _browser_action("go_to"),
    },
    "web_search": {
        "module": "actions.web_search",
        "func": "web_search",
        "map_args": _identity,
    },
    "click_element": {
        "module": "actions.browser_control",
        "func": "browser_control",
        "map_args": _browser_action("smart_click"),
    },
    "type_text_browser": {
        "module": "actions.browser_control",
        "func": "browser_control",
        "map_args": _browser_action("smart_type"),
    },
    "browser_screenshot": {
        "module": "actions.browser_control",
        "func": "browser_control",
        "map_args": _browser_action("screenshot"),
    },
    "close_browser": {
        "module": "actions.browser_control",
        "func": "browser_control",
        "map_args": _browser_action("close"),
    },

    # ── Applications ─────────────────────────────────────────────────────────
    "open_application": {
        "module": "actions.open_app",
        "func": "open_app",
        "map_args": lambda args: {"app_name": args.get("app_name", args.get("name", ""))},
    },
    "close_application": {
        "module": "actions.computer_settings",
        "func": "computer_settings",
        "map_args": lambda args: {"action": "close", "description": f"Close {args.get('name', '')}"},
    },

    # ── System ───────────────────────────────────────────────────────────────
    "adjust_volume": {
        "module": "actions.computer_settings",
        "func": "computer_settings",
        "map_args": lambda args: {"action": "volume", "value": str(args.get("level", "50"))},
    },
    "adjust_brightness": {
        "module": "actions.computer_settings",
        "func": "computer_settings",
        "map_args": lambda args: {"action": "brightness", "value": str(args.get("level", "50"))},
    },
    "system_status": {
        "module": "actions.system_monitor",
        "func": "get_system_status",
        "map_args": lambda args: {},
    },
    "lock_screen": {
        "module": "actions.computer_settings",
        "func": "computer_settings",
        "map_args": _computer_settings_action("lock"),
    },
    "restart_system": {
        "module": "actions.computer_settings",
        "func": "computer_settings",
        "map_args": _computer_settings_action("restart"),
    },
    "shutdown_system": {
        "module": "actions.computer_settings",
        "func": "computer_settings",
        "map_args": _computer_settings_action("shutdown"),
    },

    # ── Communication ────────────────────────────────────────────────────────
    "send_message": {
        "module": "actions.send_message",
        "func": "send_message",
        "map_args": _identity,
    },
    "set_reminder": {
        "module": "actions.reminder",
        "func": "reminder",
        "map_args": _identity,
    },

    # ── Media ────────────────────────────────────────────────────────────────
    "play_video": {
        "module": "actions.youtube_video",
        "func": "youtube_video",
        "map_args": lambda args: {"action": "play", **args},
    },
    "search_video": {
        "module": "actions.youtube_video",
        "func": "youtube_video",
        "map_args": lambda args: {"action": "play", "query": args.get("query", "")},
    },

    # ── Code ─────────────────────────────────────────────────────────────────
    "write_code": {
        "module": "actions.code_helper",
        "func": "code_helper",
        "map_args": _code_action("write"),
    },
    "run_code": {
        "module": "actions.code_helper",
        "func": "code_helper",
        "map_args": _code_action("run"),
    },
    "edit_code": {
        "module": "actions.code_helper",
        "func": "code_helper",
        "map_args": _code_action("edit"),
    },
    "explain_code": {
        "module": "actions.code_helper",
        "func": "code_helper",
        "map_args": _code_action("explain"),
    },

    # ── Computer control ─────────────────────────────────────────────────────
    "type_text": {
        "module": "actions.computer_control",
        "func": "computer_control",
        "map_args": _computer_control_action("type"),
    },
    "click_screen": {
        "module": "actions.computer_control",
        "func": "computer_control",
        "map_args": _computer_control_action("click"),
    },
    "press_key": {
        "module": "actions.computer_control",
        "func": "computer_control",
        "map_args": _computer_control_action("press"),
    },
    "hotkey": {
        "module": "actions.computer_control",
        "func": "computer_control",
        "map_args": _computer_control_action("hotkey"),
    },
    "scroll": {
        "module": "actions.computer_control",
        "func": "computer_control",
        "map_args": _computer_control_action("scroll"),
    },
    "screenshot": {
        "module": "actions.computer_control",
        "func": "computer_control",
        "map_args": _computer_control_action("screenshot"),
    },
}


# ── Resolver class ───────────────────────────────────────────────────────────

class ToolResolver:
    """
    Resolves abstract capabilities into concrete tool calls.

    Usage:
        resolver = ToolResolver()
        tool_func, mapped_args = resolver.resolve("list_files", {"path": "downloads"})
        result = tool_func(parameters=mapped_args, player=ui)
    """

    def __init__(self):
        self._cache: dict[str, Callable] = {}

    def can_resolve(self, capability: str) -> bool:
        """Check if a capability has a known mapping."""
        return capability in CAPABILITY_MAP

    def resolve(self, capability: str, arguments: dict) -> tuple[Callable, dict]:
        """
        Resolve a capability into (callable, mapped_arguments).

        Args:
            capability: The capability name from the plan step.
            arguments: Raw arguments from the plan step.

        Returns:
            Tuple of (tool_function, transformed_arguments)

        Raises:
            ResolutionError: If capability is unknown or import fails.
        """
        if capability not in CAPABILITY_MAP:
            raise ResolutionError(f"Unknown capability: '{capability}'")

        entry = CAPABILITY_MAP[capability]
        module_path = entry["module"]
        func_name = entry["func"]
        map_args = entry["map_args"]

        # Import and cache the function
        cache_key = f"{module_path}.{func_name}"
        if cache_key not in self._cache:
            try:
                import importlib
                mod = importlib.import_module(module_path)
                func = getattr(mod, func_name)
                self._cache[cache_key] = func
            except (ImportError, AttributeError) as e:
                raise ResolutionError(
                    f"Cannot resolve capability '{capability}': {e}"
                )

        func = self._cache[cache_key]
        mapped = map_args(arguments)

        return func, mapped

    def get_tool_name(self, capability: str) -> str:
        """Return the underlying tool/module name for logging."""
        if capability in CAPABILITY_MAP:
            return CAPABILITY_MAP[capability]["func"]
        return capability


# ── Exceptions ───────────────────────────────────────────────────────────────

class ResolutionError(Exception):
    """Raised when a capability cannot be resolved to a tool."""
    pass
