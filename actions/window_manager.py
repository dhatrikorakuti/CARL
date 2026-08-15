"""
actions/window_manager.py — Window Management for CARL.

Maximize, minimize, close, snap, arrange, virtual desktops.
"""
from __future__ import annotations
import platform, subprocess
import pyautogui

_OS = platform.system()


def window_manager(parameters: dict, player=None) -> str:
    action = parameters.get("action", "").lower()
    target = parameters.get("window", "")

    if action == "maximize":
        pyautogui.hotkey("win", "up")
        return "Window maximized."
    elif action == "minimize":
        pyautogui.hotkey("win", "down")
        return "Window minimized."
    elif action == "close":
        pyautogui.hotkey("alt", "F4")
        return "Window closed."
    elif action == "snap_left":
        pyautogui.hotkey("win", "left")
        return "Window snapped left."
    elif action == "snap_right":
        pyautogui.hotkey("win", "right")
        return "Window snapped right."
    elif action == "switch":
        pyautogui.hotkey("alt", "tab")
        return "Switched window."
    elif action == "minimize_all":
        pyautogui.hotkey("win", "d")
        return "All windows minimized."
    elif action == "new_desktop":
        pyautogui.hotkey("win", "ctrl", "d")
        return "New virtual desktop created."
    elif action == "close_desktop":
        pyautogui.hotkey("win", "ctrl", "F4")
        return "Virtual desktop closed."
    elif action == "next_desktop":
        pyautogui.hotkey("win", "ctrl", "right")
        return "Switched to next desktop."
    elif action == "prev_desktop":
        pyautogui.hotkey("win", "ctrl", "left")
        return "Switched to previous desktop."
    elif action == "task_view":
        pyautogui.hotkey("win", "tab")
        return "Task view opened."
    elif action == "list":
        return _list_windows()
    else:
        return f"Unknown window action: {action}"


def _list_windows() -> str:
    try:
        import pygetwindow as gw
        windows = [w.title for w in gw.getAllWindows() if w.title.strip() and w.visible]
        return f"Open windows ({len(windows)}): " + ", ".join(windows[:15])
    except ImportError:
        if _OS == "Windows":
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-Process | Where-Object {$_.MainWindowTitle} | Select-Object -ExpandProperty MainWindowTitle"],
                             capture_output=True, text=True, timeout=5)
            return r.stdout.strip() or "Could not list windows."
    except Exception as e:
        return f"Window listing failed: {e}"
    return "Window listing unavailable."
