"""
actions/screenshots.py — Screenshot & Screen Recording for CARL.
"""
from __future__ import annotations
import time, subprocess, platform
from pathlib import Path

_OS = platform.system()
_SAVE_DIR = Path.home() / "Pictures" / "CARL Screenshots"


def screenshot_control(parameters: dict, player=None) -> str:
    action = parameters.get("action", "full").lower()
    _SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    if action in ("full", "screenshot", "capture"):
        return _capture_full(ts)
    elif action in ("window", "active"):
        return _capture_window(ts)
    elif action in ("region", "area", "select"):
        return _capture_region(ts)
    elif action == "record_start":
        return _start_recording()
    elif action == "record_stop":
        return _stop_recording()
    else:
        return _capture_full(ts)


def _capture_full(ts: str) -> str:
    path = _SAVE_DIR / f"screenshot_{ts}.png"
    try:
        import pyautogui
        img = pyautogui.screenshot()
        img.save(str(path))
        return f"Screenshot saved: {path.name}"
    except ImportError:
        if _OS == "Windows":
            subprocess.run(["powershell", "-NoProfile", "-Command",
                          f"Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Screen]::PrimaryScreen | Out-Null; "
                          f"$b = New-Object System.Drawing.Bitmap([System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Width, [System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Height); "
                          f"$g = [System.Drawing.Graphics]::FromImage($b); "
                          f"$g.CopyFromScreen(0,0,0,0,$b.Size); $b.Save('{path}')"],
                         capture_output=True, timeout=10)
            return f"Screenshot saved: {path.name}" if path.exists() else "Screenshot failed."
    except Exception as e:
        return f"Screenshot failed: {e}"


def _capture_window(ts: str) -> str:
    path = _SAVE_DIR / f"window_{ts}.png"
    try:
        import pyautogui
        import pygetwindow as gw
        win = gw.getActiveWindow()
        if win:
            img = pyautogui.screenshot(region=(win.left, win.top, win.width, win.height))
            img.save(str(path))
            return f"Window screenshot saved: {path.name}"
        return "No active window detected."
    except Exception as e:
        return f"Window capture failed: {e}"


def _capture_region(ts: str) -> str:
    # Use Windows Snipping Tool
    if _OS == "Windows":
        subprocess.Popen(["snippingtool", "/clip"], creationflags=subprocess.CREATE_NO_WINDOW)
        return "Snipping tool opened — select your region."
    return "Region capture not available on this OS."


def _start_recording() -> str:
    if _OS == "Windows":
        # Use Xbox Game Bar
        import pyautogui
        pyautogui.hotkey("win", "alt", "r")
        return "Screen recording started (Xbox Game Bar)."
    return "Screen recording requires Windows Xbox Game Bar."


def _stop_recording() -> str:
    if _OS == "Windows":
        import pyautogui
        pyautogui.hotkey("win", "alt", "r")
        return "Screen recording stopped. Check Videos/Captures folder."
    return "No active recording."
