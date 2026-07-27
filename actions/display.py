"""
actions/display.py — Display control for CARL.

Brightness, Night Light, resolution, rotation, monitor detection.
Windows-only (uses WMI + PowerShell where needed).
"""
from __future__ import annotations
import platform
import subprocess

_OS = platform.system()


def display_control(parameters: dict, player=None) -> str:
    action = parameters.get("action", "").lower()
    value = parameters.get("value", "")

    if action == "brightness" or action == "set_brightness":
        return _set_brightness(int(value) if value else 50)
    elif action == "increase_brightness":
        return _adjust_brightness(10)
    elif action == "decrease_brightness":
        return _adjust_brightness(-10)
    elif action == "night_light_on":
        return _night_light(True)
    elif action == "night_light_off":
        return _night_light(False)
    elif action == "get_brightness":
        return _get_brightness()
    elif action == "monitors":
        return _list_monitors()
    elif action == "resolution":
        return f"Resolution change not implemented yet. Current action: {action}"
    else:
        return f"Unknown display action: {action}"


def _set_brightness(level: int) -> str:
    level = max(0, min(100, level))
    if _OS == "Windows":
        try:
            cmd = f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{level})"
            subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                         capture_output=True, timeout=5)
            return f"Brightness set to {level}%."
        except Exception as e:
            return f"Failed to set brightness: {e}"
    return "Brightness control requires Windows."


def _adjust_brightness(delta: int) -> str:
    if _OS == "Windows":
        try:
            get_cmd = "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness).CurrentBrightness"
            r = subprocess.run(["powershell", "-NoProfile", "-Command", get_cmd],
                             capture_output=True, text=True, timeout=5)
            current = int(r.stdout.strip()) if r.stdout.strip().isdigit() else 50
            new = max(0, min(100, current + delta))
            return _set_brightness(new)
        except Exception as e:
            return f"Failed to adjust brightness: {e}"
    return "Brightness control requires Windows."


def _get_brightness() -> str:
    if _OS == "Windows":
        try:
            cmd = "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness).CurrentBrightness"
            r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                             capture_output=True, text=True, timeout=5)
            return f"Current brightness: {r.stdout.strip()}%"
        except Exception as e:
            return f"Cannot read brightness: {e}"
    return "Brightness reading requires Windows."


def _night_light(enable: bool) -> str:
    if _OS == "Windows":
        try:
            # Night Light is controlled via registry
            state = "02,00,00,00" if enable else "01,00,00,00"
            action_word = "enabled" if enable else "disabled"
            # This uses the Settings app shortcut
            subprocess.run(["powershell", "-NoProfile", "-Command",
                          f"Start-Process ms-settings:nightlight"],
                         capture_output=True, timeout=5)
            return f"Night Light settings opened. Please toggle manually — registry method varies by Windows version."
        except Exception as e:
            return f"Night Light control failed: {e}"
    return "Night Light requires Windows."


def _list_monitors() -> str:
    if _OS == "Windows":
        try:
            cmd = "Get-CimInstance -Namespace root/wmi -ClassName WmiMonitorID | ForEach-Object { ($_.UserFriendlyName | Where-Object {$_ -ne 0} | ForEach-Object { [char]$_ }) -join '' }"
            r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                             capture_output=True, text=True, timeout=5)
            monitors = r.stdout.strip()
            return f"Connected monitors: {monitors}" if monitors else "1 monitor detected (name unavailable)."
        except Exception as e:
            return f"Monitor detection failed: {e}"
    return "Monitor detection requires Windows."
