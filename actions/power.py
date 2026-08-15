"""
actions/power.py — Power Management for CARL.

Shutdown, restart, sleep, hibernate, lock, sign out.
All destructive actions require confirmation from the caller.
"""
from __future__ import annotations
import platform, subprocess, os

_OS = platform.system()


def power_control(parameters: dict, player=None) -> str:
    action = parameters.get("action", "").lower()
    confirmed = parameters.get("confirmed", False)

    # Destructive actions — require confirmation
    if action in ("shutdown", "restart", "hibernate", "sign_out") and not confirmed:
        return f"CONFIRMATION_REQUIRED: Are you sure you want to {action} the computer?"

    if action == "shutdown":
        if _OS == "Windows":
            subprocess.Popen(["shutdown", "/s", "/t", "5"])
            return "Shutting down in 5 seconds."
        os.system("shutdown -h now")
        return "Shutting down."

    elif action == "restart":
        if _OS == "Windows":
            subprocess.Popen(["shutdown", "/r", "/t", "5"])
            return "Restarting in 5 seconds."
        os.system("shutdown -r now")
        return "Restarting."

    elif action == "sleep":
        if _OS == "Windows":
            subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Add-Type -Assembly System.Windows.Forms; [System.Windows.Forms.Application]::SetSuspendState('Suspend', $false, $false)"],
                         capture_output=True, timeout=5)
            return "Computer entering sleep mode."
        os.system("systemctl suspend")
        return "Sleeping."

    elif action == "hibernate":
        if _OS == "Windows":
            subprocess.Popen(["shutdown", "/h"])
            return "Hibernating."
        os.system("systemctl hibernate")
        return "Hibernating."

    elif action == "lock":
        if _OS == "Windows":
            subprocess.Popen(["rundll32.exe", "user32.dll,LockWorkStation"])
            return "Computer locked."
        os.system("loginctl lock-session")
        return "Locked."

    elif action == "sign_out":
        if _OS == "Windows":
            subprocess.Popen(["shutdown", "/l"])
            return "Signing out."
        return "Sign out not available on this OS."

    elif action == "cancel_shutdown":
        if _OS == "Windows":
            subprocess.run(["shutdown", "/a"], capture_output=True)
            return "Shutdown cancelled."
        return "No pending shutdown to cancel."

    else:
        return f"Unknown power action: {action}"
