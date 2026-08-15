"""
actions/battery.py — Battery information for CARL.

Reports: percentage, charging status, estimated time, power mode.
"""
from __future__ import annotations
import psutil
import platform

_OS = platform.system()


def battery_status(parameters: dict = None, player=None) -> str:
    """Get comprehensive battery information."""
    bat = psutil.sensors_battery()
    if bat is None:
        return "No battery detected — this appears to be a desktop computer."

    pct = bat.percent
    plugged = bat.power_plugged
    secs_left = bat.secsleft

    parts = [f"Battery: {pct}%"]

    if plugged:
        parts.append("Charging")
    else:
        parts.append("Discharging")
        if secs_left and secs_left > 0 and secs_left != psutil.POWER_TIME_UNLIMITED:
            hours = secs_left // 3600
            mins = (secs_left % 3600) // 60
            parts.append(f"Estimated {hours}h {mins}m remaining")

    # Power mode (Windows)
    if _OS == "Windows":
        try:
            import subprocess
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance -Namespace root/cimv2/power -ClassName Win32_PowerPlan | Where-Object IsActive).ElementName"],
                capture_output=True, text=True, timeout=5
            )
            plan = r.stdout.strip()
            if plan:
                parts.append(f"Power plan: {plan}")
        except Exception:
            pass

    # Health estimate
    if pct <= 20 and not plugged:
        parts.append("[!] Low battery — consider charging soon.")

    return " | ".join(parts)
