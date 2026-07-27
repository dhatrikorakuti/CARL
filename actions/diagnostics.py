"""
actions/diagnostics.py — Full System Diagnostics for CARL.

Comprehensive system health report: CPU, RAM, GPU, disk, network, devices.
"""
from __future__ import annotations
import platform, socket, subprocess, time
import psutil

_OS = platform.system()


def system_diagnostics(parameters: dict, player=None) -> str:
    action = parameters.get("action", "full").lower()

    if action == "full" or action == "health":
        return _full_report()
    elif action == "disk":
        return _disk_info()
    elif action == "processes":
        return _top_processes()
    elif action == "devices":
        return _devices()
    elif action == "os":
        return _os_info()
    else:
        return _full_report()


def _full_report() -> str:
    parts = []

    # CPU
    cpu = psutil.cpu_percent(interval=0.5)
    cpu_count = psutil.cpu_count()
    parts.append(f"CPU: {cpu}% ({cpu_count} cores)")

    # RAM
    mem = psutil.virtual_memory()
    parts.append(f"RAM: {mem.percent}% ({mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB)")

    # Disk
    disk = psutil.disk_usage('/')
    parts.append(f"Disk: {disk.percent}% ({disk.free/1024**3:.0f} GB free)")

    # Temperature
    try:
        temps = psutil.sensors_temperatures()
        for name in ["coretemp", "k10temp", "cpu_thermal"]:
            if name in temps and temps[name]:
                parts.append(f"Temp: {temps[name][0].current:.0f}°C")
                break
    except Exception:
        pass

    # Battery
    bat = psutil.sensors_battery()
    if bat:
        parts.append(f"Battery: {bat.percent}% {'⚡' if bat.power_plugged else '🔋'}")

    # Network
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        parts.append("Network: Connected")
    except OSError:
        parts.append("Network: Disconnected")

    # Uptime
    boot = psutil.boot_time()
    uptime = time.time() - boot
    h, m = int(uptime // 3600), int((uptime % 3600) // 60)
    parts.append(f"Uptime: {h}h {m}m")

    # Processes
    parts.append(f"Processes: {len(psutil.pids())}")

    return " | ".join(parts)


def _disk_info() -> str:
    parts = []
    for part in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(part.mountpoint)
            parts.append(f"{part.device}: {usage.free/1024**3:.0f}GB free / {usage.total/1024**3:.0f}GB ({usage.percent}% used)")
        except Exception:
            pass
    return "\n".join(parts) if parts else "No disk info available."


def _top_processes() -> str:
    procs = []
    for p in psutil.process_iter(['name', 'cpu_percent', 'memory_percent']):
        try:
            procs.append((p.info['name'], p.info['cpu_percent'] or 0, p.info['memory_percent'] or 0))
        except Exception:
            pass
    procs.sort(key=lambda x: x[1] + x[2], reverse=True)
    lines = [f"{name}: CPU {cpu:.0f}% RAM {mem:.1f}%" for name, cpu, mem in procs[:10]]
    return "Top processes:\n" + "\n".join(lines)


def _devices() -> str:
    parts = []
    # USB (Windows)
    if _OS == "Windows":
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-PnpDevice -Status OK -Class USB | Select-Object -First 10 -ExpandProperty FriendlyName"],
                             capture_output=True, text=True, timeout=5)
            if r.stdout.strip():
                parts.append(f"USB: {r.stdout.strip()}")
        except Exception:
            pass
    # Audio
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-PnpDevice -Status OK -Class AudioEndpoint | Select-Object -ExpandProperty FriendlyName"],
                         capture_output=True, text=True, timeout=5)
        if r.stdout.strip():
            parts.append(f"Audio: {r.stdout.strip()}")
    except Exception:
        pass
    return "\n".join(parts) if parts else "Device listing unavailable."


def _os_info() -> str:
    return (f"OS: {platform.system()} {platform.release()} "
            f"({platform.machine()}) | Python {platform.python_version()}")
