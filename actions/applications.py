"""
actions/applications.py — Application Management for CARL.

List running, force close, install, uninstall.
"""
from __future__ import annotations
import platform, subprocess
import psutil

_OS = platform.system()


def app_manager(parameters: dict, player=None) -> str:
    action = parameters.get("action", "").lower()
    name = parameters.get("name", "")
    confirmed = parameters.get("confirmed", False)

    if action == "list" or action == "running":
        return _list_running()
    elif action == "force_close" or action == "kill":
        return _force_close(name)
    elif action == "close":
        return _close_app(name)
    elif action == "install":
        return _install(name, confirmed)
    elif action == "uninstall":
        return _uninstall(name, confirmed)
    elif action == "restart":
        return _restart_app(name)
    else:
        return f"Unknown app action: {action}"


def _list_running() -> str:
    procs = []
    seen = set()
    for p in psutil.process_iter(['name', 'pid']):
        name = p.info['name']
        if name and name not in seen and not name.startswith("svchost"):
            seen.add(name)
            procs.append(name)
    procs.sort()
    top = procs[:20]
    return f"Running ({len(procs)} total): {', '.join(top)}"


def _force_close(name: str) -> str:
    if not name:
        return "Please specify which application to close."
    killed = 0
    for p in psutil.process_iter(['name', 'pid']):
        if name.lower() in (p.info['name'] or '').lower():
            try:
                p.kill()
                killed += 1
            except Exception:
                pass
    if killed:
        return f"Force closed {name} ({killed} process{'es' if killed > 1 else ''})."
    return f"No running process found matching '{name}'."


def _close_app(name: str) -> str:
    if not name:
        return "Please specify which application to close."
    if _OS == "Windows":
        r = subprocess.run(["taskkill", "/IM", f"{name}*", "/F"],
                         capture_output=True, text=True, timeout=10)
        if "SUCCESS" in r.stdout:
            return f"Closed {name}."
        return f"Could not close {name}: {r.stderr.strip()[:80]}"
    return _force_close(name)


def _restart_app(name: str) -> str:
    _force_close(name)
    import time
    time.sleep(1)
    try:
        if _OS == "Windows":
            subprocess.Popen(["start", name], shell=True)
        else:
            subprocess.Popen([name])
        return f"Restarted {name}."
    except Exception as e:
        return f"Closed {name} but couldn't relaunch: {e}"


def _install(name: str, confirmed: bool) -> str:
    if not confirmed:
        return f"CONFIRMATION_REQUIRED: Install '{name}'? This will download and run an installer."
    # Use winget on Windows
    if _OS == "Windows":
        try:
            r = subprocess.run(["winget", "install", name, "--accept-package-agreements", "--accept-source-agreements"],
                             capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                return f"Installed {name} successfully."
            return f"Installation issue: {r.stdout[-200:]}"
        except FileNotFoundError:
            return "winget not available. Please install Windows Package Manager."
        except Exception as e:
            return f"Install failed: {e}"
    return "Package installation requires winget (Windows)."


def _uninstall(name: str, confirmed: bool) -> str:
    if not confirmed:
        return f"CONFIRMATION_REQUIRED: Uninstall '{name}'? This cannot be undone."
    if _OS == "Windows":
        try:
            r = subprocess.run(["winget", "uninstall", name],
                             capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                return f"Uninstalled {name}."
            return f"Uninstall issue: {r.stdout[-200:]}"
        except Exception as e:
            return f"Uninstall failed: {e}"
    return "Package removal requires winget (Windows)."
