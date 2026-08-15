"""
actions/network.py — Network control for CARL.

WiFi on/off, Bluetooth, IP info, speed test, connectivity check.
"""
from __future__ import annotations
import platform
import subprocess
import socket

_OS = platform.system()


def network_control(parameters: dict, player=None) -> str:
    action = parameters.get("action", "").lower()
    target = parameters.get("target", "")

    if action == "wifi_on":
        return _wifi(True)
    elif action == "wifi_off":
        return _wifi(False)
    elif action == "bluetooth_on":
        return _bluetooth(True)
    elif action == "bluetooth_off":
        return _bluetooth(False)
    elif action == "ip":
        return _get_ip()
    elif action == "speed_test":
        return _speed_test()
    elif action == "ping":
        return _ping(target or "google.com")
    elif action == "status":
        return _network_status()
    elif action == "diagnose":
        return _diagnose()
    else:
        return _network_status()


def _wifi(enable: bool) -> str:
    if _OS == "Windows":
        state = "enable" if enable else "disable"
        try:
            subprocess.run(["netsh", "interface", "set", "interface", "Wi-Fi", state],
                         capture_output=True, timeout=5)
            return f"Wi-Fi {'enabled' if enable else 'disabled'}."
        except Exception as e:
            return f"Wi-Fi control failed: {e}"
    return "Wi-Fi control requires Windows."


def _bluetooth(enable: bool) -> str:
    if _OS == "Windows":
        action_word = "enabled" if enable else "disabled"
        try:
            # Open Bluetooth settings (direct toggle requires device-specific APIs)
            subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Start-Process ms-settings:bluetooth"], capture_output=True, timeout=5)
            return f"Bluetooth settings opened. Toggle manually — direct API varies by adapter."
        except Exception as e:
            return f"Bluetooth control failed: {e}"
    return "Bluetooth control requires Windows."


def _get_ip() -> str:
    try:
        # Local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()

        # Public IP
        import urllib.request
        public_ip = urllib.request.urlopen("https://api.ipify.org", timeout=5).read().decode()

        return f"Local IP: {local_ip} | Public IP: {public_ip}"
    except Exception as e:
        return f"IP lookup failed: {e}"


def _speed_test() -> str:
    """Basic speed test using a small download."""
    import time
    import urllib.request
    url = "https://speed.cloudflare.com/__down?bytes=1000000"  # 1MB
    try:
        start = time.time()
        urllib.request.urlopen(url, timeout=15).read()
        duration = time.time() - start
        mbps = (1.0 / duration) * 8  # 1MB * 8 bits / seconds
        return f"Download speed: approximately {mbps:.1f} Mbps (tested with 1MB file)."
    except Exception as e:
        return f"Speed test failed: {e}"


def _ping(host: str) -> str:
    try:
        flag = "-n" if _OS == "Windows" else "-c"
        r = subprocess.run(["ping", flag, "3", host],
                         capture_output=True, text=True, timeout=10)
        # Extract average time
        output = r.stdout
        if "Average" in output:
            avg = output.split("Average = ")[-1].strip()
            return f"Ping to {host}: {avg}"
        elif "avg" in output:
            # Linux format
            stats = output.split("/")[-3] if "/" in output else ""
            return f"Ping to {host}: {stats}ms average" if stats else f"Ping to {host} successful."
        return f"Ping to {host}: {'successful' if r.returncode == 0 else 'failed'}."
    except Exception as e:
        return f"Ping failed: {e}"


def _network_status() -> str:
    import psutil
    parts = []
    # Check connectivity
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        parts.append("Internet: Connected")
    except OSError:
        parts.append("Internet: Disconnected")

    # Network interfaces
    stats = psutil.net_if_stats()
    for name, stat in stats.items():
        if stat.isup and "Loopback" not in name:
            speed = f"{stat.speed}Mbps" if stat.speed else ""
            parts.append(f"{name}: UP {speed}")

    return " | ".join(parts) if parts else "No network information available."


def _diagnose() -> str:
    """Run basic network diagnostics."""
    results = []
    # DNS
    try:
        socket.gethostbyname("google.com")
        results.append("DNS: OK")
    except Exception:
        results.append("DNS: FAILED")

    # Gateway ping
    try:
        import psutil
        gateways = psutil.net_if_addrs()
        results.append(_ping("8.8.8.8"))
    except Exception:
        results.append("Gateway: Unknown")

    return " | ".join(results)
