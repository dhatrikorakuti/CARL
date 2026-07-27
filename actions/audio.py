"""
actions/audio.py — Audio control for CARL.

Volume, mute, output device switching, connected devices.
"""
from __future__ import annotations
import platform
import subprocess

_OS = platform.system()


def audio_control(parameters: dict, player=None) -> str:
    action = parameters.get("action", "").lower()
    value = parameters.get("value", "")

    if action == "set_volume":
        return _set_volume(int(value) if value else 50)
    elif action == "increase_volume":
        return _adjust_volume(10)
    elif action == "decrease_volume":
        return _adjust_volume(-10)
    elif action == "mute":
        return _mute(True)
    elif action == "unmute":
        return _mute(False)
    elif action == "get_volume":
        return _get_volume()
    elif action == "list_devices":
        return _list_audio_devices()
    elif action == "switch_output":
        return _switch_output(value)
    else:
        return f"Unknown audio action: {action}"


def _set_volume(level: int) -> str:
    level = max(0, min(100, level))
    if _OS == "Windows":
        try:
            # Use nircmd or PowerShell COM
            ps = f"""
$obj = New-Object -ComObject WScript.Shell
$current = 0
1..50 | ForEach-Object {{ $obj.SendKeys([char]174) }}
$steps = [math]::Round({level} / 2)
1..$steps | ForEach-Object {{ $obj.SendKeys([char]175) }}
"""
            # Simpler approach using pycaw if available
            try:
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                volume.SetMasterVolumeLevelScalar(level / 100.0, None)
                return f"Volume set to {level}%."
            except ImportError:
                pass

            # Fallback: nircmd
            subprocess.run(["nircmd", "setsysvolume", str(int(level * 655.35))],
                         capture_output=True, timeout=5)
            return f"Volume set to {level}%."
        except Exception as e:
            return f"Failed to set volume: {e}"
    return "Volume control requires Windows."


def _adjust_volume(delta: int) -> str:
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        current = volume.GetMasterVolumeLevelScalar()
        new_level = max(0.0, min(1.0, current + delta / 100.0))
        volume.SetMasterVolumeLevelScalar(new_level, None)
        return f"Volume {'increased' if delta > 0 else 'decreased'} to {int(new_level * 100)}%."
    except ImportError:
        # Fallback: use key simulation
        import pyautogui
        if delta > 0:
            for _ in range(abs(delta) // 2):
                pyautogui.press("volumeup")
            return f"Volume increased."
        else:
            for _ in range(abs(delta) // 2):
                pyautogui.press("volumedown")
            return f"Volume decreased."
    except Exception as e:
        return f"Volume adjustment failed: {e}"


def _mute(mute: bool) -> str:
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMute(1 if mute else 0, None)
        return "Audio muted." if mute else "Audio unmuted."
    except ImportError:
        import pyautogui
        pyautogui.press("volumemute")
        return "Toggled mute."
    except Exception as e:
        return f"Mute failed: {e}"


def _get_volume() -> str:
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        level = volume.GetMasterVolumeLevelScalar()
        muted = volume.GetMute()
        return f"Volume: {int(level*100)}%{' (muted)' if muted else ''}."
    except Exception:
        return "Cannot read volume level."


def _list_audio_devices() -> str:
    try:
        from pycaw.pycaw import AudioUtilities
        devices = AudioUtilities.GetAllDevices()
        names = [d.FriendlyName for d in devices if d.FriendlyName]
        return f"Audio devices: {', '.join(names[:10])}" if names else "No audio devices found."
    except ImportError:
        if _OS == "Windows":
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-AudioDevice -List | Select-Object -ExpandProperty Name"],
                             capture_output=True, text=True, timeout=5)
            return r.stdout.strip() or "Could not list audio devices."
    except Exception as e:
        return f"Audio device listing failed: {e}"
    return "Audio device listing requires pycaw or Windows."


def _switch_output(device_name: str) -> str:
    return f"Audio output switching to '{device_name}' requires the AudioDeviceCmdlets PowerShell module. Install with: Install-Module -Name AudioDeviceCmdlets"
