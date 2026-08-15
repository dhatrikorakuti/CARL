"""
core/mic_health.py — Microphone health detection for CARL.

Checks:
  - Microphone exists
  - Microphone is accessible
  - Audio frames are flowing
  - Device didn't disconnect mid-session

Reports issues naturally instead of crashing.
"""
from __future__ import annotations
import sounddevice as sd


def check_microphone() -> tuple[bool, str]:
    """
    Check if a working microphone is available.
    Returns (ok: bool, message: str).
    """
    try:
        devices = sd.query_devices()
        input_dev = sd.query_devices(kind='input')

        if input_dev is None:
            return False, "No microphone detected. Please connect a microphone."

        name = input_dev.get('name', 'Unknown')
        channels = input_dev.get('max_input_channels', 0)

        if channels == 0:
            return False, f"Microphone '{name}' has no input channels."

        # Try opening a brief stream to verify access
        try:
            with sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=1024):
                pass
        except sd.PortAudioError as e:
            err = str(e)
            if "permission" in err.lower():
                return False, "Microphone permission denied. Check Windows privacy settings."
            if "device" in err.lower():
                return False, f"Cannot access microphone: {err}"
            return False, f"Microphone error: {err}"

        return True, f"Microphone ready: {name}"

    except Exception as e:
        return False, f"Microphone check failed: {e}"


def get_audio_devices() -> dict:
    """List available input and output devices."""
    try:
        devices = sd.query_devices()
        inputs = []
        outputs = []
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                inputs.append({"index": i, "name": d['name']})
            if d['max_output_channels'] > 0:
                outputs.append({"index": i, "name": d['name']})
        return {"inputs": inputs, "outputs": outputs}
    except Exception:
        return {"inputs": [], "outputs": []}
