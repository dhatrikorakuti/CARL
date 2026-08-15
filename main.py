import platform as _platform
import subprocess as _subprocess

# ── Nuclear: force CREATE_NO_WINDOW on EVERY subprocess call on Windows ───────
# This patches Popen itself, so no per-file flag is needed anywhere.
if _platform.system() == "Windows":
    _OrigPopen = _subprocess.Popen

    class _Popen(_OrigPopen):
        def __init__(self, args, **kw):
            kw["creationflags"] = kw.get("creationflags", 0) | _subprocess.CREATE_NO_WINDOW
            kw.pop("startupinfo", None)   # drop any stale/shared STARTUPINFO
            super().__init__(args, **kw)

    _subprocess.Popen = _Popen
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import re
import threading
import time
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import sounddevice as sd
from google import genai
from google.genai import types
from ui import JarvisUI
from memory.memory_manager import (
    load_memory, update_memory, format_memory_for_prompt,
)
from memory.memory_store import MemoryStore
from memory.session_memory import SessionMemory
from memory.context_engine import ContextEngine
from memory.summarizer import ConversationSummarizer
from memory.cognitive import CognitiveMemory

from actions.file_processor import file_processor
from actions.flight_finder     import flight_finder
from actions.open_app          import open_app
from actions.weather_report    import weather_action
from actions.send_message      import send_message
from actions.reminder          import reminder
from actions.computer_settings import computer_settings
from actions.screen_processor  import _capture_camera, _capture_screen
from actions.youtube_video     import youtube_video
from actions.desktop           import desktop_control
from actions.browser_control   import browser_control
from actions.file_controller   import file_controller
from actions.code_helper       import code_helper
from actions.dev_agent         import dev_agent
from actions.web_search        import web_search as web_search_action
from actions.computer_control  import computer_control
from actions.game_updater      import game_updater
from actions.system_monitor    import SystemMonitor, get_system_status
from actions.proactive         import ProactiveEngine
from actions.display           import display_control
from actions.audio             import audio_control
from actions.battery           import battery_status
from actions.network           import network_control
from actions.downloads         import download_file
from actions.file_intake       import file_intake
from actions.applications      import app_manager
from actions.power             import power_control
from actions.diagnostics       import system_diagnostics
from actions.screenshots       import screenshot_control
from actions.window_manager    import window_manager
from actions.data_cleaning     import data_cleaning
from actions.ml_preparation    import ml_preparation
from actions.smart_data_scientist import smart_data_scientist
from actions.ml_pipeline       import ml_pipeline
from reasoning.engine          import ReasoningEngine, FallbackResult
from reasoning.models          import ExecutionLevel
from core.emotional_intelligence import EmotionalIntelligence
from core.mic_health import check_microphone


def get_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


BASE_DIR        = get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
VOICE_CONFIG_PATH = BASE_DIR / "config" / "voice_config.json"
PROMPT_PATH     = BASE_DIR / "core" / "prompt.txt"
LIVE_MODEL          = "models/gemini-2.5-flash-native-audio-preview-12-2025"
CHANNELS            = 1
SEND_SAMPLE_RATE    = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE          = 1024

def _get_api_key() -> str:
    with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["gemini_api_key"]


def _load_voice_config() -> dict:
    """Load voice configuration from config/voice_config.json."""
    try:
        return json.loads(VOICE_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_gemini_voice() -> str:
    """Return the configured Gemini Live voice name, default 'Puck'."""
    cfg = _load_voice_config()
    return cfg.get("gemini_live", {}).get("voice_name", "Puck")


def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "You are CARL, a professional AI Operating System. "
            "Be concise, direct, and always use the provided tools to complete tasks. "
            "Never simulate or guess results — always call the appropriate tool. "
            "Default language is English."
        )

_CTRL_RE = re.compile(r"<ctrl\d+>", re.IGNORECASE)

def _clean_transcript(text: str) -> str:    
    text = _CTRL_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
    return text.strip()

TOOL_DECLARATIONS = [
    {
        "name": "open_app",
        "description": (
            "Opens any application on the computer. "
            "Use this whenever the user asks to open, launch, or start any app, "
            "website, or program. Always call this tool — never just say you opened it."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "app_name": {
                    "type": "STRING",
                    "description": "Exact name of the application (e.g. 'WhatsApp', 'Chrome', 'Spotify')"
                }
            },
            "required": ["app_name"]
        }
    },
    {
        "name": "web_search",
        "description": (
            "Searches the web. Use for ANY question about current facts, events, prices, "
            "or topics — always prefer this over guessing. "
            "Modes: 'search' (default), 'news' (latest headlines on a topic), "
            "'research' (deep comprehensive answer), 'price' (product cost lookup), "
            "'compare' (side-by-side comparison of items)."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query":  {"type": "STRING", "description": "Search query or topic"},
                "mode":   {"type": "STRING", "description": "search | news | research | price | compare"},
                "items":  {"type": "ARRAY",  "items": {"type": "STRING"}, "description": "Items to compare (compare mode)"},
                "aspect": {"type": "STRING", "description": "Comparison aspect: price | specs | reviews | features"},
            },
            "required": ["query"]
        }
    },
    {
        "name": "system_status",
        "description": (
            "Returns real-time system metrics: CPU usage, RAM, GPU load, CPU temperature, "
            "uptime, and process count. Use when the user asks about computer performance, "
            "temperature, memory, or resource usage."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        }
    },
    {
        "name": "weather_report",
        "description": "Gives the weather report to user",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "city": {"type": "STRING", "description": "City name"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "send_message",
        "description": "Sends a text message via WhatsApp, Telegram, or other messaging platform.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "receiver":     {"type": "STRING", "description": "Recipient contact name"},
                "message_text": {"type": "STRING", "description": "The message to send"},
                "platform":     {"type": "STRING", "description": "Platform: WhatsApp, Telegram, etc."}
            },
            "required": ["receiver", "message_text", "platform"]
        }
    },
    {
        "name": "reminder",
        "description": "Sets a timed reminder using Task Scheduler.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "date":    {"type": "STRING", "description": "Date in YYYY-MM-DD format"},
                "time":    {"type": "STRING", "description": "Time in HH:MM format (24h)"},
                "message": {"type": "STRING", "description": "Reminder message text"}
            },
            "required": ["date", "time", "message"]
        }
    },
    {
        "name": "youtube_video",
        "description": (
            "Controls YouTube. Use for: playing videos, summarizing a video's content, "
            "getting video info, or showing trending videos."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "play | summarize | get_info | trending (default: play)"},
                "query":  {"type": "STRING", "description": "Search query for play action"},
                "save":   {"type": "BOOLEAN", "description": "Save summary to Notepad (summarize only)"},
                "region": {"type": "STRING", "description": "Country code for trending e.g. TR, US"},
                "url":    {"type": "STRING", "description": "Video URL for get_info action"},
            },
            "required": []
        }
    },
    {
        "name": "computer_settings",
        "description": "Controls computer: volume, brightness, wifi, close apps, fullscreen, shortcuts, power.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "The action to perform"},
                "description": {"type": "STRING", "description": "Natural language description of what to do"},
                "value":       {"type": "STRING", "description": "Optional value: volume level, text to type, etc."}
            },
            "required": []
        }
    },
    {
        "name": "browser_control",
        "description": "Controls web browsers: open URLs, search, click, type, scroll, navigate, screenshot.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "go_to|search|click|type|scroll|smart_click|smart_type|press|new_tab|close_tab|screenshot|back|forward|reload|close"},
                "browser":     {"type": "STRING", "description": "chrome|edge|firefox|brave (omit=active)"},
                "url":         {"type": "STRING"},
                "query":       {"type": "STRING"},
                "text":        {"type": "STRING"},
                "description": {"type": "STRING", "description": "Element description for smart_click/smart_type"},
                "direction":   {"type": "STRING", "description": "up|down"},
                "key":         {"type": "STRING"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "file_controller",
        "description": "Manages files and folders: list, create, delete, move, copy, rename, read, write, find, disk usage.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "list | create_file | create_folder | delete | move | copy | rename | read | write | find | largest | disk_usage | organize_desktop | info"},
                "path":        {"type": "STRING", "description": "File/folder path or shortcut: desktop, downloads, documents, home"},
                "destination": {"type": "STRING", "description": "Destination path for move/copy"},
                "new_name":    {"type": "STRING", "description": "New name for rename"},
                "content":     {"type": "STRING", "description": "Content for create_file/write"},
                "name":        {"type": "STRING", "description": "File name to search for"},
                "extension":   {"type": "STRING", "description": "File extension to search (e.g. .pdf)"},
                "count":       {"type": "INTEGER", "description": "Number of results for largest"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "code_helper",
        "description": "Writes, edits, explains, runs, or builds code files.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "write | edit | explain | run | build | auto (default: auto)"},
                "description": {"type": "STRING", "description": "What the code should do or what change to make"},
                "language":    {"type": "STRING", "description": "Programming language (default: python)"},
                "output_path": {"type": "STRING", "description": "Where to save the file"},
                "file_path":   {"type": "STRING", "description": "Path to existing file for edit/explain/run/build"},
                "code":        {"type": "STRING", "description": "Raw code string for explain"},
                "args":        {"type": "STRING", "description": "CLI arguments for run/build"},
                "timeout":     {"type": "INTEGER", "description": "Execution timeout in seconds (default: 30)"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "computer_control",
        "description": "Direct computer control: type, click, hotkeys, scroll, move mouse, screenshots, find elements.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "type|click|double_click|right_click|hotkey|press|scroll|move|screenshot|wait|focus_window|screen_click"},
                "text":        {"type": "STRING"},
                "x":           {"type": "INTEGER"},
                "y":           {"type": "INTEGER"},
                "keys":        {"type": "STRING", "description": "Key combo e.g. ctrl+c"},
                "key":         {"type": "STRING"},
                "direction":   {"type": "STRING", "description": "up|down|left|right"},
                "description": {"type": "STRING", "description": "Element description for screen_click"},
                "title":       {"type": "STRING", "description": "Window title for focus_window"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "shutdown_jarvis",
        "description": "Shuts down CARL when user says goodbye or stop.",
        "parameters": {"type": "OBJECT", "properties": {}}
    },
    {
    "name": "file_processor",
    "description": "Processes uploaded files: images, PDFs, docs, CSV, code, audio, video, archives. Use when user has a file.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "file_path":   {"type": "STRING", "description": "Path to file (empty=current upload)"},
            "action":      {"type": "STRING", "description": "describe|ocr|resize|compress|convert|summarize|extract_text|analyze|stats|filter|sort|explain|review|fix|run|transcribe|trim|info|list|extract"},
            "instruction": {"type": "STRING", "description": "Free-form instruction"},
            "format":      {"type": "STRING", "description": "Target format (mp3/pdf/csv/png)"},
        },
        "required": []
    }
},
    {
        "name": "save_memory",
        "description": "Save a personal fact about the user to memory. Call silently when user reveals preferences, name, habits, projects.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {"type": "STRING", "description": "identity|preferences|projects|relationships|wishes|notes"},
                "key":   {"type": "STRING", "description": "Short snake_case key"},
                "value": {"type": "STRING", "description": "Value in English"},
            },
            "required": ["category", "key", "value"]
        }
    },
    {
        "name": "display_control",
        "description": "Controls display: brightness, night light, monitors.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "set_brightness|increase_brightness|decrease_brightness|night_light_on|night_light_off|get_brightness|monitors"},
                "value":  {"type": "STRING", "description": "Brightness percentage (0-100)"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "audio_control",
        "description": "Controls audio: volume, mute, output devices.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "set_volume|increase_volume|decrease_volume|mute|unmute|get_volume|list_devices"},
                "value":  {"type": "STRING", "description": "Volume percentage"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "battery_status",
        "description": "Reports battery percentage, charging status, time remaining, power mode.",
        "parameters": {"type": "OBJECT", "properties": {}}
    },
    {
        "name": "network_control",
        "description": "Network: WiFi on/off, Bluetooth, IP, speed test, ping, diagnose.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "wifi_on|wifi_off|bluetooth_on|bluetooth_off|ip|speed_test|ping|status|diagnose"},
                "target": {"type": "STRING", "description": "Host for ping"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "download_file",
        "description": "Downloads files, repos, images from URLs to Downloads folder.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "url":      {"type": "STRING", "description": "URL to download"},
                "filename": {"type": "STRING", "description": "Optional filename"},
                "action":   {"type": "STRING", "description": "download|github"},
            },
            "required": ["url"]
        }
    },
    {
        "name": "file_intake",
        "description": "Opens native file or folder picker when user wants to upload, analyze, or open a file/dataset/folder. Call this when user mentions uploading, analyzing, or opening files.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "mode":        {"type": "STRING", "description": "file|folder"},
                "file_type":   {"type": "STRING", "description": "dataset|image|pdf|document|code|video|audio|archive|excel|any"},
                "description": {"type": "STRING", "description": "What the user asked for"},
            },
            "required": ["mode"]
        }
    },
    {
        "name": "app_manager",
        "description": "Manage applications: list running, force close, install, uninstall, restart apps.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "list|close|force_close|install|uninstall|restart"},
                "name":   {"type": "STRING", "description": "Application name"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "power_control",
        "description": "Power: shutdown, restart, sleep, hibernate, lock, sign out. Confirms destructive actions.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "shutdown|restart|sleep|hibernate|lock|sign_out|cancel_shutdown"},
                "confirmed": {"type": "BOOLEAN", "description": "User confirmed the action"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "system_diagnostics",
        "description": "Full system health report: CPU, RAM, disk, battery, network, processes, devices, OS info.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "full|disk|processes|devices|os"},
            },
            "required": []
        }
    },
    {
        "name": "screenshot_control",
        "description": "Screenshots and screen recording: capture full screen, active window, region, start/stop recording.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "full|window|region|record_start|record_stop"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "window_manager",
        "description": "Window management: maximize, minimize, close, snap left/right, switch, virtual desktops, list windows.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "maximize|minimize|close|snap_left|snap_right|switch|minimize_all|new_desktop|next_desktop|prev_desktop|task_view|list"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "data_cleaning",
        "description": "Enterprise data cleaning: audit datasets, detect missing values/duplicates/outliers, clean intelligently, generate quality reports. Use when user uploads or mentions cleaning data.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":   {"type": "STRING", "description": "audit|clean|query|report|quality"},
                "path":     {"type": "STRING", "description": "Path to dataset file"},
                "question": {"type": "STRING", "description": "Question about the dataset (for query action)"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "ml_preparation",
        "description": "ML data preparation: semantic column analysis, intelligent cleaning with policies (IDs protected), feature engineering, encoding recommendations, ML readiness report. Use for machine learning preparation.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":   {"type": "STRING", "description": "analyze|prepare|features|report|query"},
                "path":     {"type": "STRING", "description": "Path to dataset"},
                "question": {"type": "STRING", "description": "Question about the dataset"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "smart_data_scientist",
        "description": "CARL's data science brain. ALWAYS ask the user's goal first before analyzing. Actions: analyze (understand columns), clean (fix data based on goal), prepare_ml (full ML pipeline), explain (show decisions), query (answer questions about data). Requires a goal: prediction, insights, cleaning, visualization, or specific question.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":   {"type": "STRING", "description": "analyze|clean|prepare_ml|explain|query"},
                "path":     {"type": "STRING", "description": "Path to dataset"},
                "goal":     {"type": "STRING", "description": "User's objective: predict diabetes, find insights, clean data, count patients, etc."},
                "question": {"type": "STRING", "description": "Specific question about the data"},
            },
            "required": ["action", "path"]
        }
    },
    {
        "name": "ml_pipeline",
        "description": "COMPLETE autonomous ML pipeline. Executes end-to-end: clean, engineer, encode, split, scale, train 8-12 models, cross-validate, evaluate, select best, save artifacts. Use when user says 'build a model', 'train ML', 'predict', 'machine learning'. Never stops mid-pipeline.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "path":   {"type": "STRING", "description": "Path to dataset"},
                "goal":   {"type": "STRING", "description": "ML goal: classification, regression, clustering"},
                "target": {"type": "STRING", "description": "Target column name (auto-detected if empty)"},
            },
            "required": ["path"]
        }
    },
]

# --- Plugin system ---


class JarvisLive:  # Legacy class name — this is the CARL core engine

    def __init__(self, ui: JarvisUI):
        self.ui             = ui
        self.session              = None
        self.audio_in_queue       = None
        self.out_queue            = None
        self._loop                = None
        self._is_speaking         = False
        self._speaking_lock       = threading.Lock()
        self._speaking_ended_at   = 0.0     # monotonic time when speaking stopped (echo guard)
        self._phone_active        = False   # True while phone mic is streaming; pauses PC mic
        self._pending_vision       = None    # (img_bytes, mime_type, question, angle) to inject after tool response
        self._vision_cam_active    = False   # True if camera was opened for vision → auto-close after response
        self._vision_close_pending = False   # True after vision injected; next turn_complete closes camera
        self._vision_last_time     = 0.0     # monotonic time of last screen_process call (cooldown guard)
        self._vision_busy          = False   # True while a vision capture/inject cycle is in flight
        self._interrupted          = False   # True while draining audio after user interrupt
        self.ui.on_text_command   = self._on_text_command
        self.ui.on_remote_clicked = self._make_remote_key
        self.ui.on_interrupt      = self.interrupt
        self._turn_done_event: asyncio.Event | None = None
        self._dashboard     = None
        self._boot_time        = time.monotonic()  # used for proactive grace period
        self._sys_monitor      = SystemMonitor()  # persistent cooldown state
        self._proactive        = ProactiveEngine()
        self._reasoning        = ReasoningEngine()  # Phase 16: internal planning engine
        self._last_user_speech = time.monotonic()  # updated on every user utterance

        # Phase 17: Persistent semantic memory
        self._memory_store   = MemoryStore()
        self._session_memory = SessionMemory()
        self._context_engine = ContextEngine(self._memory_store, self._session_memory)
        self._summarizer     = ConversationSummarizer(self._memory_store)

        # Phase 18: Cognitive memory (unified interface)
        self._brain = CognitiveMemory()

        # Phase 27: Emotional Intelligence
        self._ei = EmotionalIntelligence()

        # Connect context engine to reasoning engine
        self._reasoning.set_context_engine(self._context_engine)

    def _make_remote_key(self):
        """Called from Qt main thread when user presses Remote Control."""
        if self._dashboard is None:
            self.ui.write_log(
                "SYS: Dashboard unavailable. "
                "Run: pip install fastapi \"uvicorn[standard]\" cryptography"
            )
            return None
        key    = self._dashboard.new_key()
        url    = self._dashboard.get_url()
        manual = self._dashboard.get_manual_url()
        return url, key, f"{url}/auto-login?key={key}", manual

    def _on_text_command(self, text: str):
        if not self._loop or not self.session:
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"parts": [{"text": text}]},
                turn_complete=True
            ),
            self._loop
        )

    def set_speaking(self, value: bool):
        with self._speaking_lock:
            was_speaking = self._is_speaking
            self._is_speaking = value
            # Record when CARL stops speaking — echo suppression uses this
            if was_speaking and not value:
                self._speaking_ended_at = time.monotonic()
        if value:
            self.ui.set_state("SPEAKING")
        elif not self.ui.muted:
            self.ui.set_state("LISTENING")

    def interrupt(self) -> None:
        """Stop CARL mid-speech: drain queued audio and open mic immediately."""
        self._interrupted = True
        q = self.audio_in_queue
        if q:
            drained = 0
            while True:
                try:
                    q.get_nowait()
                    drained += 1
                except Exception:
                    break
            if drained:
                print(f"[CARL] [X] Interrupted — {drained} audio chunks discarded")
        self.set_speaking(False)
        if self._turn_done_event:
            self._turn_done_event.clear()
        # No log message — silent return to listening

    def speak(self, text: str):
        if not self._loop or not self.session:
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"parts": [{"text": text}]},
                turn_complete=True
            ),
            self._loop
        )

    def speak_error(self, tool_name: str, error: str):
        short = str(error)[:120]
        self.ui.write_log(f"ERR: {tool_name} — {short}")
        self.speak(f"Sir, {tool_name} encountered an error. {short}")

    def _build_config(self) -> types.LiveConnectConfig:
        from datetime import datetime

        memory     = load_memory()
        mem_str    = format_memory_for_prompt(memory)
        sys_prompt = _load_system_prompt()

        now      = datetime.now()
        time_str = now.strftime("%A, %B %d, %Y — %I:%M %p")
        time_ctx = (
            f"[CURRENT DATE & TIME]\n"
            f"Right now it is: {time_str}\n"
            f"Use this to calculate exact times for reminders.\n\n"
        )

        parts = [time_ctx]
        if mem_str:
            parts.append(mem_str)
        parts.append(sys_prompt)

        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription=types.AudioTranscriptionConfig(),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            system_instruction="\n".join(parts),
            tools=[{"function_declarations": TOOL_DECLARATIONS}],
            session_resumption=types.SessionResumptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=_get_gemini_voice()
                    )
                )
            ),
        )

    async def _execute_tool(self, fc) -> types.FunctionResponse:
        name = fc.name
        args = dict(fc.args or {})

        print(f"[CARL] [T] {name}  {args}")
        # Set task-specific state based on tool type
        _TOOL_STATES = {
            "data_cleaning": "CLEANING", "smart_data_scientist": "CLEANING",
            "ml_preparation": "TRAINING", "ml_pipeline": "TRAINING",
            "web_search": "PROCESSING", "browser_control": "PROCESSING",
            "file_controller": "PROCESSING", "file_intake": "PROCESSING",
            "display_control": "PROCESSING", "audio_control": "PROCESSING",
            "network_control": "PROCESSING", "power_control": "PROCESSING",
            "system_diagnostics": "PROCESSING", "screenshot_control": "PROCESSING",
            "window_manager": "PROCESSING", "app_manager": "PROCESSING",
            "download_file": "PROCESSING", "code_helper": "PROCESSING",
        }
        self.ui.set_state(_TOOL_STATES.get(name, "THINKING"))

        if name == "save_memory":
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if key and value:
                update_memory({category: {key: {"value": value}}})
                print(f"[Memory] [S] save_memory: {category}/{key} = {value}")
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": "ok", "silent": True}
            )

        loop   = asyncio.get_event_loop()
        result = "Done."

        # ── Reasoning Engine gate: check if this needs multi-step planning ───
        execution_level = self._reasoning.evaluate(name, args)
        if execution_level == ExecutionLevel.PLAN:
            goal = self._reasoning.extract_goal(name, args)
            print(f"[CARL] [B] Reasoning Engine activated for: {goal}")
            self.ui.set_state("THINKING")

            # Build context from memory
            mem_ctx = format_memory_for_prompt(load_memory())

            # Execute the plan (blocking — runs on executor thread)
            plan_result = await loop.run_in_executor(
                None,
                lambda: self._reasoning.plan_and_execute(
                    goal=goal,
                    ui=self.ui,
                    speak_fn=None,
                    confirm_fn=None,
                    context=mem_ctx,
                )
            )

            # Handle fallback results — planner failed, execute directly
            if plan_result == FallbackResult.DIRECT:
                # Fall through to normal tool execution below
                print(f"[CARL] [<] Fallback: executing {name} directly")
                pass  # continues to the normal try block below
            elif plan_result == FallbackResult.CLARIFY:
                # Ask user for clarification
                result = ("I couldn't build a reliable execution plan. "
                          "Could you be a little more specific?")
                if not self.ui.muted:
                    self.ui.set_state("LISTENING")
                return types.FunctionResponse(
                    id=fc.id, name=name,
                    response={"result": result}
                )
            else:
                # Successful plan execution — return the result
                if not self.ui.muted:
                    self.ui.set_state("LISTENING")
                print(f"[CARL] [>] {name} (planned) → {str(plan_result)[:80]}")
                return types.FunctionResponse(
                    id=fc.id, name=name,
                    response={"result": plan_result}
                )
        # ── End reasoning gate ───────────────────────────────────────────────

        try:
            if name == "open_app":
                r = await loop.run_in_executor(None, lambda: open_app(parameters=args, response=None, player=self.ui))
                result = r or f"Opened {args.get('app_name')}."

            elif name == "weather_report":
                r = await loop.run_in_executor(None, lambda: weather_action(parameters=args, player=self.ui))
                result = r or "Weather delivered."

            elif name == "browser_control":
                r = await loop.run_in_executor(None, lambda: browser_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "file_controller":
                r = await loop.run_in_executor(None, lambda: file_controller(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "send_message":
                r = await loop.run_in_executor(None, lambda: send_message(parameters=args, response=None, player=self.ui, session_memory=None))
                result = r or f"Message sent to {args.get('receiver')}."

            elif name == "reminder":
                r = await loop.run_in_executor(None, lambda: reminder(parameters=args, response=None, player=self.ui))
                result = r or "Reminder set."

            elif name == "youtube_video":
                r = await loop.run_in_executor(None, lambda: youtube_video(parameters=args, response=None, player=self.ui))
                result = r or "Done."

            elif name == "screen_process":
                import time as _t_mod
                _now = _t_mod.monotonic()
                _cooldown = 4.0  # seconds — covers echo window after speaking ends
                if self._vision_busy or (_now - self._vision_last_time) < _cooldown:
                    _wait = max(0, _cooldown - (_now - self._vision_last_time))
                    print(f"[Vision] [W] Cooldown active ({_wait:.1f}s remaining) — ignoring duplicate call")
                    result = "Vision is still processing the previous request. I will not call this again."
                else:
                    self._vision_busy      = True
                    self._vision_last_time = _now
                    angle     = args.get("angle", "screen").lower()
                    user_text = args.get("text", "What do you see?")
                    if angle == "camera":
                        img_b, mime_t = await loop.run_in_executor(None, _capture_camera)
                        self.ui.start_camera_stream()
                        self._vision_cam_active = True
                        print(f"[Vision] [C] Camera: {len(img_b):,} bytes")
                        _stall = "camera"
                    else:
                        img_b, mime_t = await loop.run_in_executor(None, _capture_screen)
                        print(f"[Vision] [S]  Screen: {len(img_b):,} bytes")
                        _stall = "screen"
                    self._pending_vision = (img_b, mime_t, user_text, angle)
                    result = (
                        f"[VISION_ACTIVE] {_stall.capitalize()} captured. "
                        f"Immediately say ONE natural sentence in the user's language "
                        f"(e.g. 'Looking at your {_stall} now, sir' / "
                        f"'{'Kameraya' if _stall == 'camera' else 'Ekrana'} bakıyorum efendim'). "
                        f"Do NOT describe or guess content — the actual image arrives in the NEXT message."
                    )

            elif name == "close_camera":
                self.ui.stop_camera_stream()
                result = "Camera closed."

            elif name == "computer_settings":
                r = await loop.run_in_executor(None, lambda: computer_settings(parameters=args, response=None, player=self.ui))
                result = r or "Done."

            elif name == "desktop_control":
                r = await loop.run_in_executor(None, lambda: desktop_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "code_helper":
                r = await loop.run_in_executor(None, lambda: code_helper(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "dev_agent":
                r = await loop.run_in_executor(None, lambda: dev_agent(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "web_search":
                r = await loop.run_in_executor(None, lambda: web_search_action(parameters=args, player=self.ui))
                result = r or "Done."
                # Mirror results to the on-screen content panel
                _mode = args.get("mode", "search")
                if r and not r.startswith("No results") and not r.startswith("Search failed"):
                    _query = args.get("query") or ", ".join(args.get("items", []))
                    _label = f"{_mode.upper()} — {_query[:38]}" if _query else _mode.upper()
                    self.ui.show_content(_label, r)
            elif name == "file_processor":
                if not args.get("file_path") and self.ui.current_file:
                    args["file_path"] = self.ui.current_file
                r = await loop.run_in_executor(
                    None,
                    lambda: file_processor(parameters=args, player=self.ui, speak=self.speak)
                )
                result = r or "Done."

            elif name == "computer_control":
                r = await loop.run_in_executor(None, lambda: computer_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "game_updater":
                r = await loop.run_in_executor(None, lambda: game_updater(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "flight_finder":
                r = await loop.run_in_executor(None, lambda: flight_finder(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "system_status":
                r = await loop.run_in_executor(None, get_system_status)
                result = str(r)

            elif name == "display_control":
                r = await loop.run_in_executor(None, lambda: display_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "audio_control":
                r = await loop.run_in_executor(None, lambda: audio_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "battery_status":
                r = await loop.run_in_executor(None, lambda: battery_status(parameters=args, player=self.ui))
                result = r or "No battery info."

            elif name == "network_control":
                r = await loop.run_in_executor(None, lambda: network_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "download_file":
                r = await loop.run_in_executor(None, lambda: download_file(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "file_intake":
                # file_intake returns immediately — no QFileDialog, no blocking
                r = file_intake(parameters=args, player=self.ui, speak=self.speak)
                if r == "CANCELLED":
                    result = "No problem, sir. Upload cancelled."
                elif r.startswith("ERROR:"):
                    result = f"I encountered an issue: {r[6:].strip()}"
                else:
                    result = r or "No file selected."
                    if "Path:" in result:
                        path_line = [l for l in result.split("\n") if l.startswith("Path:")]
                        if path_line:
                            self._selected_file = path_line[0].replace("Path: ", "").strip()

            elif name == "app_manager":
                r = await loop.run_in_executor(None, lambda: app_manager(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "power_control":
                r = await loop.run_in_executor(None, lambda: power_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "system_diagnostics":
                r = await loop.run_in_executor(None, lambda: system_diagnostics(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "screenshot_control":
                r = await loop.run_in_executor(None, lambda: screenshot_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "window_manager":
                r = await loop.run_in_executor(None, lambda: window_manager(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "data_cleaning":
                r = await loop.run_in_executor(None, lambda: data_cleaning(parameters=args, player=self.ui))
                result = r or "Done."
                if r and len(r) > 200:
                    self.ui.show_content("DATA ANALYSIS", r)

            elif name == "ml_preparation":
                r = await loop.run_in_executor(None, lambda: ml_preparation(parameters=args, player=self.ui))
                result = r or "Done."
                if r and len(r) > 200:
                    self.ui.show_content("ML PREPARATION", r)

            elif name == "smart_data_scientist":
                r = await loop.run_in_executor(None, lambda: smart_data_scientist(parameters=args, player=self.ui))
                result = r or "Done."
                if r and len(r) > 200:
                    self.ui.show_content("DATA SCIENCE", r)

            elif name == "ml_pipeline":
                self.ui.set_state("TRAINING")
                r = await loop.run_in_executor(None, lambda: ml_pipeline(parameters=args, player=self.ui, speak=self.speak))
                self.ui.set_state("COMPLETED")
                result = r or "Done."
                if r and len(r) > 200:
                    self.ui.show_content("ML PIPELINE", r)

            elif name == "shutdown_jarvis":
                self.ui.write_log("SYS: Shutdown requested.")
                self.speak("Goodbye, sir.")
                def _shutdown():
                    import time, os
                    time.sleep(1)
                    os._exit(0)
                threading.Thread(target=_shutdown, daemon=True).start()

            else:
                result = f"Unknown tool: {name}"

        except Exception as e:
            result = f"Tool '{name}' failed: {e}"
            traceback.print_exc()
            self.speak_error(name, e)

        if not self.ui.muted:
            self.ui.set_state("LISTENING")

        print(f"[CARL] [>] {name} → {str(result)[:80]}")

        # Phase 17: track tool usage and auto-save session
        self._session_memory.record_tool_call(name)
        self._memory_store.record_usage(name, category="tool")
        if name == "open_app":
            app = args.get("app_name", "")
            if app:
                self._session_memory.record_app_opened(app)
                self._memory_store.record_usage(app, category="app")
        self._session_memory.auto_save_if_dirty()

        return types.FunctionResponse(
            id=fc.id, name=name,
            response={"result": result}
        )

    async def _send_realtime(self):
        _sent_count = 0
        while True:
            msg = await self.out_queue.get()
            await self.session.send_realtime_input(audio=msg)
            _sent_count += 1
            if _sent_count == 1:
                print("[DEBUG] [S] First audio chunk sent to Gemini")
            elif _sent_count == 50:
                print("[DEBUG] [S] 50 chunks sent — audio stream to Gemini confirmed")

    async def _listen_audio(self):
        print("[CARL] [M] Mic started")
        loop = asyncio.get_event_loop()
        import numpy as np

        _ECHO_GUARD_SECS = 0.15  # 150ms silence after CARL stops speaking
        _frame_count = 0
        _GAIN = 4  # Amplification factor for quiet mics

        def _safe_put(item):
            """Put item into out_queue without raising QueueFull."""
            try:
                self.out_queue.put_nowait(item)
            except asyncio.QueueFull:
                pass  # drop frame — queue consumer is behind

        def callback(indata, frames, time_info, status):
            nonlocal _frame_count
            with self._speaking_lock:
                carl_speaking = self._is_speaking
                ended_at = self._speaking_ended_at
            # Suppress mic while speaking AND for 300ms after (echo guard)
            if carl_speaking:
                return
            if (time.monotonic() - ended_at) < _ECHO_GUARD_SECS:
                return
            if self.ui.muted or self._phone_active:
                return
            # indata is numpy int16 array from sounddevice — amplify and send
            amplified = np.clip(indata * _GAIN, -32767, 32767).astype(np.int16)
            data = amplified.tobytes()
            _frame_count += 1
            if _frame_count == 1:
                print(f"[DEBUG] [M] First mic frame — {len(data)} bytes, peak={int(np.max(np.abs(indata)))}")
            elif _frame_count == 50:
                print("[DEBUG] [M] 50 frames captured — mic confirmed working")
            loop.call_soon_threadsafe(_safe_put, {"data": data, "mime_type": "audio/pcm"})

        try:
            with sd.InputStream(
                samplerate=SEND_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                callback=callback,
            ):
                print("[CARL] [M] Mic stream open")
                while True:
                    await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[CARL] [!] Mic: {e}")
            raise

    async def _receive_audio(self):
        print("[CARL] [R] Recv started")
        out_buf, in_buf = [], []

        try:
            while True:
                async for response in self.session.receive():

                    if response.data:
                        if self._interrupted:
                            pass  # discard: interrupted
                        else:
                            if self._turn_done_event and self._turn_done_event.is_set():
                                self._turn_done_event.clear()
                            # Split into ~50 ms chunks so interrupt() stops audio within 50 ms
                            # (24000 Hz × 2 bytes/sample × 0.05 s = 2400 bytes per slice)
                            _audio_data = response.data
                            if not hasattr(self, '_resp_logged'):
                                self._resp_logged = True
                                print(f"[DEBUG] [P] First audio response from Gemini ({len(_audio_data)} bytes)")
                            _SLICE = 2400
                            for _i in range(0, len(_audio_data), _SLICE):
                                await self.audio_in_queue.put(_audio_data[_i : _i + _SLICE])

                    if response.server_content:
                        sc = response.server_content

                        if sc.output_transcription and sc.output_transcription.text:
                            txt = _clean_transcript(sc.output_transcription.text)
                            if txt and txt != (out_buf[-1] if out_buf else ""):
                                out_buf.append(txt)

                        if sc.input_transcription and sc.input_transcription.text:
                            txt = _clean_transcript(sc.input_transcription.text)
                            if txt:
                                in_buf.append(txt)
                                self._last_user_speech = time.monotonic()
                                print(f"[DEBUG] [V] Transcript: {txt[:60]}")

                        if sc.turn_complete:
                            if self._turn_done_event:
                                self._turn_done_event.set()

                            # If this turn_complete ends an interrupted response, clear the
                            # flag and skip all further processing for that turn.
                            if self._interrupted:
                                self._interrupted = False
                                in_buf  = []
                                out_buf = []
                                continue

                            full_in = " ".join(in_buf).strip()
                            if full_in:
                                self.ui.write_log(f"You: {full_in}")

                                # Lightweight tracking (non-blocking)
                                self._session_memory.add_conversation_turn("user", full_in)
                                self._brain.record_conversation_turn("user", full_in)

                                # Check memory commands ONLY if it looks like one
                                if full_in.lower().startswith(("remember", "forget", "what do you remember", "what do you know", "clear temp")):
                                    mem_resp = self._brain.handle_command(full_in)
                                    if mem_resp:
                                        self.ui.write_log(f"CARL: {mem_resp}")
                                        if self.session:
                                            await self.session.send_client_content(
                                                turns={"parts": [{"text": f"[MEMORY] {mem_resp} — say this to the user naturally."}]},
                                                turn_complete=True,
                                            )

                                # Check reasoning commands ONLY if plan is active
                                elif self._reasoning.is_plan_active:
                                    session_resp = self._reasoning.handle_session_command(full_in)
                                    if session_resp:
                                        self.ui.write_log(f"CARL: {session_resp}")
                                        if self.session:
                                            await self.session.send_client_content(
                                                turns={"parts": [{"text": f"[SYSTEM] {session_resp} — say this to the user naturally."}]},
                                                turn_complete=True,
                                            )

                                # Background: track interests + summarize (non-blocking)
                                asyncio.get_event_loop().call_soon(
                                    lambda t=full_in: (
                                        self._summarizer.add_turn("user", t),
                                        self._ei.process_input(t),  # track mood silently, don't inject
                                    )
                                )

                                if self._dashboard:
                                    asyncio.create_task(self._dashboard.broadcast({
                                        "type": "log", "speaker": "user",
                                        "text": full_in,
                                        "ts": datetime.now().isoformat(),
                                    }))
                                    asyncio.create_task(self._dashboard.broadcast({
                                        "type": "log", "speaker": "user",
                                        "text": full_in,
                                        "ts": datetime.now().isoformat(),
                                    }))
                            in_buf = []

                            full_out = " ".join(out_buf).strip()
                            if full_out:
                                self.ui.write_log(f"CARL: {full_out}")

                                # Track CARL's response in session memory and summarizer
                                self._session_memory.add_conversation_turn("carl", full_out)
                                self._summarizer.add_turn("carl", full_out)

                                if self._dashboard:
                                    asyncio.create_task(self._dashboard.broadcast({
                                        "type": "log", "speaker": "carl",
                                        "text": full_out,
                                        "ts": datetime.now().isoformat(),
                                    }))
                            out_buf = []

                            # Empty response — Gemini said nothing; silently return to LISTENING
                            if not full_out and not full_in:
                                if not self.ui.muted:
                                    self.ui.set_state("LISTENING")

                            # Vision injection: model finished tool-response turn → now send the image
                            if self._pending_vision and self.session:
                                import base64 as _b64
                                img_b, mime_t, question, angle = self._pending_vision
                                self._pending_vision = None
                                b64 = _b64.b64encode(img_b).decode("ascii")
                                print(f"[Vision] [>] {len(img_b):,} bytes (angle={angle}) → main session")
                                await self.session.send_client_content(
                                    turns={"parts": [
                                        {"inline_data": {"mime_type": mime_t, "data": b64}},
                                        {"text": question},
                                    ]},
                                    turn_complete=True,
                                )
                                # Mark next turn_complete behaviour depending on angle
                                if self._vision_cam_active:
                                    # Camera: keep busy until JARVIS finishes speaking the answer
                                    self._vision_cam_active    = False
                                    self._vision_close_pending = True
                                else:
                                    # Screen-only: no camera to close; release busy flag now
                                    self._vision_busy = False
                            elif self._vision_close_pending:
                                # This turn_complete IS the vision answer — close camera + release busy flag
                                self._vision_close_pending = False
                                self._vision_busy = False
                                async def _cam_close():
                                    await asyncio.sleep(2.0)
                                    self.ui.stop_camera_stream()
                                asyncio.create_task(_cam_close())

                    if response.tool_call:
                        fn_responses = []
                        for fc in response.tool_call.function_calls:
                            print(f"[CARL] [C] {fc.name}")
                            fr = await self._execute_tool(fc)
                            fn_responses.append(fr)
                        await self.session.send_tool_response(
                            function_responses=fn_responses
                        )
        except Exception as e:
            print(f"[CARL] [!] Recv: {e}")
            traceback.print_exc()
            raise

    async def _play_audio(self):
        print("[CARL] [P] Play started")

        stream = sd.RawOutputStream(
            samplerate=RECEIVE_SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
        )
        stream.start()

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        self.audio_in_queue.get(),
                        timeout=0.3
                    )
                except asyncio.TimeoutError:
                    if (
                        self._turn_done_event
                        and self._turn_done_event.is_set()
                        and self.audio_in_queue.empty()
                    ):
                        self.set_speaking(False)
                        self._turn_done_event.clear()
                    continue
                self.set_speaking(True)
                try:
                    await asyncio.to_thread(stream.write, chunk)
                except (RuntimeError, asyncio.CancelledError):
                    break   # executor shutting down — exit cleanly
        except Exception as e:
            print(f"[CARL] [!] Play: {e}")
            raise
        finally:
            self.set_speaking(False)
            stream.stop()
            stream.close()

    # ── (Briefing system removed — CARL starts silently) ───────────────────────

    # ── System monitor ──────────────────────────────────────────────────────────

    async def _run_system_monitor(self) -> None:
        """Background task: voice alerts when metrics exceed thresholds."""
        while True:
            await asyncio.sleep(10)
            alert = await asyncio.to_thread(self._sys_monitor.check)
            if alert and self.session:
                try:
                    await self.session.send_client_content(
                        turns={"parts": [{"text": alert}]},
                        turn_complete=True,
                    )
                except Exception as e:
                    print(f"[Monitor] [!] Could not send alert: {e}")

    # ── Proactive mode ──────────────────────────────────────────────────────────

    async def _run_proactive_mode(self) -> None:
        """
        Background task: periodically checks if the user has been silent long enough,
        then hands time + memory context to Gemini so it can decide what (if anything)
        to say proactively. No hardcoded rules — Gemini makes the call.
        """
        _BOOT_GRACE = 1800  # 30 minutes — never speak proactively right after startup

        while True:
            await asyncio.sleep(60)   # evaluate once per minute

            if not self.session:
                continue

            # Don't trigger proactive mode within 30 min of boot
            if (time.monotonic() - self._boot_time) < _BOOT_GRACE:
                continue

            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking:
                continue

            if not self._proactive.should_trigger(self._last_user_speech):
                continue

            self._proactive.mark_triggered()

            try:
                memory = await asyncio.to_thread(load_memory)
                prompt = self._proactive.build_prompt(memory)
                await self.session.send_client_content(
                    turns={"parts": [{"text": prompt}]},
                    turn_complete=True,
                )
                self.ui.write_log("SYS: Proactive check-in.")
            except Exception as e:
                print(f"[Proactive] [!] {e}")

    # ── Memory cleanup ───────────────────────────────────────────────────────────

    async def _run_memory_cleanup(self) -> None:
        """Background task: periodic memory cleanup (expired entries, duplicates, session save)."""
        _CLEANUP_INTERVAL = 21600  # 6 hours
        # Initial delay — don't run cleanup immediately on startup
        await asyncio.sleep(300)  # wait 5 minutes

        while True:
            try:
                # Clean expired memories
                removed = await asyncio.to_thread(self._memory_store.cleanup_expired)
                if removed:
                    print(f"[Memory] [C] Cleanup: removed {removed} expired entries")

                # Auto-save session state
                await asyncio.to_thread(self._session_memory.auto_save_if_dirty)

                # Force summarize if buffer has content (session might end unexpectedly)
                summary = await asyncio.to_thread(self._summarizer.force_summarize)
                if summary:
                    print(f"[Memory] [N] Periodic summary saved")

            except Exception as e:
                print(f"[Memory] [!] Cleanup error: {e}")

            await asyncio.sleep(_CLEANUP_INTERVAL)

    # ── Phone audio relay ────────────────────────────────────────────────────────

    async def _relay_phone_audio(self) -> None:
        """Forward phone mic PCM chunks from dashboard queue into the Gemini Live session."""
        q = self._dashboard._phone_audio_queue
        while True:
            try:
                chunk = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # No audio for 1 s → phone mic inactive, give PC mic back
                self._phone_active = False
                continue
            self._phone_active = True   # phone is streaming — silence PC mic
            with self._speaking_lock:
                speaking = self._is_speaking
            if not speaking and not self.ui.muted:
                try:
                    self.out_queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    pass

    def _on_phone_connected(self) -> None:
        self.ui.write_log("SYS: Phone connected via Remote Dashboard.")
        self.ui.notify_phone_connected()

    # ── dashboard command relay ─────────────────────────────────────────────

    async def _process_dashboard_commands(self) -> None:
        while True:
            try:
                text = await asyncio.wait_for(
                    self._dashboard._command_queue.get(), timeout=0.5
                )
                if not text:
                    continue
                # Wait up to 8s for session to become ready after a wake
                for _ in range(80):
                    if self.session:
                        break
                    await asyncio.sleep(0.1)
                if self.session:
                    await self.session.send_client_content(
                        turns={"parts": [{"text": text}]},
                        turn_complete=True,
                    )
                    self.ui.write_log(f"[Web]: {text}")
                else:
                    print(f"[Dashboard] Dropped command (no session): {text}")
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print(f"[Dashboard] Command error: {e}")
                await asyncio.sleep(0.5)

    # ── main loop ───────────────────────────────────────────────────────────

    async def run(self):
        self._loop = asyncio.get_event_loop()

        # Start dashboard (optional — needs: pip install fastapi "uvicorn[standard]" cryptography)
        try:
            from dashboard.server import DashboardServer
            self._dashboard = DashboardServer()
            self._dashboard.set_connect_callback(self._on_phone_connected)
            asyncio.create_task(self._dashboard.serve())
            # Runs for the whole lifetime, not just inside an active session
            asyncio.create_task(self._process_dashboard_commands())
        except Exception as e:
            print(f"[Dashboard] Disabled: {e}")
            self._dashboard = None

        # ── One-time planner health check ────────────────────────────────────
        try:
            planner_ok = await asyncio.to_thread(self._reasoning.startup_check)
            status_msg = self._reasoning.health.status_text()
            self.ui.write_log(f"SYS: {status_msg}")
        except Exception as e:
            print(f"[Reasoning] Startup check failed: {e}")
            self.ui.write_log("SYS: Planner Offline — Direct Command Mode")

        # ── Restore session memory ───────────────────────────────────────────
        try:
            restored = self._session_memory.restore()
            if restored and self._session_memory.get_active_project():
                proj = self._session_memory.get_active_project()
                self.ui.write_log(f"SYS: Session restored — project: {proj}")
        except Exception as e:
            print(f"[Memory] Session restore failed: {e}")

        # ── Microphone health check ─────────────────────────────────────────
        mic_ok, mic_msg = check_microphone()
        if mic_ok:
            print(f"[CARL] [OK] {mic_msg}")
        else:
            print(f"[CARL] [!] {mic_msg}")
            self.ui.write_log(f"SYS: {mic_msg}")

        while True:
            try:
                print("[CARL] Connecting...")
                self.ui.set_state("THINKING")
                config = self._build_config()

                # Fresh client on every reconnect — avoids stale HTTP session state
                client = genai.Client(
                    api_key=_get_api_key(),
                    http_options={"api_version": "v1beta"}
                )

                async with (
                    client.aio.live.connect(model=LIVE_MODEL, config=config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session          = session
                    self.audio_in_queue   = asyncio.Queue(maxsize=500)
                    self.out_queue        = asyncio.Queue(maxsize=100)
                    self._turn_done_event = asyncio.Event()

                    # Reset transient state that must not carry over from a previous session
                    self._pending_vision       = None
                    self._vision_cam_active    = False
                    self._vision_close_pending = False
                    self._vision_busy          = False
                    self._vision_last_time     = 0.0
                    self._interrupted          = False

                    print("[CARL] Connected.")
                    self.ui.set_state("LISTENING")
                    self.ui.write_log("SYS: CARL online.")

                    if self._dashboard:
                        await self._dashboard.broadcast({"type": "status", "state": "active"})

                    tg.create_task(self._send_realtime())
                    tg.create_task(self._listen_audio())
                    tg.create_task(self._receive_audio())
                    tg.create_task(self._play_audio())
                    tg.create_task(self._run_memory_cleanup())
                    if self._dashboard:
                        tg.create_task(self._relay_phone_audio())

                    # Silent startup — CARL waits for user to speak first

            except KeyboardInterrupt:
                raise
            except SystemExit:
                raise
            except BaseException as e:
                # Catches both Exception and BaseExceptionGroup (Python 3.11+
                # TaskGroup raises BaseExceptionGroup when tasks are cancelled
                # externally, which `except Exception` would miss, letting the
                # exception escape the while-loop and causing asyncio.run() to
                # start shutdown — resulting in "executor after shutdown" errors).
                err_str = str(e)
                print(f"[CARL] Error ({type(e).__name__}): {e}")
                traceback.print_exc()

                # Invalid API key — stop hammering the API, prompt re-configuration
                if "API key not valid" in err_str or "1007" in err_str:
                    self.ui.write_log("ERR: API key invalid — please re-enter your key.")
                    self.ui.set_state("SLEEPING")
                    self.ui.prompt_reconfig()
                    while not self.ui._win._ready:
                        await asyncio.sleep(1)
                    print("[CARL] New API key saved — reconnecting...")
                    _conn_backoff = 3
                    continue

                # Network / timeout errors — log clearly and back off
                is_net_err = any(k in err_str for k in (
                    "TimeoutError", "timed out", "getaddrinfo", "CancelledError",
                    "ConnectionRefusedError", "OSError", "Cannot connect",
                ))
                if is_net_err:
                    _conn_backoff = min(getattr(self, "_conn_backoff", 3) * 2, 60)
                    self._conn_backoff = _conn_backoff
                    self.ui.write_log(
                        f"NET: Bağlantı kurulamadı — {_conn_backoff}s sonra tekrar deneniyor. "
                        "(VPN gerekiyor olabilir)"
                    )
                else:
                    self._conn_backoff = 3
            finally:
                self.session = None

            self.set_speaking(False)
            self.ui.set_state("SLEEPING")

            if self._dashboard:
                await self._dashboard.broadcast({"type": "status", "state": "sleeping"})

            delay = getattr(self, "_conn_backoff", 3)
            print(f"[CARL] Reconnecting in {delay}s...")
            await asyncio.sleep(delay)

def main():
    ui = JarvisUI("face.png")

    def runner():
        ui.wait_for_api_key()
        jarvis = JarvisLive(ui)
        try:
            asyncio.run(jarvis.run())
        except KeyboardInterrupt:
            print("\n🔴 Shutting down...")

    threading.Thread(target=runner, daemon=True).start()
    ui.root.mainloop()

if __name__ == "__main__":
    main()