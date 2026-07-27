"""
memory/preferences.py — Preference Memory Layer for CARL.

Automatically learns recurring user preferences from behavior.
No explicit "save preference" needed — learns from patterns.

Examples:
  - User always says "no news" → preference: no_news
  - User consistently uses aqua blue → preference: color_aqua
  - User prefers British voice → preference: voice_british_male
  - User always uses VS Code → preference: ide_vscode
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

_STORE_PATH = Path(__file__).resolve().parent.parent / "memory" / "preferences_store.json"
_lock = threading.Lock()
_LEARN_THRESHOLD = 3  # occurrences before becoming a learned preference


@dataclass
class Preference:
    key: str
    value: str
    confidence: float = 0.5  # 0-1, increases with reinforcement
    occurrences: int = 1
    learned: bool = False  # True once threshold reached
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    source: str = "observed"  # observed, explicit, inferred

    def reinforce(self):
        """User exhibited this preference again."""
        self.occurrences += 1
        self.last_seen = time.time()
        self.confidence = min(1.0, self.confidence + 0.15)
        if self.occurrences >= _LEARN_THRESHOLD:
            self.learned = True

    def to_dict(self) -> dict:
        return {
            "key": self.key, "value": self.value, "confidence": self.confidence,
            "occurrences": self.occurrences, "learned": self.learned,
            "created_at": self.created_at, "last_seen": self.last_seen, "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Preference":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PreferenceMemory:
    """Auto-learns and stores user preferences."""

    def __init__(self):
        self._prefs: Dict[str, Preference] = {}
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self._load()
            self._loaded = True

    def observe(self, key: str, value: str, source: str = "observed") -> Preference:
        """Observe a preference occurrence. Auto-learns after threshold."""
        self._ensure_loaded()
        if key in self._prefs:
            self._prefs[key].reinforce()
        else:
            self._prefs[key] = Preference(key=key, value=value, source=source)
        self._save()
        return self._prefs[key]

    def set_explicit(self, key: str, value: str) -> Preference:
        """User explicitly stated a preference."""
        self._ensure_loaded()
        pref = Preference(key=key, value=value, confidence=0.9, learned=True,
                          occurrences=_LEARN_THRESHOLD, source="explicit")
        self._prefs[key] = pref
        self._save()
        return pref

    def get(self, key: str) -> str | None:
        """Get a learned preference value."""
        self._ensure_loaded()
        p = self._prefs.get(key)
        if p and p.learned:
            return p.value
        return None

    def get_all_learned(self) -> list[Preference]:
        """Get all confirmed learned preferences."""
        self._ensure_loaded()
        return [p for p in self._prefs.values() if p.learned]

    def format_for_context(self) -> str:
        """Format learned preferences for prompt injection."""
        learned = self.get_all_learned()
        if not learned:
            return ""
        lines = ["[USER PREFERENCES]"]
        for p in sorted(learned, key=lambda x: -x.confidence)[:15]:
            lines.append(f"  {p.key}: {p.value}")
        return "\n".join(lines)

    def _load(self):
        if not _STORE_PATH.exists():
            return
        try:
            with _lock:
                data = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
            self._prefs = {d["key"]: Preference.from_dict(d) for d in data}
        except Exception as e:
            print(f"[Memory] Preferences load error: {e}")

    def _save(self):
        with _lock:
            _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _STORE_PATH.write_text(
                json.dumps([p.to_dict() for p in self._prefs.values()], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
