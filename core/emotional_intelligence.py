"""
core/emotional_intelligence.py — Emotional Intelligence Engine for CARL.

Detects conversational emotion from language patterns, tracks session mood,
and generates adaptive response hints that CARL injects into context.

Does NOT fake empathy or claim feelings. Responds as a calm, emotionally
aware professional assistant.

Integrates with:
  - Reasoning Engine (provides mood context for planning)
  - Cognitive Memory (stores mood patterns over time)
  - System Prompt (injects response tone guidance)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List


# ═══════════════════════════════════════════════════════════════════════════════
# MOOD CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class Mood(Enum):
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"
    URGENT = "urgent"
    GRATEFUL = "grateful"
    EXCITED = "excited"
    DISAPPOINTED = "disappointed"
    TIRED = "tired"


# ═══════════════════════════════════════════════════════════════════════════════
# LANGUAGE PATTERNS FOR DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

_FRUSTRATED_PATTERNS = [
    r"\b(doesn't|doesn't|doesnt|does not) work\b",
    r"\b(not working|broken|keeps? failing|still wrong|again|ugh|damn|shit)\b",
    r"\b(why (won't|wont|can't|cant|isn't|isnt))\b",
    r"\b(frustrated|annoyed|irritated|fed up|sick of)\b",
    r"\b(this is (stupid|ridiculous|terrible|awful|garbage))\b",
    r"\b(i (told|said|asked) you)\b",
    r"\b(how many times)\b",
    r"\b(come on|for (god's|gods) sake)\b",
]

_CONFUSED_PATTERNS = [
    r"\b(i don't understand|i dont understand|confused|what do you mean)\b",
    r"\b(what\?|huh\?|excuse me\?)\b",
    r"\b(that doesn't make sense|makes no sense)\b",
    r"\b(wait,? what|hold on)\b",
    r"\b(can you (explain|clarify|rephrase))\b",
    r"\b(i'm lost|im lost|lost me)\b",
    r"\b(which one|what exactly)\b",
]

_URGENT_PATTERNS = [
    r"\b(urgent|asap|immediately|right now|hurry|quick|fast)\b",
    r"\b(deadline|running out of time|emergency)\b",
    r"\b(need this (now|today|done|immediately))\b",
    r"\b(as soon as possible)\b",
]

_GRATEFUL_PATTERNS = [
    r"\b(thanks?|thank you|cheers|appreciate|grateful|nice one)\b",
    r"\b(well done|good job|perfect|brilliant|excellent|great work)\b",
    r"\b(that('s| is) (great|perfect|exactly|wonderful|helpful))\b",
]

_POSITIVE_PATTERNS = [
    r"\b(awesome|cool|nice|love it|looks good|impressive)\b",
    r"\b(happy|glad|pleased|excited|looking forward)\b",
    r"\b(finally|yes!|it works|working now)\b",
]

_DISAPPOINTED_PATTERNS = [
    r"\b(disappointing|let down|expected (more|better))\b",
    r"\b(not (good|great|what i (wanted|expected)))\b",
    r"\b(worse than|went wrong|failed)\b",
]

_TIRED_PATTERNS = [
    r"\b(tired|exhausted|long day|been at this|hours)\b",
    r"\b(need a break|enough for today|calling it)\b",
    r"\b(can't think|brain (is )?dead|done for today)\b",
]


# ═══════════════════════════════════════════════════════════════════════════════
# MOOD TRACKER — Session-level emotional state
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MoodEntry:
    mood: Mood
    confidence: float  # 0.0 - 1.0
    timestamp: float = field(default_factory=time.time)
    trigger_text: str = ""


class MoodTracker:
    """
    Tracks conversational mood across a session.
    Uses a weighted recent-history approach — last 5 interactions matter most.
    """

    def __init__(self):
        self._history: list[MoodEntry] = []
        self._current: Mood = Mood.NEUTRAL
        self._frustration_count: int = 0  # consecutive frustrated messages

    @property
    def current_mood(self) -> Mood:
        return self._current

    @property
    def frustration_level(self) -> int:
        """0 = none, 1 = mild, 2 = moderate, 3+ = high"""
        return self._frustration_count

    def record(self, mood: Mood, confidence: float, trigger: str = ""):
        """Record a detected mood and update session state."""
        self._history.append(MoodEntry(mood=mood, confidence=confidence, trigger_text=trigger[:80]))
        # Keep last 20
        if len(self._history) > 20:
            self._history = self._history[-20:]

        # Update frustration counter
        if mood == Mood.FRUSTRATED:
            self._frustration_count += 1
        elif mood in (Mood.POSITIVE, Mood.GRATEFUL):
            self._frustration_count = max(0, self._frustration_count - 2)
        else:
            self._frustration_count = max(0, self._frustration_count - 1)

        # Update current mood (weighted toward recent)
        self._current = mood

    def get_session_summary(self) -> dict:
        """Summary of emotional patterns this session."""
        if not self._history:
            return {"dominant": "neutral", "shifts": 0, "frustration_level": 0}
        moods = [e.mood.value for e in self._history]
        from collections import Counter
        counts = Counter(moods)
        dominant = counts.most_common(1)[0][0]
        shifts = sum(1 for i in range(1, len(moods)) if moods[i] != moods[i-1])
        return {
            "dominant": dominant,
            "shifts": shifts,
            "frustration_level": self._frustration_count,
            "total_interactions": len(self._history),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EMOTION DETECTOR — Analyzes text for emotional signals
# ═══════════════════════════════════════════════════════════════════════════════

class EmotionDetector:
    """Detects mood from text using pattern matching. Fast, no ML needed."""

    def detect(self, text: str) -> tuple[Mood, float]:
        """
        Analyze text and return (detected_mood, confidence).
        Confidence 0.0-1.0 indicates how certain the detection is.
        """
        lower = text.lower().strip()
        if not lower:
            return Mood.NEUTRAL, 0.0

        scores: dict[Mood, float] = {}

        scores[Mood.FRUSTRATED] = self._score_patterns(lower, _FRUSTRATED_PATTERNS)
        scores[Mood.CONFUSED] = self._score_patterns(lower, _CONFUSED_PATTERNS)
        scores[Mood.URGENT] = self._score_patterns(lower, _URGENT_PATTERNS)
        scores[Mood.GRATEFUL] = self._score_patterns(lower, _GRATEFUL_PATTERNS)
        scores[Mood.POSITIVE] = self._score_patterns(lower, _POSITIVE_PATTERNS)
        scores[Mood.DISAPPOINTED] = self._score_patterns(lower, _DISAPPOINTED_PATTERNS)
        scores[Mood.TIRED] = self._score_patterns(lower, _TIRED_PATTERNS)

        # Additional signals
        if lower.endswith("?") and len(lower.split()) <= 4:
            scores[Mood.CONFUSED] = max(scores.get(Mood.CONFUSED, 0), 0.3)

        # Exclamation marks boost intensity
        excl_count = lower.count("!")
        if excl_count >= 2:
            for mood in scores:
                scores[mood] *= 1.3

        # Caps detection (shouting)
        if text.isupper() and len(text) > 5:
            scores[Mood.FRUSTRATED] = max(scores.get(Mood.FRUSTRATED, 0), 0.6)

        # Find winner
        if not scores or max(scores.values()) < 0.2:
            return Mood.NEUTRAL, 0.1

        best_mood = max(scores, key=scores.get)
        confidence = min(1.0, scores[best_mood])

        return best_mood, confidence

    def _score_patterns(self, text: str, patterns: list[str]) -> float:
        """Score how many patterns match. More matches = higher confidence."""
        matches = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
        if matches == 0:
            return 0.0
        return min(1.0, 0.3 + matches * 0.25)


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE ADVISOR — Generates tone/style hints for the system prompt
# ═══════════════════════════════════════════════════════════════════════════════

class ResponseAdvisor:
    """
    Generates response guidance based on detected mood.
    Output is injected into context so CARL adapts naturally.
    """

    def advise(self, mood: Mood, frustration_level: int = 0) -> str:
        """
        Generate a concise instruction for how CARL should respond.
        Returns empty string for neutral mood (no modification needed).
        """
        if mood == Mood.FRUSTRATED:
            if frustration_level >= 3:
                return (
                    "[TONE] User is very frustrated. Be extra patient. "
                    "Acknowledge the difficulty briefly. Don't repeat what failed. "
                    "Offer a clear alternative approach. Be direct and helpful."
                )
            return (
                "[TONE] User seems frustrated. Stay calm and patient. "
                "Acknowledge the issue. Focus on solutions, not explanations."
            )

        if mood == Mood.CONFUSED:
            return (
                "[TONE] User seems confused. Explain simply and clearly. "
                "Break down complex ideas. Ask if they'd like clarification."
            )

        if mood == Mood.URGENT:
            return (
                "[TONE] User is in a hurry. Be extremely concise. "
                "Skip pleasantries. Act immediately. No unnecessary words."
            )

        if mood == Mood.GRATEFUL:
            return (
                "[TONE] User expressed gratitude. Acknowledge briefly — "
                "'Glad I could help' or 'Of course, sir.' Then move on."
            )

        if mood == Mood.DISAPPOINTED:
            return (
                "[TONE] User seems disappointed. Acknowledge it briefly. "
                "Don't over-apologize. Offer to improve or try again."
            )

        if mood == Mood.TIRED:
            return (
                "[TONE] User seems tired. Keep responses short. "
                "Offer to handle things or suggest resuming later."
            )

        if mood == Mood.POSITIVE or mood == Mood.EXCITED:
            return ""  # No modification needed — CARL stays calm regardless

        return ""  # Neutral — no guidance needed


# ═══════════════════════════════════════════════════════════════════════════════
# EMOTIONAL INTELLIGENCE ENGINE — Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

class EmotionalIntelligence:
    """
    Main class integrated into CARL's pipeline.

    Usage:
        ei = EmotionalIntelligence()
        tone_hint = ei.process_input("this isn't working!")
        # → "[TONE] User seems frustrated. Stay calm..."
        # Inject tone_hint into Gemini context before generating response.
    """

    def __init__(self):
        self._detector = EmotionDetector()
        self._tracker = MoodTracker()
        self._advisor = ResponseAdvisor()

    def process_input(self, text: str) -> str:
        """
        Analyze user input and return a tone guidance string.
        Returns empty string if no adaptation needed (neutral mood).
        """
        mood, confidence = self._detector.detect(text)

        # Only track if confidence is meaningful
        if confidence >= 0.3:
            self._tracker.record(mood, confidence, text)

        # Generate advice based on detected mood + frustration history
        return self._advisor.advise(mood, self._tracker.frustration_level)

    @property
    def current_mood(self) -> str:
        return self._tracker.current_mood.value

    @property
    def frustration_level(self) -> int:
        return self._tracker.frustration_level

    def get_session_summary(self) -> dict:
        return self._tracker.get_session_summary()

    def reset(self):
        """Reset for new session."""
        self._tracker = MoodTracker()
