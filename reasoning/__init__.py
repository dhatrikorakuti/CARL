"""
CARL Reasoning Engine — Phase 16

Internal planning and execution system that transforms complex goals
into structured multi-step plans, executes them safely, and tracks progress.

Architecture:
    engine.py        — Main entry point (ReasoningEngine)
    models.py        — Data classes (Plan, Step, ExecutionLevel, etc.)
    planner.py       — LLM-based goal decomposition
    tool_resolver.py — Maps capabilities to action modules
    safety.py        — Validates destructive actions
    task_manager.py  — Plan state machine (pause/resume/cancel)
    executor.py      — Step-by-step execution with retries
    progress.py      — Natural language progress reporting
    history.py       — Mission history persistence
"""

from reasoning.models import (
    ExecutionLevel,
    Step,
    StepStatus,
    Plan,
    PlanResult,
    MissionRecord,
)
from reasoning.engine import ReasoningEngine

__all__ = [
    "ReasoningEngine",
    "ExecutionLevel",
    "Step",
    "StepStatus",
    "Plan",
    "PlanResult",
    "MissionRecord",
]
