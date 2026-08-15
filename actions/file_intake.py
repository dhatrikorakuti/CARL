"""
actions/file_intake.py — Intelligent File & Folder Intake for CARL.

CRITICAL FIX: QFileDialog CANNOT be called from asyncio executor threads.
It causes deadlocks. Instead, this module signals the UI to open the picker
and returns immediately. The UI handles the dialog on its own thread.

If the UI is unavailable, CARL asks the user to provide the path verbally.
"""
from __future__ import annotations

import os
import platform
import threading
from pathlib import Path
from typing import Optional

_OS = platform.system()

# Folders to ignore when scanning
_IGNORE_FOLDERS = {
    "node_modules", ".venv", "venv", ".git", "__pycache__",
    "build", "dist", ".cache", ".tox", ".mypy_cache",
    ".pytest_cache", "env", ".env", ".idea", ".vs",
}

# Type detection keywords
_TYPE_KEYWORDS = {
    "dataset": ["dataset", "data", "csv", "excel", "spreadsheet", "xlsx", "parquet", "table"],
    "image": ["image", "picture", "photo", "screenshot", "scan"],
    "pdf": ["pdf"],
    "document": ["document", "doc", "report", "paper", "file"],
    "code": ["code", "script", "source", "python", "javascript", "program", "repository"],
    "video": ["video", "clip", "recording", "mp4"],
    "audio": ["audio", "music", "song", "recording", "mp3", "wav"],
    "archive": ["zip", "archive", "compressed", "rar"],
    "presentation": ["presentation", "slides", "powerpoint", "pptx"],
    "word": ["word", "docx"],
    "excel": ["excel", "xlsx", "xls", "spreadsheet"],
}


def file_intake(parameters: dict, player=None, speak=None) -> str:
    """
    Handle file upload requests. Returns IMMEDIATELY — never blocks.

    Strategy:
    1. If user provided a path in their request → use it directly
    2. Otherwise → ask user to provide the path or drag-drop the file

    This function NEVER opens QFileDialog (which causes deadlocks).
    """
    mode = parameters.get("mode", "file").lower()
    file_type = parameters.get("file_type", "any").lower()
    description = parameters.get("description", "")
    path = parameters.get("path", "")

    print(f"[TOOL] file_intake called: mode={mode}, type={file_type}, path='{path}'")

    # Auto-detect type
    if file_type == "any" and description:
        file_type = _detect_type(description)

    # If a path was provided, process it directly
    if path:
        return _process_path(path)

    # Check common locations for recently added files
    recent = _check_recent_downloads(file_type)
    if recent:
        return f"I found a recent file in Downloads: {recent.name} ({_format_size(recent.stat().st_size)}). Would you like me to use this? Path: {recent}"

    # No path provided — ask user naturally
    type_hint = f" ({file_type})" if file_type != "any" else ""
    if mode == "folder":
        return "Please tell me the folder path, or drag and drop it into the window. For example: 'Use C:\\Users\\Admin\\Documents\\MyProject'"
    else:
        return f"Please provide the file path{type_hint}, or drag and drop it into the window. You can also say the full path."


def _process_path(path: str) -> str:
    """Process a file or folder path that was provided."""
    p = Path(path)

    if not p.exists():
        print(f"[TOOL] Path does not exist: {path}")
        return f"Path not found: {path}. Please check the path and try again."

    if p.is_dir():
        summary = _scan_folder(p)
        return f"Folder received: {p.name}\n{summary}\nPath: {path}"
    else:
        try:
            size = p.stat().st_size
            size_str = _format_size(size)
            ext = p.suffix.lower()
            print(f"[TOOL] File processed: {p.name} ({size_str})")
            return f"File received: {p.name} ({size_str}, {ext})\nPath: {path}"
        except Exception as e:
            return f"ERROR: Cannot access file: {e}"


def _check_recent_downloads(file_type: str) -> Optional[Path]:
    """Check Downloads folder for recently added files matching the type."""
    downloads = Path.home() / "Downloads"
    if not downloads.exists():
        return None

    ext_map = {
        "dataset": {".csv", ".xlsx", ".xls", ".json", ".parquet", ".tsv"},
        "excel": {".xlsx", ".xls", ".csv"},
        "image": {".png", ".jpg", ".jpeg", ".gif", ".webp"},
        "pdf": {".pdf"},
        "code": {".py", ".js", ".ts", ".java", ".cpp"},
    }

    target_exts = ext_map.get(file_type, set())
    if not target_exts:
        return None

    import time
    recent_threshold = 3600  # last hour

    try:
        candidates = []
        for f in downloads.iterdir():
            if f.is_file() and f.suffix.lower() in target_exts:
                age = time.time() - f.stat().st_mtime
                if age < recent_threshold:
                    candidates.append((age, f))
        if candidates:
            candidates.sort()
            return candidates[0][1]  # Most recent
    except Exception:
        pass

    return None


def _detect_type(text: str) -> str:
    """Detect file type from natural language description."""
    lower = text.lower()
    for ftype, keywords in _TYPE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return ftype
    return "any"


def _scan_folder(folder: Path) -> str:
    """Recursively scan a folder and summarize contents."""
    stats = {"files": 0, "folders": 0, "types": {}}
    total_size = 0

    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in _IGNORE_FOLDERS]
        stats["folders"] += len(dirs)
        for f in files:
            stats["files"] += 1
            ext = Path(f).suffix.lower()
            if ext:
                stats["types"][ext] = stats["types"].get(ext, 0) + 1
            try:
                total_size += (Path(root) / f).stat().st_size
            except OSError:
                pass

    top_types = sorted(stats["types"].items(), key=lambda x: -x[1])[:5]
    types_str = ", ".join(f"{ext} ({count})" for ext, count in top_types)

    return (
        f"Contents: {stats['files']} files, {stats['folders']} subfolders, "
        f"{_format_size(total_size)} total\n"
        f"File types: {types_str}"
    )


def _format_size(size: int) -> str:
    if size < 1024: return f"{size} B"
    elif size < 1024**2: return f"{size/1024:.1f} KB"
    elif size < 1024**3: return f"{size/1024**2:.1f} MB"
    else: return f"{size/1024**3:.1f} GB"
