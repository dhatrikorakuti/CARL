"""
actions/downloads.py — Managed file downloads for CARL.

Downloads files, images, PDFs, repos from URLs.
Saves to ~/Downloads with progress reporting.
"""
from __future__ import annotations
import os
import platform
import subprocess
import urllib.request
import urllib.parse
import time
from pathlib import Path

_DOWNLOADS_DIR = Path.home() / "Downloads"


def download_file(parameters: dict, player=None) -> str:
    url = parameters.get("url", "")
    filename = parameters.get("filename", "")
    action = parameters.get("action", "download")

    if not url:
        return "No URL provided. Please specify what to download."

    if action == "github" or "github.com" in url:
        return _clone_repo(url)

    return _download(url, filename)


def _download(url: str, filename: str = "") -> str:
    """Download a file from URL to Downloads folder."""
    try:
        # Determine filename
        if not filename:
            parsed = urllib.parse.urlparse(url)
            filename = os.path.basename(parsed.path)
            if not filename or "." not in filename:
                filename = f"download_{int(time.time())}"

        dest = _DOWNLOADS_DIR / filename
        # Avoid overwrite
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            counter = 1
            while dest.exists():
                dest = _DOWNLOADS_DIR / f"{stem}_{counter}{suffix}"
                counter += 1

        # Download with progress
        start = time.time()
        urllib.request.urlretrieve(url, str(dest))
        duration = time.time() - start

        size = dest.stat().st_size
        size_str = _format_size(size)

        return f"Downloaded: {dest.name} ({size_str}) in {duration:.1f}s. Saved to Downloads."

    except urllib.error.HTTPError as e:
        return f"Download failed: HTTP {e.code} — {e.reason}"
    except urllib.error.URLError as e:
        return f"Download failed: Cannot reach URL — {e.reason}"
    except Exception as e:
        return f"Download failed: {e}"


def _clone_repo(url: str) -> str:
    """Clone a GitHub repository."""
    try:
        # Extract repo name
        parts = url.rstrip("/").split("/")
        repo_name = parts[-1].replace(".git", "") if parts else "repo"
        dest = _DOWNLOADS_DIR / repo_name

        if dest.exists():
            return f"Repository '{repo_name}' already exists in Downloads."

        r = subprocess.run(
            ["git", "clone", url, str(dest)],
            capture_output=True, text=True, timeout=120
        )
        if r.returncode == 0:
            return f"Cloned '{repo_name}' to Downloads."
        return f"Git clone failed: {r.stderr[:100]}"
    except FileNotFoundError:
        return "Git is not installed. Install Git from https://git-scm.com"
    except Exception as e:
        return f"Clone failed: {e}"


def _format_size(size: int) -> str:
    if size < 1024: return f"{size} B"
    elif size < 1024**2: return f"{size/1024:.1f} KB"
    elif size < 1024**3: return f"{size/1024**2:.1f} MB"
    else: return f"{size/1024**3:.1f} GB"
