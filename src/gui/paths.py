"""Project locations shared by the GUI modules (no Qt, no ML imports)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "config.toml"
CONFIG_EXAMPLE_PATH = CONFIG_DIR / "config.toml.example"
PROMPTS_PATH = CONFIG_DIR / "prompts.toml"
STATE_PATH = CONFIG_DIR / "gui-state.ini"
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs"
TRANSCRIPT_SCRIPT = SRC_DIR / "transcript.py"
PROBE_SCRIPT = Path(__file__).resolve().parent / "probe.py"
INSTALL_SCRIPT = PROJECT_ROOT / "install.bat"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

MEDIA_EXTENSIONS = (
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".3gp",
)


def is_media(path: Path) -> bool:
    return path.suffix.lower() in MEDIA_EXTENSIONS


def python_executable() -> Path:
    """Console interpreter of the running environment (never pythonw)."""
    executable = Path(sys.executable)
    if executable.name.lower() == "pythonw.exe":
        candidate = executable.with_name("python.exe")
        if candidate.exists():
            return candidate
    return executable


def bundled_ffmpeg_dirs(root: Path | None = None) -> list[Path]:
    """Optional portable ffmpeg next to the app; used before the system PATH."""
    root = PROJECT_ROOT if root is None else root
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    candidates = (root / "tools" / "ffmpeg" / "bin", root / "tools" / "ffmpeg", root / "ffmpeg" / "bin", root)
    return [directory for directory in candidates if (directory / name).is_file()]


def display_path(path: Path, root: Path | None = None) -> str:
    """Config-friendly spelling: project-relative with forward slashes when inside the project."""
    root = PROJECT_ROOT if root is None else root
    try:
        resolved = path.resolve()
        return resolved.relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def resolve_requested(requested: str, root: Path | None = None) -> Path:
    """Mirror transcript.resolve_media_path closely enough for display purposes."""
    root = PROJECT_ROOT if root is None else root
    path = Path(requested).expanduser()
    if path.is_absolute():
        return path
    direct = root / path
    if direct.exists() or path.parts[:1] == ("input",):
        return direct
    nested = root / "input" / path
    return nested if nested.exists() else direct
