"""Read and update config/config.toml for the GUI (no Qt imports).

The file stays the single source of truth shared with the console workflow.
Updates go through tomlkit so the explanatory comments survive, are written
atomically, and a one-time ``config.toml.bak`` keeps the pre-GUI version.
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

# Mirrors transcript.DEFAULT_CONFIG (tests assert the two stay identical).
DEFAULTS = {
    "files": [],
    "model_size": "large-v3-turbo",
    "device": "cuda",
    "language": "uk",
    "output_mode": "all",
    "remove_fillers": False,
    "initial_prompt": "",
    "auto_preprocess_block_seconds": 60,
    "diarize": False,
    "hf_token": "",
    "min_speakers": 0,
    "max_speakers": 0,
    "batch_size": 8,
}
OUTPUT_MODES = ("all", "timecode", "simple", "srt")
DEVICES = ("cuda", "cpu")


class ConfigError(Exception):
    pass


def ensure_config(path: Path, example: Path) -> bool:
    """Seed a missing config from the bundled template; True when created."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(example.read_bytes())
    except FileExistsError:
        return False
    return True


def _coerce(key: str, value: object) -> object:
    """Return a usable value or raise ValueError; mirrors transcript.validate_config."""
    default = DEFAULTS[key]
    if key == "files":
        if not isinstance(value, list):
            raise ValueError
        return [item for item in value if isinstance(item, str) and item.strip()]
    if isinstance(default, bool):
        if type(value) is not bool:
            raise ValueError
        return value
    if key == "auto_preprocess_block_seconds":
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError
        return value
    if isinstance(default, int):
        minimum = 1 if key == "batch_size" else 0
        if type(value) is not int or value < minimum:
            raise ValueError
        return value
    if not isinstance(value, str):
        raise ValueError
    if key == "output_mode":
        value = value.lower()
        if value not in OUTPUT_MODES:
            raise ValueError
    elif key == "device" and value not in DEVICES:
        raise ValueError
    elif key in ("model_size", "language") and not value.strip():
        raise ValueError
    return value


def load_settings(path: Path) -> tuple[dict, list[str]]:
    """Return (settings with defaults applied, keys whose stored value was unusable)."""
    import tomlkit

    try:
        document = tomlkit.parse(path.read_text(encoding="utf-8-sig"))
    except OSError as error:
        raise ConfigError(str(error)) from error
    except Exception as error:  # tomlkit raises its own ParseError family
        raise ConfigError(str(error)) from error
    raw = document.unwrap()
    settings, problems = dict(DEFAULTS), []
    settings["files"] = []
    for key in DEFAULTS:
        if key not in raw:
            continue
        try:
            settings[key] = _coerce(key, raw[key])
        except (ValueError, OverflowError):
            problems.append(key)
    return settings, problems


def save_settings(path: Path, values: dict) -> bool:
    """Write the given keys into the config, keeping comments; True if the file changed."""
    import tomlkit

    try:
        raw_bytes = path.read_bytes()
        crlf = b"\r\n" in raw_bytes
        original = raw_bytes.decode("utf-8-sig").replace("\r\n", "\n")
        document = tomlkit.parse(original)
    except OSError as error:
        raise ConfigError(str(error)) from error
    except Exception as error:
        raise ConfigError(str(error)) from error
    stored = document.unwrap()
    for key, value in values.items():
        if key not in DEFAULTS:
            continue
        value = _coerce(key, value)
        if key in stored and type(stored[key]) is type(value) and stored[key] == value:
            continue
        if key == "files":
            array = tomlkit.array()
            for item in value:
                array.append(item)
            array.multiline(bool(value))
            value = array
        document[key] = value
    updated = tomlkit.dumps(document)
    if updated == original:
        return False
    backup = path.with_name(path.name + ".bak")
    if not backup.exists():
        try:
            shutil.copy2(path, backup)
        except OSError:
            pass
    temporary = path.with_name(path.name + ".gui.tmp")
    try:
        # Keep the file's existing line-ending style (Windows editors use CRLF).
        temporary.write_text(updated, encoding="utf-8", newline="\r\n" if crlf else "\n")
        os.replace(temporary, path)
    except OSError as error:
        raise ConfigError(str(error)) from error
    return True
