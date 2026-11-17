"""Progress protocol between transcript.py and the GUI (no Qt imports).

transcript.py prints ``@@WB {json}`` lines when WHISPERBLOCKS_EVENTS=1.
"""

from __future__ import annotations

import json

PREFIX = "@@WB "
ENV_NAME = "WHISPERBLOCKS_EVENTS"

# Share of one file's work per stage; diarization only counts when enabled.
STAGE_ORDER = ("load", "transcribe", "align", "diarize")
_WEIGHTS = {"load": 0.08, "transcribe": 0.62, "align": 0.30, "diarize": 0.0}
_WEIGHTS_DIARIZE = {"load": 0.06, "transcribe": 0.46, "align": 0.20, "diarize": 0.28}


def parse_line(line: str) -> dict | None:
    """Return the event dict for a protocol line, else None (ordinary log text)."""
    if not line.startswith(PREFIX):
        return None
    try:
        data = json.loads(line[len(PREFIX):])
    except ValueError:
        return None
    return data if isinstance(data, dict) and isinstance(data.get("event"), str) else None


def split_stream(buffer: str) -> tuple[list[str], str]:
    """Split decoded process output into complete lines and the unfinished tail.

    A carriage return rewrites the current line (tqdm download bars), so only
    the text after the last ``\\r`` of each line is kept.
    """
    *lines, tail = buffer.split("\n")
    cleaned = []
    for line in lines:
        line = line.rstrip("\r")
        if "\r" in line:
            line = line.rsplit("\r", 1)[1]
        cleaned.append(line)
    return cleaned, tail


class ProgressModel:
    """Turns file/stage/percent events into one overall 0..1 fraction."""

    def __init__(self, *, diarize: bool = False) -> None:
        self.weights = _WEIGHTS_DIARIZE if diarize else _WEIGHTS
        self.total = 0
        self.index = 0          # 1-based index of the file in progress
        self.finished = 0
        self.stage = ""
        self.percent = 0

    def apply(self, event: dict) -> None:
        kind = event.get("event")
        if kind == "plan":
            self.total = sum(1 for item in event.get("items", []) if item.get("status") == "runnable")
        elif kind == "file_start":
            self.index = int(event.get("index", self.index + 1))
            self.total = int(event.get("total", self.total)) or self.total
            self.stage, self.percent = "", 0
        elif kind == "stage":
            self.stage, self.percent = str(event.get("stage", "")), 0
        elif kind == "progress":
            self.stage = str(event.get("stage", self.stage))
            self.percent = max(0, min(100, int(event.get("percent", 0))))
        elif kind == "file_done":
            self.finished += 1
            self.stage, self.percent = "", 0
        elif kind == "summary":
            self.finished = self.total

    def file_fraction(self) -> float:
        if self.stage not in self.weights:
            return 0.0
        done = 0.0
        for name in STAGE_ORDER:
            if name == self.stage:
                return done + self.weights[name] * self.percent / 100
            done += self.weights[name]
        return done

    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return max(0.0, min(1.0, (min(self.finished, self.total) + (
            self.file_fraction() if self.finished < self.total else 0.0
        )) / self.total))
