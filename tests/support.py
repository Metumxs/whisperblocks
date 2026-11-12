"""Offline test fixtures; no ML/runtime packages, media or network required."""

import importlib
import io
import json
import socket
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock


PROJECT_SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_SOURCE / "src"))


class PlainText(str):
    def __new__(cls, text="", **kwargs):
        return super().__new__(cls, text)


class PlainTable:
    def __init__(self, *args, **kwargs):
        self.rows = []

    def add_column(self, *args, **kwargs):
        pass

    def add_row(self, *values):
        self.rows.append(values)

    def __str__(self):
        return "\n".join(" | ".join(map(str, row)) for row in self.rows)


class PlainPanel:
    def __init__(self, renderable, title="", **kwargs):
        self.renderable = renderable
        self.title = title

    def __str__(self):
        return f"{self.title}\n{self.renderable}"


class RecordingConsole:
    def __init__(self, *args, **kwargs):
        self.file = io.StringIO()

    def print(self, *args, **kwargs):
        print(*args, file=self.file)

    def rule(self, *args, **kwargs):
        self.print(*args)

    def status(self, *args, **kwargs):
        return nullcontext()

    @property
    def output(self):
        return self.file.getvalue()


# The production UI remains Rich. Import under minimal UI stubs so running
# unittest itself needs only Python, even without installing runtime packages.
rich_stubs = {name: types.ModuleType(name) for name in (
    "rich", "rich.console", "rich.panel", "rich.table", "rich.text",
)}
rich_stubs["rich.console"].Console = RecordingConsole
rich_stubs["rich.panel"].Panel = PlainPanel
rich_stubs["rich.table"].Table = PlainTable
rich_stubs["rich.text"].Text = PlainText
previous_rich_modules = {name: sys.modules.get(name) for name in rich_stubs}
sys.modules.update(rich_stubs)
try:
    transcript = importlib.import_module("transcript")
finally:
    # Restore only UI entries, not new standard-library modules imported by
    # transcript.py (restoring all sys.modules would create duplicate modules).
    for name, previous in previous_rich_modules.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
preprocess = importlib.import_module("preprocess")
file_io = importlib.import_module("file_io")


def fake_torch(*, available=True, cuda_version="12.8"):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=mock.Mock(return_value=available), empty_cache=mock.Mock(),
        ),
        version=types.SimpleNamespace(cuda=cuda_version), __version__="2.8.0+test",
    )


def transcript_result():
    return {"language": "uk", "segments": [
        {"start": 0.0, "end": 2.005, "text": " Тестовий текст. "},
    ]}


def toml_value(value):
    if type(value) is bool:
        return "true" if value else "false"
    if isinstance(value, str) or isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value).lower()


class ProjectTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="whisperblocks-test-")
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.root = self.workspace / "project"
        self.root.mkdir()
        for folder in ("input", "output", "config"):
            (self.root / folder).mkdir()
        self.console = RecordingConsole()
        self.patch(transcript, "PROJECT_ROOT", self.root)
        self.patch(transcript, "CONFIG_PATH", self.root / "config/config.toml")
        self.patch(transcript, "CONFIG_EXAMPLE_PATH", PROJECT_SOURCE / "config/config.toml.example")
        self.patch(transcript, "console", self.console)
        self.patch(transcript, "torch", None)
        self.patch(transcript, "whisperx", None)
        environment = mock.patch.dict("os.environ", {"HF_TOKEN": ""})
        environment.start()
        self.addCleanup(environment.stop)
        # A failed boundary mock must never turn into a real network request.
        self.patch(socket.socket, "connect", mock.Mock(side_effect=AssertionError("Network forbidden in tests")))
        self.patch(socket, "create_connection", mock.Mock(side_effect=AssertionError("Network forbidden in tests")))

    def patch(self, target, name, value):
        patcher = mock.patch.object(target, name, value)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def media(self, relative="input/a.mp3"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic offline fixture, not real audio")
        return path.resolve()

    def config(self, **updates):
        raw = {"files": ["input/a.mp3"], "auto_preprocess_block_seconds": 0, **updates}
        return transcript.validate_config(raw, environ={})

    def write_config(self, *, path=None, **updates):
        path = self.root / "config/config.toml" if path is None else path
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = {"files": ["input/a.mp3"], "auto_preprocess_block_seconds": 0, **updates}
        path.write_text("\n".join(f"{key} = {toml_value(value)}" for key, value in raw.items()) + "\n", encoding="utf-8")
        return path

    def plan(self, **updates):
        return transcript.build_preflight(self.config(**updates), project_root=self.root)

    def run_main(self, args=None, *, available=True, result=None):
        runtime = fake_torch(available=available)
        self.patch(transcript, "load_torch", mock.Mock(return_value=runtime))
        boundary = self.patch(transcript, "transcribe_file", mock.Mock(
            return_value=transcript_result() if result is None else result,
        ))
        summaries = []
        original = transcript.display_summary

        def capture(counts):
            summaries.append(transcript.BatchCounts(**vars(counts)))
            original(counts)

        self.patch(transcript, "display_summary", capture)
        code = transcript.main([] if args is None else args)
        return code, boundary, summaries
