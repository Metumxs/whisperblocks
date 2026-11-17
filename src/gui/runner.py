"""Runs transcript.py (and the environment probe) as child processes."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from . import events, paths


def child_environment() -> QProcessEnvironment:
    environment = QProcessEnvironment.systemEnvironment()
    for name, value in (
        ("PYTHONIOENCODING", "utf-8"), ("PYTHONUTF8", "1"), ("PYTHONUNBUFFERED", "1"),
        ("NO_COLOR", "1"), ("COLUMNS", "110"), ("HF_HUB_DISABLE_SYMLINKS_WARNING", "1"),
    ):
        environment.insert(name, value)
    extra = [str(directory) for directory in paths.bundled_ffmpeg_dirs()]
    if extra:
        environment.insert("PATH", os.pathsep.join([*extra, environment.value("PATH", "")]))
    return environment


class LineProcess(QObject):
    """QProcess wrapper emitting decoded lines; stdout and stderr are merged."""

    line = Signal(str)
    finished = Signal(int, bool)   # exit code, crashed-or-killed
    failed_to_start = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self._buffer = ""
        self.killed = False

    def start(self, arguments: list[str], *, extra_env: dict[str, str] | None = None) -> None:
        environment = child_environment()
        for name, value in (extra_env or {}).items():
            environment.insert(name, value)
        self._buffer, self.killed = "", False
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(paths.PROJECT_ROOT))
        self.process.start(str(paths.python_executable()), arguments)

    def is_running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def kill(self) -> None:
        """Stop the whole process tree (ffmpeg children included on Windows)."""
        if not self.is_running():
            return
        self.killed = True
        pid = self.process.processId()
        if os.name == "nt" and pid:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=15,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except (OSError, subprocess.SubprocessError):
                pass
        self.process.kill()

    def _read(self) -> None:
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        lines, self._buffer = events.split_stream(self._buffer + data)
        for text in lines:
            self.line.emit(text)

    def _finished(self, code: int, status: QProcess.ExitStatus) -> None:
        self._read()
        if self._buffer.strip():
            self.line.emit(self._buffer.rsplit("\r", 1)[-1])
        self._buffer = ""
        self.finished.emit(code, self.killed or status == QProcess.ExitStatus.CrashExit)

    def _error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.failed_to_start.emit(self.process.errorString())


class EnvironmentProbe(QObject):
    """One-shot torch/CUDA/ffmpeg check; result is a dict (see probe.py)."""

    result = Signal(dict)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.runner = LineProcess(self)
        self.runner.line.connect(self._line)
        self.runner.finished.connect(self._done)
        self.runner.failed_to_start.connect(lambda message: self._deliver({"error": message}))
        self._info: dict | None = None
        self._delivered = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._timeout)

    def start(self) -> None:
        self._info, self._delivered = None, False
        self.runner.start([str(paths.PROBE_SCRIPT)])
        self._timer.start(120_000)

    def _line(self, text: str) -> None:
        if text.startswith("@@PROBE "):
            try:
                self._info = json.loads(text[len("@@PROBE "):])
            except ValueError:
                pass

    def _done(self, code: int, crashed: bool) -> None:
        self._deliver(self._info if self._info is not None else {"error": f"probe exited with code {code}"})

    def _timeout(self) -> None:
        self.runner.kill()
        self._deliver({"error": "probe timed out"})

    def _deliver(self, info: dict) -> None:
        if self._delivered:
            return
        self._delivered = True
        self._timer.stop()
        self.result.emit(info)


def open_in_file_manager(path: Path) -> None:
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices

    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def run_install_script() -> bool:
    """Open install.bat in its own console window (Windows only)."""
    if os.name != "nt" or not paths.INSTALL_SCRIPT.exists():
        return False
    subprocess.Popen(
        ["cmd.exe", "/c", str(paths.INSTALL_SCRIPT)], cwd=str(paths.PROJECT_ROOT),
        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
    )
    return True
