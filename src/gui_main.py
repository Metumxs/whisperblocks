"""WhisperBlocks GUI entry point.

Run with the project's virtual environment, normally through
WhisperBlocks.exe or run-gui.bat:

    .venv\\Scripts\\pythonw.exe src\\gui_main.py [media files...]
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def report_fatal(message: str) -> None:
    """Startup failures must be visible even under pythonw (no console)."""
    try:
        log_dir = SRC_DIR.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        (log_dir / "gui-error.log").write_text(message, encoding="utf-8")
    except OSError:
        pass
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message[-1800:], "WhisperBlocks", 0x10)
            return
        except Exception:
            pass
    if sys.stderr is not None:
        print(message, file=sys.stderr)


def main() -> int:
    try:
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication

        import tomlkit  # noqa: F401  (fail early with a clear message)
        from gui import i18n, paths, theme
        from gui.main_window import MainWindow
    except ImportError as error:
        report_fatal(
            "Не вдалося запустити інтерфейс WhisperBlocks: бракує компонентів.\n"
            "The WhisperBlocks GUI components are missing.\n\n"
            "Запустіть run-gui.bat або виконайте / run:\n"
            "  .venv\\Scripts\\python.exe -m pip install -r requirements\\requirements-gui.txt\n\n"
            f"{type(error).__name__}: {error}"
        )
        return 1

    if os.name == "nt":
        try:
            import ctypes

            # Own taskbar identity (icon/grouping) instead of the generic pythonw one.
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("WhisperBlocks.GUI")
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("WhisperBlocks")
    app.setWindowIcon(theme.app_icon())
    for directory in (paths.CONFIG_DIR, paths.INPUT_DIR, paths.OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    state = QSettings(str(paths.STATE_PATH), QSettings.Format.IniFormat)
    i18n.set_language(str(state.value("ui_language", "uk")))
    theme.apply(app, str(state.value("theme", "system")))

    windows: list[MainWindow] = []

    def build(initial_files: list[str] | None = None) -> None:
        window = MainWindow(initial_files)
        window.rebuild_requested.connect(rebuild)
        windows.append(window)
        window.show()

    def rebuild() -> None:
        # Interface language changed: recreate the window with the new strings.
        old = windows.pop()
        build()
        old.config_ok = False      # the new window owns the config now
        old.close()
        old.deleteLater()

    build([argument for argument in sys.argv[1:] if not argument.startswith("-")])
    return app.exec()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        report_fatal("WhisperBlocks GUI crashed:\n\n" + traceback.format_exc())
        raise SystemExit(1)
