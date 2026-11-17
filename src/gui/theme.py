"""Look and feel: Fusion style, system/light/dark scheme, palette-based QSS."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import QApplication

from . import paths

ACCENT = "#2f6fed"
ACCENT_HOVER = "#255fd6"
ACCENT_PRESSED = "#1d4fb8"
DANGER = "#d64545"

STATUS_COLORS = {
    "pending": "#9aa0a6", "ready": "#5b8def", "running": ACCENT, "done": "#2e9d5b",
    "skipped": "#d9962b", "failed": DANGER, "missing": DANGER, "cancelled": "#9aa0a6",
}

_dot_cache: dict[str, QIcon] = {}

SCHEMES = {
    "system": Qt.ColorScheme.Unknown, "light": Qt.ColorScheme.Light, "dark": Qt.ColorScheme.Dark,
}

def build_style(dark: bool) -> str:
    warning_text = "#f2b963" if dark else "#8a5200"
    return f"""
QFrame#card {{
    background: palette(base);
    border: 1px solid palette(mid);
    border-radius: 10px;
}}
QLabel#cardTitle {{ font-weight: 600; font-size: 11pt; }}
QLabel#appTitle {{ font-weight: 700; font-size: 14pt; }}
QLabel#hint {{ color: palette(placeholder-text); }}
QLabel#warning {{
    color: {warning_text}; background: rgba(217, 150, 43, 0.14);
    border: 1px solid rgba(217, 150, 43, 0.5); border-radius: 6px; padding: 6px 8px;
}}
QLabel#statusText {{ font-weight: 600; }}
QPushButton {{
    padding: 6px 14px; border-radius: 6px; border: 1px solid palette(mid); background: palette(button);
}}
QPushButton:hover {{ background: palette(midlight); }}
QPushButton:pressed {{ background: palette(mid); }}
QPushButton:disabled {{ color: palette(placeholder-text); border-color: palette(midlight); }}
QToolButton {{ padding: 6px 12px; border-radius: 6px; border: 1px solid palette(mid); background: palette(button); }}
QToolButton:hover {{ background: palette(midlight); }}
QPushButton#primary {{
    background: {ACCENT}; color: white; border: none; font-weight: 600; padding: 8px 22px;
}}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#primary:pressed {{ background: {ACCENT_PRESSED}; }}
QPushButton#primary:disabled {{ background: palette(mid); color: palette(placeholder-text); }}
QPushButton#danger {{
    background: {DANGER}; color: white; border: none; font-weight: 600; padding: 8px 22px;
}}
QToolButton#chip {{
    border: 1px solid palette(mid); border-radius: 13px; padding: 5px 12px; background: palette(base);
}}
QToolButton#chip::menu-indicator {{ image: none; }}
QToolButton#flat {{ border: none; background: transparent; padding: 4px 6px; }}
QToolButton#flat:hover {{ background: palette(midlight); }}
QToolButton#sectionToggle {{
    border: none; background: transparent; font-weight: 600; font-size: 11pt; padding: 2px 0;
}}
QTreeWidget {{ border: 1px solid palette(mid); border-radius: 10px; padding: 4px; }}
QTreeWidget::item {{ padding: 5px 2px; }}
QListWidget {{ border: 1px solid palette(mid); border-radius: 8px; padding: 4px; }}
QListWidget::item {{ padding: 6px 4px; border-radius: 4px; }}
QPlainTextEdit#log {{ border: 1px solid palette(mid); border-radius: 8px; }}
QProgressBar {{
    border: none; border-radius: 4px; background: palette(midlight); max-height: 8px; min-height: 8px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QScrollArea {{ border: none; background: transparent; }}
QComboBox, QLineEdit, QSpinBox {{ padding: 4px 6px; min-height: 22px; }}
"""


def is_dark(app: QApplication) -> bool:
    return app.palette().color(QPalette.ColorRole.Window).lightness() < 128


def apply(app: QApplication, scheme: str = "system") -> None:
    app.setStyle("Fusion")
    set_scheme(scheme)
    refresh_style(app)
    hints = QGuiApplication.styleHints()
    if hasattr(hints, "colorSchemeChanged"):
        hints.colorSchemeChanged.connect(lambda _scheme: refresh_style(app))


def link_color() -> str:
    app = QApplication.instance()
    return "#7fb0ff" if app is not None and is_dark(app) else ACCENT


def refresh_style(app: QApplication) -> None:
    _dot_cache.clear()
    app.setStyleSheet(build_style(is_dark(app)))


def set_scheme(scheme: str) -> None:
    hints = QGuiApplication.styleHints()
    if hasattr(hints, "setColorScheme"):   # Qt 6.8+
        hints.setColorScheme(SCHEMES.get(scheme, Qt.ColorScheme.Unknown))


def app_icon() -> QIcon:
    icon = QIcon()
    for name in ("icon.ico", "icon.png"):
        path = paths.ASSETS_DIR / name
        if path.exists():
            icon.addFile(str(path))
    return icon


def status_dot(status: str) -> QIcon:
    """Small filled circle used as the per-file state marker."""
    if status not in _dot_cache:
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(STATUS_COLORS.get(status, STATUS_COLORS["pending"]))
        painter.setPen(Qt.PenStyle.NoPen)
        if status == "pending":
            painter.setBrush(Qt.BrushStyle.NoBrush)
            pen = painter.pen()
            pen.setStyle(Qt.PenStyle.SolidLine)
            pen.setColor(color)
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawEllipse(8, 8, 16, 16)
        else:
            painter.setBrush(color)
            painter.drawEllipse(7, 7, 18, 18)
        painter.end()
        _dot_cache[status] = QIcon(pixmap)
    return _dot_cache[status]
