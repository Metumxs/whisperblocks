"""Regenerate the app icon (src/gui/assets/icon.png + icon.ico).

Usage (needs PySide6 and Pillow):  python launcher/make_icon.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QLinearGradient, QPainter

ASSETS = Path(__file__).resolve().parents[1] / "src" / "gui" / "assets"
BARS = (0.30, 0.55, 0.85, 0.50, 0.70, 0.35)   # waveform heights


def render(size: int) -> QImage:
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    gradient = QLinearGradient(0, 0, size, size)
    gradient.setColorAt(0, QColor("#4a8bff"))
    gradient.setColorAt(1, QColor("#1f4fc4"))
    painter.setBrush(gradient)
    painter.setPen(Qt.PenStyle.NoPen)
    radius = size * 0.22
    painter.drawRoundedRect(QRectF(0, 0, size, size), radius, radius)
    painter.setBrush(QColor("white"))
    margin, gap = size * 0.19, size * 0.045
    width = (size - 2 * margin - gap * (len(BARS) - 1)) / len(BARS)
    for index, height in enumerate(BARS):
        bar = size * 0.62 * height
        x = margin + index * (width + gap)
        painter.drawRoundedRect(QRectF(x, (size - bar) / 2, width, bar), width / 2, width / 2)
    painter.end()
    return image


def main() -> int:
    app = QGuiApplication(sys.argv)
    ASSETS.mkdir(parents=True, exist_ok=True)
    png = ASSETS / "icon.png"
    render(256).save(str(png))
    Image.open(png).save(ASSETS / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("written", png, ASSETS / "icon.ico")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
