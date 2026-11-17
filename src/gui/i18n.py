"""Two-language UI strings: every call site passes both spellings."""

from __future__ import annotations

LANGUAGES = {"uk": "Українська", "en": "English"}
_current = "uk"


def set_language(code: str) -> None:
    global _current
    _current = code if code in LANGUAGES else "uk"


def language() -> str:
    return _current


def t(uk: str, en: str) -> str:
    return en if _current == "en" else uk
