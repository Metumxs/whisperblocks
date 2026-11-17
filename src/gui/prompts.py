"""Decoder prompt library: three locked built-ins plus the user's own (no Qt).

Custom prompts live in config/prompts.toml, so updating the app never
overwrites them. Built-in prompts cannot be edited or deleted; duplicate one
to get an editable copy.
"""

from __future__ import annotations

import os
import tomllib
import uuid
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Prompt:
    id: str
    name: str
    language: str   # ISO code, or "" for "any language"
    text: str
    builtin: bool = False


BUILTIN_PROMPTS = (
    Prompt("builtin-uk", "Лекція · українська", "uk",
           "Це лекція. Грамотний текст українською мовою без слів-паразитів: ", True),
    Prompt("builtin-en", "Lecture · English", "en",
           "This is a lecture. Well-written English text without filler words: ", True),
    Prompt("builtin-ru", "Лекция · русский", "ru",
           "Это лекция. Грамотный текст на русском языке без слов-паразитов: ", True),
)


class PromptError(ValueError):
    pass


class PromptStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.custom: list[Prompt] = []
        self.load_error = ""
        self.load()

    # ── reading ────────────────────────────────────────────────────────
    def load(self) -> None:
        self.custom, self.load_error = [], ""
        try:
            with self.path.open("rb") as handle:
                raw = tomllib.load(handle)
        except FileNotFoundError:
            return
        except (OSError, ValueError) as error:
            self.load_error = str(error)
            return
        seen = {prompt.id for prompt in BUILTIN_PROMPTS}
        entries = raw.get("prompt", [])
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            text, name = entry.get("text"), entry.get("name")
            if not isinstance(text, str) or not text.strip() or not isinstance(name, str):
                continue
            identifier = entry.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                identifier = self._new_id(seen)
            seen.add(identifier)
            language = entry.get("language", "")
            self.custom.append(Prompt(
                identifier, name.strip() or "—",
                language.strip().lower() if isinstance(language, str) else "", text,
            ))

    @property
    def prompts(self) -> list[Prompt]:
        return [*BUILTIN_PROMPTS, *self.custom]

    def get(self, identifier: str) -> Prompt | None:
        return next((prompt for prompt in self.prompts if prompt.id == identifier), None)

    def find_by_text(self, text: str) -> Prompt | None:
        return next((prompt for prompt in self.prompts if prompt.text == text), None) or next(
            (prompt for prompt in self.prompts if prompt.text.strip() == text.strip()), None)

    def best_for_language(self, language: str, current: Prompt | None = None) -> Prompt | None:
        """Keep a matching/any-language current prompt, else prefer the user's own."""
        language = language.strip().lower()
        if language in ("", "auto"):
            return current
        if current is not None and current.language in ("", language):
            return current
        matches = [prompt for prompt in self.prompts if prompt.language == language]
        own = [prompt for prompt in matches if not prompt.builtin]
        return (own or matches or [current])[0]

    # ── editing ────────────────────────────────────────────────────────
    @staticmethod
    def _new_id(taken) -> str:
        while True:
            identifier = "user-" + uuid.uuid4().hex[:8]
            if identifier not in taken:
                return identifier

    def _validated(self, name: str, text: str, language: str, *, ignore: str = "") -> tuple[str, str, str]:
        name, language = name.strip(), language.strip().lower()
        if not name:
            raise PromptError("name")
        if not text.strip():
            raise PromptError("text")
        if any(prompt.name.casefold() == name.casefold() and prompt.id != ignore for prompt in self.prompts):
            raise PromptError("duplicate")
        return name, text, language

    def unique_name(self, base: str) -> str:
        names = {prompt.name.casefold() for prompt in self.prompts}
        if base.casefold() not in names:
            return base
        number = 2
        while f"{base} ({number})".casefold() in names:
            number += 1
        return f"{base} ({number})"

    def add(self, name: str, text: str, language: str = "") -> Prompt:
        name, text, language = self._validated(name, text, language)
        prompt = Prompt(self._new_id({item.id for item in self.prompts}), name, language, text)
        self.custom.append(prompt)
        self.save()
        return prompt

    def update(self, identifier: str, name: str, text: str, language: str = "") -> Prompt:
        current = self.get(identifier)
        if current is None:
            raise PromptError("missing")
        if current.builtin:
            raise PromptError("builtin")
        name, text, language = self._validated(name, text, language, ignore=identifier)
        updated = replace(current, name=name, text=text, language=language)
        self.custom[self.custom.index(current)] = updated
        self.save()
        return updated

    def delete(self, identifier: str) -> None:
        current = self.get(identifier)
        if current is None:
            return
        if current.builtin:
            raise PromptError("builtin")
        self.custom.remove(current)
        self.save()

    def duplicate(self, identifier: str, suffix: str = "копія") -> Prompt:
        current = self.get(identifier)
        if current is None:
            raise PromptError("missing")
        return self.add(self.unique_name(f"{current.name} — {suffix}"), current.text, current.language)

    # ── writing ────────────────────────────────────────────────────────
    def save(self) -> None:
        import tomlkit

        document = tomlkit.document()
        document.add(tomlkit.comment("WhisperBlocks — your own decoder prompts (managed by the GUI)."))
        document.add(tomlkit.comment("The three built-in prompts are part of the app and are not stored here."))
        tables = tomlkit.aot()
        for prompt in self.custom:
            table = tomlkit.table()
            table.add("id", prompt.id)
            table.add("name", prompt.name)
            table.add("language", prompt.language)
            table.add("text", prompt.text)
            tables.append(table)
        if self.custom:
            document.add("prompt", tables)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(tomlkit.dumps(document), encoding="utf-8", newline="\n")
        os.replace(temporary, self.path)
