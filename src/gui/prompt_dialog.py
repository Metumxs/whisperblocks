"""Dialog for managing decoder prompts (built-ins are locked)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QSizePolicy, QVBoxLayout,
    QWidget,
)

from . import catalog
from .i18n import t
from .prompts import Prompt, PromptError, PromptStore


class ChoiceCombo(QComboBox):
    """Drop-down of (label, code) choices with an optional "Other…" entry.

    Picking "Other…" asks for a free-form code, so rare languages or models
    stay reachable without making the whole box an editable text field.
    """

    def __init__(self) -> None:
        super().__init__()
        self._custom_label = ""
        self._ask_title = ""
        self._ask_label = ""
        self._last_index = 0
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(18)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.activated.connect(self._activated)
        self.currentIndexChanged.connect(self._remember)

    def allow_custom(self, label: str, title: str, question: str) -> None:
        self._custom_label, self._ask_title, self._ask_label = label, title, question
        self.addItem(label, None)

    def _remember(self, index: int) -> None:
        if index >= 0 and self.itemData(index) is not None:
            self._last_index = index

    def _activated(self, index: int) -> None:
        if self.itemData(index) is not None or not self._custom_label:
            return
        text, accepted = QInputDialog.getText(self, self._ask_title, self._ask_label)
        text = text.strip()
        if accepted and text:
            self.set_code(text)
        else:
            self.setCurrentIndex(self._last_index)

    def code(self) -> str:
        data = self.currentData()
        return data if isinstance(data, str) else (self.itemData(self._last_index) or "")

    def set_code(self, code: str) -> None:
        index = self.findData(code)
        if index < 0:
            # Unknown value (typed by hand or from config.toml): keep it as its own entry.
            index = self.count() - 1 if self._custom_label else self.count()
            self.insertItem(index, code, code)
        self.setCurrentIndex(index)


def language_combo(*, any_label: str | None = None) -> ChoiceCombo:
    combo = ChoiceCombo()
    if any_label is not None:
        combo.addItem(any_label, "")
    for code, _uk, _en in catalog.AUDIO_LANGUAGES:
        if code == "auto" and any_label is not None:
            continue
        combo.addItem(catalog.language_label(code), code)
    combo.allow_custom(
        t("Інша мова…", "Another language…"), t("Інша мова", "Another language"),
        t("Код мови за ISO-639-1, наприклад «hu» чи «fi»:", "ISO-639-1 language code, e.g. “hu” or “fi”:"))
    return combo


class PromptDialog(QDialog):
    def __init__(self, store: PromptStore, parent: QWidget | None = None, *,
                 select: str = "", new_language: str | None = None) -> None:
        super().__init__(parent)
        self.store = store
        self.current: Prompt | None = None
        self._loading = False
        self.setWindowTitle(t("Підказки для моделі", "Model prompts"))
        self.resize(760, 440)

        self.list = QListWidget()
        self.list.setMinimumWidth(240)
        self.list.currentItemChanged.connect(self._selected)
        self.new_button = QPushButton(t("Нова", "New"))
        self.copy_button = QPushButton(t("Дублювати", "Duplicate"))
        self.delete_button = QPushButton(t("Видалити", "Delete"))
        self.new_button.clicked.connect(lambda: self._create())
        self.copy_button.clicked.connect(self._duplicate)
        self.delete_button.clicked.connect(self._delete)
        buttons = QHBoxLayout()
        for button in (self.new_button, self.copy_button, self.delete_button):
            buttons.addWidget(button)
        left = QVBoxLayout()
        left.addWidget(self.list, 1)
        left.addLayout(buttons)

        self.name_edit = QLineEdit()
        self.name_edit.setMaxLength(80)
        self.language_combo = language_combo(any_label=t("Будь-яка мова", "Any language"))
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText(t(
            "Наприклад: «Це інтерв'ю. Грамотний текст українською мовою без слів-паразитів: »",
            "For example: “This is an interview. Well-written English text without filler words: ”",
        ))
        self.lock_label = QLabel()
        self.lock_label.setObjectName("warning")
        self.lock_label.setWordWrap(True)
        hint = QLabel(t(
            "Підказка — це короткий початок тексту мовою запису. Модель продовжує в тому ж стилі, "
            "тому пишіть її тією мовою, якою говорять в аудіо.",
            "A prompt is a short opening written in the language of the recording. The model continues "
            "in the same style, so write it in the language spoken in the audio.",
        ))
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        form = QFormLayout()
        form.addRow(t("Назва", "Name"), self.name_edit)
        form.addRow(t("Мова", "Language"), self.language_combo)
        self.save_button = QPushButton(t("Зберегти зміни", "Save changes"))
        self.save_button.setObjectName("primary")
        self.save_button.clicked.connect(self._save)
        right = QVBoxLayout()
        right.addWidget(self.lock_label)
        right.addLayout(form)
        right.addWidget(self.text_edit, 1)
        right.addWidget(hint)
        save_row = QHBoxLayout()
        save_row.addStretch(1)
        save_row.addWidget(self.save_button)
        right.addLayout(save_row)

        body = QHBoxLayout()
        body.addLayout(left, 2)
        body.addLayout(right, 3)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.button(QDialogButtonBox.StandardButton.Close).setText(t("Закрити", "Close"))
        close.rejected.connect(self.close)
        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addWidget(close)

        for widget in (self.name_edit, self.text_edit):
            widget.textChanged.connect(self._edited)
        self.language_combo.currentIndexChanged.connect(self._edited)
        self._reload(select or "builtin-uk")
        if new_language is not None:
            self._create(new_language)

    # ── list ───────────────────────────────────────────────────────────
    def _reload(self, select: str = "") -> None:
        self._loading = True
        self.list.clear()
        for prompt in self.store.prompts:
            label = prompt.name + (t("  · системна", "  · built-in") if prompt.builtin else "")
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, prompt.id)
            item.setToolTip(prompt.text)
            self.list.addItem(item)
            if prompt.id == select:
                self.list.setCurrentItem(item)
        self._loading = False
        if self.list.currentItem() is None and self.list.count():
            self.list.setCurrentRow(0)
        else:
            self._selected(self.list.currentItem(), None)

    def _selected(self, item: QListWidgetItem | None, _previous) -> None:
        if self._loading:
            return
        prompt = self.store.get(item.data(Qt.ItemDataRole.UserRole)) if item else None
        self.current = prompt
        self._loading = True
        self.name_edit.setText(prompt.name if prompt else "")
        self.language_combo.set_code(prompt.language if prompt else "")
        self.text_edit.setPlainText(prompt.text if prompt else "")
        self._loading = False
        locked = prompt is None or prompt.builtin
        self.name_edit.setReadOnly(locked)
        self.text_edit.setReadOnly(locked)
        self.language_combo.setEnabled(not locked)
        self.delete_button.setEnabled(not locked)
        self.copy_button.setEnabled(prompt is not None)
        self.lock_label.setVisible(bool(prompt and prompt.builtin))
        self.lock_label.setText(t(
            "Системну підказку не можна змінити чи видалити. Натисніть «Дублювати», щоб зробити власну на її основі.",
            "A built-in prompt cannot be changed or deleted. Press “Duplicate” to make your own based on it.",
        ))
        self.save_button.setVisible(not locked)
        self.save_button.setEnabled(False)

    def selected_id(self) -> str:
        return self.current.id if self.current else ""

    def _edited(self, *_args) -> None:
        if not self._loading and self.current and not self.current.builtin:
            self.save_button.setEnabled(True)

    # ── actions ────────────────────────────────────────────────────────
    def _explain(self, error: PromptError) -> None:
        messages = {
            "name": t("Вкажіть назву підказки.", "Enter a prompt name."),
            "text": t("Текст підказки не може бути порожнім.", "The prompt text cannot be empty."),
            "duplicate": t("Підказка з такою назвою вже є.", "A prompt with this name already exists."),
            "builtin": t("Системну підказку не можна змінити.", "A built-in prompt cannot be changed."),
        }
        QMessageBox.warning(self, self.windowTitle(), messages.get(str(error), str(error)))

    def _confirm_unsaved(self) -> None:
        if self.save_button.isVisible() and self.save_button.isEnabled():
            answer = QMessageBox.question(
                self, self.windowTitle(), t("Зберегти зміни в підказці?", "Save changes to the prompt?"))
            if answer == QMessageBox.StandardButton.Yes:
                self._save()

    def _create(self, language: str = "") -> None:
        self._confirm_unsaved()
        try:
            prompt = self.store.add(
                self.store.unique_name(t("Моя підказка", "My prompt")),
                t("Грамотний текст без слів-паразитів: ", "Well-written text without filler words: "),
                "" if language == "auto" else language,
            )
        except (PromptError, OSError) as error:
            QMessageBox.warning(self, self.windowTitle(), str(error))
            return
        self._reload(prompt.id)
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    def _duplicate(self) -> None:
        if self.current is None:
            return
        self._confirm_unsaved()
        try:
            prompt = self.store.duplicate(self.current.id, t("копія", "copy"))
        except (PromptError, OSError) as error:
            QMessageBox.warning(self, self.windowTitle(), str(error))
            return
        self._reload(prompt.id)
        self.text_edit.setFocus()

    def _delete(self) -> None:
        if self.current is None or self.current.builtin:
            return
        answer = QMessageBox.question(
            self, self.windowTitle(),
            t("Видалити підказку «{name}»?", "Delete the prompt “{name}”?").format(name=self.current.name))
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.store.delete(self.current.id)
        except (PromptError, OSError) as error:
            QMessageBox.warning(self, self.windowTitle(), str(error))
            return
        self._reload()

    def _save(self) -> None:
        if self.current is None or self.current.builtin:
            return
        try:
            prompt = self.store.update(
                self.current.id, self.name_edit.text(), self.text_edit.toPlainText(),
                self.language_combo.code().strip().lower(),
            )
        except PromptError as error:
            self._explain(error)
            return
        except OSError as error:
            QMessageBox.warning(self, self.windowTitle(), str(error))
            return
        self._reload(prompt.id)

    def closeEvent(self, event) -> None:
        self._confirm_unsaved()
        super().closeEvent(event)

    def reject(self) -> None:
        self.close()
