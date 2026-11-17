"""Main WhisperBlocks window: file queue, settings, run control and log."""

from __future__ import annotations

import os
import time
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QDesktopServices, QFontDatabase, QPainter
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QSplitter, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from . import catalog, config_store, events, i18n, paths, theme
from .i18n import t
from .prompt_dialog import ChoiceCombo, PromptDialog, language_combo
from .prompts import Prompt, PromptStore
from .runner import EnvironmentProbe, LineProcess, open_in_file_manager, run_install_script

ROLE_REQUESTED = Qt.ItemDataRole.UserRole
ROLE_STATUS = Qt.ItemDataRole.UserRole + 1
ROLE_OUTPUT = Qt.ItemDataRole.UserRole + 2
PYANNOTE_URL = "https://huggingface.co/pyannote/speaker-diarization-community-1"
TOKEN_URL = "https://huggingface.co/settings/tokens"


def format_clock(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def format_remaining(seconds: float) -> str:
    minutes = max(1, round(seconds / 60))
    if minutes < 60:
        return t("≈ {n} хв", "≈ {n} min").format(n=minutes)
    return t("≈ {h} год {m} хв", "≈ {h} h {m} min").format(h=minutes // 60, m=minutes % 60)


class Card(QFrame):
    """Rounded settings group with a title; optionally collapsible."""

    def __init__(self, title: str, *, collapsible: bool = False, expanded: bool = True) -> None:
        super().__init__()
        self.setObjectName("card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)
        self.content = QWidget()
        self.body = QVBoxLayout(self.content)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(10)
        if collapsible:
            self.toggle = QToolButton()
            self.toggle.setObjectName("sectionToggle")
            self.toggle.setText(title)
            self.toggle.setCheckable(True)
            self.toggle.setChecked(expanded)
            self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self.toggle.toggled.connect(self._toggled)
            outer.addWidget(self.toggle)
            self._toggled(expanded)
        else:
            self.toggle = None
            label = QLabel(title)
            label.setObjectName("cardTitle")
            outer.addWidget(label)
        outer.addWidget(self.content)

    def _toggled(self, expanded: bool) -> None:
        self.content.setVisible(expanded)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)

    def form(self) -> QFormLayout:
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.body.addLayout(form)
        return form


def hint_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("hint")
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    label.setOpenExternalLinks(True)
    return label


class NoWheelFilter(QObject):
    """Stops combo boxes and spin boxes from changing while the page is scrolled."""

    def eventFilter(self, watched, event) -> bool:
        if event.type() == event.Type.Wheel and not watched.hasFocus():
            event.ignore()
            return True
        return False


class FileQueue(QTreeWidget):
    """File list accepting drag & drop of media files and folders."""

    dropped = Signal(list)
    delete_pressed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setColumnCount(2)
        self.setHeaderLabels([t("Файл", "File"), t("Стан", "State")])
        self.setRootIsDecorated(False)
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(False)
        self.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setDragDropMode(QTreeWidget.DragDropMode.DropOnly)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.header().setStretchLastSection(True)
        self.setColumnWidth(0, 250)
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        files = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if files:
            event.acceptProposedAction()
            self.dropped.emit(files)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_pressed.emit()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.topLevelItemCount():
            return
        painter = QPainter(self.viewport())
        painter.setPen(self.palette().color(self.palette().ColorRole.PlaceholderText))
        painter.drawText(
            self.viewport().rect().adjusted(24, 24, -24, -24),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            t("Перетягніть сюди аудіо чи відео\nабо натисніть «Додати файли»",
              "Drop audio or video here\nor press “Add files”"),
        )


class MainWindow(QMainWindow):
    rebuild_requested = Signal()

    def __init__(self, initial_files: list[str] | None = None) -> None:
        super().__init__()
        self.state = QSettings(str(paths.STATE_PATH), QSettings.Format.IniFormat)
        self.store = PromptStore(paths.PROMPTS_PATH)
        self.runner = LineProcess(self)
        self.runner.line.connect(self._on_line)
        self.runner.finished.connect(self._on_finished)
        self.runner.failed_to_start.connect(self._on_failed_to_start)
        self.probe = EnvironmentProbe(self)
        self.probe.result.connect(self._on_probe)
        self.environment: dict | None = None
        self.config_ok = False
        self.mode = ""                      # "", "run" or "plan"
        self.progress = events.ProgressModel()
        self.started_at = 0.0
        self.stage_started = 0.0
        self.summary: dict | None = None
        self.run_error = ""
        self.active_item: QTreeWidgetItem | None = None
        self.clock = QTimer(self)
        self.clock.setInterval(1000)
        self.clock.timeout.connect(self._tick)
        self._wheel_filter = NoWheelFilter(self)
        self._loading = False

        self.setWindowTitle("WhisperBlocks")
        self.setWindowIcon(theme.app_icon())
        self.setAcceptDrops(True)
        self._build_ui()
        self._build_menu()
        self._restore_state()
        self._load_config()
        for name in initial_files or []:
            self._add_paths([Path(name)])
        self._update_controls()
        QTimer.singleShot(150, self.probe.start)

    # ══ UI construction ════════════════════════════════════════════════
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # Header
        header = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(theme.app_icon().pixmap(28, 28))
        title = QLabel("WhisperBlocks")
        title.setObjectName("appTitle")
        self.env_chip = QToolButton()
        self.env_chip.setObjectName("chip")
        self.env_chip.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.env_chip.setIcon(theme.status_dot("pending"))
        self.env_chip.setText(t("Перевірка системи…", "Checking system…"))
        self.env_chip.clicked.connect(self._show_environment)
        self.menu_button = QToolButton()
        self.menu_button.setObjectName("chip")
        self.menu_button.setText(t("Меню", "Menu"))
        self.menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        header.addWidget(icon)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.env_chip)
        header.addWidget(self.menu_button)
        root.addLayout(header)

        # Banners (environment problems)
        self.banners = QVBoxLayout()
        self.banners.setSpacing(6)
        root.addLayout(self.banners)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self._build_queue())
        self.splitter.addWidget(self._build_settings())
        self.splitter.setStretchFactor(0, 5)
        self.splitter.setStretchFactor(1, 6)
        root.addWidget(self.splitter, 1)
        root.addWidget(self._build_footer())

    def _build_queue(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(8)
        top = QHBoxLayout()
        self.queue_title = QLabel()
        self.queue_title.setObjectName("cardTitle")
        self.add_files_button = QPushButton(t("Додати файли…", "Add files…"))
        self.add_folder_button = QPushButton(t("Додати папку…", "Add folder…"))
        self.add_files_button.clicked.connect(self._browse_files)
        self.add_folder_button.clicked.connect(self._browse_folder)
        top.addWidget(self.queue_title)
        top.addStretch(1)
        top.addWidget(self.add_files_button)
        top.addWidget(self.add_folder_button)
        layout.addLayout(top)

        self.queue = FileQueue()
        self.queue.dropped.connect(self._add_paths)
        self.queue.delete_pressed.connect(self._remove_selected)
        self.queue.customContextMenuRequested.connect(self._queue_menu)
        self.queue.itemDoubleClicked.connect(lambda item, _column: self._open_item_output(item))
        self.queue.itemSelectionChanged.connect(self._update_controls)
        layout.addWidget(self.queue, 1)

        bottom = QHBoxLayout()
        self.remove_button = QPushButton(t("Прибрати вибрані", "Remove selected"))
        self.clear_button = QPushButton(t("Очистити список", "Clear list"))
        self.remove_button.clicked.connect(self._remove_selected)
        self.clear_button.clicked.connect(self._clear_queue)
        bottom.addWidget(self.remove_button)
        bottom.addWidget(self.clear_button)
        bottom.addStretch(1)
        layout.addLayout(bottom)
        return panel

    def _watch(self, widget: QWidget) -> QWidget:
        widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        widget.installEventFilter(self._wheel_filter)
        return widget

    def _build_settings(self) -> QWidget:
        self.settings_panel = QWidget()
        column = QVBoxLayout(self.settings_panel)
        column.setContentsMargins(8, 0, 6, 0)
        column.setSpacing(12)

        # ── Main ───────────────────────────────────────────────────────
        main = Card(t("Основне", "General"))
        form = main.form()
        self.language_combo = self._watch(language_combo())
        self.language_combo.setToolTip(t(
            "Мова, якою говорять у записі. Для українських записів завжди вибирайте «Українська»: "
            "автовизначення часто плутає її з російською.",
            "The language spoken in the recording. Choosing it explicitly is more reliable than auto-detection.",
        ))
        self.model_combo = self._watch(ChoiceCombo())
        for code, label in catalog.models():
            self.model_combo.addItem(label, code)
        self.model_combo.allow_custom(
            t("Інша модель…", "Another model…"), t("Інша модель", "Another model"),
            t("Назва моделі Whisper (наприклад «distil-large-v3»):", "Whisper model name (e.g. “distil-large-v3”):"))
        self.model_combo.setToolTip(t(
            "Модель завантажується з інтернету під час першого використання (1.5–3 ГБ) і далі працює локально.",
            "The model is downloaded on first use (1.5–3 GB) and then runs locally.",
        ))
        self.output_combo = self._watch(ChoiceCombo())
        for code, label in catalog.output_modes():
            self.output_combo.addItem(label, code)
        self.blocks_check = QCheckBox(t("Об'єднувати репліки в блоки по", "Merge lines into blocks of"))
        self.blocks_spin = self._watch(QSpinBox())
        self.blocks_spin.setRange(5, 3600)
        self.blocks_spin.setSingleStep(10)
        self.blocks_spin.setSuffix(t(" с", " s"))
        self.blocks_spin.setMinimumWidth(90)
        blocks = QHBoxLayout()
        blocks.addWidget(self.blocks_check)
        blocks.addWidget(self.blocks_spin)
        blocks.addStretch(1)
        self.blocks_hint = hint_label(t(
            "Додатковий файл …blocks.txt, де текст із таймкодами зібрано в абзаци заданої тривалості. "
            "Працює для форматів із таймкодами.",
            "An extra …blocks.txt file where timecoded text is grouped into paragraphs of the given length. "
            "Works for formats with timecodes.",
        ))
        form.addRow(t("Мова запису", "Audio language"), self.language_combo)
        form.addRow(t("Модель", "Model"), self.model_combo)
        form.addRow(t("Результат", "Output"), self.output_combo)
        main.body.addLayout(blocks)
        main.body.addWidget(self.blocks_hint)
        column.addWidget(main)

        # ── Prompt ─────────────────────────────────────────────────────
        prompt = Card(t("Підказка для моделі", "Model prompt"))
        self.prompt_check = QCheckBox(t(
            "Підказувати стиль тексту (менше слів-паразитів)", "Hint the text style (fewer filler words)"))
        self.prompt_combo = self._watch(ChoiceCombo())
        self.prompt_manage = QPushButton(t("Керувати…", "Manage…"))
        self.prompt_manage.clicked.connect(lambda: self._manage_prompts())
        row = QHBoxLayout()
        row.addWidget(self.prompt_combo, 1)
        row.addWidget(self.prompt_manage)
        self.prompt_preview = hint_label("")
        self.prompt_warning = QLabel()
        self.prompt_warning.setObjectName("warning")
        self.prompt_warning.setWordWrap(True)
        self.prompt_warning.linkActivated.connect(lambda _link: self._manage_prompts(new_language=self._language()))
        prompt.body.addWidget(self.prompt_check)
        prompt.body.addLayout(row)
        prompt.body.addWidget(self.prompt_preview)
        prompt.body.addWidget(self.prompt_warning)
        column.addWidget(prompt)

        # ── Diarization ────────────────────────────────────────────────
        speakers = Card(t("Спікери", "Speakers"))
        self.diarize_check = QCheckBox(t(
            "Розпізнавати, хто говорить (мітки SPEAKER_00, SPEAKER_01…)",
            "Detect who is speaking (SPEAKER_00, SPEAKER_01… labels)"))
        self.diarize_box = QWidget()
        box = QVBoxLayout(self.diarize_box)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(10)
        self.token_edit = QLineEdit()
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText("hf_…")
        self.token_edit.setClearButtonEnabled(True)
        self.token_show = QToolButton()
        self.token_show.setText(t("Показати", "Show"))
        self.token_show.setCheckable(True)
        self.token_show.toggled.connect(self._toggle_token)
        token_row = QHBoxLayout()
        token_row.setSpacing(8)
        token_row.addWidget(self.token_edit, 1)
        token_row.addWidget(self.token_show)
        token_form = QFormLayout()
        token_form.setContentsMargins(0, 0, 0, 0)
        token_form.setHorizontalSpacing(14)
        token_form.setVerticalSpacing(10)
        token_form.addRow(t("Токен Hugging Face", "Hugging Face token"), token_row)
        self.min_speakers = self._watch(QSpinBox())
        self.max_speakers = self._watch(QSpinBox())
        for spin in (self.min_speakers, self.max_speakers):
            spin.setRange(0, 50)
            spin.setMinimumWidth(84)
            spin.setSpecialValueText(t("авто", "auto"))
        self.min_speakers.valueChanged.connect(self._speakers_changed)
        self.max_speakers.valueChanged.connect(self._speakers_changed)
        count_row = QHBoxLayout()
        count_row.setSpacing(8)
        count_row.addWidget(QLabel(t("від", "from")))
        count_row.addWidget(self.min_speakers)
        count_row.addWidget(QLabel(t("до", "to")))
        count_row.addWidget(self.max_speakers)
        count_row.addStretch(1)
        token_form.addRow(t("Кількість спікерів", "Number of speakers"), count_row)
        box.addLayout(token_form)
        self.token_hint = hint_label("")
        box.addWidget(self.token_hint)
        speakers.body.addWidget(self.diarize_check)
        speakers.body.addWidget(self.diarize_box)
        column.addWidget(speakers)

        # ── Advanced ───────────────────────────────────────────────────
        self.advanced = Card(t("Додатково", "Advanced"), collapsible=True, expanded=False)
        advanced_form = self.advanced.form()
        self.device_combo = self._watch(ChoiceCombo())
        for code, label in catalog.devices():
            self.device_combo.addItem(label, code)
        self.batch_spin = self._watch(QSpinBox())
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setMinimumWidth(84)
        self.batch_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.batch_spin.setToolTip(t(
            "Скільки фрагментів обробляти одночасно. Рекомендовано 8. Зменшіть до 4 або 2, "
            "якщо з'являється помилка «CUDA out of memory».",
            "How many chunks to process at once. 8 is recommended. Lower it to 4 or 2 on “CUDA out of memory”.",
        ))
        self.overwrite_check = QCheckBox(t(
            "Перезаписувати наявні результати", "Overwrite existing results"))
        self.overwrite_check.setToolTip(t(
            "Без цієї позначки файли, для яких результат уже є, пропускаються.",
            "Without this, files that already have results are skipped.",
        ))
        advanced_form.addRow(t("Пристрій", "Device"), self.device_combo)
        advanced_form.addRow(t("Розмір пакета", "Batch size"), self.batch_spin)
        self.advanced.body.addWidget(self.overwrite_check)
        column.addWidget(self.advanced)
        column.addStretch(1)

        # signals
        self.language_combo.currentIndexChanged.connect(self._language_changed)
        self.model_combo.currentIndexChanged.connect(self._update_controls)
        self.output_combo.currentIndexChanged.connect(self._update_controls)
        self.blocks_check.toggled.connect(self._update_controls)
        self.prompt_check.toggled.connect(self._update_controls)
        self.prompt_combo.currentIndexChanged.connect(self._update_controls)
        self.diarize_check.toggled.connect(self._update_controls)
        self.token_edit.textChanged.connect(self._update_controls)
        self.device_combo.currentIndexChanged.connect(self._refresh_banners)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self.settings_panel)
        scroll.setMinimumWidth(430)
        scroll.viewport().setAutoFillBackground(False)
        self.settings_panel.setAutoFillBackground(False)
        return scroll

    def _build_footer(self) -> QWidget:
        footer = QWidget()
        layout = QVBoxLayout(footer)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)
        status_row = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setObjectName("statusText")
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.time_label = QLabel()
        self.time_label.setObjectName("hint")
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.time_label)
        layout.addLayout(status_row)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        buttons = QHBoxLayout()
        self.log_toggle = QToolButton()
        self.log_toggle.setObjectName("flat")
        self.log_toggle.setCheckable(True)
        self.log_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.log_toggle.setText(t("Журнал", "Log"))
        self.log_toggle.toggled.connect(self._toggle_log)
        self.open_output_button = QPushButton(t("Відкрити результати", "Open results"))
        self.open_output_button.clicked.connect(lambda: self._open_folder(paths.OUTPUT_DIR))
        self.plan_button = QPushButton(t("Перевірити", "Check"))
        self.plan_button.setToolTip(t(
            "Швидка перевірка без запуску моделі: які файли будуть оброблені, які пропущені і куди ляже результат.",
            "A quick check without loading the model: which files will run, which are skipped and where results go.",
        ))
        self.plan_button.clicked.connect(lambda: self._start("plan"))
        self.start_button = QPushButton()
        self.start_button.setObjectName("primary")
        self.start_button.setDefault(True)
        self.start_button.clicked.connect(self._start_or_cancel)
        buttons.addWidget(self.log_toggle)
        buttons.addStretch(1)
        buttons.addWidget(self.open_output_button)
        buttons.addWidget(self.plan_button)
        buttons.addWidget(self.start_button)
        layout.addLayout(buttons)

        self.log = QPlainTextEdit()
        self.log.setObjectName("log")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(6000)
        self.log.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.log.setMinimumHeight(150)
        self.log.setMaximumHeight(260)
        self.log.setVisible(False)
        layout.addWidget(self.log)
        return footer

    def _build_menu(self) -> None:
        menu = QMenu(self)
        language_menu = menu.addMenu(t("Мова інтерфейсу", "Interface language"))
        group = QActionGroup(self)
        for code, name in i18n.LANGUAGES.items():
            action = QAction(name, self, checkable=True, checked=code == i18n.language())
            action.triggered.connect(lambda _checked, code=code: self._set_ui_language(code))
            group.addAction(action)
            language_menu.addAction(action)
        self.language_menu = language_menu
        theme_menu = menu.addMenu(t("Тема", "Theme"))
        theme_group = QActionGroup(self)
        current = str(self.state.value("theme", "system"))
        for code, name in (
            ("system", t("Як у системі", "Follow system")), ("light", t("Світла", "Light")),
            ("dark", t("Темна", "Dark")),
        ):
            action = QAction(name, self, checkable=True, checked=code == current)
            action.triggered.connect(lambda _checked, code=code: self._set_theme(code))
            theme_group.addAction(action)
            theme_menu.addAction(action)
        menu.addSeparator()
        menu.addAction(t("Підказки для моделі…", "Model prompts…"), lambda: self._manage_prompts())
        menu.addAction(t("Перевірити систему ще раз", "Re-check system"), self._recheck)
        menu.addSeparator()
        menu.addAction(t("Відкрити папку input", "Open input folder"), lambda: self._open_folder(paths.INPUT_DIR))
        menu.addAction(t("Відкрити папку проєкту", "Open project folder"), lambda: self._open_folder(paths.PROJECT_ROOT))
        menu.addAction(t("Відкрити config.toml", "Open config.toml"), self._open_config)
        menu.addSeparator()
        menu.addAction(t("Про програму", "About"), self._about)
        self.menu_button.setMenu(menu)

    # ══ State ══════════════════════════════════════════════════════════
    def _restore_state(self) -> None:
        geometry = self.state.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            available = self.screen().availableGeometry() if self.screen() else None
            width, height = 1180, 860
            if available is not None:
                width, height = min(width, available.width() - 40), min(height, available.height() - 60)
            self.resize(width, height)
        splitter = self.state.value("splitter")
        if splitter is not None:
            self.splitter.restoreState(splitter)
        else:
            self.splitter.setSizes([500, 600])
        self.log_toggle.setChecked(self.state.value("log_visible", "false") == "true")
        self._toggle_log(self.log_toggle.isChecked())
        if self.state.value("advanced_open", "false") == "true":
            self.advanced.toggle.setChecked(True)

    def _save_state(self) -> None:
        self.state.setValue("geometry", self.saveGeometry())
        self.state.setValue("splitter", self.splitter.saveState())
        self.state.setValue("log_visible", "true" if self.log_toggle.isChecked() else "false")
        self.state.setValue("advanced_open", "true" if self.advanced.toggle.isChecked() else "false")
        self.state.sync()

    # ══ Config ═════════════════════════════════════════════════════════
    def _load_config(self) -> None:
        self._loading = True
        settings, problems = dict(config_store.DEFAULTS), []
        try:
            config_store.ensure_config(paths.CONFIG_PATH, paths.CONFIG_EXAMPLE_PATH)
            settings, problems = config_store.load_settings(paths.CONFIG_PATH)
            self.config_ok = True
        except (config_store.ConfigError, OSError) as error:
            self.config_ok = False
            self._loading = False
            if self._offer_config_reset(str(error)):
                return self._load_config()
            self._loading = True

        self.language_combo.set_code(settings["language"])
        self.model_combo.set_code(settings["model_size"])
        self.output_combo.setCurrentIndex(max(0, self.output_combo.findData(settings["output_mode"])))
        seconds = settings["auto_preprocess_block_seconds"]
        self.blocks_check.setChecked(seconds > 0)
        self.blocks_spin.setValue(max(5, round(seconds)) if seconds > 0 else 60)
        self.diarize_check.setChecked(settings["diarize"])
        self.token_edit.setText(settings["hf_token"])
        self.min_speakers.setValue(settings["min_speakers"])
        self.max_speakers.setValue(settings["max_speakers"])
        self.device_combo.setCurrentIndex(max(0, self.device_combo.findData(settings["device"])))
        self.batch_spin.setValue(min(64, settings["batch_size"]))
        self.prompt_check.setChecked(settings["remove_fillers"])

        # Prompt text typed straight into config.toml becomes a managed prompt.
        text = settings["initial_prompt"]
        selected = self.store.find_by_text(text) if text.strip() else None
        if text.strip() and selected is None:
            try:
                selected = self.store.add(
                    self.store.unique_name(t("Моя підказка (з config.toml)", "My prompt (from config.toml)")), text)
            except (ValueError, OSError):
                selected = None
        if selected is None:
            selected = self.store.get(str(self.state.value("prompt_id", ""))) or self.store.best_for_language(
                settings["language"], None) or self.store.prompts[0]
        self._fill_prompts(selected.id)

        self.queue.clear()
        for requested in settings["files"]:
            self._add_requested(requested)
        self._loading = False
        if problems:
            self._log(t(
                "У config.toml були некоректні значення ({keys}); для них узято типові.",
                "config.toml had invalid values ({keys}); defaults are used for them.",
            ).format(keys=", ".join(problems)))
        self._set_status(t("Додайте файли й натисніть «Старт».", "Add files and press “Start”.")
                         if not self.queue.topLevelItemCount() else
                         t("Готово до запуску.", "Ready to start."))
        self._update_controls()

    def _offer_config_reset(self, detail: str) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("WhisperBlocks")
        box.setText(t("Не вдалося прочитати config/config.toml.", "Could not read config/config.toml."))
        box.setInformativeText(detail)
        reset = box.addButton(t("Створити новий (старий зберегти)", "Create a new one (keep the old)"),
                              QMessageBox.ButtonRole.AcceptRole)
        box.addButton(t("Продовжити без збереження", "Continue without saving"), QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is not reset:
            return False
        try:
            broken = paths.CONFIG_PATH.with_name(f"config.toml.broken-{time.strftime('%Y%m%d-%H%M%S')}")
            os.replace(paths.CONFIG_PATH, broken)
            return True
        except OSError as error:
            QMessageBox.warning(self, "WhisperBlocks", str(error))
            return False

    def _collect(self) -> dict:
        prompt = self._current_prompt()
        return {
            "files": [self.queue.topLevelItem(row).data(0, ROLE_REQUESTED)
                      for row in range(self.queue.topLevelItemCount())],
            "model_size": self._model(),
            "device": self.device_combo.currentData(),
            "language": self._language(),
            "output_mode": self.output_combo.currentData(),
            "remove_fillers": self.prompt_check.isChecked(),
            "initial_prompt": prompt.text if prompt else "",
            "auto_preprocess_block_seconds": self.blocks_spin.value() if self.blocks_check.isChecked() else 0,
            "diarize": self.diarize_check.isChecked(),
            "hf_token": self.token_edit.text().strip(),
            "min_speakers": self.min_speakers.value(),
            "max_speakers": self.max_speakers.value(),
            "batch_size": self.batch_spin.value(),
        }

    def _save_config(self, *, quiet: bool = False) -> bool:
        if not self.config_ok:
            return False
        try:
            config_store.save_settings(paths.CONFIG_PATH, self._collect())
            prompt = self._current_prompt()
            self.state.setValue("prompt_id", prompt.id if prompt else "")
            return True
        except (config_store.ConfigError, ValueError, OSError) as error:
            if not quiet:
                QMessageBox.warning(self, "WhisperBlocks", t(
                    "Не вдалося зберегти налаштування:\n{error}", "Could not save settings:\n{error}",
                ).format(error=error))
            return False

    # ══ Small getters ══════════════════════════════════════════════════
    def _language(self) -> str:
        return self.language_combo.code().strip() or "uk"

    def _model(self) -> str:
        return self.model_combo.code().strip() or "large-v3-turbo"

    def _current_prompt(self) -> Prompt | None:
        return self.store.get(self.prompt_combo.currentData() or "")

    # ══ Prompts ════════════════════════════════════════════════════════
    def _fill_prompts(self, select: str) -> None:
        self.prompt_combo.blockSignals(True)
        self.prompt_combo.clear()
        for prompt in self.store.prompts:
            self.prompt_combo.addItem(prompt.name, prompt.id)
        index = self.prompt_combo.findData(select)
        self.prompt_combo.setCurrentIndex(index if index >= 0 else 0)
        self.prompt_combo.blockSignals(False)
        self._update_controls()

    def _manage_prompts(self, *, new_language: str | None = None) -> None:
        current = self._current_prompt()
        dialog = PromptDialog(self.store, self, select=current.id if current else "", new_language=new_language)
        dialog.exec()
        chosen = dialog.selected_id() if new_language is not None else (current.id if current else "")
        self._fill_prompts(chosen if self.store.get(chosen) else "builtin-uk")

    def _language_changed(self, *_args) -> None:
        if self._loading:
            return
        best = self.store.best_for_language(self._language(), self._current_prompt())
        if best is not None and best.id != self.prompt_combo.currentData():
            index = self.prompt_combo.findData(best.id)
            if index >= 0:
                self.prompt_combo.setCurrentIndex(index)
        self._update_controls()

    # ══ Queue ══════════════════════════════════════════════════════════
    def _requested_set(self) -> set[str]:
        return {
            os.path.normcase(str(paths.resolve_requested(self.queue.topLevelItem(row).data(0, ROLE_REQUESTED))))
            for row in range(self.queue.topLevelItemCount())
        }

    def _add_requested(self, requested: str) -> None:
        target = paths.resolve_requested(requested)
        item = QTreeWidgetItem([target.name, ""])
        item.setData(0, ROLE_REQUESTED, requested)
        item.setToolTip(0, str(target))
        self.queue.addTopLevelItem(item)
        self._set_item_status(item, "pending" if target.is_file() else "missing")

    def _add_paths(self, entries: list[Path]) -> None:
        if self.mode:
            return
        known = self._requested_set()
        added = skipped = 0
        for entry in entries:
            candidates: list[Path]
            if entry.is_dir():
                candidates = sorted(path for path in entry.rglob("*") if path.is_file() and paths.is_media(path))
            elif entry.is_file():
                candidates = [entry]
            else:
                candidates = []
            for path in candidates:
                key = os.path.normcase(str(path.resolve()))
                if key in known:
                    skipped += 1
                    continue
                known.add(key)
                self._add_requested(paths.display_path(path))
                added += 1
        if added:
            self._set_status(t("Додано файлів: {n}.", "Files added: {n}.").format(n=added))
        elif skipped:
            self._set_status(t("Ці файли вже є у списку.", "These files are already in the list."))
        elif entries:
            self._set_status(t("Не знайдено аудіо чи відео для додавання.", "No audio or video found to add."))
        self._update_controls()
        self._save_config(quiet=True)

    def _start_directory(self) -> str:
        stored = str(self.state.value("last_dir", ""))
        if stored and Path(stored).is_dir():
            return stored
        paths.INPUT_DIR.mkdir(parents=True, exist_ok=True)
        return str(paths.INPUT_DIR)

    def _browse_files(self) -> None:
        patterns = " ".join("*" + extension for extension in paths.MEDIA_EXTENSIONS)
        selected, _filter = QFileDialog.getOpenFileNames(
            self, t("Виберіть аудіо або відео", "Choose audio or video"), self._start_directory(),
            t("Аудіо та відео ({p});;Усі файли (*)", "Audio and video ({p});;All files (*)").format(p=patterns),
        )
        if selected:
            self.state.setValue("last_dir", str(Path(selected[0]).parent))
            self._add_paths([Path(name) for name in selected])

    def _browse_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, t("Виберіть папку із записами", "Choose a folder with recordings"), self._start_directory())
        if selected:
            self.state.setValue("last_dir", selected)
            self._add_paths([Path(selected)])

    def _remove_selected(self) -> None:
        if self.mode:
            return
        for item in self.queue.selectedItems():
            self.queue.takeTopLevelItem(self.queue.indexOfTopLevelItem(item))
        self._update_controls()
        self._save_config(quiet=True)

    def _clear_queue(self) -> None:
        if self.mode or not self.queue.topLevelItemCount():
            return
        self.queue.clear()
        self._update_controls()
        self._save_config(quiet=True)

    def _queue_menu(self, position) -> None:
        item = self.queue.itemAt(position)
        if item is None:
            return
        menu = QMenu(self)
        target = paths.resolve_requested(item.data(0, ROLE_REQUESTED))
        output = item.data(0, ROLE_OUTPUT)
        if output and Path(output).is_dir():
            menu.addAction(t("Відкрити результат", "Open result"), lambda: self._open_folder(Path(output)))
        menu.addAction(t("Показати файл у папці", "Show file in folder"), lambda: self._open_folder(target.parent))
        if not self.mode:
            menu.addSeparator()
            menu.addAction(t("Прибрати зі списку", "Remove from list"), self._remove_selected)
        menu.exec(self.queue.viewport().mapToGlobal(position))

    def _open_item_output(self, item: QTreeWidgetItem) -> None:
        output = item.data(0, ROLE_OUTPUT)
        if output and Path(output).is_dir():
            self._open_folder(Path(output))

    def _set_item_status(self, item: QTreeWidgetItem, status: str, detail: str = "", reason: str = "") -> None:
        item.setData(0, ROLE_STATUS, status)
        item.setIcon(0, theme.status_dot(status))
        item.setText(1, detail or catalog.status_name(status))
        item.setToolTip(1, reason or detail or catalog.status_name(status))
        color = QColor(theme.STATUS_COLORS[status]) if status in ("failed", "missing", "skipped", "done") else None
        item.setForeground(1, color if color else self.palette().text())

    def _item_for(self, requested: str) -> QTreeWidgetItem | None:
        for row in range(self.queue.topLevelItemCount()):
            item = self.queue.topLevelItem(row)
            if item.data(0, ROLE_REQUESTED) == requested:
                return item
        return None

    # ══ Controls state ═════════════════════════════════════════════════
    def _update_controls(self, *_args) -> None:
        if self._loading:
            return
        running = bool(self.mode)
        count = self.queue.topLevelItemCount()
        self.queue_title.setText(t("Файли ({n})", "Files ({n})").format(n=count) if count else t("Файли", "Files"))
        self.settings_panel.setEnabled(not running)
        for widget in (self.add_files_button, self.add_folder_button, self.clear_button):
            widget.setEnabled(not running)
        self.remove_button.setEnabled(not running and bool(self.queue.selectedItems()))
        self.clear_button.setEnabled(not running and count > 0)
        self.plan_button.setEnabled(not running and count > 0)
        self.language_menu.setEnabled(not running)
        if self.mode == "run":
            self.start_button.setText(t("Скасувати", "Cancel"))
            self.start_button.setObjectName("danger")
            self.start_button.setEnabled(True)
        else:
            self.start_button.setText(t("Старт", "Start"))
            self.start_button.setObjectName("primary")
            self.start_button.setEnabled(not running and count > 0)
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)

        timecoded = self.output_combo.currentData() in ("all", "timecode")
        self.blocks_check.setEnabled(timecoded)
        self.blocks_spin.setEnabled(timecoded and self.blocks_check.isChecked())
        self.blocks_hint.setEnabled(timecoded)

        use_prompt = self.prompt_check.isChecked()
        self.prompt_combo.setEnabled(use_prompt)
        prompt = self._current_prompt()
        self.prompt_preview.setText(f"«{prompt.text.strip()}»" if prompt else "")
        self.prompt_preview.setEnabled(use_prompt)
        language = self._language()
        mismatch = bool(use_prompt and prompt and prompt.language and language != prompt.language)
        self.prompt_warning.setVisible(mismatch)
        if mismatch:
            if language == "auto":
                message = t(
                    "З автовизначенням мови підказка мовою «{p}» може погіршити результат для інших мов.",
                    "With auto-detection, a prompt in “{p}” can make other languages worse.",
                ).format(p=catalog.language_label(prompt.language))
            else:
                message = t(
                    "Підказка написана для мови «{p}», а запис — «{a}». Якість може впасти. "
                    "<a href='#new' style='color:{c}'>Створити підказку для цієї мови</a>",
                    "The prompt is written for “{p}”, but the audio is “{a}”. Quality may drop. "
                    "<a href='#new' style='color:{c}'>Create a prompt for this language</a>",
                ).format(p=catalog.language_label(prompt.language), a=catalog.language_label(language),
                         c=theme.link_color())
            self.prompt_warning.setText(message)

        self.diarize_box.setVisible(self.diarize_check.isChecked())
        if os.environ.get("HF_TOKEN", "").strip():
            self.token_hint.setText(t(
                "Знайдено змінну середовища HF_TOKEN — буде використано її, а не це поле.",
                "The HF_TOKEN environment variable is set and will be used instead of this field."))
        else:
            self.token_hint.setText(t(
                "Потрібно один раз: <a href='{license}' style='color:{c}'>прийняти ліцензію pyannote</a> і "
                "<a href='{token}' style='color:{c}'>створити токен</a>. "
                "Токен зберігається лише у вашому config/config.toml.",
                "One-time setup: <a href='{license}' style='color:{c}'>accept the pyannote license</a> and "
                "<a href='{token}' style='color:{c}'>create a token</a>. "
                "The token is stored only in your config/config.toml.",
            ).format(license=PYANNOTE_URL, token=TOKEN_URL, c=theme.link_color()))

    def _speakers_changed(self, *_args) -> None:
        low, high = self.min_speakers.value(), self.max_speakers.value()
        if low > 0 and high > 0 and low > high:
            sender = self.sender()
            if sender is self.min_speakers:
                self.max_speakers.setValue(low)
            else:
                self.min_speakers.setValue(high)

    def _toggle_token(self, shown: bool) -> None:
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Normal if shown else QLineEdit.EchoMode.Password)
        self.token_show.setText(t("Сховати", "Hide") if shown else t("Показати", "Show"))

    def _toggle_log(self, shown: bool) -> None:
        self.log.setVisible(shown)
        self.log_toggle.setArrowType(Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _log(self, text: str) -> None:
        self.log.appendPlainText(text)

    # ══ Environment ════════════════════════════════════════════════════
    def _recheck(self) -> None:
        self.environment = None
        self.env_chip.setIcon(theme.status_dot("pending"))
        self.env_chip.setText(t("Перевірка системи…", "Checking system…"))
        self.probe.start()

    def _on_probe(self, info: dict) -> None:
        self.environment = info
        if info.get("torch") is None:
            self.env_chip.setIcon(theme.status_dot("failed"))
            self.env_chip.setText(t("Середовище не готове", "Environment not ready"))
        elif info.get("cuda_available"):
            self.env_chip.setIcon(theme.status_dot("done"))
            vram = f" · {info['vram_gb']:g} {t('ГБ', 'GB')}" if info.get("vram_gb") else ""
            self.env_chip.setText(f"{info.get('gpu') or 'GPU'}{vram}")
        else:
            self.env_chip.setIcon(theme.status_dot("skipped"))
            self.env_chip.setText(t("Без відеокарти NVIDIA", "No NVIDIA GPU"))
        self._refresh_banners()

    def _clear_banners(self) -> None:
        while self.banners.count():
            widget = self.banners.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()

    def _banner(self, text: str, button: str = "", action=None) -> None:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(text)
        label.setObjectName("warning")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(label, 1)
        if button:
            control = QPushButton(button)
            control.clicked.connect(action)
            layout.addWidget(control)
        self.banners.addWidget(row)

    def _refresh_banners(self, *_args) -> None:
        self._clear_banners()
        info = self.environment
        if not info:
            return
        if info.get("torch") is None or info.get("whisperx") is None:
            self._banner(
                t("Компоненти для розпізнавання (PyTorch, WhisperX) не встановлено в цьому середовищі. "
                  "Запустіть install.bat — це одноразово, потрібен інтернет і близько 10 ГБ на диску.",
                  "The transcription components (PyTorch, WhisperX) are not installed in this environment. "
                  "Run install.bat once; it needs internet and about 10 GB of disk space."),
                t("Запустити install.bat", "Run install.bat") if os.name == "nt" else "", self._run_install)
        elif not info.get("cuda_available") and self.device_combo.currentData() == "cuda":
            reason = (t("встановлено збірку PyTorch без CUDA", "the installed PyTorch build has no CUDA")
                      if not info.get("cuda_build") else
                      t("відеокарту NVIDIA або її драйвер не знайдено", "no NVIDIA GPU or driver was found"))
            self._banner(
                t("Вибрано роботу на відеокарті, але {reason}. На процесорі розпізнавання працює, "
                  "тільки значно повільніше.", "GPU mode is selected, but {reason}. Transcription works "
                  "on the processor, only much slower.").format(reason=reason),
                t("Перемкнути на процесор", "Switch to processor"),
                lambda: self.device_combo.setCurrentIndex(self.device_combo.findData("cpu")))
        if "python" in info and not info.get("ffmpeg"):
            self._banner(t(
                "Не знайдено ffmpeg — без нього файли не відкриються. Встановіть його командою "
                "«winget install Gyan.FFmpeg» і перезапустіть програму, або покладіть ffmpeg.exe "
                "у папку tools\\ffmpeg поруч із програмою.",
                "ffmpeg was not found; media files cannot be opened without it. Install it with "
                "“winget install Gyan.FFmpeg” and restart the app, or put ffmpeg.exe into the "
                "tools\\ffmpeg folder next to the app."))

    def _run_install(self) -> None:
        if run_install_script():
            self._set_status(t(
                "install.bat запущено в окремому вікні. Коли він завершиться, виберіть «Меню → Перевірити систему ще раз».",
                "install.bat is running in a separate window. When it finishes, choose “Menu → Re-check system”."))

    def _show_environment(self) -> None:
        info = self.environment
        if not info:
            return
        yes, no = t("так", "yes"), t("ні", "no")
        rows = [
            ("Python", info.get("python") or "—"),
            ("PyTorch", info.get("torch") or t("не встановлено", "not installed")),
            ("WhisperX", info.get("whisperx") or t("не встановлено", "not installed")),
            (t("Збірка CUDA", "CUDA build"), info.get("cuda_build") or "—"),
            (t("Відеокарта доступна", "GPU available"), yes if info.get("cuda_available") else no),
            (t("Відеокарта", "GPU"), info.get("gpu") or "—"),
            (t("Відеопам'ять", "Video memory"), f"{info['vram_gb']:g} {t('ГБ', 'GB')}" if info.get("vram_gb") else "—"),
            ("ffmpeg", info.get("ffmpeg") or t("не знайдено", "not found")),
        ]
        if info.get("error"):
            rows.append((t("Помилка", "Error"), str(info["error"])))
        text = "<table cellspacing='6'>" + "".join(
            f"<tr><td><b>{name}</b></td><td>{value}</td></tr>" for name, value in rows) + "</table>"
        box = QMessageBox(self)
        box.setWindowTitle(t("Система", "System"))
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(text)
        box.exec()

    # ══ Running ════════════════════════════════════════════════════════
    def _start_or_cancel(self) -> None:
        if self.mode == "run":
            self._cancel()
        else:
            self._start("run")

    def _start(self, mode: str) -> None:
        if self.mode or not self.queue.topLevelItemCount():
            return
        settings = self._collect()
        if mode == "run":
            if (settings["diarize"] and not settings["hf_token"]
                    and not os.environ.get("HF_TOKEN", "").strip()):
                answer = QMessageBox.question(self, "WhisperBlocks", t(
                    "Увімкнено розпізнавання спікерів, але токен Hugging Face не вказано. "
                    "Без нього текст буде без міток спікерів.\n\nПродовжити без міток?",
                    "Speaker detection is on, but no Hugging Face token is set. Without it the text "
                    "will have no speaker labels.\n\nContinue without labels?"))
                if answer != QMessageBox.StandardButton.Yes:
                    self.token_edit.setFocus()
                    return
            info = self.environment or {}
            if info and info.get("torch") is None:
                QMessageBox.warning(self, "WhisperBlocks", t(
                    "Компоненти для розпізнавання не встановлено. Спершу запустіть install.bat.",
                    "The transcription components are not installed. Run install.bat first."))
                return
        if not self.config_ok:
            QMessageBox.warning(self, "WhisperBlocks", t(
                "config/config.toml не читається, тому запуск неможливий. Виправте або видаліть файл і перезапустіть програму.",
                "config/config.toml cannot be read, so nothing can run. Fix or delete the file and restart the app."))
            return
        if not self._save_config():
            return

        self.mode = mode
        self.progress = events.ProgressModel(diarize=settings["diarize"])
        self.summary, self.run_error, self.active_item = None, "", None
        for row in range(self.queue.topLevelItemCount()):
            item = self.queue.topLevelItem(row)
            item.setData(0, ROLE_OUTPUT, None)
            self._set_item_status(item, "pending")
        arguments = ["-u", str(paths.TRANSCRIPT_SCRIPT)]
        if mode == "plan":
            arguments.append("--dry-run")
        if self.overwrite_check.isChecked():
            arguments.append("--overwrite")
        self.log.clear()
        self.progress_bar.setValue(0)
        if mode == "run":
            self.progress_bar.setRange(0, 0)       # busy until the first event
            self.started_at = time.monotonic()
            self.clock.start()
            self._set_status(t("Запуск…", "Starting…"))
        else:
            self._set_status(t("Перевірка…", "Checking…"))
        self.time_label.setText("")
        self._update_controls()
        self.runner.start(arguments, extra_env={events.ENV_NAME: "1"})

    def _cancel(self) -> None:
        answer = QMessageBox.question(self, "WhisperBlocks", t(
            "Зупинити обробку? Уже готові файли залишаться, поточний файл доведеться почати заново.",
            "Stop processing? Finished files are kept; the current file will have to start over."))
        if answer == QMessageBox.StandardButton.Yes and self.mode == "run":
            self._set_status(t("Зупинка…", "Stopping…"))
            self.runner.kill()

    def _on_failed_to_start(self, message: str) -> None:
        self._log(message)
        self.run_error = t("Не вдалося запустити Python: {m}", "Could not start Python: {m}").format(m=message)
        self._on_finished(-1, False)

    def _on_line(self, text: str) -> None:
        event = events.parse_line(text)
        if event is None:
            if text.strip() or self.log.blockCount() > 1:
                self._log(text)
            return
        self.progress.apply(event)
        kind = event["event"]
        if kind in ("stage", "file_start"):
            self.stage_started = time.monotonic()
        if kind == "plan":
            for entry in event.get("items", []):
                item = self._item_for(entry.get("requested", ""))
                if item is None:
                    continue
                item.setData(0, ROLE_OUTPUT, entry.get("output_dir"))
                status = {"runnable": "ready" if self.mode == "plan" else "pending",
                          "skipped_existing": "skipped", "invalid_or_missing": "missing"}.get(entry.get("status"), "pending")
                self._set_item_status(item, status, reason=entry.get("reason") or "")
        elif kind == "file_start":
            self.active_item = self._item_for(event.get("requested", ""))
            if self.active_item is not None:
                self._set_item_status(self.active_item, "running")
                self.queue.scrollToItem(self.active_item)
        elif kind == "file_done":
            item = self._item_for(event.get("requested", ""))
            if item is not None:
                status = {"succeeded": "done", "skipped_existing": "skipped"}.get(event.get("status"), "failed")
                detail = ""
                if status == "done" and event.get("seconds") is not None:
                    detail = t("Готово за {time}", "Done in {time}").format(time=format_clock(event["seconds"])[3:]
                               if event["seconds"] < 3600 else format_clock(event["seconds"]))
                if event.get("output_dir"):
                    item.setData(0, ROLE_OUTPUT, event["output_dir"])
                self._set_item_status(item, status, detail, event.get("reason") or "")
            self.active_item = None
        elif kind == "error":
            self.run_error = str(event.get("message", ""))
        elif kind == "summary":
            self.summary = event
        self._render_progress()

    def _render_progress(self) -> None:
        if self.mode != "run":
            return
        model = self.progress
        if model.total > 0:
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setValue(round(model.fraction() * 1000))
        if model.stage == "preprocess":
            self._set_status(catalog.stage_name("preprocess") + "…")
            return
        if model.index and model.stage:
            stage = catalog.stage_name(model.stage)
            percent = f" {model.percent}%" if model.stage != "load" and model.percent else "…"
            text = t("Файл {i} з {n} · {stage}{percent}", "File {i} of {n} · {stage}{percent}").format(
                i=model.index, n=max(model.total, model.index), stage=stage, percent=percent)
            if model.stage == "load" and time.monotonic() - self.stage_started > 10:
                text += " " + t(
                    "(нова модель спершу завантажується з інтернету — це може тривати кілька хвилин)",
                    "(a new model is downloaded first; this can take a few minutes)")
            self._set_status(text)
            if self.active_item is not None:
                self.active_item.setText(1, f"{stage}{percent}")

    def _tick(self) -> None:
        if self.mode != "run":
            return
        elapsed = time.monotonic() - self.started_at
        text = t("Минуло {time}", "Elapsed {time}").format(time=format_clock(elapsed))
        if self.progress.stage == "load":
            self._render_progress()
        fraction = self.progress.fraction()
        if fraction >= 0.05 and elapsed >= 30 and fraction < 1:
            text += " · " + t("залишилось {eta}", "{eta} left").format(
                eta=format_remaining(elapsed * (1 - fraction) / fraction))
        self.time_label.setText(text)

    def _on_finished(self, code: int, crashed: bool) -> None:
        mode, self.mode = self.mode, ""
        if not mode:
            return
        self.clock.stop()
        self.progress_bar.setRange(0, 1000)
        cancelled = crashed and self.runner.killed
        for row in range(self.queue.topLevelItemCount()):
            item = self.queue.topLevelItem(row)
            status = item.data(0, ROLE_STATUS)
            if status == "running" or (mode == "run" and status == "pending" and (cancelled or crashed or self.run_error)):
                self._set_item_status(item, "cancelled" if cancelled else "failed" if status == "running" else "pending")
        self.active_item = None
        counts = {name: 0 for name in ("ready", "done", "skipped", "failed", "missing")}
        for row in range(self.queue.topLevelItemCount()):
            status = self.queue.topLevelItem(row).data(0, ROLE_STATUS)
            if status in counts:
                counts[status] += 1
        if mode == "plan":
            self.progress_bar.setValue(0)
            if self.run_error:
                self._set_status(t("Помилка налаштувань: {m}", "Configuration error: {m}").format(m=self.run_error))
            else:
                parts = [t("до обробки: {n}", "to process: {n}").format(n=counts["ready"])]
                if counts["skipped"]:
                    parts.append(t("уже мають результат: {n}", "already have results: {n}").format(n=counts["skipped"]))
                if counts["missing"]:
                    parts.append(t("недоступні: {n}", "unavailable: {n}").format(n=counts["missing"]))
                self._set_status(t("Перевірено — ", "Checked — ") + ", ".join(parts) + ".")
        elif cancelled:
            self._set_status(t("Зупинено. Готових файлів: {n}.", "Stopped. Files finished: {n}.").format(n=counts["done"]))
        elif self.run_error:
            self._set_status(t("Помилка: {m}", "Error: {m}").format(m=self.run_error))
        elif self.summary is None:
            self._set_status(t(
                "Обробка несподівано завершилась (код {code}). Подробиці в журналі.",
                "Processing ended unexpectedly (code {code}). See the log for details.").format(code=code))
            self.log_toggle.setChecked(True)
        else:
            failed = counts["failed"] + counts["missing"]
            parts = [t("готово: {n}", "done: {n}").format(n=counts["done"])]
            if counts["skipped"]:
                parts.append(t("пропущено: {n}", "skipped: {n}").format(n=counts["skipped"]))
            if failed:
                parts.append(t("з помилками: {n}", "failed: {n}").format(n=failed))
            if self.summary.get("failed_preprocess"):
                parts.append(t("блоки не створено: {n}", "blocks not created: {n}").format(
                    n=self.summary["failed_preprocess"]))
            text = t("Завершено — ", "Finished — ") + ", ".join(parts) + "."
            if counts["skipped"] and not counts["done"] and not failed:
                text += " " + t(
                    "Щоб обробити ще раз, увімкніть «Перезаписувати наявні результати» в розділі «Додатково».",
                    "To process again, enable “Overwrite existing results” under “Advanced”.")
            self._set_status(text)
            self.progress_bar.setValue(1000 if not failed else self.progress_bar.value())
            if failed:
                self.log_toggle.setChecked(True)
        if mode == "run":
            elapsed = time.monotonic() - self.started_at
            self.time_label.setText(t("Тривалість {time}", "Took {time}").format(time=format_clock(elapsed)))
            QApplication.alert(self)
        self._update_controls()

    # ══ Menu actions ═══════════════════════════════════════════════════
    def _open_folder(self, path: Path) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        open_in_file_manager(path)

    def _open_config(self) -> None:
        self._save_config(quiet=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(paths.CONFIG_PATH)))

    def _set_ui_language(self, code: str) -> None:
        if code == i18n.language() or self.mode:
            return
        self._save_config(quiet=True)
        self._save_state()
        self.state.setValue("ui_language", code)
        self.state.sync()
        i18n.set_language(code)
        self.rebuild_requested.emit()

    def _set_theme(self, code: str) -> None:
        self.state.setValue("theme", code)
        theme.set_scheme(code)
        theme.refresh_style(QApplication.instance())
        for row in range(self.queue.topLevelItemCount()):
            item = self.queue.topLevelItem(row)
            item.setIcon(0, theme.status_dot(item.data(0, ROLE_STATUS) or "pending"))
        self._update_controls()

    def _about(self) -> None:
        QMessageBox.about(self, t("Про програму", "About"), t(
            "<b>WhisperBlocks</b><br>Локальна транскрипція аудіо та відео на основі WhisperX.<br><br>"
            "Записи обробляються на вашому комп'ютері й нікуди не надсилаються. "
            "Інтернет потрібен лише для першого завантаження моделей.<br><br>"
            "Налаштування: config/config.toml · власні підказки: config/prompts.toml",
            "<b>WhisperBlocks</b><br>Local audio and video transcription built on WhisperX.<br><br>"
            "Recordings are processed on your computer and are not sent anywhere. "
            "Internet is only needed to download models the first time.<br><br>"
            "Settings: config/config.toml · your prompts: config/prompts.toml"))

    # ══ Window events ══════════════════════════════════════════════════
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls() and not self.mode:
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        files = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if files:
            event.acceptProposedAction()
            self._add_paths(files)

    def closeEvent(self, event) -> None:
        if self.mode == "run":
            answer = QMessageBox.question(self, "WhisperBlocks", t(
                "Обробка ще триває. Зупинити її та закрити програму?",
                "Processing is still running. Stop it and close the app?"))
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        if self.runner.is_running():
            self.mode = ""
            self.runner.kill()
            self.runner.process.waitForFinished(3000)
        self.probe.runner.kill()
        self._save_config(quiet=True)
        self._save_state()
        super().closeEvent(event)
