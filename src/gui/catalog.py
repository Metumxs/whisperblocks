"""Choices offered by the GUI drop-downs (no Qt imports)."""

from __future__ import annotations

from .i18n import t

# (code, Ukrainian name, English name). Any other ISO-639-1 code can be typed in.
AUDIO_LANGUAGES = (
    ("uk", "Українська", "Ukrainian"), ("en", "Англійська", "English"),
    ("auto", "Визначити автоматично", "Detect automatically"),
    ("pl", "Польська", "Polish"), ("de", "Німецька", "German"), ("fr", "Французька", "French"),
    ("es", "Іспанська", "Spanish"), ("it", "Італійська", "Italian"), ("pt", "Португальська", "Portuguese"),
    ("cs", "Чеська", "Czech"), ("sk", "Словацька", "Slovak"), ("ro", "Румунська", "Romanian"),
    ("bg", "Болгарська", "Bulgarian"), ("lt", "Литовська", "Lithuanian"), ("lv", "Латиська", "Latvian"),
    ("et", "Естонська", "Estonian"), ("nl", "Нідерландська", "Dutch"), ("sv", "Шведська", "Swedish"),
    ("tr", "Турецька", "Turkish"), ("ja", "Японська", "Japanese"), ("zh", "Китайська", "Chinese"),
    ("ko", "Корейська", "Korean"), ("ru", "Російська", "Russian"),
)


def language_label(code: str) -> str:
    for known, uk, en in AUDIO_LANGUAGES:
        if known == code:
            return t(uk, en) if code == "auto" else f"{t(uk, en)} ({code})"
    return code


def models() -> list[tuple[str, str]]:
    return [
        ("large-v3-turbo", t("large-v3-turbo — швидка й точна (рекомендовано)", "large-v3-turbo — fast and accurate (recommended)")),
        ("large-v3", t("large-v3 — найточніша, повільніша", "large-v3 — most accurate, slower")),
        ("large-v2", t("large-v2 — попереднє покоління", "large-v2 — previous generation")),
        ("medium", t("medium — середня якість, менше пам'яті", "medium — medium quality, less memory")),
        ("small", t("small — швидка чернетка", "small — quick draft")),
        ("base", t("base — дуже швидка, низька якість", "base — very fast, low quality")),
        ("tiny", t("tiny — лише для перевірки", "tiny — for testing only")),
    ]


def output_modes() -> list[tuple[str, str]]:
    return [
        ("all", t("Усі формати: таймкоди, простий текст і субтитри", "All formats: timecodes, plain text and subtitles")),
        ("timecode", t("Текст із таймкодами (.txt)", "Text with timecodes (.txt)")),
        ("simple", t("Простий текст без таймкодів (.txt)", "Plain text without timecodes (.txt)")),
        ("srt", t("Субтитри (.srt)", "Subtitles (.srt)")),
    ]


def devices() -> list[tuple[str, str]]:
    return [
        ("cuda", t("Відеокарта NVIDIA (CUDA) — швидко", "NVIDIA GPU (CUDA) — fast")),
        ("cpu", t("Процесор (CPU) — повільно, але працює всюди", "Processor (CPU) — slow, works everywhere")),
    ]


def stage_name(stage: str) -> str:
    return {
        "load": t("Завантаження моделі", "Loading model"),
        "transcribe": t("Розпізнавання", "Transcribing"),
        "align": t("Вирівнювання слів", "Aligning words"),
        "diarize": t("Розпізнавання спікерів", "Identifying speakers"),
        "preprocess": t("Об'єднання в блоки", "Merging into blocks"),
    }.get(stage, stage)


def status_name(status: str) -> str:
    return {
        "pending": t("У черзі", "Queued"),
        "ready": t("Готовий до обробки", "Ready"),
        "running": t("Обробляється…", "Processing…"),
        "done": t("Готово", "Done"),
        "skipped": t("Пропущено: результат уже є", "Skipped: result exists"),
        "failed": t("Помилка", "Failed"),
        "missing": t("Файл недоступний", "File unavailable"),
        "cancelled": t("Скасовано", "Cancelled"),
    }.get(status, status)
