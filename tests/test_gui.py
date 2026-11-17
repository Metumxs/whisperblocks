"""Offline checks for the GUI's non-Qt logic, launchers and progress events.

No PySide6, GPU, models or network. Tests needing tomlkit (a GUI-only
dependency) are skipped when it is not installed.
"""

import importlib.util
import io
import json
import sys
import tomllib
import unittest
from contextlib import redirect_stdout
from unittest import mock

from support import PROJECT_SOURCE, ProjectTestCase, transcript

from gui import config_store, events, paths, prompts

HAVE_TOMLKIT = importlib.util.find_spec("tomlkit") is not None
needs_tomlkit = unittest.skipUnless(HAVE_TOMLKIT, "tomlkit (GUI dependency) is not installed")


class GuiDefaultsTests(unittest.TestCase):
    def test_gui_defaults_match_transcript_defaults(self):
        self.assertEqual(config_store.DEFAULTS, transcript.DEFAULT_CONFIG)

    def test_builtin_prompts_are_exactly_the_three_template_hints(self):
        template = (PROJECT_SOURCE / "config/config.toml.example").read_text(encoding="utf-8")
        self.assertEqual([prompt.id for prompt in prompts.BUILTIN_PROMPTS], ["builtin-uk", "builtin-en", "builtin-ru"])
        for prompt in prompts.BUILTIN_PROMPTS:
            with self.subTest(prompt=prompt.id):
                self.assertTrue(prompt.builtin)
                self.assertIn(f'initial_prompt = "{prompt.text}"', template)

    def test_gui_modules_without_qt_do_not_import_qt_or_ml(self):
        for name in ("config_store", "events", "paths", "prompts", "i18n", "catalog", "probe"):
            text = (PROJECT_SOURCE / "src/gui" / f"{name}.py").read_text(encoding="utf-8")
            with self.subTest(module=name):
                self.assertNotRegex(text, r"(?m)^\s*(from|import) (PySide6|whisperx)")
        self.assertNotIn("PySide6", sys.modules)


@needs_tomlkit
class ConfigStoreTests(ProjectTestCase):
    def setUp(self):
        super().setUp()
        self.path = self.root / "config/config.toml"
        self.example = PROJECT_SOURCE / "config/config.toml.example"

    def test_seed_load_and_save_preserve_comments_and_line_endings(self):
        self.assertTrue(config_store.ensure_config(self.path, self.example))
        self.assertFalse(config_store.ensure_config(self.path, self.example))
        original = self.path.read_bytes()
        settings, problems = config_store.load_settings(self.path)
        self.assertEqual(problems, [])
        self.assertEqual(settings["files"], [])
        self.assertFalse(config_store.save_settings(self.path, settings), "unchanged settings must not rewrite the file")
        self.assertEqual(self.path.read_bytes(), original)

        settings.update(files=["input/a b.mp3", "D:/Записи/лекція.mp4"], language="en", batch_size=4, diarize=True)
        self.assertTrue(config_store.save_settings(self.path, settings))
        saved = self.path.read_bytes()
        self.assertEqual(saved.count(b"\r\n") > 0, original.count(b"\r\n") > 0)
        text = saved.decode("utf-8")
        for comment in ("# RECOMMENDED: \"large-v3-turbo\"", "# SECURITY: hf_token is a secret.", "# TRANSCRIPTION SETTINGS"):
            self.assertIn(comment, text)
        parsed = tomllib.loads(text)
        self.assertEqual(parsed["files"], ["input/a b.mp3", "D:/Записи/лекція.mp4"])
        self.assertEqual((parsed["language"], parsed["batch_size"], parsed["diarize"]), ("en", 4, True))
        # The pre-GUI version is kept once and never overwritten afterwards.
        backup = self.path.with_name("config.toml.bak")
        self.assertEqual(backup.read_bytes(), original)
        config_store.save_settings(self.path, {"language": "pl"})
        self.assertEqual(backup.read_bytes(), original)

    def test_saved_config_is_accepted_by_transcript_validation(self):
        config_store.ensure_config(self.path, self.example)
        settings, _ = config_store.load_settings(self.path)
        settings.update(files=["input/a.mp3"], min_speakers=2, max_speakers=3, auto_preprocess_block_seconds=0)
        config_store.save_settings(self.path, settings)
        validated = transcript.validate_config(tomllib.loads(self.path.read_text(encoding="utf-8")), environ={})
        self.assertEqual(validated["files"], ["input/a.mp3"])
        self.assertEqual((validated["min_speakers"], validated["max_speakers"]), (2, 3))

    def test_unusable_values_fall_back_to_defaults_and_are_reported(self):
        self.path.write_text(
            'files = "oops"\nbatch_size = 0\ndevice = "gpu"\noutput_mode = "SRT"\ndiarize = "yes"\nextra = 1\n',
            encoding="utf-8",
        )
        settings, problems = config_store.load_settings(self.path)
        self.assertEqual(sorted(problems), ["batch_size", "device", "diarize", "files"])
        self.assertEqual(settings["batch_size"], 8)
        self.assertEqual(settings["output_mode"], "srt")
        config_store.save_settings(self.path, settings)
        self.assertEqual(tomllib.loads(self.path.read_text(encoding="utf-8"))["extra"], 1, "unknown keys are kept")

    def test_broken_toml_raises_config_error_and_invalid_values_are_not_written(self):
        self.path.write_text("files = [\n", encoding="utf-8")
        with self.assertRaises(config_store.ConfigError):
            config_store.load_settings(self.path)
        with self.assertRaises(config_store.ConfigError):
            config_store.save_settings(self.path, {"language": "uk"})
        self.path.write_text('language = "uk"\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            config_store.save_settings(self.path, {"batch_size": 0})
        self.assertEqual(self.path.read_text(encoding="utf-8"), 'language = "uk"\n')


@needs_tomlkit
class PromptStoreTests(ProjectTestCase):
    def store(self):
        return prompts.PromptStore(self.root / "config/prompts.toml")

    def test_builtin_prompts_cannot_be_changed_or_deleted(self):
        store = self.store()
        self.assertEqual(len(store.prompts), 3)
        for prompt in prompts.BUILTIN_PROMPTS:
            with self.subTest(prompt=prompt.id):
                with self.assertRaises(prompts.PromptError):
                    store.delete(prompt.id)
                with self.assertRaises(prompts.PromptError):
                    store.update(prompt.id, "x", "y")
        self.assertEqual(store.prompts[:3], list(prompts.BUILTIN_PROMPTS))
        self.assertFalse(store.path.exists())

    def test_custom_prompts_round_trip_and_never_shadow_builtins(self):
        store = self.store()
        own = store.add("Інтерв'ю", 'Це інтерв\'ю. "Чистий" текст:\nдругий рядок ', "UK")
        copy = store.duplicate("builtin-en", "copy")
        self.assertEqual((own.language, copy.text), ("uk", prompts.BUILTIN_PROMPTS[1].text))
        self.assertFalse(copy.builtin)
        with self.assertRaises(prompts.PromptError):
            store.add("інтерв'ю", "duplicate name differs only by case")
        with self.assertRaises(prompts.PromptError):
            store.add("Порожня", "   ")
        store.update(own.id, "Інтерв'ю 2", own.text, "")
        reloaded = self.store()
        self.assertEqual([p.name for p in reloaded.custom], ["Інтерв'ю 2", copy.name])
        self.assertEqual(reloaded.get(own.id).text, own.text)
        reloaded.delete(copy.id)
        self.assertEqual(len(self.store().prompts), 4)
        # A hand-edited file reusing a built-in id gets a fresh id instead of replacing it.
        store.path.write_text('[[prompt]]\nid = "builtin-uk"\nname = "Fake"\ntext = "x"\n', encoding="utf-8")
        tampered = self.store()
        self.assertEqual(tampered.get("builtin-uk"), prompts.BUILTIN_PROMPTS[0])
        self.assertEqual(len(tampered.custom), 1)

    def test_language_matching_prefers_current_then_own_then_builtin(self):
        store = self.store()
        uk, en, _ru = prompts.BUILTIN_PROMPTS
        self.assertEqual(store.best_for_language("en", uk), en)
        self.assertEqual(store.best_for_language("uk", uk), uk)
        self.assertEqual(store.best_for_language("pl", uk), uk, "no match keeps the current prompt")
        self.assertEqual(store.best_for_language("auto", en), en)
        own = store.add("My English", "Clean text: ", "en")
        self.assertEqual(store.best_for_language("en", uk), own)
        anything = store.add("Any", "Text: ", "")
        self.assertEqual(store.best_for_language("pl", anything), anything)
        self.assertEqual(store.find_by_text(uk.text), uk)
        self.assertIsNone(store.find_by_text("unknown"))

    def test_broken_prompt_file_keeps_builtins(self):
        store = self.store()
        store.path.write_text("[[prompt]\n", encoding="utf-8")
        store.load()
        self.assertTrue(store.load_error)
        self.assertEqual(store.prompts, list(prompts.BUILTIN_PROMPTS))


class EventProtocolTests(ProjectTestCase):
    def capture_main(self, args, *, enabled=True, **run):
        output = io.StringIO()
        with mock.patch.dict("os.environ", {transcript.EVENTS_ENV: "1" if enabled else ""}):
            with redirect_stdout(output):
                code, _boundary, _summaries = self.run_main(args, **run)
        parsed = [events.parse_line(line) for line in output.getvalue().splitlines()]
        return code, [event for event in parsed if event], output.getvalue()

    def test_protocol_constants_are_shared(self):
        self.assertEqual((events.PREFIX, events.ENV_NAME), (transcript.EVENT_PREFIX, transcript.EVENTS_ENV))

    def test_console_runs_emit_nothing(self):
        self.media()
        self.write_config()
        code, parsed, raw = self.capture_main([], enabled=False)
        self.assertEqual((code, parsed, raw), (0, [], ""))
        self.assertEqual(transcript.stage_progress("transcribe"), {})

    def test_batch_events_drive_the_progress_model(self):
        self.media("input/a.mp3")
        self.media("input/b.mp3")
        self.write_config(files=["input/a.mp3", "input/b.mp3", "input/missing.mp3"], output_mode="simple")
        code, parsed, _raw = self.capture_main([])
        self.assertEqual(code, 1)
        kinds = [event["event"] for event in parsed]
        self.assertEqual(kinds, ["plan", "file_start", "file_done", "file_start", "file_done", "summary"])
        plan = parsed[0]
        self.assertEqual([item["status"] for item in plan["items"]], ["runnable", "runnable", "invalid_or_missing"])
        self.assertEqual([item["requested"] for item in plan["items"]], ["input/a.mp3", "input/b.mp3", "input/missing.mp3"])
        self.assertEqual(parsed[2]["status"], "succeeded")
        self.assertTrue(parsed[2]["output_dir"].endswith("a"))
        self.assertEqual((parsed[-1]["succeeded"], parsed[-1]["missing_or_invalid"]), (2, 1))

        model = events.ProgressModel()
        fractions = []
        for event in parsed:
            model.apply(event)
            fractions.append(model.fraction())
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual((model.total, fractions[-1]), (2, 1.0))

    def test_dry_run_emits_only_the_plan_and_config_errors_are_reported(self):
        self.media()
        self.write_config()
        code, parsed, _raw = self.capture_main(["--dry-run"])
        self.assertEqual((code, [event["event"] for event in parsed]), (0, ["plan"]))
        self.write_config(batch_size=0)
        code, parsed, _raw = self.capture_main([])
        self.assertEqual((code, parsed[0]["event"], parsed[0]["kind"]), (2, "error", "config"))

    def test_events_never_contain_the_token(self):
        self.media()
        self.write_config(diarize=True, hf_token="hf_secretsecretsecret")
        failing = mock.Mock(side_effect=RuntimeError("401 for token hf_secretsecretsecret"))
        output = io.StringIO()
        with mock.patch.dict("os.environ", {transcript.EVENTS_ENV: "1"}), redirect_stdout(output):
            self.patch(transcript, "load_torch", mock.Mock(return_value=mock.MagicMock()))
            self.patch(transcript, "transcribe_file", failing)
            code = transcript.main([])
        self.assertEqual(code, 1)
        self.assertNotIn("hf_secretsecretsecret", output.getvalue())
        self.assertIn('"status": "failed_transcription"', output.getvalue())

    def test_stage_progress_reports_whole_percent_changes_once(self):
        output = io.StringIO()
        with mock.patch.dict("os.environ", {transcript.EVENTS_ENV: "1"}), redirect_stdout(output):
            callback = transcript.stage_progress("align")["progress_callback"]
            for value in (0.4, 0.9, 12.2, 12.7, 250, "bad"):
                callback(value)
        percents = [json.loads(line[len(events.PREFIX):])["percent"] for line in output.getvalue().splitlines()]
        self.assertEqual(percents, [0, 12, 100])

    def test_progress_model_weights_and_stream_splitting(self):
        model = events.ProgressModel(diarize=True)
        for event in (
            {"event": "plan", "items": [{"status": "runnable"}, {"status": "skipped_existing"}]},
            {"event": "file_start", "index": 1, "total": 1},
            {"event": "stage", "stage": "transcribe"},
            {"event": "progress", "stage": "transcribe", "percent": 50},
        ):
            model.apply(event)
        self.assertAlmostEqual(model.fraction(), 0.06 + 0.46 / 2)
        model.apply({"event": "stage", "stage": "diarize"})
        self.assertAlmostEqual(model.fraction(), 0.72)
        self.assertEqual(sum(events._WEIGHTS.values()), 1.0)
        self.assertAlmostEqual(sum(events._WEIGHTS_DIARIZE.values()), 1.0)
        lines, tail = events.split_stream("one\r\n 10%\r 55%\r100%\npart")
        self.assertEqual((lines, tail), (["one", "100%"], "part"))
        self.assertIsNone(events.parse_line("@@WB not json"))
        self.assertIsNone(events.parse_line("plain text"))


class GuiLauncherTests(ProjectTestCase):
    def test_gui_batch_launcher_uses_the_project_venv_and_forwards_files(self):
        data = (PROJECT_SOURCE / "run-gui.bat").read_bytes()
        self.assertNotIn(b"\n", data.replace(b"\r\n", b""), "Windows launcher lost CRLF line endings")
        text = data.decode("utf-8").replace("\r\n", "\n")
        self.assertIn('start "" ".venv\\Scripts\\pythonw.exe" src\\gui_main.py %*', text)
        self.assertIn("-m pip install -r requirements\\requirements-gui.txt", text)
        self.assertNotRegex(text.lower(), r"(?m)^\s*(rmdir|rd|del)\s")

    def test_gui_requirements_are_separate_and_never_pull_torch(self):
        text = (PROJECT_SOURCE / "requirements/requirements-gui.txt").read_text(encoding="utf-8")
        dependencies = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
        self.assertEqual(dependencies, ["PySide6-Essentials>=6.8,<7", "tomlkit>=0.12"])

    def test_native_launcher_source_only_starts_local_project_files(self):
        source = (PROJECT_SOURCE / "launcher/launcher.c").read_text(encoding="utf-8")
        self.assertIn('\\\\.venv\\\\Scripts\\\\pythonw.exe', source)
        self.assertIn('src\\\\gui_main.py', source)
        self.assertNotRegex(source, r"(?i)https?://|URLDownload|WinHttp|InternetOpen|RegSetValue|DeleteFile")
        executable = PROJECT_SOURCE / "WhisperBlocks.exe"
        if executable.exists():
            self.assertEqual(executable.read_bytes()[:2], b"MZ")
            self.assertLess(executable.stat().st_size, 1_000_000, "the launcher must stay a tiny stub")

    def test_path_helpers(self):
        root = self.root
        inside = self.media("input/курс/a b.mp3")
        self.assertEqual(paths.display_path(inside, root), "input/курс/a b.mp3")
        outside = self.workspace / "external.wav"
        outside.write_bytes(b"x")
        self.assertEqual(paths.display_path(outside, root), outside.as_posix())
        self.assertEqual(paths.resolve_requested("input/курс/a b.mp3", root), root / "input/курс/a b.mp3")
        self.assertEqual(paths.resolve_requested("курс/a b.mp3", root), root / "input/курс/a b.mp3")
        self.assertTrue(paths.is_media(inside))
        self.assertFalse(paths.is_media(root / "notes.txt"))
        self.assertEqual(paths.bundled_ffmpeg_dirs(root), [])


if __name__ == "__main__":
    unittest.main()
