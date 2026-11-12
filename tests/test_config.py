import tomllib
import unittest

from support import PROJECT_SOURCE, transcript


class ConfigTests(unittest.TestCase):
    def validate(self, **updates):
        return transcript.validate_config({"files": ["input/a.mp3"], **updates}, environ={})

    def test_valid_defaults_match_existing_runtime(self):
        config = self.validate()
        for key, expected in (
            ("model_size", "large-v3-turbo"), ("device", "cuda"), ("language", "uk"),
            ("output_mode", "all"), ("batch_size", 8), ("auto_preprocess_block_seconds", 60),
            ("diarize", False), ("remove_fillers", False), ("initial_prompt", ""),
        ):
            self.assertEqual(config[key], expected)

    def test_template_is_toml_with_consistent_defaults(self):
        raw = tomllib.loads((PROJECT_SOURCE / "config/config.toml.example").read_text(encoding="utf-8"))
        self.assertEqual(raw["auto_preprocess_block_seconds"], 60)
        self.assertEqual(raw["batch_size"], 8)
        config = transcript.validate_config(raw, ["input/a.mp3"], environ={})
        self.assertTrue(config["remove_fillers"])
        self.assertIn("Це лекція", config["initial_prompt"])

    def test_output_mode_is_case_insensitive(self):
        for mode in ("ALL", "TimeCode", "Simple", "SRT"):
            with self.subTest(mode=mode):
                self.assertEqual(self.validate(output_mode=mode)["output_mode"], mode.lower())

    def test_invalid_output_modes_never_silently_disable_all_outputs(self):
        for mode in ("", "unknown", " all ", 42, True):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "output_mode"):
                self.validate(output_mode=mode)
        with self.assertRaises(ValueError):
            transcript.resolve_outputs("unknown")

    def test_files_must_be_a_list_of_nonempty_strings(self):
        for value in ("a.mp3", [""], ["  "], [1], [True], None, []):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "files"):
                self.validate(files=value)

    def test_cli_paths_replace_files_and_permit_empty_config_list(self):
        config = transcript.validate_config({"files": []}, ["input/cli.mp3"], environ={})
        self.assertEqual(config["files"], ["input/cli.mp3"])
        original = {"files": ["original.mp3"]}
        config = transcript.validate_config(original, ["cli.mp3"], environ={})
        self.assertEqual(config["files"], ["cli.mp3"])
        self.assertEqual(original["files"], ["original.mp3"])

    def test_cli_override_does_not_hide_a_wrong_config_files_type(self):
        with self.assertRaisesRegex(ValueError, "files"):
            transcript.validate_config({"files": "wrong"}, ["cli.mp3"], environ={})

    def test_boolean_as_numeric_is_rejected_for_every_numeric_key(self):
        for key in ("min_speakers", "max_speakers", "batch_size", "auto_preprocess_block_seconds"):
            for value in (True, False):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    self.validate(**{key: value})

    def test_integer_settings_are_strict_and_have_bounds(self):
        for key in ("min_speakers", "max_speakers", "batch_size"):
            for value in (-1, 1.0, "2", None):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    self.validate(**{key: value})
        with self.assertRaisesRegex(ValueError, "batch_size"):
            self.validate(batch_size=0)
        self.assertEqual(self.validate(batch_size=1)["batch_size"], 1)

    def test_speaker_bounds(self):
        with self.assertRaisesRegex(ValueError, "min_speakers"):
            self.validate(min_speakers=3, max_speakers=2)
        for minimum, maximum in ((0, 0), (4, 0), (0, 4), (4, 4), (2, 4)):
            with self.subTest(minimum=minimum, maximum=maximum):
                self.validate(min_speakers=minimum, max_speakers=maximum)

    def test_preprocess_duration_must_be_finite_nonnegative(self):
        for value in (-1, float("nan"), float("inf"), -float("inf"), "60", None, 10 ** 1000):
            with self.subTest(value=str(value)[:30]), self.assertRaisesRegex(ValueError, "auto_preprocess"):
                self.validate(auto_preprocess_block_seconds=value)
        for value in (0, 0.5, 60, 120):
            with self.subTest(value=value):
                self.assertEqual(self.validate(auto_preprocess_block_seconds=value)["auto_preprocess_block_seconds"], value)

    def test_unsafe_model_names(self):
        for value in ("", " ", "../a", "a..b", "a/b", "a\\b", 'a:b', 'a"b', "a<b", "a>b", "a|b", "a?b", "a*b", "a\x00b", 12):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "model_size"):
                self.validate(model_size=value)

    def test_model_names_are_not_limited_to_an_allowlist(self):
        for value in ("large-v3-turbo", "tiny.en", "custom-model_v1.2", "model@revision"):
            with self.subTest(value=value):
                self.assertEqual(self.validate(model_size=value)["model_size"], value)

    def test_string_and_boolean_types(self):
        for key in ("remove_fillers", "diarize"):
            for value in (0, 1, "true", None):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    self.validate(**{key: value})
        for key in ("initial_prompt", "hf_token"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.validate(**{key: False})
        for key in ("language", "device"):
            for value in ("", " ", 1):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    self.validate(**{key: value})
        with self.assertRaisesRegex(ValueError, "device"):
            self.validate(device="mps")
        self.assertEqual(self.validate(language="custom-language-code")["language"], "custom-language-code")

    def test_hf_token_environment_precedence_and_empty_fallback(self):
        raw = {"files": ["a.mp3"], "hf_token": "config-secret"}
        config = transcript.validate_config(raw, environ={"HF_TOKEN": " env-secret "})
        self.assertEqual(config["hf_token"], "env-secret")
        self.assertEqual(config["_token_source"], "HF_TOKEN environment")
        self.assertEqual(raw["hf_token"], "config-secret")
        config = transcript.validate_config(raw, environ={"HF_TOKEN": "  "})
        self.assertEqual(config["hf_token"], "config-secret")
        self.assertEqual(config["_token_source"], "config file")

    def test_no_token_warns_upfront_without_changing_diarization_option(self):
        config = self.validate(diarize=True)
        messages = transcript.configuration_warnings(config)
        self.assertTrue(any("HF_TOKEN" in message and "skipped" in message for message in messages))
        self.assertTrue(config["diarize"])

    def test_ukrainian_prompt_mismatch_including_auto_warns(self):
        prompt = "Це лекція. Грамотний текст українською."
        for language in ("en", "auto", "pl"):
            with self.subTest(language=language):
                config = self.validate(remove_fillers=True, initial_prompt=prompt, language=language)
                messages = transcript.configuration_warnings(config)
                self.assertTrue(any("initial_prompt" in message for message in messages))
                self.assertEqual(config["initial_prompt"], prompt)
        for updates in ({"language": "uk"}, {"remove_fillers": False}, {"initial_prompt": "English text"}):
            config = self.validate(**{**{"remove_fillers": True, "initial_prompt": prompt, "language": "en"}, **updates})
            self.assertFalse(any("initial_prompt" in msg for msg in transcript.configuration_warnings(config)))

    def test_auto_preprocess_non_timecode_mode_warns(self):
        for mode in ("simple", "srt"):
            with self.subTest(mode=mode):
                messages = transcript.configuration_warnings(self.validate(output_mode=mode))
                self.assertTrue(any("only" in msg and "timecode" in msg for msg in messages))
        self.assertFalse(transcript.configuration_warnings(self.validate(output_mode="all")))

    def test_error_messages_redact_active_and_token_shaped_secrets(self):
        message = transcript.error_detail(RuntimeError("private-token hf_123456789ABC failed"), {"hf_token": "private-token"})
        self.assertNotIn("private-token", message)
        self.assertNotIn("hf_123456789ABC", message)
        self.assertIn("redacted", message)


if __name__ == "__main__":
    unittest.main()
