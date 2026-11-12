from pathlib import Path
from unittest import mock

from support import ProjectTestCase, transcript


class PathTests(ProjectTestCase):
    def test_absolute_path(self):
        target = self.media("elsewhere/foo.mp3")
        self.assertEqual(transcript.resolve_media_path(str(target)), target)

    def test_explicit_input_path(self):
        target = self.media("input/foo.mp3")
        self.assertEqual(transcript.resolve_media_path("input/foo.mp3"), target)

    def test_legacy_root_path(self):
        target = self.media("foo.mp3")
        self.assertEqual(transcript.resolve_media_path("foo.mp3"), target)

    def test_input_fallback(self):
        target = self.media("input/foo.mp3")
        self.assertEqual(transcript.resolve_media_path("foo.mp3"), target)

    def test_ambiguity_uses_root_and_warns(self):
        target = self.media("foo.mp3")
        self.media("input/foo.mp3")
        messages = []
        self.assertEqual(transcript.resolve_media_path("foo.mp3", warn=messages.append), target)
        self.assertEqual(len(messages), 1)
        self.assertIn("input/...", messages[0])

    def test_explicit_input_does_not_fallback_to_input_input(self):
        self.media("input/input/a.mp3")
        plan = self.plan()
        self.assertEqual(len(plan.invalid_or_missing), 1)
        self.assertEqual(plan.items[0].target, self.root / "input/a.mp3")

    def test_existing_root_directory_is_invalid_not_fallback(self):
        (self.root / "foo.mp3").mkdir()
        self.media("input/foo.mp3")
        plan = self.plan(files=["foo.mp3"])
        self.assertEqual(len(plan.invalid_or_missing), 1)
        self.assertIn("regular file", plan.items[0].reason)

    def test_unreadable_file_is_invalid(self):
        self.media()
        with mock.patch.object(Path, "open", side_effect=PermissionError("read access denied")):
            plan = self.plan()
        self.assertEqual(len(plan.invalid_or_missing), 1)
        self.assertIn("denied", plan.items[0].reason)

    def test_duplicates_are_deduplicated_by_resolved_path(self):
        target = self.media()
        plan = self.plan(files=["input/a.mp3", str(target), "input/../input/a.mp3"])
        self.assertEqual(len(plan.runnable), 1)
        self.assertEqual(plan.duplicates, 2)
        self.assertTrue(any("Duplicate" in message for message in plan.warnings))

    def test_symlink_alias_is_deduplicated(self):
        target = self.media()
        alias = self.root / "alias.mp3"
        try:
            alias.symlink_to(target)
        except OSError as error:
            self.skipTest(str(error))
        plan = self.plan(files=["input/a.mp3", "alias.mp3"])
        self.assertEqual(len(plan.runnable), 1)
        self.assertEqual(plan.duplicates, 1)

    def test_simple_input_mapping_and_exact_filenames(self):
        self.media()
        item = self.plan(auto_preprocess_block_seconds=60).runnable[0]
        self.assertEqual(item.output_dir, self.root / "output/a")
        self.assertEqual({key: path.name for key, path in item.outputs.items()}, {
            "timecode": "timecode-a--large-v3-turbo.txt",
            "simple": "simple-a--large-v3-turbo.txt",
            "srt": "a--large-v3-turbo.srt",
            "blocks": "timecode-a--large-v3-turbo.blocks.txt",
        })

    def test_nested_input_mapping_even_when_passed_absolutely(self):
        target = self.media("input/course/sub/a.mp3")
        item = self.plan(files=[str(target)]).runnable[0]
        self.assertEqual(item.output_dir, self.root / "output/course/sub/a")

    def test_external_and_root_sources_embed_only_basename(self):
        for name in ("a.mp3", "external/course/b.mp3"):
            with self.subTest(name=name):
                target = self.media(name)
                item = self.plan(files=[str(target)]).runnable[0]
                self.assertEqual(item.output_dir, self.root / "output" / target.stem)

    def test_extension_collisions_are_order_independent(self):
        self.media("input/a.mp3")
        self.media("input/a.wav")
        files = ["input/a.wav", "input/a.mp3"]
        first = self.plan(files=files)
        second = self.plan(files=list(reversed(files)))
        mapping = lambda plan: {item.target.name: item.output_dir for item in plan.items}
        self.assertEqual(mapping(first), mapping(second))
        self.assertEqual(mapping(first)["a.mp3"], self.root / "output/a")
        self.assertEqual(mapping(first)["a.wav"], self.root / "output/a_wav")
        self.assertTrue(any("collision" in message.lower() for message in first.warnings))

    def test_external_same_stem_uses_extension_then_stable_hash(self):
        files = ["input/a.mp3", "external/x/a.mp3", "external/y/a.mp3"]
        for name in files:
            self.media(name)
        first = self.plan(files=files)
        second = self.plan(files=list(reversed(files)))
        mapping = lambda plan: {item.target: item.output_dir for item in plan.items}
        self.assertEqual(mapping(first), mapping(second))
        dirs = [item.output_dir for item in first.items]
        self.assertEqual(len(set(dirs)), 3)
        self.assertEqual(dirs[0].name, "a")
        self.assertEqual(dirs[1].name, "a_mp3")
        self.assertRegex(dirs[2].name, r"^a_mp3_[0-9a-f]{10}$")

    def test_generated_suffix_does_not_take_another_natural_folder(self):
        files = ["input/a.mp3", "input/a.wav", "input/a_wav.mp3"]
        for name in files:
            self.media(name)
        plan = self.plan(files=files)
        mapping = {item.target.name: item.output_dir.name for item in plan.items}
        self.assertEqual(mapping["a_wav.mp3"], "a_wav")
        self.assertRegex(mapping["a.wav"], r"^a_wav_[0-9a-f]{10}$")

    def test_source_marker_prevents_collisions_across_separate_runs(self):
        self.media("input/a.wav")
        original = self.plan(files=["input/a.wav"]).runnable[0]
        transcript.claim_output_directory(original, self.root)
        self.media("input/a.mp3")
        later = self.plan(files=["input/a.mp3"]).runnable[0]
        self.assertEqual(original.output_dir.name, "a")
        self.assertEqual(later.output_dir.name, "a_mp3")
        replay = self.plan(files=["input/a.mp3", "input/a.wav"])
        mapping = {item.target.name: item.output_dir for item in replay.items}
        self.assertEqual(mapping["a.wav"], original.output_dir)
        self.assertEqual(mapping["a.mp3"], later.output_dir)

    def test_existing_owned_suffix_keeps_ownership_when_new_natural_source_appears(self):
        for name in ("input/a.mp3", "input/a.wav"):
            self.media(name)
        old = self.plan(files=["input/a.mp3", "input/a.wav"])
        for item in old.items:
            transcript.claim_output_directory(item, self.root)
        self.media("input/a_wav.mp3")
        new = self.plan(files=["input/a.mp3", "input/a.wav", "input/a_wav.mp3"])
        mapping = {item.target.name: item.output_dir.name for item in new.items}
        self.assertEqual(mapping["a.wav"], "a_wav")
        self.assertEqual(mapping["a_wav.mp3"], "a_wav_mp3")

    def test_windows_case_collisions_are_disambiguated_on_linux_too(self):
        first = self.media("input/a.mp3")
        second = self.media("input/A.wav")
        plan = self.plan(files=[str(first), str(second)])
        keys = [str(item.output_dir).casefold() for item in plan.items]
        self.assertEqual(len(set(keys)), 2)

    def test_parent_traversal_in_source_never_traverses_output(self):
        target = self.media("external/a.mp3")
        item = self.plan(files=["input/../external/a.mp3"]).runnable[0]
        self.assertEqual(item.target, target)
        self.assertEqual(item.output_dir, self.root / "output/a")
        for path in item.outputs.values():
            self.assertTrue(path.resolve().is_relative_to((self.root / "output").resolve()))

    def test_sanitized_components_remain_windows_safe(self):
        for raw, expected in (("..", "source"), ("bad:name?", "bad_name_"), ("CON", "_CON"), ("NUL.txt", "_NUL.txt"), ("name. ", "name")):
            with self.subTest(raw=raw):
                self.assertEqual(transcript.safe_component(raw), expected)

    def test_output_symlink_escape_is_rejected(self):
        self.media("input/sub/a.mp3")
        outside = self.workspace / "outside"
        outside.mkdir()
        try:
            (self.root / "output/sub").symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest(str(error))
        plan = self.plan(files=["input/sub/a.mp3"])
        self.assertEqual(len(plan.invalid_or_missing), 1)
        self.assertIn("Unsafe output", plan.items[0].reason)
        self.assertEqual(list(outside.iterdir()), [])

    def test_output_file_symlink_escape_is_rejected_even_with_overwrite(self):
        self.media()
        item = self.plan().runnable[0]
        item.output_dir.mkdir()
        outside = self.workspace / "private.txt"
        outside.write_text("preserve", encoding="utf-8")
        try:
            item.outputs["timecode"].symlink_to(outside)
        except OSError as error:
            self.skipTest(str(error))
        plan = transcript.build_preflight(self.config(), overwrite=True)
        self.assertEqual(len(plan.invalid_or_missing), 1)
        self.assertEqual(outside.read_text(encoding="utf-8"), "preserve")

    def test_relative_output_directory_is_not_accepted_outside_root(self):
        with self.assertRaisesRegex(ValueError, "Unsafe output"):
            transcript.validate_output_path(self.root / "output/../private.txt")
        with self.assertRaisesRegex(ValueError, "Unsafe output"):
            transcript.validate_output_path(self.root / "output")

    def test_preflight_itself_writes_no_outputs_or_markers(self):
        self.media()
        self.plan(auto_preprocess_block_seconds=60)
        self.assertEqual(list((self.root / "output").iterdir()), [])

    def test_non_directory_output_target_is_invalid_in_preflight(self):
        self.media()
        (self.root / "output/a").write_bytes(b"not a directory")
        plan = self.plan()
        self.assertEqual(len(plan.invalid_or_missing), 1)
        self.assertIn("not a directory", plan.items[0].reason)
        self.assertEqual((self.root / "output/a").read_bytes(), b"not a directory")

    def test_unknown_historical_output_ownership_warns_without_changing_files(self):
        self.media()
        folder = self.root / "output/a"
        folder.mkdir()
        (folder / "previous-model.txt").write_bytes(b"keep me")
        plan = self.plan()
        self.assertEqual(len(plan.runnable), 1)
        self.assertTrue(any("ownership is unknown" in message for message in plan.warnings))
        self.assertEqual((folder / "previous-model.txt").read_bytes(), b"keep me")

    def test_non_directory_output_ancestor_is_invalid_before_transcription(self):
        self.media("input/sub/a.mp3")
        (self.root / "output/sub").write_bytes(b"do not change")
        plan = self.plan(files=["input/sub/a.mp3"])
        self.assertEqual(len(plan.invalid_or_missing), 1)
        self.assertIn("not a directory", plan.items[0].reason)
        self.assertEqual((self.root / "output/sub").read_bytes(), b"do not change")
