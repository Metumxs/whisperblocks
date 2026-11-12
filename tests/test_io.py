import os
from unittest import mock

from support import ProjectTestCase, file_io, preprocess, transcript


class AtomicWriteTests(ProjectTestCase):
    def test_atomic_write_new_file(self):
        path = self.root / "output/a/file.txt"
        with file_io.atomic_text_writer(path) as handle:
            handle.write("Привіт\n")
            self.assertFalse(path.exists())
            self.assertEqual(len(list(path.parent.glob(".*.tmp"))), 1)
        self.assertEqual(path.read_text(encoding="utf-8"), "Привіт\n")
        self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_default_writer_refuses_existing_destination(self):
        path = self.root / "output/file.txt"
        path.write_bytes(b"old")
        with self.assertRaisesRegex(FileExistsError, "overwrite"):
            with file_io.atomic_text_writer(path) as handle:
                handle.write("new")
        self.assertEqual(path.read_bytes(), b"old")

    def test_explicit_overwrite_replaces_after_close(self):
        path = self.root / "output/file.txt"
        path.write_bytes(b"old")
        original = os.replace
        handles = []

        def replace(source, target):
            self.assertTrue(handles[0].closed)
            self.assertEqual(source.parent, path.parent)
            self.assertEqual(path.read_bytes(), b"old")
            return original(source, target)

        with mock.patch.object(file_io.os, "replace", side_effect=replace):
            with file_io.atomic_text_writer(path, overwrite=True) as handle:
                handles.append(handle)
                handle.write("new")
                self.assertEqual(path.read_bytes(), b"old")
        self.assertEqual(path.read_text(), "new")

    def test_midstream_error_cleans_temp_and_preserves_old_file(self):
        path = self.root / "output/file.txt"
        path.write_bytes(b"old")
        with self.assertRaisesRegex(OSError, "disk full"):
            with file_io.atomic_text_writer(path, overwrite=True) as handle:
                handle.write("partial new content")
                raise OSError("disk full")
        self.assertTrue(handle.closed)
        self.assertEqual(path.read_bytes(), b"old")
        self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_interrupts_cleanup_without_swallowing(self):
        for exception in (KeyboardInterrupt, SystemExit):
            with self.subTest(exception=exception):
                path = self.root / "output/file.txt"
                path.write_bytes(b"old")
                with self.assertRaises(exception):
                    with file_io.atomic_text_writer(path, overwrite=True) as handle:
                        handle.write("partial")
                        raise exception()
                self.assertTrue(handle.closed)
                self.assertEqual(path.read_bytes(), b"old")
                self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_replace_error_is_reported_and_old_destination_survives(self):
        path = self.root / "output/file.txt"
        path.write_bytes(b"old")
        with mock.patch.object(file_io.os, "replace", side_effect=PermissionError("destination locked")):
            with self.assertRaisesRegex(PermissionError, "locked"):
                with file_io.atomic_text_writer(path, overwrite=True) as handle:
                    handle.write("complete but not committed")
        self.assertEqual(path.read_bytes(), b"old")
        self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_output_appearing_midstream_is_not_overwritten(self):
        path = self.root / "output/file.txt"
        with self.assertRaisesRegex(FileExistsError, "appeared"):
            with file_io.atomic_text_writer(path) as handle:
                handle.write("ours")
                path.write_text("other writer", encoding="utf-8")
        self.assertEqual(path.read_text(), "other writer")
        self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_broken_symlink_is_existing_for_protection(self):
        path = self.root / "output/file.txt"
        try:
            path.symlink_to(self.root / "nonexistent")
        except OSError as error:
            self.skipTest(str(error))
        self.assertTrue(file_io.path_exists(path))
        with self.assertRaises(FileExistsError):
            with file_io.atomic_text_writer(path):
                self.fail("must not yield")

    def test_write_outputs_preserves_formats_and_only_relocates_parent(self):
        source = self.media()
        item = self.plan().runnable[0]
        result = {"segments": [
            {"start": 2.005, "end": 3.9999, "text": " Привіт. ", "speaker": "SPEAKER_00"},
            {"start": 4, "end": 3602, "text": "Далі."},
        ]}
        tc = transcript.write_outputs(
            result, source, "large-v3-turbo", True, True, True, output_dir=item.output_dir,
        )
        self.assertEqual(tc, item.outputs["timecode"])
        self.assertEqual(tc.read_text(encoding="utf-8"),
                         "[00:02 -> 00:03] [SPEAKER_00] Привіт.\n[00:04 -> 01:00:02] Далі.\n")
        self.assertEqual(item.outputs["simple"].read_text(encoding="utf-8"), "[SPEAKER_00] Привіт.\nДалі.\n")
        self.assertEqual(item.outputs["srt"].read_text(encoding="utf-8"),
                         "1\n00:00:02,005 --> 00:00:04,000\n[SPEAKER_00] Привіт.\n\n"
                         "2\n00:00:04,000 --> 01:00:02,000\nДалі.\n\n")
        self.assertFalse(any(source.parent.glob("*.txt")))
        self.assertEqual(list(item.output_dir.glob(".*.tmp")), [])

    def test_output_formatting_failure_cleans_all_temps_before_committing(self):
        source = self.media()
        item = self.plan().runnable[0]
        item.output_dir.mkdir()
        for path in item.outputs.values():
            path.write_bytes(b"old")
        result = {"segments": [
            {"start": 0, "end": 1, "text": "valid"},
            {"start": 1, "end": 2},  # error after some text was written
        ]}
        with self.assertRaises(KeyError):
            transcript.write_outputs(
                result, source, "large-v3-turbo", True, True, True,
                output_dir=item.output_dir, overwrite=True,
            )
        for path in item.outputs.values():
            self.assertEqual(path.read_bytes(), b"old")
        self.assertEqual(list(item.output_dir.glob(".*.tmp")), [])

    def test_preprocess_atomic_replace_error_preserves_existing_blocks(self):
        source = self.root / "output/a/timecode-a--large-v3-turbo.txt"
        source.parent.mkdir()
        source.write_text("[00:00 -> 00:02] text\n", encoding="utf-8")
        target = preprocess.blocks_path_for(source)
        target.write_bytes(b"old blocks")
        with mock.patch.object(file_io.os, "replace", side_effect=OSError("replace failure")):
            with self.assertRaisesRegex(OSError, "replace failure"):
                preprocess.process_file(source, 60, overwrite=True)
        self.assertEqual(target.read_bytes(), b"old blocks")
        self.assertEqual(list(source.parent.glob(".*.tmp")), [])

    def test_skip_existing_includes_every_expected_output_and_blocks(self):
        self.media()
        config = self.config(auto_preprocess_block_seconds=60)
        initial = transcript.build_preflight(config).runnable[0]
        initial.output_dir.mkdir()
        for kind, path in initial.outputs.items():
            with self.subTest(kind=kind):
                path.write_bytes(b"existing")
                plan = transcript.build_preflight(config)
                self.assertEqual(len(plan.skipped_existing), 1)
                self.assertIn(str(path), plan.items[0].reason)
                overwrite = transcript.build_preflight(config, overwrite=True)
                self.assertEqual(len(overwrite.runnable), 1)
                self.assertEqual(path.read_bytes(), b"existing")
                path.unlink()

    def test_inactive_output_does_not_trigger_skip(self):
        self.media()
        config = self.config(output_mode="simple", auto_preprocess_block_seconds=60)
        item = transcript.build_preflight(config).runnable[0]
        item.output_dir.mkdir()
        (item.output_dir / "timecode-a--large-v3-turbo.blocks.txt").write_text("old")
        self.assertEqual(set(item.outputs), {"simple"})
        self.assertEqual(len(transcript.build_preflight(config).runnable), 1)
