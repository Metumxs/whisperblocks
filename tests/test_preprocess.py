import contextlib
import io
from pathlib import Path
from unittest import mock

from support import ProjectTestCase, preprocess, transcript


class TimestampAndPreprocessTests(ProjectTestCase):
    def transcript_file(self, text="[00:02 -> 00:14] hello\n[00:14 -> 00:22] world\n", name="timecode-a--model.txt"):
        path = self.root / "output/a" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_srt_millisecond_precision_and_rollover(self):
        for seconds, expected in (
            (2.005, "00:00:02,005"), (1.9999, "00:00:02,000"),
            (59.9999, "00:01:00,000"), (3599.9999, "01:00:00,000"),
            (-1, "00:00:00,000"), (3602.005, "01:00:02,005"),
        ):
            with self.subTest(seconds=seconds):
                self.assertEqual(transcript.format_timestamp(seconds, True), expected)

    def test_plain_timestamps_still_truncate_to_whole_seconds(self):
        for seconds, expected in ((2.9999, "00:02"), (59.9999, "00:59"), (3602.9, "01:00:02")):
            with self.subTest(seconds=seconds):
                self.assertEqual(transcript.format_timestamp(seconds), expected)

    def test_valid_timestamp_components(self):
        for value, expected in (("00:00", 0), ("01:59", 119), ("01:00:02", 3602), ("120:59:59", 435599)):
            with self.subTest(value=value):
                self.assertEqual(preprocess.parse_timestamp(value), expected)

    def test_invalid_timestamp_components(self):
        for value in ("-01:00", "00:-1", "01:75", "60:00", "01:60:00", "01:00:60", "00:xx", "1.0:00", "00", "00:00:00:00", "00:", "nan:01", "+1:00"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                preprocess.parse_timestamp(value)

    def test_wrapped_text_speaker_tags_and_other_brackets_are_preserved(self):
        header, segments = preprocess.parse_transcript(
            "[00:02 -> 00:14] First line\nwrapped text\n[SPEAKER_00] voice\n[inaudible]\n"
            "[нерозбірливо]\n[00:14 -> 00:22] [SPEAKER_01] Next\n"
        )
        self.assertEqual(header, "")
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, "First line wrapped text [SPEAKER_00] voice [inaudible] [нерозбірливо]")
        self.assertEqual(segments[1].text, "[SPEAKER_01] Next")

    def test_optional_header_passthrough_and_existing_block_format(self):
        path = self.transcript_file("Duration: 00:22 | Language: uk\n\n[00:02 -> 00:14] hello\n[00:14 -> 00:22] world\n")
        out = preprocess.process_file(path, 60)
        self.assertEqual(out.read_text(encoding="utf-8"),
                         f"Source: {path.name}\nDuration: 00:22 | Language: uk\n"
                         "Segments: 2 | Blocks: 1 (~60s each)\n\n"
                         "=== BLOCK 01 | 00:02 → 00:22 ===\nhello world\n")

    def test_no_duration_header_is_invented(self):
        path = self.transcript_file()
        out = preprocess.process_file(path, 60)
        self.assertNotIn("Duration:", out.read_text(encoding="utf-8"))
        self.assertEqual(out.parent, path.parent)

    def test_overlapping_and_equal_starts_are_valid(self):
        _, segments = preprocess.parse_transcript(
            "[00:00 -> 00:10] first\n[00:02 -> 00:12] overlapping\n[00:02 -> 00:03] same start\n"
        )
        self.assertEqual([segment.start for segment in segments], [0, 2, 2])

    def test_end_before_start_is_rejected_with_line_number(self):
        with self.assertRaisesRegex(ValueError, "Line 2.*end"):
            preprocess.parse_transcript("header\n[00:10 -> 00:09] bad\n")

    def test_nonchronological_starts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Line 2.*non-decreasing"):
            preprocess.parse_transcript("[00:05 -> 00:10] first\n[00:04 -> 00:12] second\n")

    def test_malformed_arrow_headers_are_not_continuation_text(self):
        for line in ("[00:xx -> 00:10] bad", "[-00:01 -> 00:10] bad", "[00:01 -> nan] bad", "[00:00:00.1 -> 00:01] bad", "[00:00 -> 01:75] bad"):
            with self.subTest(line=line), self.assertRaisesRegex(ValueError, "Line 2"):
                preprocess.parse_transcript("[00:00 -> 00:01] first\n" + line)

    def test_zero_segments_rejected(self):
        for text in ("", "header only\n", "[SPEAKER_00] no timestamp\n"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "No timestamped"):
                preprocess.parse_transcript(text)

    def test_greedy_block_merging_remains_unchanged(self):
        _, segments = preprocess.parse_transcript(
            "[00:00 -> 00:30] one\n[00:30 -> 01:00] two\n"
            "[01:00 -> 01:01] three\n[01:01 -> 02:30] long segment\n"
        )
        blocks = preprocess.merge_into_blocks(segments, 60)
        self.assertEqual([(b.start, b.end, b.segment_count, b.text) for b in blocks], [
            (0, 60, 2, "one two"), (60, 61, 1, "three"), (61, 150, 1, "long segment"),
        ])

    def test_invalid_block_values_are_rejected_programmatically(self):
        path = self.transcript_file()
        for value in (0, -1, float("nan"), float("inf"), -float("inf"), True, "60", 10 ** 1000):
            with self.subTest(value=str(value)[:25]), self.assertRaisesRegex(ValueError, "block duration"):
                preprocess.process_file(path, value)

    def test_invalid_block_cli_values_return_argparse_error(self):
        path = self.transcript_file()
        for value in ("0", "-1", "nan", "inf", "-inf"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    preprocess.main([str(path), "--block=" + value])
                self.assertEqual(result.exception.code, 2)

    def test_readable_regular_utf8_input_validation(self):
        with self.assertRaises(FileNotFoundError):
            preprocess.process_file(self.root / "missing.txt", 60)
        with self.assertRaisesRegex(ValueError, "regular file"):
            preprocess.process_file(self.root / "input", 60)
        path = self.transcript_file()
        path.write_bytes(b"\xff\xfe")
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            preprocess.process_file(path, 60)
        with mock.patch.object(Path, "read_text", side_effect=PermissionError("not readable")):
            with self.assertRaises(PermissionError):
                preprocess.process_file(path, 60)

    def test_windows_glob_expansion_excludes_blocks_and_deduplicates(self):
        path = self.transcript_file()
        block = preprocess.blocks_path_for(path)
        block.write_text("already generated")
        pattern = str(self.root / "output") + "\\*\\timecode-*.txt"
        paths, errors = preprocess.expand_inputs([pattern, str(path)])
        self.assertEqual(paths, [path])
        self.assertEqual(errors, [])
        self.assertNotIn(block, paths)

    def test_recursive_glob_and_character_class_expansion(self):
        first = self.transcript_file(name="timecode-a--model.txt")
        second = self.root / "output/course/sub/timecode-b--model.txt"
        second.parent.mkdir(parents=True)
        second.write_text("[00:00 -> 00:01] b\n", encoding="utf-8")
        paths, errors = preprocess.expand_inputs([str(self.root / "output/**/timecode-[ab]--model.txt")])
        self.assertEqual(set(paths), {first, second})
        self.assertFalse(errors)

    def test_glob_with_only_generated_blocks_counts_as_no_match(self):
        path = self.transcript_file(name="timecode-a--model.blocks.txt")
        paths, errors = preprocess.expand_inputs([str(path.parent / "timecode-*.txt")])
        self.assertEqual(paths, [])
        self.assertEqual(len(errors), 1)

    def test_no_match_errors_do_not_suppress_other_valid_inputs(self):
        path = self.transcript_file()
        missing = str(path.parent / "nothing-*.txt")
        paths, errors = preprocess.expand_inputs([missing, str(path), str(path.parent / "absent.txt")])
        self.assertEqual(paths, [path])
        self.assertEqual(len(errors), 2)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(preprocess.main([missing, str(path)]), 1)
        self.assertTrue(preprocess.blocks_path_for(path).exists())

    def test_expansion_preserves_argument_order_and_deduplicates_aliases(self):
        first = self.transcript_file(name="timecode-a--model.txt")
        second = self.transcript_file(name="timecode-b--model.txt")
        paths, errors = preprocess.expand_inputs([str(second), str(first), str(second)])
        self.assertEqual(paths, [second, first])
        self.assertFalse(errors)

    def test_standalone_skip_and_explicit_overwrite(self):
        path = self.transcript_file()
        target = preprocess.blocks_path_for(path)
        target.write_bytes(b"old")
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertEqual(preprocess.main([str(path)]), 0)
        self.assertIn("--overwrite", log.getvalue())
        self.assertEqual(target.read_bytes(), b"old")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(preprocess.main([str(path), "--overwrite"]), 0)
        self.assertNotEqual(target.read_bytes(), b"old")

    def test_standalone_invalid_transcript_returns_nonzero(self):
        path = self.transcript_file("not a transcript")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(preprocess.main([str(path)]), 1)

    def test_existing_literal_filename_with_brackets_is_not_a_glob(self):
        path = self.transcript_file(name="timecode-lecture[1]--model.txt")
        paths, errors = preprocess.expand_inputs([str(path)])
        self.assertEqual(paths, [path])
        self.assertEqual(errors, [])

    def test_existing_blocks_do_not_hide_invalid_source_text(self):
        path = self.transcript_file("not a valid transcript")
        blocks = preprocess.blocks_path_for(path)
        blocks.write_bytes(b"preserve existing output")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(preprocess.main([str(path)]), 1)
        self.assertEqual(blocks.read_bytes(), b"preserve existing output")

    def test_standalone_options_can_be_intermixed_with_files(self):
        first = self.transcript_file(name="timecode-a--model.txt")
        second = self.transcript_file(name="timecode-b--model.txt")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(preprocess.main([str(first), "--block", "90", str(second)]), 0)
        self.assertTrue(preprocess.blocks_path_for(first).exists())
        self.assertTrue(preprocess.blocks_path_for(second).exists())
