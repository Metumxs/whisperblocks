import os
from pathlib import Path
from unittest import mock

from support import ProjectTestCase, fake_torch, preprocess, transcript, transcript_result


class BatchTests(ProjectTestCase):
    def test_first_run_seeds_only_default_config_and_exits_zero(self):
        load = self.patch(transcript, "load_torch", mock.Mock())
        self.assertEqual(transcript.main([]), 0)
        path = self.root / "config/config.toml"
        self.assertEqual(path.read_bytes(), transcript.CONFIG_EXAMPLE_PATH.read_bytes())
        load.assert_not_called()
        before = path.read_bytes()
        self.assertEqual(transcript.main([]), 2)  # empty seeded file list requires editing
        self.assertEqual(path.read_bytes(), before)

    def test_missing_template_is_a_config_error_not_success(self):
        self.patch(transcript, "CONFIG_EXAMPLE_PATH", self.root / "absent-template")
        self.assertEqual(transcript.main([]), 2)
        self.assertFalse((self.root / "config/config.toml").exists())

    def test_explicit_missing_custom_config_is_not_seeded(self):
        self.assertEqual(transcript.main(["--config", "config/custom.toml"]), 2)
        self.assertFalse((self.root / "config/custom.toml").exists())
        self.assertFalse((self.root / "config/config.toml").exists())

    def test_explicit_missing_default_path_is_not_seeded_either(self):
        self.assertEqual(transcript.main(["--config", "config/config.toml"]), 2)
        self.assertFalse((self.root / "config/config.toml").exists())

    def test_existing_live_config_and_media_never_modified(self):
        source = self.media()
        config_path = self.write_config()
        original_config = config_path.read_bytes()
        original_media = source.read_bytes()
        code, boundary, summaries = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(config_path.read_bytes(), original_config)
        self.assertEqual(source.read_bytes(), original_media)
        self.assertEqual(summaries[-1].succeeded, 1)

    def test_invalid_config_fails_before_any_ml_import(self):
        self.write_config(output_mode="invalid")
        load = self.patch(transcript, "load_torch", mock.Mock())
        whisper = self.patch(transcript, "load_whisperx", mock.Mock())
        self.assertEqual(transcript.main([]), 2)
        load.assert_not_called()
        whisper.assert_not_called()
        self.assertIn("output_mode", self.console.output)

    def test_invalid_toml_is_reported_as_exit_two(self):
        path = self.root / "config/config.toml"
        path.write_text("files = [broken", encoding="utf-8")
        self.assertEqual(transcript.main([]), 2)
        self.assertIn("Configuration error", self.console.output)

    def test_all_missing_never_checks_cuda_or_loads_models(self):
        self.write_config()
        code, boundary, summaries = self.run_main()
        self.assertEqual(code, 1)
        transcript.load_torch.assert_not_called()
        boundary.assert_not_called()
        self.assertEqual(summaries[-1].missing_or_invalid, 1)
        self.assertNotIn("All tasks completed", self.console.output)

    def test_all_skipped_exit_zero_before_cuda_and_transcription(self):
        self.media()
        self.write_config()
        item = self.plan().runnable[0]
        item.output_dir.mkdir()
        item.outputs["srt"].write_bytes(b"old")
        code, boundary, summaries = self.run_main(available=False)
        self.assertEqual(code, 0)
        boundary.assert_not_called()
        transcript.load_torch.assert_not_called()
        self.assertEqual(summaries[-1].skipped_existing, 1)
        self.assertEqual(summaries[-1].succeeded, 0)
        self.assertIn("All tasks completed", self.console.output)
        self.assertEqual(item.outputs["srt"].read_bytes(), b"old")
        self.assertFalse((item.output_dir / transcript.SOURCE_MARKER).exists())

    def test_mixed_missing_and_valid_transcribes_valid_but_returns_one(self):
        source = self.media()
        self.write_config(files=["absent.mp3", "input/a.mp3"])
        code, boundary, summaries = self.run_main()
        self.assertEqual(code, 1)
        boundary.assert_called_once()
        self.assertEqual(boundary.call_args.args[0], source)
        self.assertEqual(summaries[-1].succeeded, 1)
        self.assertEqual(summaries[-1].missing_or_invalid, 1)
        self.assertNotIn("All tasks completed", self.console.output)
        self.assertLess(self.console.output.index("invalid_or_missing"), self.console.output.index("File [1/1]"))

    def test_all_valid_successes_keep_requested_batch_order(self):
        first = self.media("input/b.mp3")
        second = self.media()
        self.write_config(files=["input/b.mp3", "input/a.mp3"])
        code, boundary, summaries = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual([call.args[0] for call in boundary.call_args_list], [first, second])
        self.assertEqual(summaries[-1].succeeded, 2)
        self.assertEqual(summaries[-1].failures, 0)

    def test_overwrite_is_explicit_and_replaces_outputs(self):
        self.media()
        self.write_config()
        item = self.plan().runnable[0]
        item.output_dir.mkdir()
        for path in item.outputs.values():
            path.write_bytes(b"old output")
        code, boundary, summaries = self.run_main(["--overwrite"])
        self.assertEqual(code, 0)
        boundary.assert_called_once()
        self.assertEqual(summaries[-1].succeeded, 1)
        for path in item.outputs.values():
            self.assertNotEqual(path.read_bytes(), b"old output")

    def test_existing_blocks_skip_before_transcription_even_without_raw_text(self):
        self.media()
        self.write_config(auto_preprocess_block_seconds=60)
        item = self.plan(auto_preprocess_block_seconds=60).runnable[0]
        item.output_dir.mkdir()
        item.outputs["blocks"].write_bytes(b"existing blocks")
        code, boundary, summaries = self.run_main()
        self.assertEqual(code, 0)
        boundary.assert_not_called()
        transcript.load_torch.assert_not_called()
        self.assertEqual(summaries[-1].skipped_existing, 1)

    def test_dry_run_prints_exact_paths_without_importing_ml(self):
        self.media()
        self.write_config(auto_preprocess_block_seconds=60)
        load = self.patch(transcript, "load_torch", mock.Mock(side_effect=AssertionError("no torch")))
        whisper = self.patch(transcript, "load_whisperx", mock.Mock(side_effect=AssertionError("no whisperx")))
        self.assertEqual(transcript.main(["--dry-run"]), 0)
        load.assert_not_called()
        whisper.assert_not_called()
        self.assertIn(str(self.root / "input/a.mp3"), self.console.output)
        self.assertIn(str(self.root / "output/a/timecode-a--large-v3-turbo.blocks.txt"), self.console.output)
        self.assertEqual(list((self.root / "output").iterdir()), [])

    def test_check_config_reports_no_gpu_but_succeeds_without_models(self):
        self.media()
        self.write_config()
        load = self.patch(transcript, "load_torch", mock.Mock(return_value=fake_torch(available=False)))
        whisper = self.patch(transcript, "load_whisperx", mock.Mock())
        self.assertEqual(transcript.main(["--check-config"]), 0)
        load.assert_called_once()
        whisper.assert_not_called()
        self.assertIn("GPU available: no", self.console.output)
        self.assertEqual(list((self.root / "output").iterdir()), [])

    def test_check_config_without_torch_is_status_not_fatal(self):
        self.media()
        self.write_config()
        self.patch(transcript, "load_torch", mock.Mock(side_effect=ModuleNotFoundError("torch not installed")))
        self.assertEqual(transcript.main(["--check-config"]), 0)
        self.assertIn("status unavailable", self.console.output)

    def test_diagnostic_missing_inputs_exit_one_even_without_gpu(self):
        self.write_config()
        self.patch(transcript, "load_torch", mock.Mock(return_value=fake_torch(available=False)))
        self.assertEqual(transcript.main(["--check-config"]), 1)
        self.assertEqual(transcript.main(["--dry-run"]), 1)

    def test_cli_paths_override_empty_config_and_custom_relative_config(self):
        source = self.media("input/cli.mp3")
        self.write_config(path=self.root / "config/custom.toml", files=[])
        code, boundary, summaries = self.run_main(["--config", "config/custom.toml", "input/cli.mp3"])
        self.assertEqual(code, 0)
        self.assertEqual(boundary.call_args.args[0], source)
        self.assertEqual(boundary.call_args.args[1]["files"], ["input/cli.mp3"])
        self.assertFalse((self.root / "config/config.toml").exists())

    def test_cuda_unavailable_real_queue_fails_before_transcription(self):
        self.media()
        self.write_config()
        code, boundary, summaries = self.run_main(available=False)
        self.assertEqual(code, 1)
        boundary.assert_not_called()
        self.assertEqual(summaries[-1].failed_transcription, 1)
        self.assertIn('device = "cpu"', self.console.output)
        self.assertIn("torch.version.cuda", self.console.output)
        self.assertNotIn("All tasks completed", self.console.output)

    def test_cpu_fallback_runs_when_cuda_gpu_is_absent(self):
        self.media()
        self.write_config(device="cpu")
        code, boundary, summaries = self.run_main(available=False)
        self.assertEqual(code, 0)
        boundary.assert_called_once()
        self.assertEqual(summaries[-1].succeeded, 1)

    def test_missing_runtime_for_real_queue_is_failure(self):
        self.media()
        self.write_config()
        self.patch(transcript, "load_torch", mock.Mock(side_effect=ModuleNotFoundError("torch")))
        self.assertEqual(transcript.main([]), 1)
        self.assertNotIn("All tasks completed", self.console.output)

    def test_transcription_failure_does_not_abort_next_valid_source(self):
        for name in ("input/a.mp3", "input/b.mp3"):
            self.media(name)
        self.write_config(files=["input/a.mp3", "input/b.mp3"])
        self.patch(transcript, "load_torch", mock.Mock(return_value=fake_torch()))
        boundary = self.patch(transcript, "transcribe_file", mock.Mock(side_effect=[RuntimeError("ASR failure"), transcript_result()]))
        summaries = []
        self.patch(transcript, "display_summary", lambda counts: summaries.append(counts))
        self.assertEqual(transcript.main([]), 1)
        self.assertEqual(boundary.call_count, 2)
        self.assertEqual(summaries[-1].failed_transcription, 1)
        self.assertEqual(summaries[-1].succeeded, 1)

    def test_write_failure_is_distinct_and_does_not_abort_next_source(self):
        for name in ("input/a.mp3", "input/b.mp3"):
            self.media(name)
        self.write_config(files=["input/a.mp3", "input/b.mp3"])
        original = transcript.write_outputs

        def writer(result, target, *args, **kwargs):
            if target.stem == "a":
                raise OSError("destination locked")
            return original(result, target, *args, **kwargs)

        self.patch(transcript, "write_outputs", mock.Mock(side_effect=writer))
        code, boundary, summaries = self.run_main()
        self.assertEqual(code, 1)
        self.assertEqual(summaries[-1].failed_write, 1)
        self.assertEqual(summaries[-1].failed_transcription, 0)
        self.assertEqual(summaries[-1].succeeded, 1)
        self.assertNotIn("All tasks completed", self.console.output)

    def test_successful_auto_preprocess_same_folder_and_success_counter(self):
        self.media()
        self.write_config(auto_preprocess_block_seconds=60)
        code, boundary, summaries = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(summaries[-1].succeeded, 1)
        blocks = self.root / "output/a/timecode-a--large-v3-turbo.blocks.txt"
        self.assertTrue(blocks.exists())
        self.assertNotIn("Duration:", blocks.read_text(encoding="utf-8"))

    def test_preprocess_failure_is_distinct_and_not_counted_as_success(self):
        self.media()
        self.write_config(auto_preprocess_block_seconds=60)
        self.patch(preprocess, "process_file", mock.Mock(side_effect=ValueError("preprocess failed")))
        code, boundary, summaries = self.run_main()
        self.assertEqual(code, 1)
        self.assertEqual(summaries[-1].failed_preprocess, 1)
        self.assertEqual(summaries[-1].succeeded, 0)
        self.assertEqual(summaries[-1].failed_write, 0)
        self.assertNotIn("All tasks completed", self.console.output)

    def test_missing_preprocessor_import_counts_every_affected_source(self):
        with mock.patch.dict("sys.modules", {"preprocess": None}):
            self.assertEqual(transcript.run_preprocess([Path("a"), Path("b")], 60), 2)

    def test_outputs_appearing_after_preflight_skip_before_second_transcription(self):
        for name in ("input/a.mp3", "input/b.mp3"):
            self.media(name)
        self.write_config(files=["input/a.mp3", "input/b.mp3"])
        second = self.plan(files=["input/a.mp3", "input/b.mp3"]).items[1]
        self.patch(transcript, "load_torch", mock.Mock(return_value=fake_torch()))

        def transcribe(target, config):
            second.output_dir.mkdir()
            second.outputs["timecode"].write_bytes(b"appeared")
            return transcript_result()

        boundary = self.patch(transcript, "transcribe_file", mock.Mock(side_effect=transcribe))
        summaries = []
        self.patch(transcript, "display_summary", lambda counts: summaries.append(counts))
        self.assertEqual(transcript.main([]), 0)
        boundary.assert_called_once()
        self.assertEqual(summaries[-1].skipped_existing, 1)
        self.assertEqual(summaries[-1].succeeded, 1)
        self.assertEqual(second.outputs["timecode"].read_bytes(), b"appeared")

    def test_tokens_never_appear_in_dashboard_or_error_output(self):
        self.media()
        self.write_config(diarize=True, hf_token="config-secret")
        with mock.patch.dict(os.environ, {"HF_TOKEN": "environment-secret"}):
            self.patch(transcript, "load_torch", mock.Mock(return_value=fake_torch()))
            self.patch(transcript, "transcribe_file", mock.Mock(side_effect=RuntimeError("failed token environment-secret")))
            self.assertEqual(transcript.main([]), 1)
        self.assertIn("HF_TOKEN environment", self.console.output)
        self.assertNotIn("environment-secret", self.console.output)
        self.assertNotIn("config-secret", self.console.output)

    def test_keyboardinterrupt_and_systemexit_propagate(self):
        self.media()
        self.write_config()
        self.patch(transcript, "load_torch", mock.Mock(return_value=fake_torch()))
        for exception in (KeyboardInterrupt, SystemExit):
            with self.subTest(exception=exception), mock.patch.object(transcript, "transcribe_file", side_effect=exception):
                with self.assertRaises(exception):
                    transcript.main([])

    def test_empty_input_output_dirs_are_created_on_startup(self):
        (self.root / "input").rmdir()
        (self.root / "output").rmdir()
        self.write_config()
        self.assertEqual(transcript.main(["--dry-run"]), 1)
        self.assertTrue((self.root / "input").is_dir())
        self.assertTrue((self.root / "output").is_dir())

    def test_flags_can_be_intermixed_with_cli_file_paths(self):
        self.media("input/a.mp3")
        self.media("input/b.mp3")
        self.write_config(files=[])
        code, boundary, summaries = self.run_main(["input/a.mp3", "--overwrite", "input/b.mp3"])
        self.assertEqual(code, 0)
        self.assertEqual(boundary.call_count, 2)
        self.assertEqual(summaries[-1].succeeded, 2)

    def test_non_directory_output_never_reaches_cuda_check(self):
        self.media()
        self.write_config()
        (self.root / "output/a").write_bytes(b"not a directory")
        load = self.patch(transcript, "load_torch", mock.Mock())
        self.assertEqual(transcript.main([]), 1)
        load.assert_not_called()

    def test_failed_asr_traceback_is_gone_before_batch_cleanup_and_next_file(self):
        import gc
        import sys
        import weakref

        for name in ("input/a.mp3", "input/b.mp3"):
            self.media(name)
        self.write_config(files=["input/a.mp3", "input/b.mp3"])
        self.patch(transcript, "load_torch", mock.Mock(return_value=fake_torch()))
        references = []
        cleaned = []

        class FailingModel:
            def transcribe(self):
                raise RuntimeError("exception traceback holds this model frame")

        def boundary(target, config):
            if target.stem == "a":
                model = FailingModel()
                references.append(weakref.ref(model))
                model.transcribe()
            self.assertEqual(cleaned, [True])
            self.assertIsNone(references[0]())
            return transcript_result()

        def cleanup():
            self.assertIsNone(sys.exc_info()[0])
            gc.collect()
            self.assertIsNone(references[0]())
            cleaned.append(True)

        self.patch(transcript, "transcribe_file", boundary)
        self.patch(transcript, "flush_vram", cleanup)
        self.assertEqual(transcript.main([]), 1)
        self.assertEqual(cleaned, [True])
