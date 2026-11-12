import gc
import inspect
import os
import sys
import types
import weakref
from unittest import mock

from support import ProjectTestCase, fake_torch, transcript, transcript_result


class PipelineTests(ProjectTestCase):
    def setUp(self):
        super().setUp()
        self.target = self.media()
        self.failure = None
        self.events = []
        self.references = []
        self.arguments = {}
        owner = self

        class Model:
            def __init__(self, stage):
                self.stage = stage

            def transcribe(self, audio, *, batch_size, language):
                owner.arguments["transcribe"] = {"batch_size": batch_size, "language": language}
                owner.events.append("transcribe")
                if owner.failure == "asr":
                    raise RuntimeError("ASR failed")
                if owner.failure == "asr_interrupt":
                    raise KeyboardInterrupt()
                return transcript_result()

            def __call__(self, audio, **kwargs):
                owner.arguments["diarize"] = kwargs
                owner.events.append("diarize")
                if owner.failure == "diarize":
                    raise RuntimeError("diarization failed")
                return []

        def make_model(stage):
            # Normal and degraded optional stages must never overlap models.
            for previous_stage, reference in self.references:
                self.assertIsNone(reference(), f"{previous_stage} still alive while loading {stage}")
            model = Model(stage)
            self.references.append((stage, weakref.ref(model)))
            self.events.append("load_" + stage)
            return model

        def load_model(*args, **kwargs):
            self.arguments["load_model"] = (args, kwargs)
            if self.failure == "load_asr":
                raise RuntimeError("ASR model load failed")
            return make_model("asr")

        def load_audio(path):
            if self.failure == "audio":
                raise OSError("audio decoder failed")
            return object()

        def load_align_model(**kwargs):
            self.arguments["load_align"] = kwargs
            if self.failure == "load_align":
                raise RuntimeError("aligner load failed")
            return make_model("align"), {"language": "uk"}

        def align(segments, model, metadata, audio, device, **kwargs):
            self.events.append("align")
            if self.failure == "align":
                raise RuntimeError("alignment failed")
            return {"segments": segments}

        def diarization_pipeline(**kwargs):
            self.arguments["diarization_pipeline"] = kwargs
            if self.failure == "load_diarize":
                raise RuntimeError("diarization load failed")
            return make_model("diarize")

        def assign_word_speakers(speakers, result):
            if self.failure == "assign":
                raise RuntimeError("speaker assignment failed")
            for segment in result["segments"]:
                segment["speaker"] = "SPEAKER_00"
            return result

        wx = types.ModuleType("whisperx")
        wx.__path__ = []
        wx.load_model = load_model
        wx.load_audio = load_audio
        wx.load_align_model = load_align_model
        wx.align = align
        wx.assign_word_speakers = assign_word_speakers
        diarize = types.ModuleType("whisperx.diarize")
        diarize.DiarizationPipeline = diarization_pipeline
        patcher = mock.patch.dict(sys.modules, {"whisperx": wx, "whisperx.diarize": diarize})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.patch(transcript, "torch", fake_torch())
        self.patch(transcript, "whisperx", wx)

        def flush():
            frame = inspect.currentframe().f_back
            try:
                for name in ("model", "align_model", "align_metadata", "diarize_model"):
                    if name in frame.f_locals:
                        self.assertIsNone(frame.f_locals[name], f"{name} not cleared before cleanup")
            finally:
                del frame
            self.events.append("flush")
            gc.collect()

        self.patch(transcript, "flush_vram", flush)

    def run_pipeline(self, **updates):
        return transcript.transcribe_file(self.target, self.config(**updates))

    def assert_models_released(self):
        gc.collect()
        for stage, reference in self.references:
            self.assertIsNone(reference(), f"leaked {stage} model")

    def test_three_stage_order_and_no_simultaneously_resident_models(self):
        result = self.run_pipeline(diarize=True, hf_token="test-token")
        self.assertEqual(result["segments"][0]["speaker"], "SPEAKER_00")
        self.assertEqual(self.events, [
            "load_asr", "transcribe", "flush", "load_align", "align", "flush",
            "load_diarize", "diarize", "flush",
        ])
        self.assert_models_released()

    def test_asr_load_failure_still_flushes_without_unboundlocalerror(self):
        self.failure = "load_asr"
        with self.assertRaisesRegex(RuntimeError, "ASR model load"):
            self.run_pipeline()
        self.assertEqual(self.events, ["flush"])
        self.assert_models_released()

    def test_audio_failure_releases_asr_before_next_file(self):
        self.failure = "audio"
        with self.assertRaisesRegex(OSError, "audio decoder"):
            self.run_pipeline()
        self.assertEqual(self.events[-1], "flush")
        self.assert_models_released()
        self.failure = None
        self.run_pipeline()
        self.assert_models_released()

    def test_asr_failure_releases_model_and_next_file_runs(self):
        self.failure = "asr"
        with self.assertRaisesRegex(RuntimeError, "ASR failed"):
            self.run_pipeline()
        self.assertEqual(self.events[-1], "flush")
        self.assert_models_released()
        self.failure = None
        self.run_pipeline()
        self.assert_models_released()

    def test_keyboardinterrupt_propagates_through_finally_cleanup(self):
        self.failure = "asr_interrupt"
        with self.assertRaises(KeyboardInterrupt):
            self.run_pipeline()
        self.assertEqual(self.events[-1], "flush")
        self.assert_models_released()

    def test_alignment_load_failure_degrades_and_still_flushes(self):
        self.failure = "load_align"
        result = self.run_pipeline(diarize=True, hf_token="test-token")
        self.assertEqual(len(result["segments"]), 1)
        self.assertIn("alignment skipped", self.console.output)
        self.assertEqual(self.events.count("flush"), 3)
        self.assert_models_released()

    def test_alignment_failure_releases_model_before_diarization(self):
        self.failure = "align"
        result = self.run_pipeline(diarize=True, hf_token="test-token")
        self.assertEqual(result["segments"][0]["speaker"], "SPEAKER_00")
        self.assertIn("alignment skipped", self.console.output)
        self.assert_models_released()

    def test_diarization_constructor_failure_degrades_and_flushes(self):
        self.failure = "load_diarize"
        result = self.run_pipeline(diarize=True, hf_token="test-token")
        self.assertNotIn("speaker", result["segments"][0])
        self.assertEqual(self.events.count("flush"), 3)
        self.assert_models_released()

    def test_diarization_call_failure_degrades_and_releases_model(self):
        self.failure = "diarize"
        result = self.run_pipeline(diarize=True, hf_token="test-token")
        self.assertNotIn("speaker", result["segments"][0])
        self.assertIn("Diarization failed", self.console.output)
        self.assert_models_released()

    def test_speaker_assignment_failure_also_releases_model(self):
        self.failure = "assign"
        result = self.run_pipeline(diarize=True, hf_token="test-token")
        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(self.events[-1], "flush")
        self.assert_models_released()

    def test_missing_token_skips_only_diarization(self):
        self.run_pipeline(diarize=True, hf_token="")
        self.assertNotIn("load_diarize", self.events)
        self.assertEqual(self.events.count("flush"), 2)

    def test_batch_size_prompt_and_beam_options_preserved(self):
        prompt = "Це лекція. Без слів-паразитів: "
        self.run_pipeline(batch_size=3, remove_fillers=True, initial_prompt=prompt)
        args, kwargs = self.arguments["load_model"]
        self.assertEqual(args, ("large-v3-turbo", "cuda"))
        self.assertEqual(kwargs["compute_type"], "float16")
        self.assertEqual(kwargs["asr_options"], {"beam_size": 5, "initial_prompt": prompt})
        self.assertEqual(self.arguments["transcribe"]["batch_size"], 3)

    def test_old_configs_keep_batch_eight_and_cpu_uses_int8(self):
        self.run_pipeline(device="cpu", remove_fillers=False, initial_prompt="ignored")
        self.assertEqual(self.arguments["load_model"][1]["compute_type"], "int8")
        self.assertEqual(self.arguments["load_model"][1]["asr_options"], {"beam_size": 5})
        self.assertEqual(self.arguments["transcribe"]["batch_size"], 8)

    def test_language_auto_still_passes_none_then_aligns_detected_language(self):
        self.run_pipeline(language="auto")
        self.assertIsNone(self.arguments["load_model"][1]["language"])
        self.assertIsNone(self.arguments["transcribe"]["language"])
        self.assertEqual(self.arguments["load_align"]["language_code"], "uk")

    def test_speaker_count_hints_and_token_are_passed_to_existing_api(self):
        self.run_pipeline(diarize=True, hf_token="test-token", min_speakers=2, max_speakers=3)
        self.assertEqual(self.arguments["diarization_pipeline"], {"token": "test-token", "device": "cuda"})
        self.assertEqual(self.arguments["diarize"], {"min_speakers": 2, "max_speakers": 3})


class DllAndCleanupTests(ProjectTestCase):
    def test_dll_directories_checked_independently_and_handles_retained(self):
        first = self.workspace / "site1"
        second = self.workspace / "site2"
        cublas = first / "nvidia/cublas/bin"
        cudnn = second / "nvidia/cudnn/bin"
        cublas.mkdir(parents=True)
        cudnn.mkdir(parents=True)
        self.patch(sys, "path", [str(first), str(second)])
        handles = []
        self.patch(transcript, "_DLL_DIRECTORY_HANDLES", handles)
        self.patch(transcript, "_DLL_DIRECTORIES", set())
        objects = [object(), object()]
        with mock.patch.object(os, "add_dll_directory", side_effect=objects, create=True) as add:
            with mock.patch.dict(os.environ, {"PATH": "original"}):
                transcript.setup_cuda_dlls()
                transcript.setup_cuda_dlls()
                self.assertEqual([call.args[0] for call in add.call_args_list], [str(cublas), str(cudnn)])
                self.assertIn(str(cublas), os.environ["PATH"])
                self.assertIn(str(cudnn), os.environ["PATH"])
                self.assertEqual(handles, objects)

    def test_cudnn_only_install_does_not_require_cublas_to_exist(self):
        cudnn = self.workspace / "site/nvidia/cudnn/bin"
        cudnn.mkdir(parents=True)
        self.patch(sys, "path", [str(self.workspace / "site")])
        self.patch(transcript, "_DLL_DIRECTORY_HANDLES", [])
        self.patch(transcript, "_DLL_DIRECTORIES", set())
        with mock.patch.object(os, "add_dll_directory", create=True) as add:
            with mock.patch.dict(os.environ):
                transcript.setup_cuda_dlls()
        add.assert_called_once_with(str(cudnn))

    def test_flush_vram_gc_and_cuda_cache_boundaries(self):
        runtime = fake_torch()
        self.patch(transcript, "torch", runtime)
        with mock.patch.object(gc, "collect") as collect:
            transcript.flush_vram()
        collect.assert_called_once()
        runtime.cuda.empty_cache.assert_called_once()
        runtime.cuda.is_available.return_value = False
        runtime.cuda.empty_cache.reset_mock()
        transcript.flush_vram()
        runtime.cuda.empty_cache.assert_not_called()

    def test_flush_without_torch_does_not_import_it(self):
        self.patch(transcript, "torch", None)
        load = self.patch(transcript, "load_torch", mock.Mock())
        transcript.flush_vram()
        load.assert_not_called()
