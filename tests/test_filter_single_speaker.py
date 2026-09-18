import contextlib
import io
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import filter_single_speaker as filtering


def metrics(*speakers):
    return filtering.summarize_turns([(index, index + 1.0, name)
                                      for index, name in enumerate(speakers)], max(1, len(speakers)))




class SpawnBackend:
    """Picklable fake backend used to exercise real spawn processes without CUDA."""
    def __init__(self, model, device, token_env):
        self.device = device

    def analyze(self, path):
        item = json.loads(path.read_text(encoding="utf-8"))
        log = path.with_suffix(path.suffix + ".calls")
        with log.open("a", encoding="utf-8") as stream:
            stream.write(self.device + "\n")
        if item.get("crash"):
            os._exit(7)
        if item.get("wait_for"):
            marker = Path(item["wait_for"])
            deadline = time.monotonic() + 15
            while not marker.exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("Other GPU worker never ran")
                time.sleep(0.02)
        if item.get("marker"):
            Path(item["marker"]).write_text("done", encoding="utf-8")
        time.sleep(item.get("delay", 0))
        if item.get("oom"):
            raise RuntimeError("CUDA out of memory")
        if item.get("error"):
            raise ValueError("bad fixture audio")
        return metrics(*item.get("speakers", ["speaker0"]))


class FailingSpawnBackend:
    def __init__(self, model, device, token_env):
        raise RuntimeError("model access denied")

class SummaryTests(unittest.TestCase):
    def test_turn_taking_and_overlapping_speakers_both_filtered(self):
        for turns in ([(0, 2, "a"), (2, 4, "b")], [(0, 4, "a"), (1, 3, "b")]):
            result = filtering.summarize_turns(turns, 4)
            self.assertEqual(filtering.classify(result, 0, 0), ("multiple", 2))

    def test_repeated_speaker_and_duplicate_tracks_do_not_inflate_count(self):
        result = filtering.summarize_turns([(0, 2, "a"), (1, 3, "a"), (4, 5, "a")], 5)
        self.assertEqual(result["speaker_seconds"], {"a": 4})
        self.assertEqual(result["speech_seconds"], 4)
        self.assertEqual(filtering.classify(result, 0, 0), ("single", 1))

    def test_silence_and_short_secondary_speaker(self):
        self.assertEqual(filtering.classify(metrics(), 0, 0), ("no_speech", 0))
        result = filtering.summarize_turns([(0, 8, "a"), (8, 8.1, "b")], 10)
        self.assertEqual(filtering.classify(result, 0, 0), ("multiple", 2))
        self.assertEqual(filtering.classify(result, 0.5, 0.05), ("single", 1))
        self.assertEqual(filtering.classify(result, 20, 0), ("insufficient_speech", 0))

    def test_intervals_clipped_and_invalid_values_rejected(self):
        result = filtering.summarize_turns([(-1, 2, "a"), (8, 12, "a")], 10)
        self.assertEqual(result["speaker_seconds"], {"a": 4})
        with self.assertRaises(ValueError):
            filtering.summarize_turns([(3, 2, "a")], 10)
        with self.assertRaises(ValueError):
            filtering.summarize_turns([(0, float("nan"), "a")], 10)


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.input = self.root / "audio.scp"
        self.a = self.root / "one speaker.flac"
        self.b = self.root / "two.flac"
        self.c = self.root / "silent.flac"
        for path in (self.a, self.b, self.c):
            path.write_bytes(b"fixture audio")
        self.results = {str(self.a): metrics("a"), str(self.b): metrics("a", "b"), str(self.c): metrics()}
        self.calls = []
        self.output_text = io.StringIO()
        quiet = contextlib.redirect_stdout(self.output_text)
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def factory(self, *args):
        owner = self

        class Backend:
            def analyze(self, path):
                owner.calls.append(str(path))
                result = owner.results[str(path)]
                if isinstance(result, BaseException):
                    raise result
                return result
        return Backend()

    def args(self, *extra, jsonl=False):
        return filtering.parse_args(["--jsonl" if jsonl else "--scp", str(self.input), *extra])

    def run_filter(self, *extra, jsonl=False, factory=None):
        return filtering.run_filter(self.args(*extra, jsonl=jsonl), factory or self.factory)

    def output(self):
        return self.input.with_name(self.input.stem + "_single_speaker" + self.input.suffix)

    def state(self):
        return self.output().with_suffix(".state.jsonl")

    def write_scp(self, *paths):
        self.input.write_text("".join(str(path) + "\n" for path in paths), encoding="utf-8")

    def test_scp_format_order_duplicates_and_resume(self):
        self.write_scp(self.a, self.b, self.c, self.a)
        original = self.input.read_bytes()
        self.assertEqual(self.run_filter(), 0)
        self.assertEqual(self.output().read_text(), f"{self.a}\n{self.a}\n")
        self.assertEqual(len(self.calls), 3)
        self.calls.clear()
        self.assertEqual(self.run_filter(factory=lambda *args: self.fail("Model loaded on cached resume")), 0)
        self.assertFalse(self.calls)
        self.assertEqual(self.output().read_text(), f"{self.a}\n{self.a}\n")
        self.assertEqual(self.input.read_bytes(), original)
        self.assertTrue(all(path.exists() for path in (self.a, self.b, self.c)))

    def test_jsonl_preserves_original_pair_unicode_and_extra_fields(self):
        self.input = self.root / "flac_txt.zh.jsonl"
        first = json.dumps([str(self.a), "字幕 file.qwen.txt", {"source": "人声"}], ensure_ascii=False)
        second = json.dumps([str(self.b), "two.txt"])
        self.input.write_text(first + "\n" + second + "\n", encoding="utf-8")
        self.assertEqual(self.run_filter(jsonl=True), 0)
        self.assertEqual(self.output().name, "flac_txt.zh_single_speaker.jsonl")
        self.assertEqual(self.output().read_text(encoding="utf-8"), first + "\n")

    def test_interrupt_and_resume_skip_finished_inference(self):
        self.write_scp(self.a, self.b)
        self.results[str(self.b)] = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_filter()
        self.assertFalse(self.output().exists())
        self.assertTrue(Path(str(self.output()) + ".tmp").exists())
        self.assertEqual(len(self.state().read_text().splitlines()), 2)  # header + a
        self.calls.clear()
        self.results[str(self.b)] = metrics("b")
        self.assertEqual(self.run_filter(), 0)
        self.assertEqual(self.calls, [str(self.b)])
        self.assertEqual(self.output().read_text(), f"{self.a}\n{self.b}\n")

    def test_rebuild_output_if_interrupted_after_result_was_saved(self):
        self.write_scp(self.a)
        with patch.object(filtering.os, "replace", side_effect=OSError("simulated replace failure")), \
                self.assertRaises(OSError):
            self.run_filter()
        self.calls.clear()
        self.assertEqual(self.run_filter(factory=lambda *args: self.fail("Result not reused")), 0)
        self.assertEqual(self.output().read_text(), f"{self.a}\n")

    def test_partial_last_state_line_is_repaired(self):
        self.write_scp(self.a)
        self.run_filter()
        with self.state().open("ab") as stream:
            stream.write(b'{"kind": "res')
        self.calls.clear()
        self.assertEqual(self.run_filter(), 0)
        self.assertFalse(self.calls)
        for line in self.state().read_text().splitlines():
            json.loads(line)

    def test_complete_corrupt_state_line_is_not_silently_discarded(self):
        self.write_scp(self.a)
        self.run_filter()
        with self.state().open("ab") as stream:
            stream.write(b"not json\n")
        with self.assertRaisesRegex(ValueError, "Invalid state"):
            self.run_filter()

    def test_unrelated_state_file_without_newline_is_untouched(self):
        self.write_scp(self.a)
        self.state().write_bytes(b"important unrelated file")
        with self.assertRaisesRegex(ValueError, "recognized header"):
            self.run_filter()
        self.assertEqual(self.state().read_bytes(), b"important unrelated file")

    def test_changed_audio_is_reprocessed(self):
        self.write_scp(self.a)
        self.run_filter()
        self.calls.clear()
        self.a.write_bytes(b"replacement audio is different")
        self.results[str(self.a)] = metrics("a", "b")
        self.assertEqual(self.run_filter(), 0)
        self.assertEqual(self.calls, [str(self.a)])
        self.assertEqual(self.output().read_text(), "")

    def test_input_changes_and_new_items_reuse_audio_results(self):
        self.write_scp(self.a)
        self.run_filter()
        self.calls.clear()
        self.write_scp(self.a, self.b)
        self.run_filter()
        self.assertEqual(self.calls, [str(self.b)])

    def test_threshold_changes_reuse_metrics_without_inference(self):
        self.write_scp(self.a)
        self.results[str(self.a)] = filtering.summarize_turns([(0, 5, "a"), (5, 5.1, "b")], 6)
        self.run_filter("--decision-policy", "strict", "--min-speaker-seconds", "0",
                        "--min-speaker-ratio", "0")
        self.assertEqual(self.output().read_text(), "")
        self.calls.clear()
        self.assertEqual(self.run_filter("--decision-policy", "strict",
                                         "--min-speaker-seconds", "0.5", "--min-speaker-ratio", "0"), 0)
        self.assertFalse(self.calls)
        self.assertEqual(self.output().read_text(), str(self.a) + "\n")

    def test_changed_model_refuses_incompatible_state_before_touching_output(self):
        self.write_scp(self.a)
        self.run_filter()
        before = self.output().read_bytes()
        with self.assertRaisesRegex(ValueError, "model/config differs"):
            self.run_filter("--model", "different/model")
        self.assertEqual(self.output().read_bytes(), before)

    def test_error_is_cached_and_retry_errors_only_retries_once_per_audio(self):
        self.write_scp(self.a, self.a)
        self.results[str(self.a)] = ValueError("bad audio")
        self.assertEqual(self.run_filter(), 1)
        self.assertEqual(len(self.calls), 1)
        self.calls.clear()
        self.assertEqual(self.run_filter(), 1)
        self.assertFalse(self.calls)
        self.assertEqual(self.run_filter("--retry-errors"), 1)
        self.assertEqual(len(self.calls), 1)
        self.results[str(self.a)] = metrics("a")
        self.assertEqual(self.run_filter("--retry-errors"), 0)
        self.assertEqual(self.output().read_text(), f"{self.a}\n{self.a}\n")

    def test_missing_file_does_not_load_model_and_is_retried_when_created(self):
        missing = self.root / "missing.flac"
        self.write_scp(missing)
        self.assertEqual(self.run_filter(factory=lambda *args: self.fail("Unexpected model load")), 1)
        missing.write_bytes(b"now exists")
        self.results[str(missing)] = metrics("a")
        self.assertEqual(self.run_filter(), 0)
        self.assertEqual(self.calls, [str(missing)])

    def test_model_load_failure_is_not_cached_as_an_audio_error(self):
        self.write_scp(self.a, self.b)
        def failed(*args):
            raise RuntimeError("model unavailable")
        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            self.run_filter(factory=failed)
        self.assertEqual(len(self.state().read_text().splitlines()), 1)

    def test_relative_audio_paths_resolve_against_explicit_base(self):
        self.input.write_text("one speaker.flac\n", encoding="utf-8")
        self.assertEqual(self.run_filter("--path-base", str(self.root)), 0)
        self.assertEqual(self.calls, [str(self.a)])
        self.assertEqual(self.output().read_text(), "one speaker.flac\n")

    def test_malformed_json_does_not_replace_previous_output(self):
        self.input = self.root / "pairs.jsonl"
        self.input.write_text("not json\n", encoding="utf-8")
        self.output().write_text("previous output\n")
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            self.run_filter(jsonl=True)
        self.assertEqual(self.output().read_text(), "previous output\n")

    def test_limit_does_not_parse_extra_record(self):
        self.input = self.root / "pairs.jsonl"
        self.input.write_text(json.dumps([str(self.a), "a.txt"]) + "\nnot json\n")
        self.assertEqual(self.run_filter("--limit", "1", jsonl=True), 0)

    def test_state_input_alias_rejected_without_modification(self):
        self.write_scp(self.a)
        before = self.input.read_bytes()
        with self.assertRaisesRegex(ValueError, "different files"):
            self.run_filter("--state", str(self.input))
        self.assertEqual(self.input.read_bytes(), before)

    def test_local_config_directory_supported_and_fingerprinted(self):
        directory = self.root / "model"
        directory.mkdir()
        config = directory / "config.yaml"
        config.write_text("pipeline: example")
        identity = filtering.model_identity(str(directory))
        self.assertEqual(identity["model"], str(config))
        first_hash = identity["config_sha256"]
        config.write_text("pipeline: changed")
        self.assertNotEqual(filtering.model_identity(str(directory))["config_sha256"], first_hash)


class BackendTests(unittest.TestCase):
    def test_pyannote_annotation_and_wrapper_use_full_audio_without_speaker_constraints(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                calls = {}
                annotation = types.SimpleNamespace(itertracks=lambda **kwargs: iter([
                    (types.SimpleNamespace(start=0, end=8), 0, "speaker0"),
                    (types.SimpleNamespace(start=9, end=10), 0, "speaker1"),
                ]))
                class PipelineInstance:
                    def to(self, device):
                        calls["device"] = device

                    def __call__(self, audio, **kwargs):
                        calls["input"] = audio
                        calls["kwargs"] = kwargs
                        return types.SimpleNamespace(speaker_diarization=annotation) if wrapped else annotation

                class Pipeline:
                    @classmethod
                    def from_pretrained(cls, model, use_auth_token=None):
                        calls["model"] = model
                        if not calls.get("checkpoint_guard"):
                            raise AssertionError("Checkpoint compatibility context is missing")
                        return PipelineInstance()

                @contextlib.contextmanager
                def checkpoint_guard(torch_module):
                    calls["checkpoint_guard"] = True
                    try:
                        yield
                    finally:
                        calls["checkpoint_guard"] = False

                class Waveform:
                    def __len__(self):
                        return 160000

                    def mean(self, axis):
                        calls["mean_axis"] = axis
                        return "mono"

                torch = types.ModuleType("torch")
                torch.cuda = types.SimpleNamespace(is_available=lambda: False)
                torch.device = lambda device: types.SimpleNamespace(type=device)
                torch.from_numpy = lambda value: types.SimpleNamespace(unsqueeze=lambda dim: (value, dim))
                torch.inference_mode = contextlib.nullcontext
                soundfile = types.ModuleType("soundfile")
                soundfile.read = lambda *args, **kwargs: (Waveform(), 16000)
                package = types.ModuleType("pyannote")
                package.__path__ = []
                audio = types.ModuleType("pyannote.audio")
                audio.Pipeline = Pipeline
                modules = {"torch": torch, "soundfile": soundfile, "pyannote": package, "pyannote.audio": audio}
                with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()), \
                        patch.object(filtering, "pyannote_checkpoint_globals", side_effect=checkpoint_guard), \
                        patch.object(filtering, "local_pipeline_config",
                                     return_value=contextlib.nullcontext(Path("local.yaml"))):
                    backend = filtering.PyannoteDiarizer("local-model", "auto", "MISSING_TEST_TOKEN")
                    result = backend.analyze(Path("audio.flac"))
                self.assertEqual(result["duration"], 10)
                self.assertEqual(filtering.classify(result, 0, 0), ("multiple", 2))
                self.assertEqual(calls["kwargs"], {})
                self.assertEqual(calls["model"], "local.yaml")
                self.assertFalse(calls["checkpoint_guard"])
                self.assertEqual(calls["input"], {"waveform": ("mono", 0), "sample_rate": 16000})



class MultiGpuTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.input = self.root / "audio.scp"
        self.output = self.root / "audio_single_speaker.scp"
        self.state = self.root / "audio_single_speaker.state.jsonl"
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def audio(self, name, **config):
        path = self.root / (name + ".flac")
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def args(self, *extra):
        return filtering.parse_args(["--scp", str(self.input), "--devices", "0,1", *extra])

    def write(self, *paths):
        self.input.write_text("".join(str(path) + "\n" for path in paths), encoding="utf-8")

    def test_spawn_uses_both_gpus_persists_completion_order_and_preserves_output_order(self):
        marker = self.root / "second.started"
        first = self.audio("first", wait_for=str(marker), delay=0.2)
        second = self.audio("second", marker=str(marker))
        multiple = self.audio("multiple", speakers=["a", "b"])
        self.write(first, second, multiple, first)
        self.assertEqual(filtering.run_filter(self.args(), SpawnBackend), 0)
        self.assertEqual(self.output.read_text(), f"{first}\n{second}\n{first}\n")
        records = [json.loads(line) for line in self.state.read_text().splitlines()][1:]
        self.assertEqual(records[0]["audio"], str(second))
        devices = set()
        for path in (first, second, multiple):
            calls = path.with_suffix(".flac.calls").read_text().splitlines()
            self.assertEqual(len(calls), 1)
            devices.update(calls)
        self.assertEqual(devices, {"cuda:0", "cuda:1"})
        with patch.object(filtering, "DevicePool", side_effect=AssertionError("Cached run spawned GPU workers")):
            self.assertEqual(filtering.run_filter(self.args(), FailingSpawnBackend), 0)

    def test_jsonl_output_preserves_text_and_duplicates_across_workers(self):
        first = self.audio("first")
        second = self.audio("second", speakers=["a", "b"])
        input_path = self.root / "pairs.jsonl"
        line = json.dumps([str(first), "字幕.txt", {"extra": 1}], ensure_ascii=False)
        input_path.write_text(line + "\n" + json.dumps([str(second), "two.txt"]) + "\n" + line + "\n",
                              encoding="utf-8")
        args = filtering.parse_args(["--jsonl", str(input_path), "--devices", "0,1"])
        self.assertEqual(filtering.run_filter(args, SpawnBackend), 0)
        self.assertEqual((self.root / "pairs_single_speaker.jsonl").read_text(encoding="utf-8"),
                         line + "\n" + line + "\n")
        self.assertEqual(len(first.with_suffix(".flac.calls").read_text().splitlines()), 1)

    def test_setup_failure_leaves_checkpoint_without_per_audio_errors(self):
        first = self.audio("first")
        self.write(first)
        self.output.write_text("previous output\n")
        with self.assertRaisesRegex(RuntimeError, "model access denied"):
            filtering.run_filter(self.args(), FailingSpawnBackend)
        self.assertEqual(len(self.state.read_text().splitlines()), 1)
        self.assertEqual(self.output.read_text(), "previous output\n")

    def test_worker_hard_exit_is_reported_instead_of_hanging(self):
        first = self.audio("first", crash=True)
        self.write(first)
        with self.assertRaisesRegex(RuntimeError, "exited unexpectedly"):
            filtering.run_filter(self.args(), SpawnBackend)
        self.assertEqual(len(self.state.read_text().splitlines()), 1)

    def test_error_results_cached_and_retryable_with_multiple_workers(self):
        first = self.audio("first", error=True)
        self.write(first, first)
        self.assertEqual(filtering.run_filter(self.args(), SpawnBackend), 1)
        self.assertEqual(filtering.run_filter(self.args("--retry-errors"), SpawnBackend), 1)
        self.assertEqual(len(first.with_suffix(".flac.calls").read_text().splitlines()), 2)
        self.assertEqual(self.output.read_text(), "")

    def test_resume_from_single_gpu_checkpoint_and_process_new_file(self):
        first = self.audio("first")
        self.write(first)
        args = filtering.parse_args(["--scp", str(self.input), "--device", "cpu"])
        self.assertEqual(filtering.run_filter(args, SpawnBackend), 0)
        second = self.audio("second")
        self.write(first, second)
        self.assertEqual(filtering.run_filter(self.args(), SpawnBackend), 0)
        self.assertEqual(first.with_suffix(".flac.calls").read_text().splitlines(), ["cpu"])
        self.assertEqual(self.output.read_text(), f"{first}\n{second}\n")

    def test_oom_preserves_already_completed_results_for_resume(self):
        marker = self.root / "ready"
        first = self.audio("first", marker=str(marker))
        second = self.audio("second", wait_for=str(marker), delay=0.3, oom=True)
        self.write(first, second)
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            filtering.run_filter(self.args(), SpawnBackend)
        records = [json.loads(line) for line in self.state.read_text().splitlines()][1:]
        self.assertEqual([record["audio"] for record in records], [str(first)])
        second.write_text(json.dumps({"speakers": ["b"]}))
        self.assertEqual(filtering.run_filter(self.args(), SpawnBackend), 0)
        self.assertEqual(len(first.with_suffix(".flac.calls").read_text().splitlines()), 1)
        self.assertEqual(self.output.read_text(), f"{first}\n{second}\n")


class CacheTests(unittest.TestCase):
    def test_all_model_caches_configured_on_data_disk_before_download(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True), \
                patch.object(tempfile, "tempdir", None):
            root = Path(tmp).resolve()
            local_config = root / "local" / "config.yaml"
            with patch.object(filtering, "PyannoteDiarizer") as backend, \
                    patch.object(filtering, "prepare_model_bundle", return_value=local_config) as prepare:
                self.assertEqual(filtering.main(["--download-model", "--cache-dir", str(root)]), 0)
            prepare.assert_called_once_with(filtering.DEFAULT_MODEL, root, "HF_TOKEN")
            backend.assert_called_once_with(str(local_config), "cpu", "HF_TOKEN")
            for key in ("HF_HOME", "HF_HUB_CACHE", "PYANNOTE_CACHE", "TORCH_HOME",
                        "XDG_CACHE_HOME", "HF_XET_CACHE", "TRITON_CACHE_DIR", "TMPDIR"):
                self.assertIn(root, Path(os.environ[key]).parents)
            filtering.configure_model_cache(root, offline=True)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_gpu_list_and_download_arguments(self):
        args = filtering.parse_args(["--scp", "a.scp", "--devices", "0,1,2,3,4,5,6,7"])
        self.assertEqual(args.devices, [f"cuda:{index}" for index in range(8)])
        for arguments in (
            ["--scp", "a.scp", "--devices", "0,0"],
            ["--scp", "a.scp", "--devices", "0,"],
            ["--scp", "a.scp", "--devices", "cpu"],
            ["--scp", "a.scp", "--devices", "0,1", "--device", "cuda:0"],
            ["--download-model"],
            ["--download-model", "--cache-dir", "models", "--offline"],
            ["--scp", "a.scp", "--cpu-threads", "0"],
        ):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                filtering.parse_args(arguments)


class CliTests(unittest.TestCase):
    def test_exactly_one_input_and_valid_thresholds(self):
        for arguments in ([], ["--scp", "a.scp", "--jsonl", "a.jsonl"],
                          ["--scp", "a.scp", "--suffix", ""],
                          ["--scp", "a.scp", "--suffix", "../output"],
                          ["--scp", "a.scp", "--min-speaker-ratio", "1.2"],
                          ["--scp", "a.scp", "--min-speaker-seconds", "nan"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                filtering.parse_args(arguments)


if __name__ == "__main__":
    unittest.main()
