import json
import contextlib
import os
import shutil
import socket
import sys
import tempfile
import types
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pyannote_local as local
import filter_single_speaker as filtering


class LocalBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.cache = self.root / "cache"
        self.source = self.root / "source"
        self.source.mkdir()
        self.config = self.source / "config.yaml"
        self.segmentation = self.source / "segmentation.bin"
        self.embedding = self.source / "embedding.bin"
        self.segmentation.write_bytes(b"segmentation weights")
        self.embedding.write_bytes(b"embedding weights")
        local.write_yaml_atomic(self.config, {
            "version": "3.1.0",
            "pipeline": {"name": "pyannote.audio.pipelines.SpeakerDiarization", "params": {
                "segmentation": "pyannote/segmentation-3.0",
                "embedding": "pyannote/wespeaker-voxceleb-resnet34-LM",
                "embedding_exclude_overlap": True,
            }},
            "params": {"clustering": {"threshold": 0.7}},
        })

    def download(self, model, filename, cache_dir, token_env, local_only):
        if filename == "config.yaml":
            return self.config
        return self.embedding if "wespeaker" in model else self.segmentation

    def bundle(self):
        with patch.object(local, "hub_file", side_effect=self.download):
            return local.prepare_model_bundle(filtering.DEFAULT_MODEL, self.cache)

    def test_export_contains_real_files_and_relative_references(self):
        path = self.bundle()
        params = local.read_config(path)["pipeline"]["params"]
        for name in local.COMPONENTS:
            reference = params[name]
            self.assertFalse(Path(reference).is_absolute())
            self.assertIn("pyannote", reference)
            weight = path.parent / reference
            self.assertTrue(weight.is_file())
            self.assertFalse(weight.is_symlink())
        self.assertEqual((path.parent / params["embedding"]).read_bytes(), self.embedding.read_bytes())

    def test_bundle_portable_to_different_server_path_without_hub_calls(self):
        source = self.bundle()
        moved = self.root / "new server" / "models"
        shutil.copytree(source.parent, moved)
        shutil.rmtree(self.source)
        shutil.rmtree(self.cache)
        with patch.object(local, "hub_file", side_effect=AssertionError("Hub must not be called")), \
                local.network_disabled(), local.local_pipeline_config(str(moved)) as runtime:
            params = local.read_config(runtime)["pipeline"]["params"]
            for name in local.COMPONENTS:
                self.assertEqual(Path(params[name]).parent, moved)
                self.assertTrue(Path(params[name]).is_file())

    def test_existing_bundle_does_not_even_check_hub_cache(self):
        path = self.bundle()
        with patch.dict(os.environ, {"PYANNOTE_CACHE": str(self.cache / "pyannote")}), \
                patch.object(local, "hub_file", side_effect=AssertionError("No hub check")):
            self.assertEqual(local.offline_config_path(filtering.DEFAULT_MODEL), path)

    def test_concurrent_legacy_exports_have_no_temporary_file_collisions(self):
        with patch.object(local, "hub_file", side_effect=self.download), ThreadPoolExecutor(4) as pool:
            paths = list(pool.map(
                lambda _: local.prepare_model_bundle(filtering.DEFAULT_MODEL, self.cache, local_only=True),
                range(8)))
        self.assertEqual(len(set(paths)), 1)
        params = local.read_config(paths[0])["pipeline"]["params"]
        self.assertEqual((paths[0].parent / params["segmentation"]).read_bytes(),
                         self.segmentation.read_bytes())
        self.assertEqual((paths[0].parent / params["embedding"]).read_bytes(),
                         self.embedding.read_bytes())

    def test_legacy_cache_migration_only_requests_local_files(self):
        calls = []
        def cached(*args, **kwargs):
            calls.append(args[-1] if args else kwargs["local_only"])
            return self.download(*args, **kwargs)
        with patch.dict(os.environ, {"PYANNOTE_CACHE": str(self.cache / "pyannote")}), \
                patch.object(local, "hub_file", side_effect=cached), local.network_disabled():
            path = local.offline_config_path(filtering.DEFAULT_MODEL)
        self.assertTrue(path.is_file())
        self.assertEqual(calls, [True, True, True])

    def test_incomplete_legacy_cache_gives_no_online_fallback(self):
        with patch.dict(os.environ, {"PYANNOTE_CACHE": str(self.cache / "pyannote")}), \
                patch.object(local, "hub_file", side_effect=FileNotFoundError("not cached")):
            with self.assertRaisesRegex(RuntimeError, "No download was attempted"):
                local.offline_config_path(filtering.DEFAULT_MODEL)

    def test_missing_local_weight_is_error_without_download(self):
        config = self.bundle()
        (config.parent / "pyannote_embedding.bin").unlink()
        with patch.object(local, "hub_file", side_effect=AssertionError("No download")), \
                self.assertRaisesRegex(FileNotFoundError, "no online fallback"):
            with local.local_pipeline_config(str(config)):
                self.fail("Missing weights accepted")

    def test_local_config_rejects_remaining_remote_checkpoint(self):
        with patch.object(local, "hub_file", side_effect=AssertionError("No download")), \
                self.assertRaisesRegex(FileNotFoundError, "no online fallback"):
            with local.local_pipeline_config(str(self.config)):
                self.fail("Remote reference accepted")

    def test_relative_yaml_does_not_depend_on_current_directory(self):
        config = self.bundle()
        with local.local_pipeline_config(str(config)) as runtime:
            for path in local.read_config(runtime)["pipeline"]["params"].values():
                if isinstance(path, str) and path.endswith(".bin"):
                    self.assertTrue(Path(path).is_absolute())
                    self.assertTrue(Path(path).is_file())


class CheckpointCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.version_type = type("TorchVersion", (str,), {})
        self.specifications_type = type("Specifications", (), {})
        self.problem_type = type("Problem", (), {})
        self.resolution_type = type("Resolution", (), {})
        self.approved = [self.version_type, self.specifications_type,
                         self.problem_type, self.resolution_type]
        self.active = []
        self.additions = []

        @contextlib.contextmanager
        def safe_globals(values):
            self.additions.append(list(values))
            self.active.extend(values)
            try:
                yield
            finally:
                for value in values:
                    self.active.remove(value)

        self.torch = types.ModuleType("torch")
        self.torch.__path__ = []
        self.torch.load = object()
        self.torch.serialization = types.SimpleNamespace(
            safe_globals=safe_globals, get_safe_globals=lambda: list(self.active))
        version = types.ModuleType("torch.torch_version")
        version.TorchVersion = self.version_type
        task = types.ModuleType("pyannote.audio.core.task")
        task.Specifications = self.specifications_type
        task.Problem = self.problem_type
        task.Resolution = self.resolution_type
        modules = {"torch": self.torch, "torch.torch_version": version,
                   "pyannote.audio.core.task": task}
        for name in ("pyannote", "pyannote.audio", "pyannote.audio.core"):
            module = types.ModuleType(name)
            module.__path__ = []
            modules[name] = module
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_only_known_metadata_is_allowed_and_torch_load_is_unchanged(self):
        original_load = self.torch.load
        environment = dict(os.environ)
        with local.pyannote_checkpoint_globals(self.torch):
            self.assertEqual(self.active, self.approved)
            self.assertIs(self.torch.load, original_load)
        self.assertFalse(self.active)
        self.assertEqual(dict(os.environ), environment)
        self.assertIs(self.torch.load, original_load)

    def test_allowlist_restored_if_checkpoint_load_raises(self):
        with self.assertRaisesRegex(ValueError, "bad checkpoint"):
            with local.pyannote_checkpoint_globals(self.torch):
                raise ValueError("bad checkpoint")
        self.assertFalse(self.active)

    def test_existing_user_allowlist_is_preserved(self):
        marker = object()
        self.active.extend([marker, self.version_type])
        with local.pyannote_checkpoint_globals(self.torch):
            self.assertEqual(set(self.active), {marker, *self.approved})
        self.assertEqual(self.active, [marker, self.version_type])
        self.assertNotIn(self.version_type, self.additions[0])

    def test_nested_context_does_not_remove_outer_allowlist(self):
        with local.pyannote_checkpoint_globals(self.torch):
            with local.pyannote_checkpoint_globals(self.torch):
                self.assertEqual(self.active, self.approved)
            self.assertEqual(self.active, self.approved)
        self.assertFalse(self.active)
        self.assertEqual(self.additions[1], [])

    def test_old_torch_without_safe_globals_remains_compatible(self):
        legacy = types.SimpleNamespace(serialization=types.SimpleNamespace())
        with patch.dict(sys.modules, {"pyannote.audio.core.task": None}):
            with local.pyannote_checkpoint_globals(legacy):
                self.assertFalse(self.active)


class OfflineTests(unittest.TestCase):
    def test_download_diagnostic_includes_nested_failure_and_redacts_credentials(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf_privateExampleToken"}):
            inner = TimeoutError("HEAD request timed out https://huggingface.co/file?token=hf_privateExampleToken")
            outer = RuntimeError("cannot find the requested files in the local cache")
            outer.__cause__ = inner
            result = local.download_error_details(outer, "HF_TOKEN")
        self.assertIn("RuntimeError", result)
        self.assertIn("TimeoutError: HEAD request timed out", result)
        self.assertNotIn("hf_privateExampleToken", result)
        self.assertNotIn("token=", result)

    def test_hub_file_keeps_download_and_local_only_modes(self):
        module = types.ModuleType("huggingface_hub")
        module.hf_hub_download = Mock(return_value="local-config.yaml")
        with patch.dict(sys.modules, {"huggingface_hub": module}), \
                patch.dict(os.environ, {"HF_TOKEN": "hf_example"}):
            self.assertEqual(local.hub_file("repo/model@revision", "config.yaml", Path("cache"),
                                            "HF_TOKEN", True), Path("local-config.yaml"))
            kwargs = module.hf_hub_download.call_args.kwargs
            self.assertTrue(kwargs["local_files_only"])
            self.assertFalse(kwargs["token"])
            self.assertEqual(kwargs["revision"], "revision")
            error = RuntimeError("not cached")
            error.__cause__ = ConnectionError("Proxy connection failed")
            module.hf_hub_download.side_effect = error
            with self.assertRaisesRegex(RuntimeError, "ConnectionError: Proxy connection failed"):
                local.hub_file("repo/model", "config.yaml", Path("cache"), "HF_TOKEN", False)
            self.assertFalse(module.hf_hub_download.call_args.kwargs["local_files_only"])

    def test_guard_blocks_external_and_proxy_connections_and_restores_socket_api(self):
        connect = socket.socket.connect
        dns = socket.getaddrinfo
        with local.network_disabled():
            with self.assertRaisesRegex(RuntimeError, "DNS"):
                socket.getaddrinfo("huggingface.co", 443)
            for address in (("huggingface.co", 443), ("127.0.0.1", 7890)):
                with socket.socket() as stream:
                    with self.assertRaisesRegex(RuntimeError, "network connection"):
                        stream.connect(address)
        self.assertIs(socket.socket.connect, connect)
        self.assertIs(socket.getaddrinfo, dns)

    def test_offline_default_reaches_backend_without_explicit_offline_flag(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            audio = root / "test.flac"
            audio.write_bytes(b"audio")
            scp = root / "audio.scp"
            scp.write_text(str(audio) + "\n")
            class Backend:
                def __init__(self, *args):
                    self.assert_offline()

                @staticmethod
                def assert_offline():
                    if os.environ.get("HF_HUB_OFFLINE") != "1":
                        raise AssertionError("Offline environment not set")
                    try:
                        socket.getaddrinfo("huggingface.co", 443)
                    except RuntimeError:
                        return
                    raise AssertionError("DNS was not blocked")

                def analyze(self, path):
                    self.assert_offline()
                    return filtering.summarize_turns([(0, 1, "speaker")], 1)
            args = filtering.parse_args(["--scp", str(scp)])
            self.assertFalse(args.offline)
            self.assertEqual(filtering.run_filter(args, Backend), 0)

    def test_network_violation_during_inference_is_fatal_not_audio_error(self):
        class Backend:
            def analyze(self, path):
                socket.getaddrinfo("huggingface.co", 443)
        with local.network_disabled(), self.assertRaisesRegex(RuntimeError, "Offline inference blocked"):
            filtering.analyze_audio_record(Backend(), Path("audio.flac"), {})


if __name__ == "__main__":
    unittest.main()
