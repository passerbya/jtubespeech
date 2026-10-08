import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import cleanup_audio_language as cleanup
from videoid_state import video_id_path


VID = "-DB8_Y4rP3s"


class ErrorListGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jtubespeech-audio-language-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.videoid = self.root / "videoid"
        self.empty = video_id_path(self.videoid, "empty", "en")
        self.args = SimpleNamespace(
            lang="en",
            workers=1,
            proxy=[None],
            limit=0,
            verbose=False,
        )
        self.paths = [self.root / "video" / "en" / "flac" / "x" / f"{VID}.flac"]

    def run_result(self, status, error):
        def metadata(_vid, _args, _proxy):
            return VID, status, {}, error

        with patch.object(cleanup, "iter_video_batches", return_value=iter([{VID: self.paths}])),                 patch.object(cleanup, "fetch_metadata", side_effect=metadata),                 contextlib.redirect_stdout(io.StringIO()):
            return cleanup.clean_language(self.root, self.args, self.empty)

    def error_path(self):
        return video_id_path(self.videoid, "error", "en")

    def assert_error_list_empty(self):
        path = self.error_path()
        self.assertTrue(path.exists())
        self.assertEqual(path.read_text(encoding="utf-8"), "")

    def test_only_explicit_unavailable_signature_is_persisted(self):
        result = self.run_result(
            "unavailable",
            f"ERROR: [youtube] {VID}: Video unavailable",
        )
        self.assertEqual(result, 0)
        self.assertEqual(self.error_path().read_text(encoding="utf-8"), VID + "\n")

    def test_transient_http_error_is_retryable(self):
        result = self.run_result(
            "error",
            f"ERROR: [youtube] {VID}: HTTP Error 429: Too Many Requests",
        )
        self.assertEqual(result, 1)
        self.assert_error_list_empty()

    def test_proxy_and_timeout_errors_are_retryable(self):
        for message in (
            f"ERROR: [youtube] {VID}: Unable to download API page: "
            "Connection reset by peer",
            f"ERROR: [youtube] {VID}: timed out",
        ):
            with self.subTest(message=message):
                result = self.run_result("error", message)
                self.assertEqual(result, 1)
                self.assert_error_list_empty()

    def test_fetch_status_is_derived_from_is_unavailable_error(self):
        args = SimpleNamespace(
            yt_dlp="yt-dlp",
            socket_timeout=20,
            extractor_args="youtube:player_client=mweb",
            cookies=None,
            timeout=30,
        )
        for message, expected in (
            (f"ERROR: [youtube] {VID}: Video unavailable", "unavailable"),
            (f"ERROR: [youtube] {VID}: HTTP Error 429: Too Many Requests", "error"),
            (f"ERROR: [youtube] {VID}: Unable to download API page: Connection reset by peer", "error"),
        ):
            with self.subTest(message=message), patch.object(
                cleanup.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    ["yt-dlp"], 1, stdout="", stderr=message
                ),
            ):
                self.assertEqual(cleanup.fetch_metadata(VID, args, None)[1], expected)


if __name__ == "__main__":
    unittest.main()
