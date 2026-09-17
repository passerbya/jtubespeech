import contextlib
import io
import json
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import audio_language
import cleanup_audio_language as cleanup
from videoid_state import VideoIdLog, load_video_ids, save_video_id_queue, video_id_path

VID = "abC_def-123"
KEEP = "abC_def-124"
UNKNOWN = "abC_def-125"
ERROR = "abC_def-126"


def info(*languages):
    return {"formats": [
        {"format_id": str(i), "language": lang, "vcodec": "none", "acodec": "opus"}
        for i, lang in enumerate(languages)
    ]}


class LanguageTests(unittest.TestCase):
    def test_conservative_classification(self):
        cases = [
            (info("en"), "ja", "mismatch"),
            (info("en", "ja"), "ja", "match"),
            (info("en-US"), "en", "match"),
            (info("jw"), "jv", "match"),
            (info("en", None), "ja", "unknown"),
            (info("und"), "ja", "unknown"),
            (info("mul"), "ja", "unknown"),
            (info(), "ja", "unknown"),
        ]
        for metadata, target, expected in cases:
            with self.subTest(metadata=metadata, target=target):
                self.assertEqual(audio_language.audio_language_status(metadata, target), expected)

    def test_translated_subtitles_do_not_establish_audio_language(self):
        metadata = info("en")
        metadata["subtitles"] = {"ja": [{}]}
        metadata["automatic_captions"] = {"ja": [{}]}
        self.assertEqual(audio_language.audio_language_status(metadata, "ja"), "mismatch")

    def test_incomplete_extraction_is_not_deletion_evidence(self):
        metadata = info("en")
        metadata["audio_formats_incomplete"] = True
        self.assertEqual(audio_language.audio_language_status(metadata, "ja"), "unknown")
        self.assertEqual(audio_language.audio_language_status(metadata, "en"), "match")

    def test_network_and_bot_errors_are_retryable(self):
        for message in (
            "ERROR: [youtube] abc: HTTP Error 429",
            "ERROR: [youtube] abc: Sign in to confirm you are not a bot",
            "ERROR: [youtube] abc: The uploader has not made this video available in your country",
        ):
            self.assertFalse(audio_language.is_unavailable_error(message))
        self.assertTrue(audio_language.is_unavailable_error("ERROR: [youtube] abc: Private video"))

    def test_unknown_reason_details(self):
        self.assertEqual(audio_language.audio_language_unknown_reasons(info()),
                         ["no_audio_only_formats"])
        self.assertEqual(audio_language.audio_language_unknown_reasons(info("en", None)),
                         ["missing_or_ambiguous_language"])
        metadata = info("en")
        metadata["audio_formats_incomplete"] = True
        self.assertEqual(audio_language.audio_language_unknown_reasons(metadata),
                         ["incomplete_format_list"])

    def test_unavailable_messages_and_transient_variants(self):
        for message in (
            "ERROR: [youtube] --hcFW5x3oo: Video unavailable",
            "ERROR: [youtube] --Mj7c-jFYI: This video is unavailable",
            "ERROR: [youtube] abc: Private video. Sign in if you have been granted access",
            "ERROR: [youtube] abc: Join this channel to get access",
        ):
            self.assertTrue(audio_language.is_unavailable_error(message))
        for message in (
            "ERROR: [youtube] abc: Video unavailable. Sign in to confirm you are not a bot",
            "ERROR: [youtube] abc: Video unavailable. HTTP Error 429",
            "ERROR: [youtube] abc: Video unavailable. This video is not available in your country",
        ):
            self.assertFalse(audio_language.is_unavailable_error(message))

    def test_exact_filename_boundaries(self):
        self.assertEqual(cleanup.artifact_videoid("wav_org", VID + ".flac"), VID)
        self.assertEqual(cleanup.artifact_videoid("vtt", VID + ".en-US.vtt"), VID)
        self.assertEqual(cleanup.artifact_videoid("segs", VID + "_0000.whisper.txt"), VID)
        self.assertEqual(cleanup.artifact_videoid("segs", VID + "_10000.flac"), VID)
        for category, name in (
            ("wav_org", VID + "X.flac"), ("segs", VID + "_bad.flac"),
            ("segs", VID + "_0000_other.flac"), ("txt", "dns_mos.scp"),
        ):
            self.assertIsNone(cleanup.artifact_videoid(category, name))


class StateTests(unittest.TestCase):
    def test_append_preserves_existing_ids_and_missing_newline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.txt"
            path.write_text(VID, encoding="utf-8")
            with VideoIdLog(path) as log:
                self.assertFalse(log.append(VID))
                self.assertTrue(log.append(KEEP))
                self.assertFalse(log.append(VID))
            self.assertEqual(path.read_text().splitlines(), [VID, KEEP])

    def test_interleaved_writers_ingest_new_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.txt"
            with VideoIdLog(path) as first, VideoIdLog(path) as second:
                first.append(VID)
                second.append(KEEP)
                self.assertFalse(first.append(KEEP))
                first.append(UNKNOWN)
                self.assertFalse(second.append(UNKNOWN))
            self.assertEqual(path.read_text().splitlines(), [VID, KEEP, UNKNOWN])

    def test_queue_writer_keeps_ids_not_replayed_on_this_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.txt"
            path.write_text(VID + "\n")
            tasks = queue.Queue()
            for item in (KEEP, KEEP, "STOP"):
                tasks.put(item)
            save_video_id_queue(path, tasks)
            self.assertEqual(path.read_text().splitlines(), [VID, KEEP])


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "ocr" / "jtubespeech"
        self.other = self.base / "corpus" / "jtubespeech"
        (self.root / "video" / "ja").mkdir(parents=True)
        (self.other / "video" / "ja").mkdir(parents=True)
        self.empty = video_id_path(self.root / "videoid", "empty", "ja")
        self.args = cleanup.parse_args(["--root", str(self.root), "--lang", "ja", "--workers", "1"])
        self.output = io.StringIO()
        quiet = contextlib.redirect_stdout(self.output)
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def make_file(self, root=None, lang="ja", category="wav_org", name=None, bucket="ab"):
        path = (root or self.root) / "video" / lang / category
        if bucket:
            path = path / bucket
        path = path / (name or VID + ".flac")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test artifact")
        return path

    def add_empty(self, vid=VID):
        with VideoIdLog(self.empty) as output:
            output.append(vid)

    def run_cleanup(self):
        return cleanup.clean_language(self.root, self.args, self.empty)

    def test_immediate_delete_only_selected_root_and_language(self):
        doomed = [
            self.make_file(category=category, name=filename)
            for category, filename in (
                ("wav_org", VID + ".flac"), ("wav", VID + ".wav"),
                ("flac", VID + ".flac"), ("txt", VID + ".txt"),
                ("vtt", VID + ".vtt"), ("segs", VID + "_0000.flac"),
                ("segs", VID + "_0000.txt"), ("segs", VID + "_0000.vtt"),
                ("segs", VID + "_0000.lang.txt"), ("segs", VID + "_0000.whisper.txt"),
                ("segs", VID + "_0000.qwen.txt"),
            )
        ]
        kept = [
            self.make_file(root=self.other),
            self.make_file(lang="en"),
            self.make_file(name=VID + "X.flac"),
            self.make_file(name=KEEP + ".flac"),
            self.make_file(name=UNKNOWN + ".flac"),
            self.make_file(name=ERROR + ".flac"),
        ]

        def metadata(vid, args, proxy):
            if vid == ERROR:
                return vid, "error", {}, "429"
            return vid, "ok", info(*( {VID: ("en",), KEEP: ("ja",), UNKNOWN: (None,)}[vid] )), None

        original_unlink = Path.unlink

        def check_blacklist(path, *args, **kwargs):
            self.assertIn(VID, load_video_ids(self.empty))
            return original_unlink(path, *args, **kwargs)

        with patch.object(cleanup, "fetch_metadata", side_effect=metadata) as fetch, \
                patch.object(Path, "unlink", check_blacklist):
            self.assertEqual(self.run_cleanup(), 1)  # the 429 remains retryable
        self.assertEqual(fetch.call_count, 4)
        self.assertTrue(all(not path.exists() for path in doomed))
        self.assertTrue(all(path.exists() for path in kept))
        self.assertEqual(load_video_ids(self.empty), {VID})
        self.assertFalse((self.other / "videoid").exists())

    def test_existing_empty_deleted_before_query_without_rechecking(self):
        doomed = [self.make_file(), self.make_file(category="segs", name=VID + "_0000.txt")]
        self.make_file(name=KEEP + ".flac")
        self.add_empty()

        def metadata(vid, args, proxy):
            self.assertEqual(vid, KEEP)
            self.assertTrue(all(not path.exists() for path in doomed))
            return vid, "ok", info("ja"), None

        with patch.object(cleanup, "fetch_metadata", side_effect=metadata) as fetch:
            self.assertEqual(self.run_cleanup(), 0)
        self.assertEqual(fetch.call_count, 1)

    def test_delete_happens_before_scanning_next_bucket(self):
        first = self.make_file()
        second = self.make_file(name=KEEP + ".flac", bucket="other")

        def batches(root, lang):
            yield {VID: [first]}
            self.assertFalse(first.exists())
            self.assertIn(VID, load_video_ids(self.empty))
            yield {KEEP: [second]}

        def metadata(vid, args, proxy):
            return vid, "ok", info("en" if vid == VID else "ja"), None

        with patch.object(cleanup, "iter_video_batches", side_effect=batches), \
                patch.object(cleanup, "fetch_metadata", side_effect=metadata):
            self.assertEqual(self.run_cleanup(), 0)
        self.assertTrue(second.exists())

    def test_fast_mismatch_deleted_while_another_query_is_still_running(self):
        slow_path = self.make_file(name=KEEP + ".flac")
        doomed = self.make_file()
        deleted = threading.Event()
        self.args.workers = 2

        def metadata(vid, args, proxy):
            if vid == KEEP:
                self.assertTrue(deleted.wait(5), "Deletion waited for the slower query")
                return vid, "ok", info("ja"), None
            return vid, "ok", info("en"), None

        original_unlink = Path.unlink

        def unlink(path, *args, **kwargs):
            result = original_unlink(path, *args, **kwargs)
            if path == doomed:
                deleted.set()
            return result

        with patch.object(cleanup, "iter_video_batches", return_value=iter([{KEEP: [slow_path], VID: [doomed]}])), \
                patch.object(cleanup, "fetch_metadata", side_effect=metadata), \
                patch.object(Path, "unlink", unlink):
            self.assertEqual(self.run_cleanup(), 0)
        self.assertTrue(slow_path.exists())
        self.assertFalse(doomed.exists())

    def test_partial_deletion_failure_resumes_from_empty_without_query(self):
        paths = [self.make_file(category=category, name=VID + suffix)
                 for category, suffix in (("wav_org", ".flac"), ("txt", ".txt"), ("vtt", ".vtt"))]
        original_unlink = Path.unlink

        def fail_one(path, *args, **kwargs):
            if path == paths[1]:
                raise PermissionError("simulated permission failure")
            return original_unlink(path, *args, **kwargs)

        with patch.object(cleanup, "fetch_metadata", return_value=(VID, "ok", info("en"), None)), \
                patch.object(Path, "unlink", fail_one):
            self.assertEqual(self.run_cleanup(), 1)
        self.assertFalse(paths[0].exists())
        self.assertTrue(paths[1].exists())
        self.assertFalse(paths[2].exists())
        self.assertEqual(load_video_ids(self.empty), {VID})
        with patch.object(cleanup, "fetch_metadata", side_effect=AssertionError("Must not re-query empty")):
            self.assertEqual(self.run_cleanup(), 0)
        self.assertFalse(any(path.exists() for path in paths))
        self.assertEqual(self.empty.read_text().splitlines(), [VID])

    def test_interrupt_after_first_unlink_resumes(self):
        paths = [self.make_file(), self.make_file(category="txt", name=VID + ".txt")]
        original_unlink = Path.unlink
        count = 0

        def interrupt(path, *args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise KeyboardInterrupt
            return original_unlink(path, *args, **kwargs)

        with patch.object(cleanup, "fetch_metadata", return_value=(VID, "ok", info("en"), None)), \
                patch.object(Path, "unlink", interrupt), self.assertRaises(KeyboardInterrupt):
            self.run_cleanup()
        self.assertIn(VID, load_video_ids(self.empty))
        with patch.object(cleanup, "fetch_metadata", side_effect=AssertionError("Must not re-query empty")):
            self.assertEqual(self.run_cleanup(), 0)
        self.assertFalse(any(path.exists() for path in paths))

    def test_empty_write_failure_never_deletes_files(self):
        path = self.make_file()
        with patch.object(cleanup, "fetch_metadata", return_value=(VID, "ok", info("en"), None)), \
                patch.object(VideoIdLog, "append", side_effect=OSError("disk full")), \
                self.assertRaises(OSError):
            self.run_cleanup()
        self.assertTrue(path.exists())

    def test_orphan_segments_in_empty_are_cleaned(self):
        paths = [self.make_file(category="segs", name=VID + "_0000." + extension)
                 for extension in ("flac", "txt", "vtt")]
        self.add_empty()
        with patch.object(cleanup, "fetch_metadata", side_effect=AssertionError("Must not re-query empty")):
            self.assertEqual(self.run_cleanup(), 0)
            self.assertEqual(self.run_cleanup(), 0)
        self.assertFalse(any(path.exists() for path in paths))

    def test_outside_path_refused(self):
        outside = self.base / (VID + ".flac")
        outside.write_bytes(b"keep outside")
        self.assertEqual(cleanup.delete_video_files(self.root, "ja", VID, [outside]), (0, 0, 1))
        self.assertTrue(outside.exists())

    def test_symlink_swap_during_query_is_refused(self):
        path = self.make_file()
        outside = self.base / "outside.flac"
        outside.write_bytes(b"keep outside")
        probe = self.base / "probe"
        try:
            probe.symlink_to(outside)
        except OSError as exc:
            self.skipTest(f"Symlink creation unavailable: {exc}")
        probe.unlink()

        def metadata(vid, args, proxy):
            path.unlink()
            path.symlink_to(outside)
            return vid, "ok", info("en"), None

        with patch.object(cleanup, "fetch_metadata", side_effect=metadata):
            self.assertEqual(self.run_cleanup(), 1)
        self.assertEqual(outside.read_bytes(), b"keep outside")
        self.assertTrue(path.is_symlink())

    def test_kept_video_is_queried_once_per_run_and_again_next_run(self):
        self.make_file()
        self.make_file(category="segs", name=VID + "_0000.txt", bucket="elsewhere")
        with patch.object(cleanup, "fetch_metadata", return_value=(VID, "ok", info("ja"), None)) as fetch:
            self.assertEqual(self.run_cleanup(), 0)
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(self.run_cleanup(), 0)
            self.assertEqual(fetch.call_count, 2)

    def test_new_empty_cleans_later_buckets_without_second_query(self):
        paths = [self.make_file(), self.make_file(category="segs", name=VID + "_0000.txt", bucket="later")]
        with patch.object(cleanup, "fetch_metadata", return_value=(VID, "ok", info("en"), None)) as fetch:
            self.assertEqual(self.run_cleanup(), 0)
        self.assertEqual(fetch.call_count, 1)
        self.assertFalse(any(path.exists() for path in paths))

    def test_query_limit_still_cleans_empty_in_later_bucket(self):
        self.make_file(name=KEEP + ".flac")
        self.make_file(name=UNKNOWN + ".flac")
        leftover = self.make_file(bucket="zz")
        self.add_empty()
        self.args.limit = 1
        with patch.object(cleanup, "fetch_metadata", return_value=(KEEP, "ok", info("ja"), None)) as fetch:
            self.assertEqual(self.run_cleanup(), 0)
        self.assertEqual(fetch.call_count, 1)
        self.assertFalse(leftover.exists())

    def test_main_uses_only_empty_file_for_resume(self):
        path = self.make_file()
        with patch.object(cleanup, "fetch_metadata", return_value=(VID, "ok", info("en"), None)):
            self.assertEqual(cleanup.main(["--root", str(self.root), "--lang", "ja"]), 0)
        self.assertFalse(path.exists())
        self.assertEqual(load_video_ids(self.empty), {VID})
        self.assertFalse(list(self.base.rglob("*.sqlite")))
        self.assertFalse(list(self.base.rglob("*.jsonl")))

    def test_unavailable_persists_error_and_is_skipped_next_run(self):
        path = self.make_file()
        error_fn = video_id_path(self.root / "videoid", "error", "ja")
        message = f"ERROR: [youtube] {VID}: This video is unavailable"
        with patch.object(cleanup, "fetch_metadata", return_value=(VID, "unavailable", {}, message)) as fetch:
            self.assertEqual(self.run_cleanup(), 0)
            self.assertEqual(self.run_cleanup(), 0)
        self.assertEqual(fetch.call_count, 1)
        self.assertTrue(path.exists())
        self.assertEqual(load_video_ids(error_fn), {VID})
        self.assertNotIn(VID, load_video_ids(self.empty))
        self.assertIn("saved_error=1", self.output.getvalue())
        self.assertIn("skipped_error=1", self.output.getvalue())

    def test_unknown_persists_with_reason_and_is_skipped_next_run(self):
        path = self.make_file()
        unknown_fn = video_id_path(self.root / "videoid", "unknown", "ja")
        with patch.object(cleanup, "fetch_metadata", return_value=(VID, "ok", info(None), None)) as fetch:
            self.assertEqual(self.run_cleanup(), 0)
            self.assertEqual(self.run_cleanup(), 0)
        self.assertEqual(fetch.call_count, 1)
        self.assertTrue(path.exists())
        self.assertEqual(load_video_ids(unknown_fn), {VID})
        self.assertNotIn(VID, load_video_ids(self.empty))
        self.assertIn("reason=missing_or_ambiguous_language", self.output.getvalue())
        self.assertIn("audio=['<missing>']", self.output.getvalue())

    def test_existing_error_unknown_keep_files_while_empty_still_deletes(self):
        doomed = self.make_file()
        error_path = self.make_file(name=ERROR + ".flac")
        unknown_path = self.make_file(name=UNKNOWN + ".flac")
        for category, vids in (("empty", [VID]), ("error", [VID, ERROR]), ("unknown", [VID, UNKNOWN])):
            with VideoIdLog(video_id_path(self.root / "videoid", category, "ja")) as output:
                for vid in vids:
                    output.append(vid)
        with patch.object(cleanup, "fetch_metadata", side_effect=AssertionError("Unexpected query")):
            self.assertEqual(self.run_cleanup(), 0)
        self.assertFalse(doomed.exists())
        self.assertTrue(error_path.exists())
        self.assertTrue(unknown_path.exists())

    def test_transient_query_errors_are_not_persisted_and_are_retried(self):
        path = self.make_file()
        result = (VID, "error", {}, f"ERROR: [youtube] {VID}: HTTP Error 429")
        with patch.object(cleanup, "fetch_metadata", return_value=result) as fetch:
            self.assertEqual(self.run_cleanup(), 1)
            self.assertEqual(self.run_cleanup(), 1)
        self.assertEqual(fetch.call_count, 2)
        self.assertTrue(path.exists())
        for category in ("empty", "error", "unknown"):
            self.assertNotIn(VID, load_video_ids(video_id_path(self.root / "videoid", category, "ja")))

    def test_unknown_queue_write_failure_keeps_files(self):
        path = self.make_file()
        with patch.object(cleanup, "fetch_metadata", return_value=(VID, "ok", info(None), None)), \
                patch.object(VideoIdLog, "append", side_effect=OSError("disk full")), \
                self.assertRaises(OSError):
            self.run_cleanup()
        self.assertTrue(path.exists())

    def test_deletion_counts_include_segments_and_category_breakdown(self):
        vid = "--28iFcpozs"
        paths = [self.make_file(category=category, name=vid + suffix, bucket="--")
                 for category, suffix in (("wav_org", ".flac"), ("txt", ".txt"),
                                          ("vtt", ".vtt"), ("flac", ".flac"))]
        for index in range(10):
            for extension in ("flac", "txt", "vtt"):
                paths.append(self.make_file(category="segs",
                                            name=f"{vid}_{index:04d}.{extension}", bucket="--"))
        for extension in ("lang.txt", "whisper.txt"):
            paths.append(self.make_file(category="segs", name=f"{vid}_0000.{extension}", bucket="--"))
        expected_bytes = sum(path.stat().st_size for path in paths)
        self.args.verbose = True
        with patch.object(cleanup, "fetch_metadata", return_value=(vid, "ok", info("en"), None)):
            self.assertEqual(self.run_cleanup(), 0)
        text = self.output.getvalue()
        self.assertIn("matched=36 deleted=36 vid_deleted_total=36", text)
        self.assertIn("segs:32", text)
        self.assertIn(f"bytes={expected_bytes}", text)
        self.assertEqual(text.count("[UNLINK]"), 36)
        self.assertTrue(all(not path.exists() for path in paths))

    def test_cumulative_count_across_buckets_is_explicit(self):
        paths = [self.make_file(category=category, name=VID + suffix)
                 for category, suffix in (("wav_org", ".flac"), ("txt", ".txt"), ("vtt", ".vtt"))]
        paths.extend(self.make_file(category="segs", name=VID + "_0000." + extension, bucket="later")
                     for extension in ("flac", "txt"))
        with patch.object(cleanup, "fetch_metadata", return_value=(VID, "ok", info("en"), None)):
            self.assertEqual(self.run_cleanup(), 0)
        text = self.output.getvalue()
        self.assertIn("matched=3 deleted=3 vid_deleted_total=3", text)
        self.assertIn("matched=2 deleted=2 vid_deleted_total=5", text)
        self.assertIn("segs:2", text)
        self.assertTrue(all(not path.exists() for path in paths))

    def test_fetch_errors_and_incomplete_formats(self):
        args = SimpleNamespace(yt_dlp="yt-dlp", socket_timeout=10, timeout=20,
                               extractor_args="youtube:player_client=mweb", cookies=None)
        result = subprocess.CompletedProcess([], 0, json.dumps(dict(info("en"), id=VID)),
                                             "WARNING: Some formats have been skipped")
        with patch.object(cleanup.subprocess, "run", return_value=result):
            _vid, status, metadata, _error = cleanup.fetch_metadata(VID, args, None)
        self.assertEqual(status, "ok")
        self.assertEqual(audio_language.audio_language_status(metadata, "ja"), "unknown")
        for result, expected in (
            (subprocess.CompletedProcess([], 1, "", f"ERROR: [youtube] {VID}: Video unavailable"), "unavailable"),
            (subprocess.CompletedProcess([], 1, f"ERROR: [youtube] {VID}: This video is unavailable", ""), "unavailable"),
            (subprocess.CompletedProcess([], 1, "", f"ERROR: [youtube] {VID}: HTTP Error 429"), "error"),
            (subprocess.CompletedProcess([], 0, "invalid json", ""), "error"),
            (subprocess.CompletedProcess([], 0, json.dumps(dict(info("en"), id=KEEP)), ""), "error"),
        ):
            with patch.object(cleanup.subprocess, "run", return_value=result):
                self.assertEqual(cleanup.fetch_metadata(VID, args, None)[1], expected)


class CliTests(unittest.TestCase):
    def test_language_and_one_root_are_required(self):
        for arguments in (
            [], ["--root", "ocr"], ["--lang", "ja"],
            ["--root", "ocr", "--root", "corpus", "--lang", "ja"],
            ["--root", "ocr", "--lang", "ja", "--lang", "en"],
            ["--root", "ocr", "--lang", "../ja"],
        ):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                cleanup.parse_args(arguments)
        args = cleanup.parse_args(["--root", "ocr", "--lang", "ja"])
        self.assertEqual(args.root, Path("ocr"))
        self.assertEqual(args.lang, "ja")

    def test_no_scan_apply_or_database_arguments(self):
        for option in ("scan", "apply", "--report", "--delete"):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                cleanup.parse_args(["--root", "ocr", "--lang", "ja", option])


if __name__ == "__main__":
    unittest.main()
