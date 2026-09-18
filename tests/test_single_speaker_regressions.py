"""Regressions from user-reported short-clip false positives and symlink paths."""
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import filter_single_speaker as filtering


def lock_contender(path, active, events):
    for _ in range(2):
        with filtering.transient_file_lock(Path(path)):
            with active.get_lock():
                overlap = active.value != 0
                active.value += 1
            events.put(("entered", overlap))
            time.sleep(0.05)
            with active.get_lock():
                active.value -= 1


class FileCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.input = self.root / "audio.scp"
        self.output = self.root / "audio_single_speaker.scp"
        self.state = self.root / "audio_single_speaker.state.jsonl"
        self.review = self.root / "audio_single_speaker_review.scp"
        self.audio = self.root / "real.flac"
        self.audio.write_bytes(b"audio fixture")
        self.input.write_text(str(self.audio) + "\n", encoding="utf-8")
        self.result = filtering.summarize_turns([(0, 1, "a")], 1)
        self.calls = []
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def factory(self, *args):
        owner = self
        class Backend:
            def analyze(self, path):
                owner.calls.append(str(path))
                if isinstance(owner.result, BaseException):
                    raise owner.result
                return owner.result
        return Backend()

    def run_filter(self, *extra, factory=None):
        args = filtering.parse_args(["--scp", str(self.input), *extra])
        return filtering.run_filter(args, factory or self.factory)

    def locks(self):
        return [Path(str(self.output) + ".lock"), Path(str(self.state) + ".lock")]


class LockTests(FileCase):
    def test_normal_finish_removes_both_sidecars_including_old_empty_locks(self):
        for path in self.locks():
            path.touch()
        self.run_filter()
        self.assertTrue(all(not path.exists() for path in self.locks()))
        self.assertTrue(self.state.exists())

    def test_interrupt_removes_locks_but_keeps_resume_state(self):
        self.result = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_filter()
        self.assertTrue(all(not path.exists() for path in self.locks()))
        self.assertTrue(self.state.exists())

    def test_bad_state_also_releases_and_removes_locks(self):
        self.state.write_text("bad state\n")
        with self.assertRaises(ValueError):
            self.run_filter()
        self.assertTrue(all(not path.exists() for path in self.locks()))

    def test_waiting_processes_never_split_lock_ownership(self):
        context = filtering.mp.get_context("spawn")
        active = context.Value("i", 0)
        events = context.Queue()
        path = self.root / "concurrent.lock"
        processes = [context.Process(target=lock_contender, args=(str(path), active, events))
                     for _ in range(3)]
        try:
            for process in processes:
                process.start()
            received = [events.get(timeout=15) for _ in range(6)]
            for process in processes:
                process.join(timeout=5)
                self.assertEqual(process.exitcode, 0)
            self.assertTrue(all(event == ("entered", False) for event in received))
            self.assertFalse(path.exists())
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2)
            events.close()


class PathTests(FileCase):
    def test_record_reader_does_not_resolve_original_spelling(self):
        alias = self.root / "usr" / "local" / "audio.flac"
        self.input.write_text(str(alias) + "\n")
        with patch.object(Path, "resolve", side_effect=AssertionError("Unexpected symlink resolution")):
            record = next(filtering.iter_records(self.input, "scp", self.root))
        self.assertEqual(record[2], alias)

    def test_manually_updated_state_path_is_reused_without_rewriting_state(self):
        self.run_filter()
        logical = self.root / "usr-local.flac"
        os.link(self.audio, logical)
        self.input.write_text(str(logical) + "\n")
        # Simulate the user's completed path replacement, preserving audio signatures.
        records = [json.loads(line) for line in self.state.read_text().splitlines()]
        records[1]["audio"] = str(logical)
        self.state.write_text("".join(json.dumps(record) + "\n" for record in records))
        before = self.state.read_bytes()
        self.calls.clear()
        self.run_filter("--rebuild-only", factory=lambda *args: self.fail("Unexpected inference"))
        self.assertFalse(self.calls)
        self.assertEqual(self.output.read_text(), str(logical) + "\n")
        self.assertEqual(self.state.read_bytes(), before)

    def test_different_path_is_not_implicitly_mapped_by_inode(self):
        self.run_filter()
        logical = self.root / "another-name.flac"
        os.link(self.audio, logical)
        self.input.write_text(str(logical) + "\n")
        before = self.state.read_bytes()
        with self.assertRaisesRegex(ValueError, "No valid cached result"):
            self.run_filter("--rebuild-only", factory=lambda *args: self.fail("Unexpected inference"))
        self.assertEqual(self.state.read_bytes(), before)

    def test_actual_directory_symlink_preserves_paths_in_state_and_outputs(self):
        target = self.root / "physical"
        target.mkdir()
        alias = self.root / "logical"
        try:
            alias.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"Cannot create directory symlink: {exc}")
        audio = target / "sample.flac"
        audio.write_bytes(b"sample")
        self.input = alias / "input.scp"
        self.input.write_text(str(alias / "sample.flac") + "\n")
        self.run_filter()
        state = alias / "input_single_speaker.state.jsonl"
        record = json.loads(state.read_text().splitlines()[1])
        self.assertEqual(record["audio"], str(alias / "sample.flac"))
        self.assertEqual((alias / "input_single_speaker.scp").read_text(), str(alias / "sample.flac") + "\n")

    def test_rebuild_only_refuses_unprocessed_audio_without_touching_previous_output(self):
        self.output.write_text("previous\n")
        with self.assertRaisesRegex(ValueError, "No valid cached result"):
            self.run_filter("--rebuild-only", factory=lambda *args: self.fail("Unexpected inference"))
        self.assertEqual(self.output.read_text(), "previous\n")
        self.assertTrue(all(not path.exists() for path in self.locks()))


class TriageTests(FileCase):
    def decision(self, metrics):
        return filtering.filter_decision(metrics, filtering.parse_args(["--scp", "unused.scp"]))

    def test_six_reported_metrics_are_review_not_confirmed_single_or_multiple(self):
        samples = [
            (2.54, 2.244375, 2.244375, 0.556875),
            (1.6790022675736962, 1.535625, 1.535625, 0.91125),
            (3.25, 3.21903125, 3.0375, 0.18153125),
            (1.31, 1.27903125, 1.27903125, 0.2025),
            (1.0, 0.691875, 0.691875, 0.16875),
            (2.0760090702947847, 2.041875, 2.041875, 0.995625),
        ]
        for duration, speech, a, b in samples:
            metrics = {"duration": duration, "speech_seconds": speech,
                       "speaker_seconds": {"a": a, "b": b}}
            with self.subTest(duration=duration):
                self.assertEqual(self.decision(metrics)[0], "review")

    def test_clear_turn_taking_remains_multiple(self):
        result = filtering.summarize_turns([(0, 3, "a"), (3, 6, "b")], 6)
        self.assertEqual(self.decision(result), ("multiple", 2, "multiple_speaker_evidence"))

    def test_overlap_only_secondary_remains_review_for_new_and_legacy_metrics(self):
        result = filtering.summarize_turns([(0, 8, "a"), (2, 4, "b")], 8)
        self.assertEqual(self.decision(result)[2], "insufficient_nonoverlap_speech")
        result.pop("speaker_exclusive_seconds")
        result.pop("overlap_seconds")
        self.assertEqual(self.decision(result)[2], "insufficient_nonoverlap_speech")

    def test_overlap_and_exclusive_metrics_do_not_double_count_same_speaker_tracks(self):
        result = filtering.summarize_turns([(0, 4, "a"), (1, 3, "a"), (2, 3, "b")], 4)
        self.assertEqual(result["speech_seconds"], 4)
        self.assertEqual(result["speaker_exclusive_seconds"], {"a": 3, "b": 0})
        self.assertEqual(result["overlap_seconds"], 1)

    def test_uncertain_scp_record_goes_to_separate_review_and_can_be_rebuilt(self):
        self.result = filtering.summarize_turns([(0, 1, "a"), (0.2, 0.8, "b")], 1)
        self.run_filter()
        self.assertEqual(self.output.read_text(), "")
        self.assertEqual(self.review.read_text(), str(self.audio) + "\n")
        self.calls.clear()
        self.run_filter("--rebuild-only", "--uncertain-action", "keep",
                        factory=lambda *args: self.fail("Unexpected inference"))
        self.assertEqual(self.output.read_text(), str(self.audio) + "\n")
        self.assertEqual(self.review.read_text(), str(self.audio) + "\n")
        self.assertFalse(self.calls)

    def test_review_jsonl_keeps_original_pair_and_extra_fields(self):
        self.result = filtering.summarize_turns([(0, 1, "a"), (0.2, 0.8, "b")], 1)
        input_path = self.root / "pairs.jsonl"
        raw = json.dumps([str(self.audio), "字幕.txt", {"source": "source"}], ensure_ascii=False)
        input_path.write_text(raw + "\n", encoding="utf-8")
        args = filtering.parse_args(["--jsonl", str(input_path)])
        filtering.run_filter(args, self.factory)
        self.assertEqual((self.root / "pairs_single_speaker.jsonl").read_text(), "")
        self.assertEqual((self.root / "pairs_single_speaker_review.jsonl").read_text(encoding="utf-8"),
                         raw + "\n")

    def test_jsonl_categories_preserve_raw_records_and_rebuild_from_cache(self):
        cases = {
            "review_short_speech": filtering.summarize_turns([(0, 1, "a"), (1, 2, "b")], 2),
            "review_brief_or_low_share_secondary": filtering.summarize_turns([(0, 4, "a"), (4, 4.1, "b")], 4.1),
            "review_insufficient_nonoverlap_speech": filtering.summarize_turns([(0, 5, "a"), (1, 2, "b")], 5),
            "review_legacy_overlap_ambiguous": {
                "duration": 6, "speech_seconds": 5, "speaker_seconds": {"a": 3, "b": 2, "c": 1}},
            "multiple": filtering.summarize_turns([(0, 3, "a"), (3, 6, "b")], 6),
            "single": filtering.summarize_turns([(0, 2, "a")], 2),
        }
        expected = {}
        results = {}
        rows = []
        for index, (category, result) in enumerate(cases.items()):
            audio = self.root / f"case_{index}.flac"
            audio.write_bytes(b"fixture")
            results[str(audio)] = result
            raw = json.dumps([str(audio), f"字幕_{index}.txt", {"order": index}], ensure_ascii=False)
            rows.append(raw)
            expected[category] = raw + "\n"
        # Duplicate records keep their input order without a second inference.
        rows.append(rows[0])
        expected["review_short_speech"] += rows[0] + "\n"
        input_path = self.root / "pairs.jsonl"
        input_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        calls = []
        class Backend:
            def analyze(self, audio):
                calls.append(str(audio))
                return results[str(audio)]
        args = filtering.parse_args(["--jsonl", str(input_path)])
        self.assertEqual(filtering.run_filter(args, lambda *args: Backend()), 0)
        self.assertEqual(len(calls), len(cases))
        for category in filtering.DETAIL_OUTPUT_CATEGORIES:
            path = self.root / f"pairs_single_speaker_{category}.jsonl"
            self.assertEqual(path.read_text(encoding="utf-8"), expected[category])
        self.assertEqual((self.root / "pairs_single_speaker.jsonl").read_text(encoding="utf-8"),
                         expected["single"])
        self.assertEqual((self.root / "pairs_single_speaker_review.jsonl").read_text(encoding="utf-8"),
                         "\n".join(rows[:4] + [rows[0]]) + "\n")
        state = self.root / "pairs_single_speaker.state.jsonl"
        before = state.read_bytes()
        args.rebuild_only = True
        self.assertEqual(filtering.run_filter(args, lambda *args: self.fail("Unexpected inference")), 0)
        self.assertEqual(state.read_bytes(), before)
        for category in filtering.DETAIL_OUTPUT_CATEGORIES:
            self.assertEqual((self.root / f"pairs_single_speaker_{category}.jsonl").read_text(encoding="utf-8"),
                             expected[category])

    def test_scp_categories_replace_stale_rows_when_policy_changes(self):
        self.result = filtering.summarize_turns([(0, 1, "a"), (1, 2, "b")], 2)
        self.run_filter()
        short_path = self.root / "audio_single_speaker_review_short_speech.scp"
        multiple_path = self.root / "audio_single_speaker_multiple.scp"
        self.assertEqual(short_path.read_text(), str(self.audio) + "\n")
        self.assertEqual(multiple_path.read_text(), "")
        self.run_filter("--rebuild-only", "--decision-policy", "strict",
                        factory=lambda *args: self.fail("Unexpected inference"))
        self.assertEqual(short_path.read_text(), "")
        self.assertEqual(multiple_path.read_text(), str(self.audio) + "\n")
        self.assertEqual(self.review.read_text(), "")

    def test_failed_rebuild_does_not_replace_previous_category_files(self):
        self.result = filtering.summarize_turns([(0, 1, "a"), (1, 2, "b")], 2)
        self.run_filter()
        snapshots = {category: (self.root / f"audio_single_speaker_{category}.scp").read_bytes()
                     for category in filtering.DETAIL_OUTPUT_CATEGORIES}
        unprocessed = self.root / "new.flac"
        unprocessed.write_bytes(b"new audio")
        self.input.write_text(str(self.audio) + "\n" + str(unprocessed) + "\n")
        with self.assertRaisesRegex(ValueError, "No valid cached result"):
            self.run_filter("--rebuild-only", factory=lambda *args: self.fail("Unexpected inference"))
        for category, content in snapshots.items():
            self.assertEqual((self.root / f"audio_single_speaker_{category}.scp").read_bytes(), content)
        self.assertTrue(all(not path.exists() for path in self.locks()))

    def test_state_cannot_alias_a_category_output(self):
        category = self.root / "audio_single_speaker_multiple.scp"
        with self.assertRaisesRegex(ValueError, "different files"):
            self.run_filter("--state", str(category))


if __name__ == "__main__":
    unittest.main()
