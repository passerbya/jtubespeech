#!/usr/bin/env python3
"""Keep single-speaker audio in an SCP or [audio, text] JSONL, with resumable results."""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import signal
import tempfile
import sys
import time
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from itertools import islice
from pathlib import Path
from queue import Empty

from videoid_state import locked
from pyannote_local import (
    local_pipeline_config, network_disabled, prepare_model_bundle, pyannote_checkpoint_globals,
    set_offline_environment,
)

STATE_VERSION = 1
DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"
DETAIL_OUTPUT_CATEGORIES = (
    "review_short_speech",
    "review_brief_or_low_share_secondary",
    "review_insufficient_nonoverlap_speech",
    "review_legacy_overlap_ambiguous",
    "multiple",
)


def configure_model_cache(cache_dir, offline=False):
    """Set cache locations before importing pyannote/torch/huggingface_hub."""
    root = None
    if cache_dir is not None:
        root = Path(cache_dir).expanduser().resolve()
        paths = {
            "HF_HOME": root / "huggingface",
            "HF_HUB_CACHE": root / "huggingface" / "hub",
            "HUGGINGFACE_HUB_CACHE": root / "huggingface" / "hub",
            "HF_XET_CACHE": root / "huggingface" / "xet",
            "PYANNOTE_CACHE": root / "pyannote",
            "TORCH_HOME": root / "torch",
            "XDG_CACHE_HOME": root / "xdg",
            "TORCHINDUCTOR_CACHE_DIR": root / "inductor",
            "TRITON_CACHE_DIR": root / "triton",
            "CUDA_CACHE_PATH": root / "cuda",
            "TMPDIR": root / "tmp",
        }
        for key, path in paths.items():
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)
        tempfile.tempdir = str(paths["TMPDIR"])
    if offline:
        set_offline_environment()
    return root


def parse_devices(value):
    devices = []
    for part in value.split(","):
        part = part.strip()
        if part.isdigit():
            part = "cuda:" + part
        if not part.startswith("cuda:") or not part[5:].isdigit():
            raise argparse.ArgumentTypeError("--devices must be comma-separated GPU indices, e.g. 0,1,2,3")
        part = "cuda:" + str(int(part[5:]))
        if part in devices:
            raise argparse.ArgumentTypeError("Each GPU may only be specified once")
        devices.append(part)
    return devices


def lexical_absolute(path):
    """Make a path absolute without dereferencing the user's symlink spelling."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def same_lock_file(path, identity):
    try:
        current = path.stat()
    except FileNotFoundError:
        return False
    return (current.st_dev, current.st_ino) == identity


def remove_owned_lock(path, identity):
    try:
        if same_lock_file(path, identity):
            path.unlink()
    except FileNotFoundError:
        pass
    except PermissionError:
        # On Windows another open waiter may prevent deletion. Its owner cleans up.
        if os.name != "nt":
            print(f"[LOCK CLEANUP] could not remove {path}", flush=True)


@contextmanager
def transient_file_lock(path):
    """Revalidate the locked inode so unlinking a finished lock cannot split ownership."""
    path = Path(path)
    while True:
        stream = path.open("a+b", buffering=0)
        acquired = False
        identity = None
        try:
            with locked(stream):
                info = os.fstat(stream.fileno())
                identity = (info.st_dev, info.st_ino)
                if not same_lock_file(path, identity):
                    # Another owner removed the file while this process was waiting.
                    continue
                acquired = True
                try:
                    yield
                finally:
                    if os.name != "nt":
                        remove_owned_lock(path, identity)
            return
        finally:
            stream.close()
            if acquired and os.name == "nt":
                remove_owned_lock(path, identity)


def iter_records(input_path, input_format, path_base):
    """Retain each original record, including JSON fields and paths with spaces."""
    with input_path.open("r", encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            raw = line.rstrip("\r\n")
            if not raw.strip():
                continue
            if input_format == "jsonl":
                try:
                    item = json.loads(raw)
                except ValueError as exc:
                    raise ValueError(f"{input_path}:{line_no}: invalid JSON: {exc}") from exc
                if (not isinstance(item, list) or len(item) < 2
                        or not isinstance(item[0], str) or not item[0].strip()
                        or not isinstance(item[1], str)):
                    raise ValueError(f"{input_path}:{line_no}: expected [audio_path, text_path]")
                audio_name = item[0]
            else:
                audio_name = raw.strip()
            audio = Path(audio_name).expanduser()
            if not audio.is_absolute():
                audio = path_base / audio
            yield line_no, raw, lexical_absolute(audio)


def audio_signature(path):
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    if not path.is_file():
        raise ValueError(f"Not an audio file: {path}")
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns,
            "device": info.st_dev, "inode": info.st_ino}


def union_duration(intervals):
    total = 0.0
    end = None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
        elif stop > end:
            total += stop - end
        end = stop if end is None else max(end, stop)
    return total


def summarize_turns(turns, duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Audio duration must be finite and positive")
    by_speaker = defaultdict(list)
    for start, end, speaker in turns:
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("Invalid diarization time range")
        start, end = max(0.0, start), min(duration, end)
        if end > start:
            by_speaker[str(speaker)].append((start, end))
    events = defaultdict(list)
    for speaker, intervals in by_speaker.items():
        for start, end in intervals:
            events[start].append((speaker, 1))
            events[end].append((speaker, -1))
    active = Counter()
    exclusive = Counter()
    overlap = 0.0
    previous = 0.0
    for timestamp in sorted(events):
        elapsed = timestamp - previous
        if len(active) == 1:
            exclusive[next(iter(active))] += elapsed
        elif len(active) > 1:
            overlap += elapsed
        for speaker, change in events[timestamp]:
            active[speaker] += change
            if active[speaker] == 0:
                del active[speaker]
        previous = timestamp
    return {
        "duration": duration,
        "speech_seconds": union_duration([turn for turns in by_speaker.values() for turn in turns]),
        "speaker_seconds": {speaker: union_duration(turns) for speaker, turns in sorted(by_speaker.items())},
        "speaker_exclusive_seconds": {speaker: exclusive[speaker] for speaker in sorted(by_speaker)},
        "overlap_seconds": overlap,
    }


def validate_metrics(metrics):
    duration = metrics["duration"]
    speech = metrics["speech_seconds"]
    speakers = metrics["speaker_seconds"]
    if (not isinstance(speakers, dict) or not math.isfinite(duration) or duration <= 0
            or not math.isfinite(speech) or speech < 0 or speech > duration + 1e-6):
        raise ValueError("Invalid diarization summary")
    for speaker, seconds in speakers.items():
        if (not isinstance(speaker, str) or not math.isfinite(seconds)
                or seconds <= 0 or seconds > speech + 1e-6):
            raise ValueError("Invalid speaker duration")
    if "speaker_exclusive_seconds" in metrics:
        exclusive = metrics["speaker_exclusive_seconds"]
        if not isinstance(exclusive, dict) or set(exclusive) != set(speakers):
            raise ValueError("Invalid exclusive speaker durations")
        for speaker, seconds in exclusive.items():
            if not math.isfinite(seconds) or seconds < 0 or seconds > speakers[speaker] + 1e-6:
                raise ValueError("Invalid exclusive speaker duration")
    if "overlap_seconds" in metrics:
        overlap = metrics["overlap_seconds"]
        if not math.isfinite(overlap) or overlap < 0 or overlap > speech + 1e-6:
            raise ValueError("Invalid overlap duration")


def classify(metrics, min_speaker_seconds, min_speaker_ratio):
    validate_metrics(metrics)
    speech = metrics["speech_seconds"]
    speakers = [speaker for speaker, seconds in metrics["speaker_seconds"].items()
                if seconds >= min_speaker_seconds
                and speech > 0 and seconds / speech >= min_speaker_ratio]
    if not speakers:
        return "no_speech" if speech == 0 else "insufficient_speech", 0
    return ("single" if len(speakers) == 1 else "multiple"), len(speakers)


def filter_decision(metrics, args):
    """Conservative triage; review is not evidence that a clip really has two people."""
    decision, count = classify(metrics, args.min_speaker_seconds, args.min_speaker_ratio)
    if args.decision_policy == "strict":
        return decision, count, "strict_thresholds"
    speakers = metrics["speaker_seconds"]
    if not speakers:
        return "no_speech", 0, "no_speech"
    if len(speakers) == 1:
        return "single", 1, "one_speaker_label"
    if count < 2:
        return "review", len(speakers), "brief_or_low_share_secondary"
    speech = metrics["speech_seconds"]
    if speech < args.min_multispeaker_speech:
        return "review", count, "short_speech"
    exclusive = metrics.get("speaker_exclusive_seconds")
    if exclusive is None:
        if len(speakers) == 2:
            # For two speakers their individual durations and union determine overlap.
            exclusive = {name: max(0.0, speech - sum(value for other, value in speakers.items()
                                                     if other != name))
                         for name in speakers}
        elif abs(sum(speakers.values()) - speech) < 1e-6:
            exclusive = speakers
        else:
            return "review", count, "legacy_overlap_ambiguous"
    supported = [name for name, seconds in speakers.items()
                 if seconds >= args.min_speaker_seconds
                 and seconds / speech >= args.min_speaker_ratio
                 and exclusive.get(name, 0) >= args.min_exclusive_seconds]
    if len(supported) < 2:
        return "review", count, "insufficient_nonoverlap_speech"
    return "multiple", count, "multiple_speaker_evidence"


def model_identity(model):
    path = Path(model).expanduser()
    if path.is_dir():
        path = path / "config.yaml"
        if not path.is_file():
            raise ValueError(f"Local pipeline directory has no config.yaml: {path.parent}")
    if path.is_file():
        path = path.resolve()
        return {"model": str(path), "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if path.is_absolute() or path.suffix.lower() in {".yaml", ".yml"}:
        raise ValueError(f"Local pipeline config does not exist: {path}")
    return {"model": model, "config_sha256": None}


class ProgressJournal:
    """Append and fsync each result; only repair an interrupted final JSONL line."""

    def __init__(self, stream, config):
        self.stream = stream
        self.cache = {}
        stream.seek(0)
        header = None
        line_no = 0
        expected_header = {"kind": "header", "version": STATE_VERSION, "config": config}
        encoded_header = json.dumps(expected_header, ensure_ascii=False).encode("utf-8") + b"\n"
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            line_no += 1
            if not line.endswith(b"\n"):
                if header is None and not encoded_header.startswith(line):
                    raise ValueError("State does not start with a recognized header; file was not modified")
                print(f"[RESUME] discard incomplete final state line {line_no}", flush=True)
                stream.seek(offset)
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
                break
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("expected object")
                if header is None:
                    if record != expected_header:
                        raise ValueError("state version/model/config differs; choose another --state")
                    header = record
                else:
                    if (record.get("kind") != "result" or record.get("status") not in {"ok", "error"}
                            or not isinstance(record.get("audio"), str) or "signature" not in record):
                        raise ValueError("invalid state result")
                    if record["status"] == "ok":
                        validate_metrics(record["metrics"])
                    self.cache[record["audio"]] = record
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid state at line {line_no}: {exc}") from exc
        stream.seek(0, os.SEEK_END)
        if header is None:
            self.append(expected_header)

    def append(self, record):
        payload = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        self.stream.write(payload)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        if record["kind"] == "result":
            self.cache[record["audio"]] = record

    def lookup(self, audio, signature, retry_errors):
        # Input and state paths are already normalized by the user. Match them exactly.
        record = self.cache.get(str(audio))
        if record is None or record["signature"] != signature:
            return None
        if retry_errors and record["status"] == "error":
            return None
        return record


class PyannoteDiarizer:
    def __init__(self, model, device, token_env):
        # Validate every checkpoint locally before importing/initializing the pipeline.
        with network_disabled(), local_pipeline_config(model, token_env) as local_config:
            try:
                import soundfile as sf
                import torch
                from pyannote.audio import Pipeline
            except ImportError as exc:
                raise RuntimeError("Install pyannote.audio and soundfile in the inference environment; "
                                   "see docs/single_speaker_filter.md") from exc
            self.sf, self.torch = sf, torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
            if self.device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(f"CUDA is not available for --device {device}")
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)
            print(f"[MODEL / LOCAL ONLY] loading {model} on {self.device}", flush=True)
            try:
                with pyannote_checkpoint_globals(torch):
                    self.pipeline = Pipeline.from_pretrained(str(local_config))
            except pickle.UnpicklingError as exc:
                raise RuntimeError(
                    "Local checkpoint still contains metadata unsupported by restricted loading. "
                    "TorchVersion and pyannote Specifications/Problem/Resolution were allowed "
                    f"temporarily; no unrestricted pickle fallback was used. Details: {exc}"
                ) from exc
            if self.pipeline is None:
                raise RuntimeError("Could not load the local pyannote pipeline; check local weights/config")
            self.pipeline.to(self.device)

    def analyze(self, audio):
        waveform, sample_rate = self.sf.read(str(audio), dtype="float32", always_2d=True)
        if sample_rate <= 0 or len(waveform) == 0:
            raise ValueError("Empty or invalid audio")
        duration = len(waveform) / sample_rate
        # The pipeline performs its own resampling; do not restrict speaker count.
        waveform = self.torch.from_numpy(waveform.mean(axis=1)).unsqueeze(0)
        with network_disabled(), self.torch.inference_mode():
            result = self.pipeline({"waveform": waveform, "sample_rate": sample_rate})
        # pyannote 3 returns Annotation; newer pipelines wrap it in DiarizeOutput.
        # Use ordinary diarization so overlapping speakers remain represented.
        annotation = getattr(result, "speaker_diarization", result)
        turns = ((float(turn.start), float(turn.end), speaker)
                 for turn, _track, speaker in annotation.itertracks(yield_label=True))
        return summarize_turns(turns, duration)


def analyze_audio_record(backend, audio, signature):
    record = {"kind": "result", "audio": str(audio), "signature": signature,
              "checked_at": time.time()}
    try:
        metrics = backend.analyze(audio)
        validate_metrics(metrics)
        if audio_signature(audio) != signature:
            raise ValueError("Audio changed during inference; retry on the next run")
        record.update(status="ok", metrics=metrics)
    except Exception as exc:
        message = str(exc)
        if isinstance(exc, MemoryError) or any(part in message.lower() for part in
                                              ("out of memory", "cuda error", "offline inference blocked")):
            raise
        record.update(status="error", error=message)
    return record


def diarization_worker(model, device, token_env, cpu_threads, tasks, results, backend_factory):
    """Each spawned process owns one model on one fixed GPU."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        with network_disabled():
            if backend_factory is PyannoteDiarizer:
                import torch
                torch.set_num_threads(cpu_threads)
            backend = backend_factory(model, device, token_env)
            results.put({"kind": "ready", "device": device, "pid": os.getpid()})
            while True:
                job = tasks.get()
                if job is None:
                    return
                record = analyze_audio_record(backend, Path(job["audio"]), job["signature"])
                results.put({"kind": "result", "job_id": job["job_id"], "record": record,
                             "device": device})
    except BaseException as exc:
        results.put({"kind": "fatal", "device": device, "error": f"{type(exc).__name__}: {exc}"})


class DevicePool:
    """A bounded shared job queue dynamically feeds one process per GPU."""

    def __init__(self, model, devices, token_env, cpu_threads, backend_factory):
        self.context = mp.get_context("spawn")
        self.tasks = self.context.Queue(maxsize=2 * len(devices))
        self.results = self.context.Queue()
        self.processes = []
        self.model, self.devices = model, devices
        self.token_env, self.cpu_threads = token_env, cpu_threads
        self.backend_factory = backend_factory

    def __enter__(self):
        try:
            for device in self.devices:
                process = self.context.Process(
                    target=diarization_worker,
                    args=(self.model, device, self.token_env, self.cpu_threads,
                          self.tasks, self.results, self.backend_factory),
                    name="speaker-" + device.replace(":", "-"),
                )
                process.start()
                self.processes.append(process)
        except BaseException:
            self.close(abort=True)
            raise
        return self

    def submit(self, job_id, audio, signature):
        self.tasks.put({"job_id": job_id, "audio": str(audio), "signature": signature})

    def receive(self, block=True):
        while True:
            try:
                result = self.results.get(timeout=0.5) if block else self.results.get_nowait()
            except Empty:
                if block:
                    dead = [p for p in self.processes if p.exitcode is not None]
                    if dead:
                        raise RuntimeError("Diarization worker exited unexpectedly: " +
                                           ", ".join(f"{p.name} exit={p.exitcode}" for p in dead))
                    continue
                return None
            if result["kind"] == "ready":
                print(f"[WORKER READY] {result['device']} pid={result['pid']}", flush=True)
                continue
            if result["kind"] == "fatal":
                raise RuntimeError(f"Worker {result['device']} failed: {result['error']}")
            return result

    def close(self, abort=False):
        if not abort:
            for _ in self.processes:
                self.tasks.put(None)
            for process in self.processes:
                process.join(timeout=5)
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=2)
        for queue in (self.tasks, self.results):
            queue.cancel_join_thread()
            queue.close()

    def __exit__(self, exc_type, *_args):
        self.close(abort=exc_type is not None)


def cache_key(audio, signature):
    return str(audio), json.dumps(signature, sort_keys=True)


def ensure_distinct_paths(paths):
    values = list(paths)
    if len({path.resolve() for path in values}) != len(values):
        raise ValueError("Input, output, temporary output and state must be different files")
    for index, left in enumerate(values):
        for right in values[index + 1:]:
            if left.exists() and right.exists() and os.path.samefile(left, right):
                raise ValueError(f"Paths reference the same file: {left}, {right}")


def run_filter(args, backend_factory=None):
    configure_model_cache(args.cache_dir, offline=True)
    input_path = lexical_absolute(args.scp or args.jsonl)
    if not input_path.is_file():
        raise ValueError(f"Input file not found: {input_path}")
    input_format = "scp" if args.scp else "jsonl"
    if input_path.suffix.lower() != "." + input_format:
        raise ValueError(f"Expected a .{input_format} input file")
    output = input_path.with_name(input_path.stem + args.suffix + input_path.suffix)
    temporary = Path(str(output) + ".tmp")
    review = output.with_name(output.stem + "_review" + output.suffix)
    review_temporary = Path(str(review) + ".tmp")
    detail_paths = {
        category: output.with_name(f"{output.stem}_{category}{output.suffix}")
        for category in DETAIL_OUTPUT_CATEGORIES
    }
    detail_temporaries = {category: Path(str(path) + ".tmp")
                          for category, path in detail_paths.items()}
    state = lexical_absolute(args.state or output.with_suffix(".state.jsonl"))
    output_lock_path = Path(str(output) + ".lock")
    state_lock_path = Path(str(state) + ".lock")
    ensure_distinct_paths((input_path, output, temporary, review, review_temporary, state,
                           output_lock_path, state_lock_path,
                           *detail_paths.values(), *detail_temporaries.values()))
    path_base = lexical_absolute(args.path_base or Path.cwd())
    identity = model_identity(args.model)
    config = {"backend": "pyannote", "audio_input": "mono_native_rate_full_file", **identity}
    state.parent.mkdir(parents=True, exist_ok=True)
    devices = args.devices or [args.device]
    max_pending = 2 * len(devices)
    stats = Counter()
    processed_this_run = set()
    pending = {}
    pending_keys = set()
    backend = pool = None
    factory = backend_factory or PyannoteDiarizer
    initial_input_signature = audio_signature(input_path)
    print(f"[START] input={input_path} output={output} review={review} state={state} "
          f"devices={devices} policy={args.decision_policy}", flush=True)

    with network_disabled(), transient_file_lock(output_lock_path), transient_file_lock(state_lock_path), \
            state.open("a+b") as state_file:
        journal = ProgressJournal(state_file, config)
        with temporary.open("w", encoding="utf-8", newline="\n") as selected, \
                review_temporary.open("w", encoding="utf-8", newline="\n") as review_stream, \
                ExitStack() as resources:
            detail_streams = {
                category: resources.enter_context(path.open("w", encoding="utf-8", newline="\n"))
                for category, path in detail_temporaries.items()
            }

            def save_record(record, key, device):
                journal.append(record)
                processed_this_run.add(key)
                stats["processed"] += 1
                if record["status"] == "error":
                    print(f"[RESULT ERROR] device={device} audio={record['audio']}: {record['error']}",
                          flush=True)
                elif args.verbose:
                    decision, count, reason = filter_decision(record["metrics"], args)
                    print(f"[RESULT] device={device} decision={decision} speakers={count} "
                          f"reason={reason} audio={record['audio']}", flush=True)

            def collect(block):
                if not pending:
                    return
                result = pool.receive(block)
                if result is None:
                    return
                key = pending.pop(result["job_id"])
                pending_keys.remove(key)
                save_record(result["record"], key, result["device"])

            records = iter_records(input_path, input_format, path_base)
            for _line_no, _raw, audio in islice(records, args.limit or None):
                collect(False)
                stats["records"] += 1
                signature_error = None
                try:
                    signature = audio_signature(audio)
                except (OSError, ValueError) as exc:
                    signature, signature_error = None, str(exc)
                key = cache_key(audio, signature)
                record = journal.lookup(audio, signature,
                                        args.retry_errors and key not in processed_this_run)
                if record is not None:
                    stats["cached"] += 1
                elif key in pending_keys:
                    stats["pending_duplicates"] += 1
                elif args.rebuild_only:
                    raise ValueError(f"No valid cached result for {audio}; use the original complete state "
                                     "or omit --rebuild-only to run inference")
                elif signature is None:
                    save_record({"kind": "result", "audio": str(audio), "signature": None,
                                 "checked_at": time.time(), "status": "error",
                                 "error": signature_error or "Audio file does not exist"}, key, "main")
                elif len(devices) == 1:
                    if backend is None:
                        backend = factory(identity["model"], devices[0], args.token_env)
                    save_record(analyze_audio_record(backend, audio, signature), key, devices[0])
                else:
                    if pool is None:
                        pool = resources.enter_context(DevicePool(
                            identity["model"], devices, args.token_env, args.cpu_threads, factory))
                    job_id = stats["submitted"]
                    pending[job_id] = key
                    pending_keys.add(key)
                    pool.submit(job_id, audio, signature)
                    stats["submitted"] += 1
                    if len(pending) >= max_pending:
                        collect(True)
                if stats["records"] % 100 == 0:
                    print(f"[PROGRESS] {dict(stats)} in_flight={len(pending)}", flush=True)
            while pending:
                collect(True)

            # Results are persisted in completion order. A streaming second pass restores
            # input order without retaining an unbounded backlog behind one slow clip.
            if audio_signature(input_path) != initial_input_signature:
                raise ValueError("Input list changed during filtering; retry using the saved state")
            records = iter_records(input_path, input_format, path_base)
            for line_no, raw, audio in islice(records, args.limit or None):
                try:
                    signature = audio_signature(audio)
                except (OSError, ValueError):
                    signature = None
                record = journal.lookup(audio, signature, retry_errors=False)
                if record is None:
                    raise ValueError(f"Audio changed before output was built: {audio}; rerun to resume")
                if record["status"] == "error":
                    stats["error"] += 1
                    print(f"[ERROR] line={line_no} audio={audio}: {record['error']}", flush=True)
                else:
                    decision, count, reason = filter_decision(record["metrics"], args)
                    stats[decision] += 1
                    if decision == "single" or (decision == "review" and args.uncertain_action == "keep"):
                        selected.write(raw + "\n")
                    if decision == "review":
                        review_stream.write(raw + "\n")
                        category = "review_" + reason
                        stats[category] += 1
                        detail_streams[category].write(raw + "\n")
                    elif decision == "multiple":
                        detail_streams["multiple"].write(raw + "\n")
                    if args.verbose:
                        print(f"[{decision.upper()}] line={line_no} speakers={count} reason={reason} "
                              f"audio={audio} seconds={record['metrics']['speaker_seconds']}", flush=True)
            selected.flush()
            os.fsync(selected.fileno())
            review_stream.flush()
            os.fsync(review_stream.fileno())
            for stream in detail_streams.values():
                stream.flush()
                os.fsync(stream.fileno())
        if audio_signature(input_path) != initial_input_signature:
            raise ValueError("Input list changed during filtering; retry using the saved state")
        for category, path in detail_paths.items():
            os.replace(detail_temporaries[category], path)
        os.replace(review_temporary, review)
        os.replace(temporary, output)
    print(f"[DONE] {dict(stats)} output={output} review={review} state={state}", flush=True)
    for category, path in detail_paths.items():
        print(f"[OUTPUT] category={category} records={stats[category]} path={path}", flush=True)
    return 1 if stats["error"] else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--download-model", action="store_true",
                        help="Download/export a local model bundle on an online machine, verify on CPU, then exit")
    source.add_argument("--scp", type=Path, help="One audio path per line, as produced by filter_scp_by_jsonl.py")
    source.add_argument("--jsonl", type=Path, help="[audio, text] JSONL produced by filter_quality_jsonl.py")
    parser.add_argument("--suffix", default="_single_speaker", help="Suffix inserted before .scp/.jsonl")
    parser.add_argument("--state", type=Path, help="Append-only JSONL checkpoint; default OUTPUT_STEM.state.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="pyannote model ID, local config.yaml, or its directory")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--device", default="auto", help="Single device: auto, cpu, cuda, cuda:0, ...")
    execution.add_argument("--devices", type=parse_devices, help="One process per GPU, e.g. 0,1,2,3,4,5,6,7")
    parser.add_argument("--cpu-threads", type=int, default=2, help="Torch CPU threads per GPU worker")
    parser.add_argument("--cache-dir", type=Path, help="Store pyannote/Hugging Face/Torch caches under this directory")
    parser.add_argument("--offline", action="store_true", help="Compatibility flag: filtering is now always offline, even when omitted")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable containing the Hugging Face token")
    parser.add_argument("--path-base", type=Path, help="Base for relative audio paths; default current working directory")
    parser.add_argument("--min-speaker-seconds", type=float, default=0.5,
                        help="Minimum per-speaker duration for multi-speaker evidence (seconds)")
    parser.add_argument("--min-speaker-ratio", type=float, default=0.1,
                        help="Minimum per-speaker share of speech time for multi-speaker evidence")
    parser.add_argument("--decision-policy", choices=("conservative", "strict"), default="conservative",
                        help="Conservative triage sends short/brief/overlap-only evidence to review")
    parser.add_argument("--min-multispeaker-speech", type=float, default=3.0,
                        help="Minimum speech seconds for a multiple-speaker decision in conservative mode")
    parser.add_argument("--min-exclusive-seconds", type=float, default=0.3,
                        help="Minimum non-overlapping speech seconds for each of two speakers")
    parser.add_argument("--uncertain-action", choices=("review", "keep"), default="review",
                        help="review: separate uncertain items; keep: also include them in main output")
    parser.add_argument("--rebuild-only", action="store_true",
                        help="Rebuild outputs from the complete state without loading models or rewriting cached paths")
    parser.add_argument("--retry-errors", action="store_true", help="Reprocess cached file errors")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N input records; 0 means all")
    parser.add_argument("--verbose", action="store_true", help="Print every classification")
    args = parser.parse_args(argv)
    if not args.suffix or any(char in args.suffix for char in "/\\"):
        parser.error("--suffix must be non-empty and contain no path separators")
    if (not math.isfinite(args.min_speaker_seconds) or args.min_speaker_seconds < 0
            or not math.isfinite(args.min_speaker_ratio) or not 0 <= args.min_speaker_ratio <= 1):
        parser.error("Speaker seconds must be >= 0 and ratio must be in [0, 1]")
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if any(not math.isfinite(value) or value < 0 for value in
           (args.min_multispeaker_speech, args.min_exclusive_seconds)):
        parser.error("Speech and exclusive duration thresholds must be finite and non-negative")
    if args.rebuild_only and args.retry_errors:
        parser.error("--rebuild-only and --retry-errors cannot be used together")
    if args.rebuild_only and args.download_model:
        parser.error("--rebuild-only requires --scp or --jsonl, not --download-model")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.download_model and (args.cache_dir is None or args.offline):
        parser.error("--download-model requires --cache-dir and cannot use --offline")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.download_model:
        root = configure_model_cache(args.cache_dir)
        # This is the only mode allowed to contact model repositories.
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        identity = model_identity(args.model)
        local_config = prepare_model_bundle(identity["model"], root, args.token_env)
        PyannoteDiarizer(str(local_config), "cpu", args.token_env)
        print(f"[DOWNLOAD COMPLETE] local_model={local_config} cache_dir={root}. "
              "Copy the complete local/ directory to the offline server. Filtering never contacts "
              "online models, even without --offline.", flush=True)
        return 0
    return run_filter(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FATAL] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
