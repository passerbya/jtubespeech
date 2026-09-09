#!/usr/bin/env python3
# coding: utf-8
"""Estimate the effective bandwidth of a collection of FLAC files.

The command accepts one of the following inputs:

* ``--scp``: an SCP file containing one audio path per line;
* ``--input-jsonl`` (or ``--jsonl``): JSONL records such as
  ``["audio.flac", "audio.txt"]``;
* ``--directory`` (or ``--testset-dir``): a directory scanned recursively for
  ``.flac`` files.

Results are appended to a JSONL file as soon as they are computed.  Existing
valid records are used as a checkpoint, so an interrupted run can be started
again safely.  With ``--buckets N --bucket K`` only the deterministic K-th
bucket is processed and the output is written to ``*_K.jsonl``.

The frequency estimator follows the implementation used by the original
jtubespeech helper: a Hann-windowed STFT, a 99.9% energy roll-off for each
frame, and a high quantile over those frame roll-offs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from queue import Empty
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - gives a useful CLI error
    raise RuntimeError("get_freq.py requires numpy") from exc

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover - gives a useful CLI error
    raise RuntimeError("get_freq.py requires soundfile") from exc

try:
    import torch
except ImportError:  # pragma: no cover - numpy fallback keeps the tool usable
    torch = None  # type: ignore


DEFAULT_ROLL_PERCENT = 0.999
DEFAULT_CUTOFF_PERCENT = 0.9999
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 1024
READER_DONE = "READER_DONE"
WORKER_DONE = "WORKER_DONE"
RESULT = "RESULT"
ERROR = "ERROR"
STOP = "STOP"


def bucket_for_filename(path: Path, buckets: int) -> int:
    """Return a stable zero-based bucket, matching ``dnsmos_local.py``."""

    if buckets < 1:
        raise ValueError("buckets must be at least 1")
    if buckets == 1:
        return 0
    digest = hashlib.sha256(path.name.encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big") % buckets


def _path_from_value(value: Any) -> Optional[Path]:
    if isinstance(value, str) and value.strip():
        return Path(value.strip())
    return None


def _audio_path_from_json_record(record: Any) -> Optional[Path]:
    """Extract an audio path from the common flac/text JSONL formats."""

    if isinstance(record, (list, tuple)) and record:
        return _path_from_value(record[0])
    if isinstance(record, str):
        return _path_from_value(record)
    if isinstance(record, dict):
        for key in ("audio_path", "filename", "path", "audio"):
            path = _path_from_value(record.get(key))
            if path is not None:
                return path
    return None


def load_scp(scp_path: Path) -> Iterator[Path]:
    """Yield one path per non-empty, non-comment SCP line."""

    scp_path = Path(scp_path)
    with scp_path.open("r", encoding="utf-8-sig") as scp_file:
        for line_no, raw_line in enumerate(scp_file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            yield Path(line)


def load_jsonl(jsonl_path: Path) -> Iterator[Path]:
    """Yield audio paths from JSONL arrays, strings, or object records.

    A malformed line is reported and skipped.  Large corpus jobs can therefore
    continue when a single metadata line is damaged; the line number is kept in
    the warning to make repair straightforward.
    """

    jsonl_path = Path(jsonl_path)
    with jsonl_path.open("r", encoding="utf-8-sig") as jsonl_file:
        for line_no, raw_line in enumerate(jsonl_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as json_error:
                print(
                    "warning: skip %s:%d: invalid JSON (%s)"
                    % (jsonl_path, line_no, json_error),
                    file=sys.stderr,
                    flush=True,
                )
                continue
            path = _audio_path_from_json_record(record)
            if path is None:
                print(
                    "warning: skip %s:%d: no audio path in record"
                    % (jsonl_path, line_no),
                    file=sys.stderr,
                    flush=True,
                )
                continue
            yield path


def scan_flac(directory: Path) -> Iterator[Path]:
    """Recursively yield sorted FLAC files under ``directory``."""

    directory = Path(directory)
    # Sort one directory at a time so a large corpus does not require a
    # second in-memory list containing every path before processing can start.
    for root, dirnames, filenames in os.walk(directory):
        dirnames.sort()
        for filename in sorted(filenames):
            if filename.lower().endswith(".flac"):
                yield Path(root) / filename


def iter_audio_files(
    directory: Optional[Path] = None,
    scp_path: Optional[Path] = None,
    jsonl_path: Optional[Path] = None,
    testset_dir: Optional[Path] = None,
) -> Iterator[Path]:
    """Yield paths from exactly one supported input source."""

    if directory is None:
        directory = testset_dir
    elif testset_dir is not None:
        raise ValueError("directory and testset_dir cannot both be provided")
    selected = sum(item is not None for item in (directory, scp_path, jsonl_path))
    if selected != 1:
        raise ValueError("exactly one of directory, scp_path, and jsonl_path is required")
    if scp_path is not None:
        yield from load_scp(scp_path)
    elif jsonl_path is not None:
        yield from load_jsonl(jsonl_path)
    else:
        # ``directory`` is known to be non-None because of the check above.
        yield from scan_flac(directory)  # type: ignore[arg-type]


def _as_mono(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim == 2:
        # soundfile returns (frames, channels).
        array = array.mean(axis=1)
    elif array.ndim != 1:
        raise ValueError("audio must be a one- or two-dimensional array")
    if array.size == 0:
        raise ValueError("empty audio")
    # NaN/Inf values make cumulative energy unusable.  Treat them as silence
    # and leave the original file untouched.
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0, copy=True)


def _estimate_frequency_numpy(
    audio: np.ndarray,
    sampling_rate: int,
    n_fft: int,
    hop_length: int,
    roll_percent: float,
    cutoff_percent: float,
) -> float:
    """NumPy fallback for environments without PyTorch."""

    if audio.size < n_fft:
        padded = np.zeros(n_fft, dtype=np.float32)
        padded[: audio.size] = audio
        audio = padded
    frame_count = 1 + max(0, (audio.size - n_fft) // hop_length)
    window = np.hanning(n_fft).astype(np.float32)
    rolloffs: List[float] = []
    for index in range(frame_count):
        start = index * hop_length
        frame = audio[start : start + n_fft]
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size))
        spectrum = np.abs(np.fft.rfft(frame * window, n=n_fft)) ** 2
        cumulative = np.cumsum(spectrum)
        total = float(cumulative[-1]) if cumulative.size else 0.0
        target = total * roll_percent
        bin_index = int(np.searchsorted(cumulative, target, side="left"))
        bin_index = min(bin_index, n_fft // 2)
        rolloffs.append(bin_index * float(sampling_rate) / n_fft)
    return float(np.quantile(np.asarray(rolloffs), cutoff_percent))


def estimate_frequency(
    audio: np.ndarray,
    sampling_rate: int,
    roll_percent: float = DEFAULT_ROLL_PERCENT,
    cutoff_percent: float = DEFAULT_CUTOFF_PERCENT,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    device: str = "cpu",
) -> float:
    """Estimate the effective upper frequency in Hz.

    The result is the ``cutoff_percent`` quantile of per-frame frequencies at
    which ``roll_percent`` of cumulative spectral energy has been reached.
    """

    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be positive")
    if not 0.0 <= roll_percent <= 1.0:
        raise ValueError("roll_percent must be between 0 and 1")
    if not 0.0 <= cutoff_percent <= 1.0:
        raise ValueError("cutoff_percent must be between 0 and 1")

    mono = _as_mono(audio)
    if torch is None:
        return _estimate_frequency_numpy(
            mono,
            sampling_rate,
            n_fft,
            hop_length,
            roll_percent,
            cutoff_percent,
        )

    requested_device = device.lower()
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")

    tensor = torch.as_tensor(mono, dtype=torch.float32, device=requested_device)
    window = torch.hann_window(n_fft, dtype=tensor.dtype, device=tensor.device)
    with torch.inference_mode():
        try:
            spectrum = torch.stft(
                tensor,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                return_complex=True,
            ).abs().pow(2)
        except RuntimeError:
            # Reflect padding cannot be larger than a very short waveform.  A
            # zero-padded, non-centred STFT still gives a useful answer.
            if tensor.numel() >= n_fft:
                raise
            padded = torch.nn.functional.pad(tensor, (0, n_fft - tensor.numel()))
            spectrum = torch.stft(
                padded,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                center=False,
                return_complex=True,
            ).abs().pow(2)
        energy_cumulative = spectrum.cumsum(dim=0)
        total_energy = energy_cumulative[-1:, :]
        threshold = total_energy * roll_percent
        rolloff_bins = (energy_cumulative >= threshold).to(torch.int64).argmax(dim=0)
        rolloff_freq = rolloff_bins.to(torch.float32) * float(sampling_rate) / n_fft
        return float(torch.quantile(rolloff_freq, cutoff_percent).item())


def _sample_rate_for_cutoff(cutoff_hz: float, current_rate: int) -> int:
    for rate in (8000, 16000, 22050, 32000, 44100, 48000):
        if rate / 2.0 > cutoff_hz:
            return rate
    return current_rate


def analyze_audio(
    audio: np.ndarray,
    sampling_rate: int,
    audio_path: str,
    roll_percent: float = DEFAULT_ROLL_PERCENT,
    cutoff_percent: float = DEFAULT_CUTOFF_PERCENT,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Return one JSON-serialisable frequency record."""

    mono = _as_mono(audio)
    estimated_freq = estimate_frequency(
        mono,
        sampling_rate,
        roll_percent=roll_percent,
        cutoff_percent=cutoff_percent,
        n_fft=n_fft,
        hop_length=hop_length,
        device=device,
    )
    estimated_rate = _sample_rate_for_cutoff(estimated_freq, sampling_rate)
    return {
        "audio_path": audio_path,
        "duration": float(mono.size / sampling_rate),
        "current_sample_rate": int(sampling_rate),
        "max_freq": float(sampling_rate / 2.0),
        "estimated_freq": float(estimated_freq),
        "bandwidth_ratio": float(estimated_freq / (sampling_rate / 2.0)),
        "is_upsampled": bool(estimated_rate != sampling_rate),
        "estimated_sample_rate": int(estimated_rate),
    }


def analyze_audio_bytes(
    audio_path: str,
    audio_bytes: bytes,
    device: str,
    roll_percent: float,
    cutoff_percent: float,
    n_fft: int,
    hop_length: int,
) -> Dict[str, Any]:
    audio, sampling_rate = sf.read(
        io.BytesIO(audio_bytes), dtype="float32", always_2d=True
    )
    return analyze_audio(
        audio,
        int(sampling_rate),
        audio_path,
        roll_percent=roll_percent,
        cutoff_percent=cutoff_percent,
        n_fft=n_fft,
        hop_length=hop_length,
        device=device,
    )


def _compute_worker(
    input_queue: Any,
    output_queue: Any,
    worker_id: int,
    device: str,
    roll_percent: float,
    cutoff_percent: float,
    n_fft: int,
    hop_length: int,
) -> None:
    """Decode and analyse files until the reader sends a stop sentinel."""

    if torch is not None:
        try:
            thread_count = int(os.environ.get("GET_FREQ_CPU_THREADS", "1"))
        except ValueError:
            thread_count = 1
        if thread_count > 0:
            torch.set_num_threads(thread_count)
    worker_device = device
    if device in {"auto", "cuda"}:
        if torch is not None and torch.cuda.is_available():
            worker_device = "cuda:%d" % (worker_id % max(torch.cuda.device_count(), 1))
        elif device == "auto":
            worker_device = "cpu"
    print("compute_worker %d started on %s" % (worker_id, worker_device), flush=True)
    while True:
        item = input_queue.get()
        if item == STOP:
            break
        audio_path, audio_bytes = item
        try:
            result = analyze_audio_bytes(
                audio_path,
                audio_bytes,
                worker_device,
                roll_percent,
                cutoff_percent,
                n_fft,
                hop_length,
            )
            output_queue.put((RESULT, result))
        except Exception as worker_error:  # keep the remaining corpus moving
            output_queue.put(
                (ERROR, audio_path, type(worker_error).__name__, str(worker_error))
            )
    output_queue.put((WORKER_DONE, worker_id))
    print("compute_worker %d stopped" % worker_id, flush=True)


def _read_worker(
    directory: Optional[Path],
    scp_path: Optional[Path],
    jsonl_path: Optional[Path],
    completed: Set[str],
    buckets: int,
    bucket: int,
    input_queue: Any,
    output_queue: Any,
    worker_count: int,
) -> None:
    queued = 0
    skipped_completed = 0
    skipped_bucket = 0
    skipped_duplicate = 0
    read_failures = 0
    seen: Set[str] = set(completed)
    try:
        paths = iter_audio_files(directory, scp_path, jsonl_path)
        for path in paths:
            display_path = str(path)
            path_key = _checkpoint_key(display_path)
            if bucket_for_filename(path, buckets) != bucket:
                skipped_bucket += 1
                continue
            if path_key in seen:
                if path_key in completed:
                    skipped_completed += 1
                else:
                    skipped_duplicate += 1
                continue
            seen.add(path_key)
            try:
                audio_bytes = path.read_bytes()
            except OSError as read_error:
                read_failures += 1
                print(
                    "warning: cannot read %s: %s" % (path, read_error),
                    file=sys.stderr,
                    flush=True,
                )
                continue
            input_queue.put((display_path, audio_bytes))
            queued += 1
    except Exception as reader_error:
        output_queue.put(
            (ERROR, "<input>", type(reader_error).__name__, str(reader_error))
        )
    finally:
        for _ in range(worker_count):
            input_queue.put(STOP)
        output_queue.put(
            (
                READER_DONE,
                queued,
                skipped_completed,
                skipped_bucket,
                skipped_duplicate,
                read_failures,
            )
        )


def _bucket_output_path(output_path: Path, buckets: int, bucket: int) -> Path:
    if buckets == 1:
        return output_path
    return output_path.with_name("%s_%d%s" % (output_path.stem, bucket, output_path.suffix))


def _checkpoint_key(path: str) -> str:
    """Canonicalise paths so relative and absolute input lists can resume each other."""

    try:
        return str(Path(path).expanduser().resolve())
    except (OSError, RuntimeError):
        return path


def load_completed(output_path: Path) -> Tuple[Set[str], int]:
    """Read checkpoint keys and count malformed lines without aborting a run."""

    output_path = Path(output_path)
    completed: Set[str] = set()
    malformed = 0
    if not output_path.exists():
        return completed, malformed
    with output_path.open("r", encoding="utf-8-sig") as output_file:
        for line_no, raw_line in enumerate(output_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(record, dict):
                malformed += 1
                continue
            path = record.get("audio_path", record.get("filename"))
            if isinstance(path, str) and path.strip():
                completed.add(path.strip())
            else:
                malformed += 1
    return completed, malformed


def _truncate_partial_last_line(output_path: Path) -> None:
    """Remove a crash-truncated final JSON line before appending new records."""

    if not output_path.exists() or output_path.stat().st_size == 0:
        return
    with output_path.open("rb+") as output_file:
        output_file.seek(-1, os.SEEK_END)
        if output_file.read(1) == b"\n":
            return
        file_size = output_file.tell()
        chunk_size = 64 * 1024
        position = file_size
        line_start = 0
        while position > 0:
            read_size = min(chunk_size, position)
            position -= read_size
            output_file.seek(position)
            chunk = output_file.read(read_size)
            newline = chunk.rfind(b"\n")
            if newline >= 0:
                line_start = position + newline + 1
                break
        output_file.seek(line_start)
        trailing_line = output_file.read()
        try:
            json.loads(trailing_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            output_file.truncate(line_start)
        else:
            # A process may have been killed after writing a complete JSON
            # object but before writing its newline.  Keep that checkpoint.
            output_file.seek(0, os.SEEK_END)
            output_file.write(b"\n")


def run_pipeline(
    input_path: Path,
    input_kind: str,
    output_path: Path,
    buckets: int = 1,
    bucket: int = 0,
    workers: int = 1,
    prefetch_size: int = 64,
    device: str = "auto",
    roll_percent: float = DEFAULT_ROLL_PERCENT,
    cutoff_percent: float = DEFAULT_CUTOFF_PERCENT,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    progress_interval: float = 5.0,
) -> Dict[str, int]:
    """Run the multiprocessing pipeline and return counters for the caller."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    if buckets < 1 or not 0 <= bucket < buckets:
        raise ValueError("bucket must be in the range [0, buckets)")
    if workers < 1 or prefetch_size < 1:
        raise ValueError("workers and prefetch_size must be at least 1")
    if input_kind not in {"scp", "jsonl", "directory"}:
        raise ValueError("input_kind must be scp, jsonl, or directory")
    if input_kind != "directory" and input_path.resolve() == output_path.resolve():
        raise ValueError("output JSONL must be different from the input list")

    actual_output = _bucket_output_path(output_path, buckets, bucket)
    actual_output.parent.mkdir(parents=True, exist_ok=True)
    completed_records, malformed = load_completed(actual_output)
    completed = {_checkpoint_key(path) for path in completed_records}
    _truncate_partial_last_line(actual_output)
    print(
        "checkpoint=%d malformed=%d bucket=%d/%d output=%s"
        % (len(completed), malformed, bucket, buckets, actual_output),
        flush=True,
    )

    directory = input_path if input_kind == "directory" else None
    scp_path = input_path if input_kind == "scp" else None
    jsonl_path = input_path if input_kind == "jsonl" else None
    context = mp.get_context("spawn")
    input_queue = context.Queue(maxsize=prefetch_size)
    output_queue = context.Queue()
    compute_processes = [
        context.Process(
            target=_compute_worker,
            args=(
                input_queue,
                output_queue,
                worker_id,
                device,
                roll_percent,
                cutoff_percent,
                n_fft,
                hop_length,
            ),
        )
        for worker_id in range(workers)
    ]
    reader = context.Process(
        target=_read_worker,
        args=(
            directory,
            scp_path,
            jsonl_path,
            completed,
            buckets,
            bucket,
            input_queue,
            output_queue,
            workers,
        ),
    )
    for process in compute_processes:
        process.start()
    reader.start()

    stats: Dict[str, int] = {
        "queued": 0,
        "completed": 0,
        "failed": 0,
        "skipped_completed": 0,
        "skipped_bucket": 0,
        "skipped_duplicate": 0,
        "read_failures": 0,
        "malformed_checkpoint": malformed,
    }
    started = time.monotonic()
    last_report = started
    worker_done = 0
    reader_done = False
    with actual_output.open("a", encoding="utf-8", buffering=1024 * 1024) as output_file:
        while worker_done < workers or not reader_done:
            try:
                message = output_queue.get(timeout=0.5)
            except Empty:
                if time.monotonic() - last_report >= progress_interval:
                    print(
                        "progress completed=%d failed=%d elapsed=%.1fs"
                        % (stats["completed"], stats["failed"], time.monotonic() - started),
                        flush=True,
                    )
                    last_report = time.monotonic()
                continue
            tag = message[0]
            if tag == RESULT:
                record = message[1]
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
                stats["completed"] += 1
            elif tag == ERROR:
                stats["failed"] += 1
                print(
                    "warning: %s %s: %s" % (message[2], message[1], message[3]),
                    file=sys.stderr,
                    flush=True,
                )
            elif tag == READER_DONE:
                (
                    stats["queued"],
                    stats["skipped_completed"],
                    stats["skipped_bucket"],
                    stats["skipped_duplicate"],
                    stats["read_failures"],
                ) = message[1:]
                reader_done = True
            elif tag == WORKER_DONE:
                worker_done += 1

    reader.join()
    for process in compute_processes:
        process.join()
    failed_processes = [process.pid for process in compute_processes if process.exitcode != 0]
    if reader.exitcode != 0:
        raise RuntimeError("read worker failed with exit code %s" % reader.exitcode)
    if failed_processes:
        raise RuntimeError("compute workers failed: %s" % failed_processes)
    print(
        "done queued=%d completed=%d failed=%d skipped_completed=%d output=%s"
        % (
            stats["queued"],
            stats["completed"],
            stats["failed"],
            stats["skipped_completed"],
            actual_output,
        ),
        flush=True,
    )
    return stats


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate effective FLAC bandwidth and write frequency JSONL records."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--scp",
        "--input-scp",
        "--input_scp",
        "--scp_path",
        dest="scp",
        type=Path,
        help="SCP file with one audio path per line.",
    )
    input_group.add_argument(
        "--input-jsonl",
        "--jsonl-input",
        "--input_jsonl",
        "--jsonl",
        dest="input_jsonl",
        type=Path,
        help="JSONL containing [audio_path, text_path] records.",
    )
    input_group.add_argument(
        "-t",
        "--directory",
        "--dir",
        "--root",
        "--input-dir",
        "--input_dir",
        "--testset-dir",
        "--testset_dir",
        dest="directory",
        type=Path,
        help="Directory to scan recursively for .flac files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--output-jsonl",
        "--output_jsonl",
        "--jsonl-path",
        "--jsonl_path",
        type=Path,
        default=Path("frequency.jsonl"),
        help="Output JSONL path (default: frequency.jsonl). Bucket suffix is added when needed.",
    )
    parser.add_argument("--buckets", type=int, default=1, help="Total number of buckets.")
    parser.add_argument("--bucket", type=int, default=0, help="Zero-based bucket index.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of compute worker processes (default: 1).",
    )
    parser.add_argument(
        "--workers-per-gpu",
        "--workers_per_gpu",
        type=int,
        default=None,
        help="Compatibility option: workers per visible GPU; overrides --workers.",
    )
    parser.add_argument(
        "--prefetch-size",
        "--prefetch_size",
        type=int,
        default=64,
        help="Maximum encoded files held in RAM (default: 64).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Computation device; auto uses CUDA when available.",
    )
    parser.add_argument("--roll-percent", "--roll_percent", type=float, default=DEFAULT_ROLL_PERCENT)
    parser.add_argument("--cutoff-percent", "--cutoff_percent", type=float, default=DEFAULT_CUTOFF_PERCENT)
    parser.add_argument("--n-fft", "--n_fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--hop-length", "--hop_length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--progress-interval", "--progress_interval", type=float, default=5.0)
    args = parser.parse_args(argv)

    selected_path = args.scp or args.input_jsonl or args.directory
    if selected_path is None or not selected_path.exists():
        parser.error("input path does not exist: %s" % selected_path)
    if args.scp is not None and not args.scp.is_file():
        parser.error("--scp is not a file: %s" % args.scp)
    if args.input_jsonl is not None and not args.input_jsonl.is_file():
        parser.error("--input-jsonl is not a file: %s" % args.input_jsonl)
    if args.directory is not None and not args.directory.is_dir():
        parser.error("--directory is not a directory: %s" % args.directory)
    if args.device == "cuda" and (torch is None or not torch.cuda.is_available()):
        parser.error("--device cuda requested but no CUDA device is available")
    if args.buckets < 1:
        parser.error("--buckets must be at least 1")
    if not 0 <= args.bucket < args.buckets:
        parser.error("--bucket must be in the range [0, --buckets)")
    if args.workers_per_gpu is not None:
        if args.workers_per_gpu < 1:
            parser.error("--workers-per-gpu must be at least 1")
        gpu_count = torch.cuda.device_count() if torch is not None and torch.cuda.is_available() else 1
        args.workers = args.workers_per_gpu * gpu_count
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.prefetch_size < 1:
        parser.error("--prefetch-size must be at least 1")
    if args.n_fft < 1 or args.hop_length < 1:
        parser.error("--n-fft and --hop-length must be positive")
    if not 0.0 <= args.roll_percent <= 1.0:
        parser.error("--roll-percent must be between 0 and 1")
    if not 0.0 <= args.cutoff_percent <= 1.0:
        parser.error("--cutoff-percent must be between 0 and 1")
    if args.progress_interval <= 0:
        parser.error("--progress-interval must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.scp is not None:
        input_path, input_kind = args.scp, "scp"
    elif args.input_jsonl is not None:
        input_path, input_kind = args.input_jsonl, "jsonl"
    else:
        input_path, input_kind = args.directory, "directory"
    run_pipeline(
        input_path=input_path,
        input_kind=input_kind,
        output_path=args.output,
        buckets=args.buckets,
        bucket=args.bucket,
        workers=args.workers,
        prefetch_size=args.prefetch_size,
        device=args.device,
        roll_percent=args.roll_percent,
        cutoff_percent=args.cutoff_percent,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        progress_interval=args.progress_interval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
