#!/usr/bin/env python3
# coding: utf-8
"""Run official NISQA on a WAV SCP/list and keep files above a MOS threshold.

The script accepts both of these common input formats and preserves each
original line in the filtered output:

    /path/to/audio.wav
    utterance_id /path/to/audio.wav

NISQA predictions are cached as JSONL. Re-running the command only predicts
files that are not already present in the cache; changing --mos-threshold
therefore does not require another inference pass. Multiple persistent worker
processes can bind to different GPUs and share chunks from one task queue.
"""

import argparse
import csv
import importlib
import json
import os
import queue
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.multiprocessing as mp

PREDICTION_COLUMNS = (
    "mos_pred",
    "noi_pred",
    "dis_pred",
    "col_pred",
    "loud_pred",
)


def read_scp(scp_path: Path) -> List[Tuple[str, str]]:
    """Return (original_line, wav_path) records from a WAV list/Kaldi SCP."""
    records: List[Tuple[str, str]] = []
    with scp_path.open("r", encoding="utf-8") as scp_file:
        for line_number, raw_line in enumerate(scp_file, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split(maxsplit=1)
            wav_path = fields[-1].strip()
            if wav_path.endswith("|"):
                raise ValueError(
                    f"{scp_path}:{line_number}: NISQA requires a WAV path; "
                    "Kaldi pipe commands are not supported"
                )
            if not wav_path:
                raise ValueError(f"{scp_path}:{line_number}: empty WAV path")
            records.append((line, wav_path))
    return records


def load_jsonl(jsonl_path: Path) -> Dict[str, dict]:
    """Load successful cached predictions, keyed by filename."""
    predictions: Dict[str, dict] = {}
    if not jsonl_path.exists():
        return predictions

    with jsonl_path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, raw_line in enumerate(jsonl_file, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                filename = record["filename"]
                float(record["mos_pred"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                print(line)
                raise ValueError(
                    f"{jsonl_path}:{line_number}: invalid NISQA JSONL record"
                ) from error
            predictions[str(filename)] = record
    return predictions


def unique_in_order(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(values))


def write_input_csv(csv_path: Path, filenames: Iterable[str]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=("filename",))
        writer.writeheader()
        writer.writerows({"filename": filename} for filename in filenames)


def result_listener(result_queue, error_queue, jsonl_path: str, total: int) -> None:
    """Persist completed chunks so an interrupted run can resume."""
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a", encoding="utf-8") as jsonl_file:
        while True:
            kind, payload = result_queue.get()
            if kind == "stop":
                break
            if kind == "error":
                error_queue.put(payload)
                continue

            for record in payload:
                json.dump(record, jsonl_file, ensure_ascii=False)
                jsonl_file.write("\n")
            jsonl_file.flush()
            written += len(payload)
            print(f"NISQA progress: {written}/{total}", flush=True)


def compute_worker(
        worker_id: int,
        device_id: Optional[int],
        task_queue,
        result_queue,
        nisqa_dir: str,
        model_path: str,
        batch_size: int,
        loader_workers: int,
        max_segments: Optional[int],
        ms_channel: Optional[int],
) -> None:
    """Load one model on one GPU and repeatedly score queued chunks."""
    device_label = "cpu" if device_id is None else f"cuda:{device_id}"
    try:
        if nisqa_dir not in sys.path:
            sys.path.insert(0, nisqa_dir)

        if device_id is not None:
            torch.cuda.set_device(device_id)

        from nisqa.NISQA_model import nisqaModel

        nisqa_lib = importlib.import_module("nisqa.NISQA_lib")

        model = None
        with tempfile.TemporaryDirectory(
                prefix=f"nisqa_worker_{worker_id}_"
        ) as temp_dir_name:
            input_csv = Path(temp_dir_name) / "input.csv"
            print(f"worker {worker_id} loading model on {device_label}", flush=True)

            for filenames in iter(task_queue.get, None):
                write_input_csv(input_csv, filenames)
                if model is None:
                    model_args = {
                        "mode": "predict_csv",
                        "pretrained_model": model_path,
                        "deg": None,
                        "data_dir": "",
                        "output_dir": None,
                        "csv_file": str(input_csv),
                        "csv_deg": "filename",
                        "num_workers": loader_workers,
                        "bs": batch_size,
                        "ms_channel": ms_channel,
                        "tr_bs_val": batch_size,
                        "tr_num_workers": loader_workers,
                        "tr_parallel": False,
                        "tr_device": "cpu" if device_id is None else "cuda",
                    }
                    if max_segments is not None:
                        model_args["ms_max_segments"] = max_segments
                    model = nisqaModel(model_args)
                    if not model.args.get("dim"):
                        raise ValueError(
                            "The selected model is not multidimensional nisqa.tar"
                        )
                    print(
                        f"worker {worker_id} loaded model on {device_label}",
                        flush=True,
                    )
                else:
                    model.args["data_dir"] = ""
                    model.args["csv_file"] = str(input_csv)
                    # noinspection PyProtectedMember
                    model._loadDatasetsCSVpredict()

                nisqa_lib.predict_dim(
                    model.model,
                    model.ds_val,
                    batch_size,
                    model.dev,
                    num_workers=loader_workers,
                )
                results = []
                for row in model.ds_val.df.to_dict(orient="records"):
                    record = {"filename": str(row["filename"])}
                    record.update(
                        {column: float(row[column]) for column in PREDICTION_COLUMNS}
                    )
                    results.append(record)
                result_queue.put(("records", results))
    except BaseException:
        result_queue.put(
            (
                "error",
                f"worker {worker_id} on {device_label} failed:\n{traceback.format_exc()}",
            )
        )


def resolve_devices(device_arg: Optional[str]) -> List[Optional[int]]:
    if device_arg and device_arg.lower() == "cpu":
        return [None]

    visible_count = torch.cuda.device_count()
    if visible_count == 0:
        if device_arg:
            raise ValueError("--devices requested CUDA devices, but CUDA is unavailable")
        return [None]

    if device_arg:
        devices = [int(item.strip()) for item in device_arg.split(",")]
    else:
        devices = list(range(visible_count))
    if not devices:
        raise ValueError("--devices cannot be empty")
    invalid = [device for device in devices if not 0 <= device < visible_count]
    if invalid:
        raise ValueError(
            f"Invalid logical CUDA device(s) {invalid}; visible count is {visible_count}"
        )
    return devices


def run_nisqa_multiprocess(
        filenames: List[str],
        jsonl_path: Path,
        nisqa_dir: Path,
        model_path: Path,
        batch_size: int,
        chunk_size: int,
        num_workers: int,
        loader_workers: int,
        devices: List[Optional[int]],
        max_segments: Optional[int],
        ms_channel: Optional[int],
) -> None:
    """Distribute chunks among persistent GPU-bound worker processes."""
    worker_count = num_workers or len(devices)
    worker_devices = [devices[index % len(devices)] for index in range(worker_count)]
    if worker_count > len(devices):
        print(
            "Warning: multiple NISQA processes will share some GPUs; "
            "watch GPU memory usage.",
            flush=True,
        )

    task_queue = mp.Queue()
    result_queue = mp.Queue(maxsize=max(2, worker_count * 2))
    error_queue = mp.Queue()
    listener = mp.Process(
        target=result_listener,
        args=(result_queue, error_queue, str(jsonl_path), len(filenames)),
        name="nisqa-result-listener",
    )
    listener.start()

    workers = []
    for worker_id, device_id in enumerate(worker_devices):
        worker = mp.Process(
            target=compute_worker,
            args=(
                worker_id,
                device_id,
                task_queue,
                result_queue,
                str(nisqa_dir),
                str(model_path),
                batch_size,
                loader_workers,
                max_segments,
                ms_channel,
            ),
            name=f"nisqa-worker-{worker_id}",
        )
        worker.start()
        workers.append(worker)

    print(
        f"Running NISQA with {worker_count} inference processes on "
        f"devices {worker_devices}; loader workers per process: {loader_workers}",
        flush=True,
    )
    for start in range(0, len(filenames), chunk_size):
        task_queue.put(filenames[start:start + chunk_size])
    for _ in workers:
        task_queue.put(None)

    try:
        for worker in workers:
            worker.join()
    except KeyboardInterrupt:
        for worker in workers:
            worker.terminate()
        listener.terminate()
        for worker in workers:
            worker.join()
        listener.join()
        raise
    else:
        result_queue.put(("stop", None))
        listener.join()

    errors = []
    while True:
        try:
            errors.append(error_queue.get_nowait())
        except queue.Empty:
            break
    for worker in workers:
        if worker.exitcode:
            errors.append(f"{worker.name} exited with code {worker.exitcode}")
    listener_exitcode = listener.exitcode
    if listener_exitcode is not None and listener_exitcode != 0:
        errors.append(f"result listener exited with code {listener_exitcode}")
    if errors:
        raise RuntimeError("\n".join(errors))


def write_filtered_scp(
        output_path: Path,
        records: List[Tuple[str, str]],
        predictions: Dict[str, dict],
        mos_threshold: float,
) -> Tuple[int, int]:
    """Atomically write passing SCP lines. Missing scores fail closed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    missing = 0

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output_file:
            for original_line, filename in records:
                prediction = predictions.get(filename)
                if prediction is None:
                    missing += 1
                    continue
                if float(prediction["mos_pred"]) >= mos_threshold:
                    output_file.write(f"{original_line}\n")
                    kept += 1
        os.replace(temp_name, output_path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return kept, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official multidimensional NISQA model on a WAV SCP/list "
            "and keep entries whose predicted MOS reaches the threshold."
        )
    )
    parser.add_argument(
        "--input-scp",
        type=Path,
        default=Path("/usr/local/corpus/TidyVoice/dns_mos.scp"),
        help="Input WAV list or two-column Kaldi SCP.",
    )
    parser.add_argument(
        "--output-scp",
        type=Path,
        default=Path("/usr/local/corpus/TidyVoice/nisqa_mos.scp"),
        help="Filtered SCP output. Default: %(default)s",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("/usr/local/corpus/TidyVoice/nisqa_mos.jsonl"),
        help="Persistent NISQA prediction cache. Default: %(default)s",
    )
    parser.add_argument(
        "--nisqa-dir",
        type=Path,
        default=Path(os.environ.get("NISQA_DIR", "NISQA")),
        help="Official gabrielmittag/NISQA checkout (or set NISQA_DIR).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Multidimensional nisqa.tar; defaults to NISQA_DIR/weights/nisqa.tar.",
    )
    parser.add_argument(
        "--mos-threshold",
        type=float,
        default=3.0,
        help="Keep files with mos_pred >= this value. Default: %(default)s",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="NISQA prediction batch size. Reduce it if GPU memory is insufficient.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Number of persistent NISQA inference processes. Default: 0, "
            "meaning one process per selected/visible GPU."
        ),
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader CPU workers inside each inference process.",
    )
    parser.add_argument(
        "--devices",
        help=(
            "Comma-separated logical CUDA devices, for example 0,1,2. "
            "Defaults to all visible GPUs; use 'cpu' for CPU inference."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help=(
            "Files scored before results are appended to JSONL. "
            "Smaller chunks checkpoint more frequently. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        help=(
            "Override the model's ms_max_segments limit. NISQA uses about "
            "25 segments per second; for example 2500 supports about 100 seconds."
        ),
    )
    parser.add_argument(
        "--ms-channel",
        type=int,
        choices=(0, 1),
        help="Channel used for stereo audio. NISQA defaults to mono input.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1.0 <= args.mos_threshold <= 5.0:
        raise ValueError("--mos-threshold must be between 1.0 and 5.0")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.loader_workers < 0:
        raise ValueError("--loader-workers cannot be negative")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")
    if args.max_segments is not None and args.max_segments < 1:
        raise ValueError("--max-segments must be at least 1")
    if not args.input_scp.is_file():
        raise FileNotFoundError(f"Input SCP not found: {args.input_scp}")

    nisqa_dir = args.nisqa_dir.expanduser().resolve()
    model_path = (
        args.model.expanduser().resolve()
        if args.model
        else nisqa_dir / "weights" / "nisqa.tar"
    )
    run_predict = nisqa_dir / "run_predict.py"
    if not run_predict.is_file():
        raise FileNotFoundError(f"NISQA entry point not found: {run_predict}")
    if not model_path.is_file():
        raise FileNotFoundError(f"NISQA model not found: {model_path}")

    scp_records = read_scp(args.input_scp)
    filenames = unique_in_order(filename for _, filename in scp_records)
    predictions = load_jsonl(args.jsonl)
    pending = [filename for filename in filenames if filename not in predictions]

    print(
        f"Input: {len(scp_records)} lines, {len(filenames)} unique files; "
        f"cached: {len(filenames) - len(pending)}, pending: {len(pending)}"
    )

    if pending:
        devices = resolve_devices(args.devices)
        run_nisqa_multiprocess(
            pending,
            jsonl_path=args.jsonl,
            nisqa_dir=nisqa_dir,
            model_path=model_path,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            num_workers=args.num_workers,
            loader_workers=args.loader_workers,
            devices=devices,
            max_segments=args.max_segments,
            ms_channel=args.ms_channel,
        )
        predictions = load_jsonl(args.jsonl)

    kept, missing = write_filtered_scp(
        args.output_scp,
        scp_records,
        predictions,
        args.mos_threshold,
    )
    removed = len(scp_records) - kept
    print(
        f"Done: kept {kept}, removed {removed}, missing scores {missing}; "
        f"output: {args.output_scp}; scores: {args.jsonl}"
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
