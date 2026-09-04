#!/usr/bin/env python3
# coding: utf-8
"""Filter frequency-analysis JSONL records into an audio SCP list."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def load_records(jsonl_path: Path):
    with jsonl_path.open("r", encoding="utf-8-sig") as jsonl_file:
        for line_no, raw_line in enumerate(jsonl_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad json at {jsonl_path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"bad item at {jsonl_path}:{line_no}: expected a JSON object"
                )
            yield line_no, record


def parse_estimated_freq(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        estimated_freq = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(estimated_freq) or estimated_freq < 0:
        return None
    return estimated_freq


def filter_freq_jsonl(
    jsonl_path: Path,
    output_path: Path,
    threshold: float,
    limit: int = 0,
) -> Counter:
    stats = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with tmp_path.open("w", encoding="utf-8") as output_file:
            for line_no, record in load_records(jsonl_path):
                if 0 < limit <= stats["files"]:
                    break
                stats["files"] += 1

                audio_path = record.get("audio_path")
                if not isinstance(audio_path, str) or not audio_path.strip():
                    stats["missing_audio_path"] += 1
                    print(
                        f"warning: skip {jsonl_path}:{line_no}: "
                        "missing or invalid audio_path",
                        flush=True,
                    )
                    continue

                if "estimated_freq" not in record:
                    stats["missing_estimated_freq"] += 1
                    print(
                        f"warning: skip {jsonl_path}:{line_no}: "
                        "missing estimated_freq",
                        flush=True,
                    )
                    continue

                estimated_freq = parse_estimated_freq(record["estimated_freq"])
                if estimated_freq is None:
                    stats["invalid_estimated_freq"] += 1
                    print(
                        f"warning: skip {jsonl_path}:{line_no}: "
                        f"invalid estimated_freq={record['estimated_freq']!r}",
                        flush=True,
                    )
                    continue

                if estimated_freq > threshold:
                    output_file.write(f"{audio_path.strip()}\n")
                    stats["selected"] += 1
                else:
                    stats["below_or_equal_threshold"] += 1

        tmp_path.replace(output_path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter frequency-analysis JSONL records and write audio_path values "
            "whose estimated_freq is greater than a threshold to an SCP file."
        )
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        required=True,
        help="Input frequency-analysis JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output SCP path; each line contains one audio_path.",
    )
    parser.add_argument(
        "--threshold",
        "--freq-threshold",
        dest="threshold",
        type=float,
        required=True,
        default=14000,
        help="Keep records with estimated_freq strictly greater than this value (Hz).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N non-empty records; 0 means all records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.jsonl.is_file():
        raise SystemExit(f"jsonl is not a file: {args.jsonl}")
    if not math.isfinite(args.threshold) or args.threshold < 0:
        raise SystemExit(f"threshold must be a finite non-negative number: {args.threshold}")
    if args.limit < 0:
        raise SystemExit(f"limit cannot be negative: {args.limit}")

    stats = filter_freq_jsonl(
        jsonl_path=args.jsonl,
        output_path=args.output,
        threshold=args.threshold,
        limit=args.limit,
    )
    print(
        f"done. files={stats['files']} selected={stats['selected']} "
        f"output={args.output}",
        flush=True,
    )
    for key in sorted(stats):
        if key not in {"files", "selected"} and stats[key]:
            print(f"{key}: {stats[key]}", flush=True)


if __name__ == "__main__":
    main()
