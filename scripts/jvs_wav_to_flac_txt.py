#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf


def discover_tasks(jvs_root: Path) -> List[Path]:
    tasks: List[Path] = []
    for transcript_path in sorted(jvs_root.glob("jvs*/**/transcripts_utf8.txt")):
        task_dir = transcript_path.parent
        if (task_dir / "wav24kHz16bit").is_dir():
            tasks.append(task_dir)
    return tasks


def load_transcripts(path: Path) -> Dict[str, str]:
    transcripts: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            if ":" not in line:
                print(f"[skip] {path}:{line_no} invalid transcript line")
                continue
            utt_id, text = line.split(":", 1)
            utt_id = utt_id.strip()
            text = text.strip()
            if utt_id:
                transcripts[utt_id] = text
    return transcripts


def flac_subtype_from_wav(wav_path: Path) -> str:
    subtype = sf.info(wav_path).subtype
    if subtype in {"PCM_16", "PCM_24"}:
        return subtype
    return "PCM_16"


def write_mono_flac(wav_path: Path, flac_path: Path) -> None:
    audio, sample_rate = sf.read(wav_path, always_2d=True)
    if audio.shape[1] == 1:
        mono = audio[:, 0]
    else:
        mono = np.mean(audio, axis=1)

    flac_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        flac_path,
        mono,
        sample_rate,
        format="FLAC",
        subtype=flac_subtype_from_wav(wav_path),
    )


def process_task(
    task_dir: Path,
    output_dir_name: str,
    overwrite: bool,
    dry_run: bool,
) -> Tuple[int, int, int, int]:
    transcript_path = task_dir / "transcripts_utf8.txt"
    wav_dir = task_dir / "wav24kHz16bit"
    output_dir = task_dir / output_dir_name

    transcripts = load_transcripts(transcript_path)
    written = 0
    skipped = 0
    missing_wav = 0
    failed = 0

    for utt_id, text in transcripts.items():
        wav_path = wav_dir / f"{utt_id}.wav"
        flac_path = output_dir / f"{utt_id}.flac"
        txt_path = output_dir / f"{utt_id}.txt"

        if not wav_path.is_file():
            missing_wav += 1
            print(f"[missing-wav] {wav_path}")
            continue

        if flac_path.exists() and txt_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            if not dry_run:
                write_mono_flac(wav_path, flac_path)
                txt_path.write_text(text, encoding="utf-8")
            written += 1
        except Exception as exc:
            failed += 1
            print(f"[failed] {wav_path}: {exc}")

    rel = task_dir
    print(
        f"[done] {rel}: written={written}, skipped={skipped}, "
        f"missing_wav={missing_wav}, failed={failed}"
    )
    return written, skipped, missing_wav, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert JVS wav files to mono flac and write same-name txt transcripts."
    )
    parser.add_argument(
        "jvs_root",
        type=Path,
        nargs="?",
        default=Path("/usr/local/corpus/ja/jvs_ver1"),
        help="JVS root directory.",
    )
    parser.add_argument(
        "--output-dir-name",
        default="flac",
        help="Output directory name inside each JVS task directory. Default: flac",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing flac/txt files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count files; do not write anything.",
    )
    args = parser.parse_args()

    jvs_root = args.jvs_root.resolve()
    if not jvs_root.is_dir():
        raise SystemExit(f"JVS root not found: {jvs_root}")

    tasks = discover_tasks(jvs_root)
    if not tasks:
        raise SystemExit(f"no JVS task dirs found under: {jvs_root}")

    print(f"[tasks] {len(tasks)}")

    total_written = 0
    total_skipped = 0
    total_missing_wav = 0
    total_failed = 0

    for task_dir in tasks:
        written, skipped, missing_wav, failed = process_task(
            task_dir,
            output_dir_name=args.output_dir_name,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        total_written += written
        total_skipped += skipped
        total_missing_wav += missing_wav
        total_failed += failed

    action = "would_write" if args.dry_run else "written"
    print(
        f"[summary] {action}={total_written}, skipped={total_skipped}, "
        f"missing_wav={total_missing_wav}, failed={total_failed}"
    )


if __name__ == "__main__":
    main()
