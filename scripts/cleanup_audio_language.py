#!/usr/bin/env python3
"""Clean one language on one disk; resume from empty and skip error/unknown."""
import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from queue import Empty, Queue

from audio_language import (
    audio_language_status, audio_language_unknown_reasons, audio_only_formats,
    formats_may_be_incomplete, is_unavailable_error,
)
from videoid_state import LANG_RE, VideoIdLog, load_video_ids, locked, video_id_path

CATEGORIES = ("wav_org", "wav", "wav16k", "flac", "txt", "vtt", "segs")
EXTENSIONS = {".flac", ".wav", ".mp3", ".m4a", ".webm", ".ogg", ".opus",
              ".aac", ".vtt", ".srt", ".txt", ".json", ".part", ".ytdl"}
SOURCE_RE = re.compile(r"^([A-Za-z0-9_-]{11})(?:\.[A-Za-z0-9_-]+)+$")
SEGMENT_RE = re.compile(r"^([A-Za-z0-9_-]{11})_[0-9]{4,}(?:\.[A-Za-z0-9_-]+)+$")


def artifact_videoid(category, name):
    if Path(name).suffix.lower() not in EXTENSIONS:
        return None
    match = (SEGMENT_RE if category == "segs" else SOURCE_RE).fullmatch(name)
    return match.group(1) if match else None


def dataset_root(value, lang):
    if LANG_RE.fullmatch(lang) is None:
        raise ValueError(f"Invalid language directory: {lang!r}")
    root = Path(value).expanduser().resolve(strict=True)
    for path in (root, root / "video", root / "video" / lang):
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"Expected a real dataset directory: {path}")
    return root


def walk_files(directory):
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_symlink():
                print(f"[SKIP SYMLINK] {entry.path}", flush=True)
            elif entry.is_dir(follow_symlinks=False):
                yield from walk_files(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                yield Path(entry.path)


def iter_video_batches(root, lang):
    """Collect one two-character bucket across all artifact categories at a time."""
    directories = {}
    buckets = set()
    flat = defaultdict(list)
    for category in CATEGORIES:
        directory = root / "video" / lang / category
        if directory.is_symlink():
            print(f"[SKIP SYMLINK] {directory}", flush=True)
            continue
        if not directory.is_dir():
            print(f"[DATA DIR MISSING] category={category} path={directory}", flush=True)
            continue
        print(f"[DATA DIR] category={category} path={directory}", flush=True)
        directories[category] = directory
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink():
                    print(f"[SKIP SYMLINK] {entry.path}", flush=True)
                elif entry.is_dir(follow_symlinks=False):
                    buckets.add(entry.name)
                elif entry.is_file(follow_symlinks=False):
                    vid = artifact_videoid(category, entry.name)
                    if vid:
                        flat[vid].append(Path(entry.path))
    if flat:
        yield flat
    del flat

    for bucket in sorted(buckets):
        files = defaultdict(list)
        for category, directory in directories.items():
            path = directory / bucket
            if path.is_symlink():
                print(f"[SKIP SYMLINK] {path}", flush=True)
                continue
            if not path.is_dir():
                continue
            for artifact in walk_files(path):
                vid = artifact_videoid(category, artifact.name)
                if vid:
                    files[vid].append(artifact)
        if files:
            yield files


def fetch_metadata(vid, args, proxy):
    try:
        with tempfile.TemporaryDirectory(prefix="jtubespeech-languages-") as tmp:
            command = [
                args.yt_dlp, "--ignore-config", "-J", "--skip-download", "--no-playlist",
                "--socket-timeout", str(args.socket_timeout), "--retries", "2",
                "--js-runtimes", "node", "--extractor-args", args.extractor_args,
            ]
            if proxy:
                command.extend(["--proxy", proxy if "://" in proxy else "http://" + proxy])
            if args.cookies:
                cookies = Path(tmp) / "cookies.txt"
                shutil.copyfile(args.cookies, cookies)
                command.extend(["--cookies", str(cookies)])
            command.append(f"https://www.youtube.com/watch?v={vid}")
            result = subprocess.run(command, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=args.timeout)
        if result.returncode:
            message = "\n".join(part for part in (result.stderr, result.stdout) if part)
            status = "unavailable" if is_unavailable_error(message) else "error"
            return vid, status, {}, message[-2000:]
        info = json.loads(result.stdout)
        if not isinstance(info, dict) or info.get("id") != vid:
            raise ValueError("yt-dlp returned the wrong video ID or a non-video object")
        # Return only the metadata needed for the decision and progress log.
        formats = audio_only_formats(info)
        evidence = {
            "id": vid,
            "audio_formats_incomplete": formats_may_be_incomplete(result.stderr),
            "formats": [
                {key: fmt.get(key) for key in ("format_id", "language", "vcodec", "acodec")}
                for fmt in formats
            ],
        }
        return vid, "ok", evidence, None
    except (OSError, ValueError, TypeError, AttributeError, subprocess.TimeoutExpired) as exc:
        return vid, "error", {}, str(exc)



def safe_path(root_text, relative_path, lang, vid):
    root = Path(root_text)
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in relative_path:
        raise ValueError(f"Unsafe relative path: {relative_path}")
    if (len(relative.parts) < 4 or relative.parts[:2] != ("video", lang)
            or relative.parts[2] not in CATEGORIES
            or artifact_videoid(relative.parts[2], relative.name) != vid):
        raise ValueError(f"Path does not belong to this VID: {relative_path}")
    path = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Refusing symlink: {current}")
    if path.resolve() != path or root not in path.resolve().parents:
        raise ValueError(f"Path escapes dataset root: {path}")
    return path



def delete_video_files(root, lang, vid, paths, by_category=None, verbose=False):
    """Delete exact artifacts for one VID; leave failed files for the next run."""
    deleted = size = failures = 0
    for path in dict.fromkeys(paths):
        try:
            relative = path.relative_to(root).as_posix()
            path = safe_path(root, relative, lang, vid)
            current = path.lstat()
            if not stat.S_ISREG(current.st_mode):
                raise ValueError(f"Not a regular file: {path}")
            path.unlink()
            deleted += 1
            size += current.st_size
            if by_category is not None:
                by_category[Path(relative).parts[2]] += 1
            if verbose:
                print(f"[UNLINK] vid={vid} bytes={current.st_size} path={path}", flush=True)
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            failures += 1
            print(f"[DELETE FAILED] vid={vid} path={path}: {exc}", flush=True)
    return deleted, size, failures


def persist_pending_video_ids(queue, output, videoids):
    """Drain outcome queues in the main thread and fsync each new exclusion."""
    while True:
        try:
            vid = queue.get_nowait()
        except Empty:
            return
        output.append(vid)
        videoids.add(vid)


def clean_language(root, args, empty_fn):
    # Existing empty entries authorize deletion; error/unknown entries only skip queries.
    videoid_dir = empty_fn.parent.parent
    error_fn = video_id_path(videoid_dir, "error", args.lang)
    unknown_fn = video_id_path(videoid_dir, "unknown", args.lang)
    empty_vids = load_video_ids(empty_fn)
    error_vids = load_video_ids(error_fn)
    unknown_vids = load_video_ids(unknown_fn)
    error_queue = Queue()
    unknown_queue = Queue()
    queried_vids = set()
    deleted_by_vid = Counter()
    deleted_by_category = Counter()
    stats = Counter()
    proxies = args.proxy or [None]
    print(f"[START] root={root} lang={args.lang} empty={empty_fn} "
          f"existing_empty={len(empty_vids)} error={error_fn} existing_error={len(error_vids)} "
          f"unknown={unknown_fn} existing_unknown={len(unknown_vids)}; deletion is immediate",
          flush=True)

    def delete(vid, paths, reason):
        by_category = Counter()
        unique_paths = list(dict.fromkeys(paths))
        deleted, size, failures = delete_video_files(
            root, args.lang, vid, unique_paths, by_category, args.verbose)
        stats["deleted_files"] += deleted
        stats["deleted_bytes"] += size
        stats["delete_errors"] += failures
        deleted_by_vid[vid] += deleted
        deleted_by_category.update(by_category)
        detail = ",".join(f"{category}:{by_category[category]}" for category in CATEGORIES)
        print(f"[{reason}] lang={args.lang} vid={vid} matched={len(unique_paths)} "
              f"deleted={deleted} vid_deleted_total={deleted_by_vid[vid]} "
              f"bytes={size} failed={failures} by_dir={detail}", flush=True)

    with VideoIdLog(empty_fn) as output, VideoIdLog(error_fn) as error_output, \
            VideoIdLog(unknown_fn) as unknown_output, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        def consume(future, paths):
            vid, status, info, error = future.result()
            if status != "ok":
                if status == "unavailable" or is_unavailable_error(error):
                    error_queue.put(vid)
                    persist_pending_video_ids(error_queue, error_output, error_vids)
                    stats["unavailable"] += 1
                    print(f"[KEEP / UNAVAILABLE -> ERROR] vid={vid} saved={error_fn}: {error}",
                          flush=True)
                else:
                    stats["query_error"] += 1
                    print(f"[KEEP / QUERY ERROR / RETRY] {vid}: {error}", flush=True)
                return
            decision = audio_language_status(info, args.lang)
            stats[decision] += 1
            if decision == "mismatch":
                languages = sorted({str(fmt.get("language")) for fmt in info["formats"]})
                print(f"[MISMATCH] vid={vid} target={args.lang} audio={languages}", flush=True)
                # Persist before any unlink: interruption/partial failure resumes from empty.
                output.append(vid)
                empty_vids.add(vid)
                delete(vid, paths, "DELETED")
            elif decision == "unknown":
                unknown_queue.put(vid)
                persist_pending_video_ids(unknown_queue, unknown_output, unknown_vids)
                reasons = ",".join(audio_language_unknown_reasons(info))
                languages = sorted({str(fmt.get("language") or "<missing>")
                                    for fmt in audio_only_formats(info)})
                print(f"[KEEP / UNKNOWN AUDIO] vid={vid} target={args.lang} "
                      f"reason={reasons} audio={languages} saved={unknown_fn}", flush=True)

        for files in iter_video_batches(root, args.lang):
            # empty wins if a VID also appears in an older error/unknown list.
            for vid in list(files):
                if vid in empty_vids:
                    stats["resumed_groups"] += 1
                    delete(vid, files.pop(vid), "EMPTY / RESUME")
                elif vid in error_vids or vid in unknown_vids:
                    category = "error" if vid in error_vids else "unknown"
                    paths = files.pop(vid)
                    stats["skipped_" + category] += 1
                    if args.verbose:
                        print(f"[SKIP / {category.upper()}] vid={vid} kept_files={len(paths)}",
                              flush=True)

            pending = {}

            def drain(block):
                if not pending:
                    return
                done, _ = wait(pending, timeout=None if block else 0,
                               return_when=FIRST_COMPLETED)
                for future in done:
                    consume(future, pending.pop(future))

            for vid, paths in files.items():
                drain(False)
                if vid in queried_vids:
                    continue
                if args.limit and stats["queried"] >= args.limit:
                    continue
                proxy = proxies[stats["queried"] % len(proxies)]
                pending[pool.submit(fetch_metadata, vid, args, proxy)] = paths
                queried_vids.add(vid)
                stats["queried"] += 1
                if len(pending) >= args.workers:
                    drain(True)
                if stats["queried"] % 100 == 0:
                    print(f"[PROGRESS] queried={stats['queried']} "
                          f"deleted_files={stats['deleted_files']} "
                          f"deleted_GiB={stats['deleted_bytes'] / 2**30:.3f} "
                          f"saved_error={stats['unavailable']} saved_unknown={stats['unknown']} "
                          f"skipped_error={stats['skipped_error']} "
                          f"skipped_unknown={stats['skipped_unknown']}", flush=True)
            while pending:
                drain(True)

    detail = ",".join(f"{category}:{deleted_by_category[category]}" for category in CATEGORIES)
    print(f"[DONE] root={root} lang={args.lang} queried={stats['queried']} "
          f"match={stats['match']} mismatch={stats['mismatch']} unknown={stats['unknown']} "
          f"saved_error={stats['unavailable']} skipped_error={stats['skipped_error']} "
          f"skipped_unknown={stats['skipped_unknown']} "
          f"query_errors={stats['query_error']} resumed_groups={stats['resumed_groups']} "
          f"deleted_files={stats['deleted_files']} "
          f"deleted_GiB={stats['deleted_bytes'] / 2**30:.3f} "
          f"delete_errors={stats['delete_errors']} by_dir={detail}", flush=True)
    return 1 if stats["delete_errors"] or stats["query_error"] else 0


class SingleValue(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} may only be specified once")
        setattr(namespace, self.dest, values)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action=SingleValue, required=True,
                        help="One jtubespeech directory containing video/")
    parser.add_argument("--lang", action=SingleValue, required=True,
                        help="One language directory, for example ja")
    parser.add_argument("--videoid-dir", type=Path, action=SingleValue,
                        help="Persistent video ID lists; default ROOT/videoid")
    parser.add_argument("--proxy", action="append", help="Proxy URL or host:port; repeat to distribute requests")
    parser.add_argument("--cookies", type=Path, help="yt-dlp cookies file; each request uses a temporary copy")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--verbose", action="store_true",
                        help="Print each deleted file path and skipped error/unknown VID")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--socket-timeout", type=float, default=20)
    parser.add_argument("--extractor-args", default="youtube:player_client=mweb")
    parser.add_argument("--yt-dlp", default="yt-dlp")
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximum NEW metadata queries; existing empty entries are still cleaned; 0 means all")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.timeout <= 0 or args.socket_timeout <= 0 or args.limit < 0:
        parser.error("workers/timeouts must be positive; limit must be non-negative")
    if args.cookies and not args.cookies.is_file():
        parser.error(f"Cookies file not found: {args.cookies}")
    if LANG_RE.fullmatch(args.lang) is None:
        parser.error("Invalid language directory")
    return args


def main(argv=None):
    args = parse_args(argv)
    root = dataset_root(args.root, args.lang)
    videoid_dir = (args.videoid_dir or root / "videoid").expanduser().resolve()
    empty_fn = video_id_path(videoid_dir, "empty", args.lang)
    empty_fn.parent.mkdir(parents=True, exist_ok=True)
    # Coordinate cleanup runs without a database or separate progress manifest.
    with Path(str(empty_fn) + ".cleanup.lock").open("a+") as lock, locked(lock):
        return clean_language(root, args, empty_fn)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
