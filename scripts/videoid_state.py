"""Append-only, durable video ID lists used by the cleanup tool."""
import os
import re
from contextlib import contextmanager
from pathlib import Path

VIDEO_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")
LANG_RE = re.compile(r"[A-Za-z]{2,8}(?:[-_][A-Za-z0-9]{1,8})*")


def validate_video_id(videoid):
    if not isinstance(videoid, str) or VIDEO_ID_RE.fullmatch(videoid) is None:
        raise ValueError(f"Invalid YouTube video ID: {videoid!r}")
    return videoid


def video_id_path(videoid_dir, category, lang):
    if LANG_RE.fullmatch(lang) is None or category not in {"empty", "error", "unknown", "exceed_limit"}:
        raise ValueError("Invalid video ID list category/language")
    return Path(videoid_dir) / category / f"{lang}wiki-latest-pages-articles-multistream-index.txt"


def load_video_ids(path):
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            return {line.strip() for line in stream if line.strip()}
    except FileNotFoundError:
        return set()


@contextmanager
def locked(stream):
    # flock also coordinates writers on Linux NFS mounts with locking enabled.
    if os.name == "nt":
        import msvcrt
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class VideoIdLog:
    """Append under a lock; ingest only the new tail when another writer appends."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+", encoding="utf-8", newline="\n")
        self.seen = set()
        self.offset = 0
        self.needs_newline = False

    def append(self, videoid):
        validate_video_id(videoid)
        with locked(self.stream):
            self.stream.seek(self.offset)
            tail = self.stream.read()
            if tail:
                self.needs_newline = not tail.endswith("\n")
            self.seen.update(line.strip() for line in tail.splitlines() if line.strip())
            self.offset = self.stream.tell()
            if videoid in self.seen:
                return False
            # Also support a pre-existing list whose last line lacks a newline.
            if self.needs_newline:
                self.stream.write("\n")
            self.stream.write(videoid + "\n")
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.offset = self.stream.tell()
            self.needs_newline = False
            self.seen.add(videoid)
            return True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()


def save_video_id_queue(path, queue):
    with VideoIdLog(path) as output:
        for videoid in iter(queue.get, "STOP"):
            if videoid:
                output.append(videoid)
