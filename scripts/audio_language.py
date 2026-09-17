"""Classify yt-dlp audio-only language metadata without guessing from subtitles."""


def language_rank(candidate_lang, requested_lang):
    if not candidate_lang:
        return -1
    candidate = str(candidate_lang).strip().lower().replace("_", "-")
    requested = str(requested_lang).strip().lower().replace("_", "-")
    aliases = {"iw": "he", "jw": "jv", "in": "id"}
    parts = candidate.split("-")
    parts[0] = aliases.get(parts[0], parts[0])
    candidate = "-".join(parts)
    parts = requested.split("-")
    parts[0] = aliases.get(parts[0], parts[0])
    requested = "-".join(parts)
    if candidate == requested:
        return 2
    if candidate.startswith(requested + "-"):
        return 1
    if requested.startswith(candidate + "-"):
        return 0
    return -1


def language_matches(candidate_lang, requested_lang):
    return language_rank(candidate_lang, requested_lang) >= 0


def audio_only_formats(info):
    return [
        fmt for fmt in (info.get("formats") or [])
        if fmt.get("vcodec") == "none" and fmt.get("acodec") not in (None, "none")
    ]


def audio_language_status(info, requested_lang):
    """match / mismatch / unknown. Unlabelled tracks never prove a mismatch."""
    formats = audio_only_formats(info)
    if any(language_matches(fmt.get("language"), requested_lang) for fmt in formats):
        return "match"
    if info.get("audio_formats_incomplete"):
        return "unknown"
    languages = [str(fmt.get("language") or "").strip().lower() for fmt in formats]
    if not languages or any(lang in {"", "und", "unknown", "mul", "zxx"} for lang in languages):
        return "unknown"
    return "mismatch"


def formats_may_be_incomplete(stderr):
    message = (stderr or "").lower()
    return any(fragment in message for fragment in
               ("formats have been skipped", "formats are being skipped", "missing a url"))


def audio_language_unknown_reasons(info):
    """Explain the metadata gaps logged when a result is classified as unknown."""
    reasons = []
    formats = audio_only_formats(info)
    if not formats:
        reasons.append("no_audio_only_formats")
    if info.get("audio_formats_incomplete"):
        reasons.append("incomplete_format_list")
    if any(str(fmt.get("language") or "").strip().lower() in
           {"", "und", "unknown", "mul", "zxx"} for fmt in formats):
        reasons.append("missing_or_ambiguous_language")
    return reasons


def is_unavailable_error(message):
    message = (message or "").lower()
    if "error: [youtube]" not in message:
        return False
    # These can be reported together with "Video unavailable"; keep them retryable.
    if any(fragment in message for fragment in (
        "not a bot", "sign in to confirm", "http error 429", "too many requests",
        "http error 403", "timed out", "timeout", "not available in your country",
        "not made this video available in your country",
    )):
        return False
    return any(fragment in message for fragment in (
        "video is no longer available", "video unavailable", "this video is unavailable",
        "this video is not available", "private video", "this video has been removed",
        "join this channel to get access", "this video requires payment to watch",
    ))
