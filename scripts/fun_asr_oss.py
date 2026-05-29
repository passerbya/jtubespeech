#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests


DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-e96c0ab3540e4cff8365973a1b7d8d23")

MODEL = "fun-asr"

API_URL_UPLOADS = "https://dashscope.aliyuncs.com/api/v1/uploads"
API_URL_SUBMIT = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
API_URL_QUERY_BASE = "https://dashscope.aliyuncs.com/api/v1/tasks/"


def get_upload_policy() -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    params = {
        "action": "getPolicy",
        "model": MODEL,
    }
    response = requests.get(API_URL_UPLOADS, headers=headers, params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"get upload policy failed: HTTP {response.status_code}\n{response.text}")

    data = response.json()
    policy = data.get("data")
    if not isinstance(policy, dict):
        raise RuntimeError(f"get upload policy response has no data: {json.dumps(data, ensure_ascii=False)}")
    return policy


def upload_to_dashscope_temp_oss(audio_path: Path) -> str:
    policy = get_upload_policy()
    file_name = audio_path.name
    key = f"{policy['upload_dir']}/{file_name}"

    with audio_path.open("rb") as audio_file:
        files = {
            "OSSAccessKeyId": (None, policy["oss_access_key_id"]),
            "Signature": (None, policy["signature"]),
            "policy": (None, policy["policy"]),
            "x-oss-object-acl": (None, policy["x_oss_object_acl"]),
            "x-oss-forbid-overwrite": (None, policy["x_oss_forbid_overwrite"]),
            "key": (None, key),
            "success_action_status": (None, "200"),
            "file": (file_name, audio_file),
        }
        response = requests.post(policy["upload_host"], files=files, timeout=300)

    if response.status_code != 200:
        raise RuntimeError(f"upload temp oss failed: HTTP {response.status_code}\n{response.text}")
    return f"oss://{key}"


def submit_transcription(file_url: str) -> str:
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
        "X-DashScope-OssResourceResolve": "enable",
    }
    payload = {
        "model": MODEL,
        "input": {
            "file_urls": [file_url],
        },
        "parameters": {
            "channel_id": [0],
            "enable_itn": False,
        },
    }

    response = requests.post(API_URL_SUBMIT, headers=headers, json=payload, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"submit failed: HTTP {response.status_code}\n{response.text}")

    data = response.json()
    task_id = data.get("output", {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"submit response has no task_id: {json.dumps(data, ensure_ascii=False)}")
    return task_id


def wait_for_result(task_id: str, verbose: bool = True) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}

    while True:
        time.sleep(2)
        response = requests.get(API_URL_QUERY_BASE + task_id, headers=headers, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"query failed: HTTP {response.status_code}\n{response.text}")

        data = response.json()
        status = str(data.get("output", {}).get("task_status", "")).upper()
        if verbose:
            print(f"status: {status or 'UNKNOWN'}", file=sys.stderr)

        if status == "SUCCEEDED":
            return data
        if status in {"FAILED", "UNKNOWN"}:
            raise RuntimeError(f"task failed: {json.dumps(data, ensure_ascii=False, indent=2)}")


def download_transcription(result: dict[str, Any]) -> dict[str, Any]:
    transcription_url = result.get("output", {}).get("results", [])[0].get("transcription_url")
    if not transcription_url:
        return result

    response = requests.get(transcription_url, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"download transcription failed: HTTP {response.status_code}\n{response.text}")

    response.encoding = "utf-8"
    return response.json()


def extract_plain_text(result: dict[str, Any]) -> str:
    transcripts = result.get("transcripts")
    if isinstance(transcripts, list):
        texts = []
        for transcript in transcripts:
            if not isinstance(transcript, dict):
                continue
            text = transcript.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        if texts:
            return "\n".join(texts)

    raise RuntimeError(f"no text field found: {json.dumps(result, ensure_ascii=False, indent=2)}")


def transcribe_file(audio_path: Path, verbose: bool = True) -> str:
    audio_path = Path(audio_path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    if verbose:
        print(f"upload temp oss: {audio_path}", file=sys.stderr)
    file_url = upload_to_dashscope_temp_oss(audio_path)
    if verbose:
        print(f"file_url: {file_url}", file=sys.stderr)

    task_id = submit_transcription(file_url)
    if verbose:
        print(f"task_id: {task_id}", file=sys.stderr)

    result = wait_for_result(task_id, verbose=verbose)
    transcription = download_transcription(result)
    return extract_plain_text(transcription)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload local audio to OSS, call fun-asr-flash-filetrans, and print plain text."
    )
    parser.add_argument("audio", help="local audio file path")
    args = parser.parse_args()
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    print(transcribe_file(audio_path, verbose=True))


if __name__ == "__main__":
    main()
