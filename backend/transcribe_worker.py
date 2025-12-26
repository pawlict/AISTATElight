from __future__ import annotations

import argparse
import json
import os
import sys


def _stderr(msg: str) -> None:
    # Everything except final JSON must go to stderr.
    print(msg, file=sys.stderr, flush=True)


def _progress(pct: int) -> None:
    # Optional hook for UI progress parsing.
    print(f"PROGRESS: {pct}", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lang", required=True)
    args = parser.parse_args()

    _stderr(f"WORKER sys.executable: {sys.executable}")
    _stderr(f"WORKER sys.path[0]: {os.getcwd()}")

    # Import here to avoid GUI importing heavy deps at startup.
    from backend.legacy_adapter import whisper_transcribe

    res = whisper_transcribe(
        args.audio,
        args.model,
        args.lang,
        log_cb=_stderr,
        progress_cb=_progress,
    )

    # JSON-only on stdout
    print(json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
