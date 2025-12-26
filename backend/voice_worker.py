from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when executed as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--lang", required=True)
    p.add_argument("--hf_token", required=True)
    args = p.parse_args()

    # Diagnostics: show which interpreter is running the worker
    print(f"WORKER sys.executable: {sys.executable}", file=sys.stderr, flush=True)
    print(f"WORKER sys.path[0]: {sys.path[0] if sys.path else ''}", file=sys.stderr, flush=True)

    # Heavy deps inside worker process
    from backend.legacy_adapter import diarize_voice_whisper_pyannote

    def log(line: str) -> None:
        print(line, file=sys.stderr, flush=True)

    def progress(pct: int) -> None:
        print(f"PROGRESS:{int(pct)}", file=sys.stderr, flush=True)

    res = diarize_voice_whisper_pyannote(
        args.audio, args.model, args.lang, args.hf_token,
        log_cb=log, progress_cb=progress
    )
    print(json.dumps(res, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
