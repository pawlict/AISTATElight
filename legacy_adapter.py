from __future__ import annotations

def _mask_token(tok: str) -> str:
    if not tok:
        return ""
    if len(tok) <= 8:
        return "*" * len(tok)
    return tok[:4] + "" + tok[-4:]

def _fmt_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def _load_pyannote_pipeline(Pipeline, hf_token: str, log_cb=None):
    """Load pyannote diarization pipeline in a version-compatible way.

    Tries multiple pipeline IDs because HF repo pointers / requirements evolve.
    Uses `token=` first (newer pyannote.audio) and falls back to legacy kwargs.
    """

    def log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    # Prefer newer/community pipelines first (more stable packaging).
    pipeline_ids = [
        "pyannote/speaker-diarization-community-1",
        "pyannote/speaker-diarization-3.1",
        "pyannote/speaker-diarization",
    ]

    kw_candidates = [
        ("token", hf_token),
        ("use_auth_token", hf_token),
        ("auth_token", hf_token),
        ("access_token", hf_token),
    ]

    last_exc = None

    log("pyannote: creating pipeline (compat loader)…")
    # Log versions to help debugging
    try:
        import pyannote.audio  # type: ignore
        log(f"pyannote.audio: {getattr(pyannote.audio, '__version__', 'unknown')}")
    except Exception:
        pass
    try:
        import huggingface_hub  # type: ignore
        log(f"huggingface_hub: {getattr(huggingface_hub, '__version__', 'unknown')}")
    except Exception:
        pass

    for pid in pipeline_ids:
        for kw, val in kw_candidates:
            try:
                log(f"pyannote: trying pipeline '{pid}' with {kw}=***")
                pipe = Pipeline.from_pretrained(pid, **{kw: val})
                log(f"pyannote: pipeline loaded OK: {pid} ({kw})")
                return pipe
            except TypeError as e:
                # kw not supported by this version
                last_exc = e
                continue
            except ValueError as e:
                # Common mismatch: revision handling / pipeline requirements
                last_exc = e
                msg = str(e)
                if "Revisions must be passed with `revision`" in msg:
                    log("pyannote: revision error detected for this pipeline/version combination.")
                    # Try next pipeline id
                    break
                # otherwise try next kw
                continue
            except Exception as e:
                last_exc = e
                # could be 401/403 gated or download errors
                continue

    # Last resort: try without token (works if user did `huggingface-cli login`)
    for pid in pipeline_ids:
        try:
            log(f"pyannote: trying pipeline '{pid}' without token (HF login/env)…")
            pipe = Pipeline.from_pretrained(pid)
            log(f"pyannote: pipeline loaded OK without token: {pid}")
            return pipe
        except Exception as e:
            last_exc = e
            continue

    log("pyannote: FAILED to load pipeline. Check: token, model access (gated), dependencies, or pyannote.audio version.")
    if last_exc:
        raise last_exc
    raise RuntimeError("pyannote pipeline load failed for unknown reason.")


def whisper_transcribe(audio_path: str, model: str, language: str, log_cb=None, progress_cb=None):
    if log_cb: log_cb(f"Whisper: load '{model}' (auto-download if missing)")
    if progress_cb: progress_cb(5)
    try:
        import whisper
    except Exception as e:
        raise RuntimeError("Missing 'openai-whisper'. Install: pip install openai-whisper") from e

    wmodel = whisper.load_model(model)
    if log_cb: log_cb("Whisper: model loaded. Transcribing")
    if progress_cb: progress_cb(20)

    lang = None if language == "auto" else language
    result = wmodel.transcribe(audio_path, language=lang, verbose=False)
    if progress_cb: progress_cb(90)

    text = (result.get("text") or "").strip()
    segments = result.get("segments") or []
    lines = []
    for seg in segments:
        s0 = float(seg.get("start", 0.0))
        s1 = float(seg.get("end", 0.0))
        t = (seg.get("text") or "").strip()
        if t:
            lines.append(f"[{_fmt_ts(s0)} - {_fmt_ts(s1)}] {t}")
    text_ts = "\n".join(lines).strip() if lines else text

    if progress_cb: progress_cb(100)
    if log_cb: log_cb(f"Whisper: done. segments={len(segments)}")
    return {"kind": "transcript", "text": text, "text_ts": text_ts}

def diarize_text_simple(text: str, speakers: int, method: str, log_cb=None, progress_cb=None):
    """Lightweight text diarization heuristics (no external deps required)."""
    def log(x: str) -> None:
        if log_cb:
            log_cb(x)

    if progress_cb: progress_cb(5)
    raw = (text or "").strip()
    if not raw:
        return {"kind": "diarized_text", "text": ""}

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return {"kind": "diarized_text", "text": ""}

    m = (method or "").lower()
    log(f"Text diarization: method='{method}', speakers={speakers}, lines={len(lines)}")

    def label(i: int) -> str:
        spk = (i % max(1, speakers)) + 1
        return f"SPK{spk}"

    # Keep existing tags if present
    if "keep" in m or "zachow" in m:
        out = []
        for ln in lines:
            if re.match(r"^(spk|speaker)\s*\d+[:\-]\s*", ln, re.I):
                out.append(ln)
            else:
                out.append(f"{label(len(out))}: {ln}")
        if progress_cb: progress_cb(100)
        return {"kind": "diarized_text", "text": "\n".join(out)}

    def split_sentences(text_line: str):
        parts = re.split(r"(?<=[\.\?\!])\s+", text_line.strip())
        return [p.strip() for p in parts if p.strip()]

    # Units: lines or sentences
    units = []
    if "sentence" in m or "zdani" in m:
        for ln in lines:
            units.extend(split_sentences(ln))
    else:
        units = list(lines)

    # Merge short units
    if "merge" in m or "łącz" in m:
        merged = []
        buf = ""
        for u in units:
            if len(buf) < 40:
                buf = (buf + " " + u).strip()
            else:
                merged.append(buf)
                buf = u
        if buf:
            merged.append(buf)
        units = merged
        log(f"Text diarization: merged units -> {len(units)}")

    out = []

    if ("naprzem" in m) or ("alternate" in m):
        for i, u in enumerate(units):
            out.append(f"{label(i)}: {u}")

    elif ("blok" in m) or ("block" in m):
        block = max(1, len(units) // max(1, speakers))
        spk = 1
        count = 0
        for u in units:
            out.append(f"SPK{spk}: {u}")
            count += 1
            if count >= block and spk < speakers:
                spk += 1
                count = 0

    elif "paragraph" in m or "akapit" in m:
        i = 0
        spk = 1
        while i < len(units):
            chunk = units[i:i+2]
            for u in chunk:
                out.append(f"SPK{spk}: {u}")
            spk = (spk % max(1, speakers)) + 1
            i += 2

    else:
        for i, u in enumerate(units):
            out.append(f"{label(i)}: {u}")

    if progress_cb: progress_cb(100)
    return {"kind": "diarized_text", "text": "\n".join(out)}


def diarize_voice_whisper_pyannote(
    audio_path: str,
    model: str,
    language: str,
    hf_token: str,
    log_cb=None,
    progress_cb=None,
):
    """Whisper transcription + pyannote speaker diarization (worker-safe).

    Compatible with:
      - pyannote.audio < 4.x: Pipeline(...) returns Annotation (has itertracks)
      - pyannote.audio >= 4.x: Pipeline(...) returns DiarizeOutput with
        .exclusive_speaker_diarization / .speaker_diarization (both are Annotation)
    """
    if log_cb:
        log_cb("Start: Whisper + pyannote")
        log_cb(f"HF token: {'OK' if hf_token else 'MISSING'} (hf_...)")

    # --- Whisper (segments) ---
    try:
        import whisper  # openai-whisper
    except Exception as e:
        raise RuntimeError("Missing 'openai-whisper'. Install: pip install openai-whisper") from e

    wmodel = whisper.load_model(model)
    lang = None if language == "auto" else language
    if log_cb:
        log_cb("Whisper: transcribe with segments")
    wres = wmodel.transcribe(audio_path, language=lang, verbose=False)
    segments = wres.get("segments") or []

    # --- pyannote ---
    from pyannote.audio import Pipeline

    if not hf_token:
        raise RuntimeError("HF token missing. Set it in the app settings.")

    if log_cb:
        log_cb("pyannote: load speaker-diarization pipeline (auto-download if missing)")
    pipeline = _load_pyannote_pipeline(Pipeline, hf_token, log_cb)

    if log_cb:
        log_cb("pyannote: diarizing file")
    diar = pipeline(audio_path)

    def get_annotation(diar_output):
        if hasattr(diar_output, "exclusive_speaker_diarization"):
            if log_cb:
                log_cb("pyannote: using output.exclusive_speaker_diarization")
            return diar_output.exclusive_speaker_diarization
        if hasattr(diar_output, "speaker_diarization"):
            if log_cb:
                log_cb("pyannote: using output.speaker_diarization")
            return diar_output.speaker_diarization
        if hasattr(diar_output, "itertracks"):
            if log_cb:
                log_cb("pyannote: using output (Annotation)")
            return diar_output
        raise RuntimeError(
            f"Unknown pyannote output type: {type(diar_output)}. Expected DiarizeOutput or Annotation."
        )

    annotation = get_annotation(diar)

    turns = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append((float(turn.start), float(turn.end), str(speaker)))

    if log_cb:
        log_cb(f"pyannote: found {len(turns)} speaker turns")

    def overlap(a0, a1, b0, b1):
        return max(0.0, min(a1, b1) - max(a0, b0))

    out_lines = []
    for seg in segments:
        s0 = float(seg.get("start", 0.0))
        s1 = float(seg.get("end", 0.0))
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue

        best_spk = "UNKNOWN"
        best_ov = 0.0
        for t0, t1, spk in turns:
            ov = overlap(s0, s1, t0, t1)
            if ov > best_ov:
                best_ov = ov
                best_spk = spk

        out_lines.append(f"[{s0:.2f}-{s1:.2f}] {best_spk}: {txt}")

    text = "\n".join(out_lines) if out_lines else (wres.get("text") or "").strip()
    return {"kind": "diarized_voice", "text": text, "ok": True}



def diarize_voice_whisper_pyannote_safe(
    audio_path: str,
    model: str,
    language: str,
    hf_token: str,
    log_cb=None,
    progress_cb=None,
):
    """Run Whisper+pyannote diarization in a separate process.

    Why:
      - Isolates native crashes (SIGSEGV) from GUI
      - Keeps stdout clean JSON (for UI) and streams stderr as logs/progress

    Returns a dict with keys:
      - kind: "diarized_voice"
      - text: diarized text OR failure message
      - ok: bool
    """
    import json
    import os
    import re
    import subprocess
    import sys

    worker_py = sys.executable  # always use the same venv/interpreter as GUI
    if log_cb:
        log_cb(f"pyannote(worker): using python: {worker_py}")

    cmd = [
        worker_py,
        "-m",
        "backend.voice_worker",
        "--audio",
        audio_path,
        "--model",
        model,
        "--lang",
        language,
        "--hf_token",
        hf_token,
    ]

    if log_cb:
        log_cb("pyannote(worker): starting separate process…")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),  # project root
            env=os.environ.copy(),
        )
        out, err = proc.communicate()

        # Stream stderr lines into GUI logs (+progress)
        if err and log_cb:
            for line in err.splitlines():
                if line.startswith("PROGRESS:"):
                    try:
                        pct = int(line.split(":", 1)[1].strip())
                        if progress_cb:
                            progress_cb(pct)
                    except Exception:
                        log_cb(line)
                else:
                    log_cb(line)

        # If worker failed, return a helpful error
        if proc.returncode != 0:
            tail = (err or "").strip().splitlines()[-10:]
            tail_txt = "\n".join(tail)
            fail_text = (
                "[Voice diarization failed]\n"
                "The worker process crashed or exited with an error.\n"
                f"Exit code: {proc.returncode}\n\n"
                f"Last logs:\n{tail_txt}\n"
            )
            return {"kind": "diarized_voice", "text": fail_text, "ok": False}

        # Worker succeeded but stdout might be empty or contain non-JSON (shouldn't happen, but guard)
        out_str = (out or "").strip()
        if not out_str:
            tail = (err or "").strip().splitlines()[-15:]
            tail_txt = "\n".join(tail)
            fail_text = (
                "[Voice diarization exception]\n"
                "Worker returned no JSON on stdout.\n\n"
                f"Last logs:\n{tail_txt}\n"
            )
            return {"kind": "diarized_voice", "text": fail_text, "ok": False}

        # Try parse JSON
        try:
            data = json.loads(out_str)
        except Exception:
            # Try to recover: find last JSON object in stdout (in case something polluted stdout)
            m = list(re.finditer(r"\{[\s\S]*\}\s*$", out_str))
            if m:
                try:
                    data = json.loads(m[-1].group(0))
                except Exception as e2:
                    raise e2
            else:
                raise

        if not isinstance(data, dict):
            return {"kind": "diarized_voice", "text": str(data), "ok": True}

        data.setdefault("kind", "diarized_voice")
        data.setdefault("ok", True)
        return data

    except Exception as e:
        if log_cb:
            log_cb("pyannote(worker): exception: " + str(e))
        fail_text = (
            "[Voice diarization exception]\n"
            f"{e}"
        )
        return {"kind": "diarized_voice", "text": fail_text, "ok": False}
