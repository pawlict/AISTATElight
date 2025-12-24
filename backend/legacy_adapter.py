from __future__ import annotations

import re
import tempfile
import subprocess


def _mask_token(tok: str) -> str:
    if not tok:
        return ""
    if len(tok) <= 8:
        return "*" * len(tok)
    return tok[:4] + "…" + tok[-4:]

def _fmt_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _convert_to_pcm_wav_16k_mono(src_path: str, log_cb=None) -> str:
    """Convert any audio to stable PCM WAV 16kHz mono (ffmpeg required).

    Some formats (m4a/mp3/VBR) can cause slight sample-count mismatches when
    pyannote crops by time. Converting to PCM avoids this.
    Returns path to a temporary wav file; caller should delete it.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out_path = tmp.name

    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        out_path,
    ]

    if log_cb:
        log_cb("ffmpeg: converting input to PCM WAV 16kHz mono (stabilize diarization)…")

    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found. Install it (e.g., sudo apt install ffmpeg).") from e

    if p.returncode != 0:
        tail = (p.stderr or "")[-1200:]
        raise RuntimeError(f"ffmpeg conversion failed (code={p.returncode}). stderr tail:\n{tail}")

    return out_path

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


def whisper_transcribe_safe(
    audio_path: str,
    model: str,
    language: str,
    log_cb=None,
    progress_cb=None,
):
    """Run Whisper transcription in a separate process and stream ALL stderr into GUI logs.

    This captures:
      - openai-whisper warnings (e.g. FP16 -> FP32 on CPU)
      - language detection lines
      - tqdm progress bar (\r updates)

    Stdout is reserved strictly for JSON result.
    """

    import json
    import os
    import subprocess
    import sys
    import threading
    import queue
    import re

    worker_py = sys.executable  # use same venv/interpreter as GUI
    if log_cb:
        log_cb(f"whisper(worker): using python: {worker_py}")

    cmd = [
        worker_py,
        "-u",  # unbuffered IO
        "-m",
        "backend.transcribe_worker",
        "--audio",
        audio_path,
        "--model",
        model,
        "--lang",
        language,
    ]

    if log_cb:
        log_cb("whisper(worker): starting separate process…")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,  # IMPORTANT: binary mode for live tqdm (\r) streaming
        bufsize=0,
        cwd=os.path.dirname(os.path.dirname(__file__)),  # project root
        env={**os.environ.copy(), "PYTHONUNBUFFERED": "1"},
    )

    stderr_q: "queue.Queue[str]" = queue.Queue()
    stderr_all: list[str] = []

    def _push_line(line: str) -> None:
        if not line:
            return
        stderr_all.append(line)
        if not log_cb:
            return

        # Optional progress hook: allow worker to emit explicit progress lines
        if line.startswith("PROGRESS:"):
            try:
                pct = int(line.split(":", 1)[1].strip())
                if progress_cb:
                    progress_cb(pct)
            except Exception:
                log_cb(line)
            return

        log_cb(line)

    def _stderr_reader(p: subprocess.Popen) -> None:
        # Read raw bytes; tqdm prints with '\r' and without '\n'
        while True:
            chunk = p.stderr.read(1024)  # type: ignore[union-attr]
            if not chunk:
                break
            txt = chunk.decode("utf-8", errors="replace")
            txt = txt.replace("\r", "\n")
            for ln in txt.splitlines():
                ln = ln.strip("\n")
                if ln.strip():
                    stderr_q.put(ln)

    t = threading.Thread(target=_stderr_reader, args=(proc,), daemon=True)
    t.start()

    # Pump stderr LIVE while process runs
    while proc.poll() is None:
        try:
            ln = stderr_q.get(timeout=0.2)
            _push_line(ln)
        except queue.Empty:
            pass

    # Flush remaining stderr lines
    while True:
        try:
            ln = stderr_q.get_nowait()
            _push_line(ln)
        except queue.Empty:
            break

    # Read stdout JSON after worker ends
    out_bytes = b""
    if proc.stdout:
        out_bytes = proc.stdout.read() or b""
    out = out_bytes.decode("utf-8", errors="replace")
    err = "\n".join(stderr_all)

    if proc.returncode != 0:
        tail_txt = "\n".join((err or "").strip().splitlines()[-12:])
        fail_text = (
            "[Transcription failed]\n"
            "The worker process crashed or exited with an error.\n"
            f"Exit code: {proc.returncode}\n\n"
            f"Last logs:\n{tail_txt}\n"
        )
        return {"kind": "transcript", "text": fail_text, "text_ts": "", "ok": False}

    out_str = (out or "").strip()
    if not out_str:
        tail_txt = "\n".join((err or "").strip().splitlines()[-20:])
        fail_text = (
            "[Transcription exception]\n"
            "Worker returned no JSON on stdout.\n\n"
            f"Last logs:\n{tail_txt}\n"
        )
        return {"kind": "transcript", "text": fail_text, "text_ts": "", "ok": False}

    # Parse JSON (guard against any accidental pollution)
    try:
        data = json.loads(out_str)
    except Exception:
        m = list(re.finditer(r"\{[\s\S]*\}\s*$", out_str))
        if m:
            data = json.loads(m[-1].group(0))
        else:
            raise

    if not isinstance(data, dict):
        return {"kind": "transcript", "text": str(data), "text_ts": "", "ok": True}

    data.setdefault("kind", "transcript")
    data.setdefault("ok", True)
    data.setdefault("text", "")
    data.setdefault("text_ts", data.get("text", ""))
    return data

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

    IMPORTANT (stability):
      - We **do not** pass file paths directly into pyannote when possible.
        On some systems pyannote >= 4.x may try to use torchcodec/AudioDecoder
        for metadata/decoding and crash with:
          NameError: name 'AudioDecoder' is not defined
      - Instead we convert the input to PCM WAV (16kHz mono) and feed an
        in-memory waveform dict: {"waveform": Tensor, "sample_rate": int}.

    Compatible with:
      - pyannote.audio < 4.x: Pipeline(...) returns Annotation (has itertracks)
      - pyannote.audio >= 4.x: Pipeline(...) returns DiarizeOutput with
        .exclusive_speaker_diarization / .speaker_diarization (both are Annotation)
    """
    import os

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
    if progress_cb:
        progress_cb(5)

    wres = wmodel.transcribe(audio_path, language=lang, verbose=False)
    segments = wres.get("segments") or []

    if progress_cb:
        progress_cb(35)

    # --- pyannote ---
    from pyannote.audio import Pipeline

    if not hf_token:
        raise RuntimeError("HF token missing. Set it in the app settings.")

    if log_cb:
        log_cb("pyannote: load speaker-diarization pipeline (auto-download if missing)")
    pipeline = _load_pyannote_pipeline(Pipeline, hf_token, log_cb)

    if progress_cb:
        progress_cb(45)

    if log_cb:
        log_cb("pyannote: diarizing file (PCM WAV -> in-memory waveform)")

    # Always stabilize input to avoid sample-count mismatch and pyannote internal decoding.
    wav_path = _convert_to_pcm_wav_16k_mono(audio_path, log_cb=log_cb)
    try:
        try:
            import torch
            import soundfile as sf
        except Exception as e:
            raise RuntimeError(
                "Missing deps for robust pyannote audio loading. "
                "Install: pip install soundfile (and ensure torch is installed)."
            ) from e

        audio_np, sr = sf.read(wav_path, dtype="float32", always_2d=True)  # (time, channels)
        waveform = torch.from_numpy(audio_np.T)  # -> (channels, time)

        file_dict = {
            "waveform": waveform,
            "sample_rate": int(sr),
            "uri": os.path.basename(audio_path),
            "duration": float(waveform.shape[1]) / float(sr) if int(sr) > 0 else None,
        }

        diar = pipeline(file_dict)

    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass

    if progress_cb:
        progress_cb(80)

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

    # Join diarized segments into final text.
    text = "\n".join(out_lines) if out_lines else (wres.get("text") or "").strip()

    if progress_cb:
        progress_cb(100)

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

    - Streams worker stderr LIVE into GUI logs (so you see progress)
    - Keeps stdout for JSON result (read after process ends)
    """
    import json
    import os
    import re
    import subprocess
    import sys
    import threading
    import queue

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
            text=False,  # IMPORTANT: binary mode for live tqdm (\r) streaming
            bufsize=0,
            cwd=os.path.dirname(os.path.dirname(__file__)),  # project root
            env=os.environ.copy(),
        )

        stderr_q: "queue.Queue[str]" = queue.Queue()
        stderr_all: list[str] = []

        def _push_line(line: str) -> None:
            if not line:
                return
            stderr_all.append(line)
            if not log_cb:
                return
            if line.startswith("PROGRESS:"):
                try:
                    pct = int(line.split(":", 1)[1].strip())
                    if progress_cb:
                        progress_cb(pct)
                except Exception:
                    log_cb(line)
            else:
                log_cb(line)

        def _stderr_reader(p: subprocess.Popen) -> None:
            # Read raw bytes; tqdm prints with '\r' (carriage return) and no '\n'
            while True:
                chunk = p.stderr.read(1024)  # type: ignore[union-attr]
                if not chunk:
                    break
                txt = chunk.decode("utf-8", errors="replace")
                # Make tqdm lines visible in GUI
                txt = txt.replace("\r", "\n")
                for ln in txt.splitlines():
                    if ln.strip():
                        stderr_q.put(ln)

        t = threading.Thread(target=_stderr_reader, args=(proc,), daemon=True)
        t.start()

        # Pump stderr lines LIVE while process runs
        while proc.poll() is None:
            try:
                ln = stderr_q.get(timeout=0.2)
                _push_line(ln)
            except queue.Empty:
                pass

        # Flush remaining stderr lines
        while True:
            try:
                ln = stderr_q.get_nowait()
                _push_line(ln)
            except queue.Empty:
                break

        # Now read stdout (JSON) after worker ends
        out_bytes = b""
        if proc.stdout:
            out_bytes = proc.stdout.read() or b""
        out = out_bytes.decode("utf-8", errors="replace")
        err = "\n".join(stderr_all)

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

        # Worker succeeded but stdout might be empty or contain non-JSON
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
                data = json.loads(m[-1].group(0))
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
        fail_text = "[Voice diarization exception]\n" + str(e)
        return {"kind": "diarized_voice", "text": fail_text, "ok": False}
