## AISTATE Light — *Artificial Intelligence Speech‑To‑Analysis‑Translation Engine*
![Version](https://img.shields.io/badge/version-v2.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-yellow)

---

**AISTATE Light** is a transcription and diarization tool.  
#### Feedback / Support

If you have any issues, suggestions, or feature requests, please contact me at: **pawlict@proton.me**

---

## ✨ Main functionalities
- **Transcribe audio to text** using **Whisper** (`openai-whisper`).
- **Diarize speakers in audio** (who spoke when) using **pyannote.audio** (Hugging Face pipeline) + Whisper segments.
- **“Text diarization” (heuristics)** — a simple alternating / block labeling of lines or sentences (no ML), useful when you already have plain text.
- Show **live logs** inside the app (including worker/tqdm output from diarization and transcription when enabled).
## ✨ Updates v 2.0: Segment playback + transcription/diarization corrections

This update introduces a **segment-level review workflow**:

- **Segment playback (start–end):** you can play the audio for a selected segment directly in the app.
- **“Segment correction” panel:** edit the **transcription text**, adjust **diarization** (speaker assignment), and rename the **speaker label** for that segment.
- **Safe saving (no block merging):** saving edits preserves segment boundaries, preventing the previous issue where edited text could merge into the next segment and break playback mapping in subsequent edits.
- **Improved UX:** corrections are applied immediately to the segment list / editor without re-running the whole pipeline.

---
---

## Requirements
### System (Linux)
Install FFmpeg (used to convert audio to stable PCM WAV when needed):
```bash
sudo apt update -y
sudo apt install -y \
  python3 python3-venv python3-pip git \
  ffmpeg \
  libsndfile1 \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav
  libgl1 \
  libxkbcommon-x11-0 \
  libxcb-cursor0 \
  libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
  libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xinput0


```
### Python
Recommended: **Python 3.11+** (project is known to run on newer versions too, but PyTorch wheels may be easiest on 3.11).

---
### Hugging Face Token (pyannote)
- Voice diarization requires an HF token. Paste the token in the Settings tab
---
### Program installation
```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/pawlict/AISTATElight.git
cd AISTATElight

python3 -m venv .AISTATElight
source .AISTATElight/bin/activate

python -m pip install --upgrade pip wheel setuptools
pip install -U soundfile
pip install -r requirements.txt
```
---
### Run
```bash
python3 AISTATElight.py
```
---
### Troubleshooting
## Hugging Face token (pyannote diarization)

Voice diarization uses a Hugging Face pipeline (default tried first):
- `pyannote/speaker-diarization-community-1`

You typically need:
1. A Hugging Face account
2. A token (Settings tab → paste token)
3. Acceptance of the model’s terms (some models are gated)

> **Token storage:** the app stores settings (including HF token) in:  
> `~/.AISTATElight/backend/settings.json` (field: `hf_token`).

---

## Where “Text diarization” comes from (no Whisper / no pyannote)

The “Text diarization” button on the Home tab uses **simple heuristics** (alternating speakers / block assignment).  
It is implemented in:

- `backend/legacy_adapter.py` → `diarize_text_simple(...)`

This is **not** ML diarization — it just labels text units (lines/sentences) as `SPK1`, `SPK2`, etc.

---

## Project structure (important files)

- `AISTATElight.py` — application entry point (+ splash screen)
- `gui_pyside.py` — main window UI (tabs, actions, logging)
- `backend/legacy_adapter.py` — Whisper transcription + pyannote diarization + helper utilities
- `backend/voice_worker.py` — worker process for diarization (keeps GUI stable)
- `backend/transcribe_worker.py` — optional worker process for Whisper logs (if enabled)
- `backend/settings_store.py` — settings load/save (stores HF token)
- `ui/theme.py` — themes / palettes
- `ui/Info_pl.md`, `ui/Info_en.md` — Info tab content (Markdown)

---

## License
This project is released under the **AISTATElight License v1.2 (Source-Available)**: **Licence.md*

- ✅ Personal, educational, research, and **internal commercial use** allowed
- ✅ **Modifications allowed** for internal use
- ❌ No resale / no commercial redistribution
- ❌ No distribution of binaries/installers without permission
- ✅ Provided **“AS IS”** (no warranty)

Third-party license notices: see **THIRD_PARTY_NOTICES.md**.

