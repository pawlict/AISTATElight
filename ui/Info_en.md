## {{APP_NAME}} – Artificial Intelligence Speech-To-Analysis-Translation Engine ({{APP_VERSION}})

**Author:** pawlict
---
(Contact / support)
If you encounter bugs, technical issues, have improvement suggestions, or ideas for new features — please contact the author at: pawlict@proton.me
<p align="center">
  <img src="assets/logo.png" alt="Logo" width="280" />
</p>

---
## What this app does

### Speech-to-text (AI)
- **Whisper (openai-whisper)** is used to transcribe audio into text.  
  Models are **auto-downloaded on first use** (e.g., tiny/base/small/medium/large).

### Voice diarization (AI)
- **pyannote.audio** is used to identify “who spoke when” (speaker diarization) using a Hugging Face pipeline
  (e.g., pyannote/speaker-diarization-community-1).  
  This may require:
  - a valid **Hugging Face token**,
  - accepting model-specific terms (“gated” repositories),
  - compliance with the license/terms on the model card.

### Text “diarization” (non-AI / heuristic)
- The **Text diarization** options (e.g., alternating speakers / block speakers) do **not** use AI.
- They work purely on an existing transcript (plain text) and assign labels like **SPK1, SPK2, …**
  using simple rules such as:
  - line/sentence splitting,
  - alternating speaker assignment,
  - block grouping,
  - optional merging of short fragments.

> This is formatting/structuring of text, not real speaker recognition from audio.

---

## Third-party libraries & components

> Note: many packages are installed as transitive dependencies (dependencies of dependencies).
> The list below focuses on the main building blocks used by the app.

### GUI
- **PySide6 (Qt for Python)** — application GUI (tabs, widgets, dialogs, QTextBrowser).  
  License: **LGPL** (Qt for Python licensing).

### Speech / diarization
- **openai-whisper** — speech-to-text transcription (Whisper).  
- **pyannote.audio** — speaker diarization pipeline.  
- **huggingface_hub** — downloads models/pipelines from Hugging Face Hub.  

### Core ML / audio stack (typically installed as dependencies)
- **torch** (PyTorch) — neural network runtime used by Whisper / pyannote.
- **torchaudio** — audio utilities for PyTorch (commonly required by pyannote).
- **numpy** — numeric computations (common dependency).
- **tqdm** — progress bars (visible in logs during transcription).
- **soundfile / librosa** — audio I/O / utilities (commonly used in audio pipelines; depends on your environment).

### Audio conversion (external tool)
- **FFmpeg** — used to convert audio to stable **PCM WAV** (e.g., 16kHz mono) when needed.  
  License: depends on distribution/build (LGPL/GPL variants).

---

## AI models (weights) & usage terms

### Whisper model weights
Whisper model weights may be downloaded automatically on first use.

**Model weight licenses/terms are not always identical to the Python wrapper/package license.**  
Always document:
- the source of model files,
- the license/terms attached to the model weights.

### Pyannote diarization pipelines (Hugging Face)
Voice diarization uses a Hugging Face pipeline repository.  
**You are responsible for following the license/terms shown on the model card of the specific repository you use.**

---
---

## License and disclaimer

This project is released under:  
**AISTATElight GitHub-Compatible No-Resale License v1.2 (Source-Available)**

### ✅ Permissions
You may:
- use the software for **personal, educational, and research** purposes, and for **internal commercial use**,
- **modify** the code for your own internal needs.

### ❌ Restrictions
Without prior **written permission** from the author, you may **not**:
- resell or **commercially redistribute** the software (source code or binaries),
- distribute **binaries/installers/containers** or any packaged builds,
- publish modified versions outside GitHub’s allowed fork/view mechanisms, to the extent GitHub permits those actions under its Terms.

### © Attribution (required)
You must retain attribution to the author (**pawlict**) and keep the license text.
If used commercially within an organization, attribution must remain visible in the app’s **About/Info** section (or equivalent documentation).

### “AS IS” — no warranty
The software is provided **“AS IS”**, without warranty of any kind, express or implied, including but not limited to
merchantability, fitness for a particular purpose, and noninfringement.

### Limitation of liability
In no event shall the author or copyright holder be liable for any claim, damages, or other liability, whether in an action
of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

### Third-party licenses
This project depends on third-party components (e.g., Whisper, pyannote.audio, PySide6/Qt, FFmpeg), each under its own license.
See: **THIRD_PARTY_NOTICES.md**.

---


