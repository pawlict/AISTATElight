## {{APP_NAME}} – Artificial Intelligence Speech-To-Analysis-Translation Engine ({{APP_VERSION}})

**Author:** pawlict
---
(Contact / support)
If you encounter bugs, technical issues, have improvement suggestions, or ideas for new features — please contact the author at: pawlict@proton.me
<p align="center">
  <img src="assets/logo.png" alt="Logo" width="280" />
</p>

---

## License (MIT) & Disclaimer (AS IS)

This project is distributed under the **MIT License**, which permits use, copying, modification, merging, publishing, distribution, sublicensing, and/or selling copies of the software, provided that the copyright notice and license text are included.

**Warranty disclaimer (“AS IS”):**  
The software is provided *“as is”*, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement.

**Limitation of liability:**  
In no event shall the author(s) or copyright holder(s) be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

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


