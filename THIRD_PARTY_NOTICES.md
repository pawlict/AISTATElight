# Third-Party Notices (AISTATElight)

This project depends on third-party software and system components.
Each component is subject to its own license terms.

**Note:** This repository primarily contains pawlict's source code. Third-party
components are generally installed via package managers (pip/apt). If you bundle
or redistribute binaries, you may have additional obligations (especially for
LGPL/GPL components like Qt/PySide6 and FFmpeg).

---

## Whisper / openai-whisper — MIT
- Project: OpenAI Whisper (openai/whisper)
- Python package: `openai-whisper`
- License: MIT (code and model weights)
- Source: https://github.com/openai/whisper

---

## pyannote.audio — MIT
- Project: pyannote.audio
- License: MIT
- Source: https://github.com/pyannote/pyannote-audio
- Note: Some pretrained pipelines/models are downloaded from Hugging Face and may
  require accepting access conditions; check the model card for the exact terms.

---

## huggingface_hub — Apache-2.0
- Project: Hugging Face Hub Python client (`huggingface_hub`)
- License: Apache License 2.0
- Source: https://github.com/huggingface/huggingface_hub

---

## PyTorch — BSD 3-Clause
- Packages: `torch`, `torchaudio`
- License: BSD 3-Clause (project license)
- Source: https://github.com/pytorch/pytorch
- Note: torchaudio pretrained models may have separate licenses/terms.

---

## NumPy — BSD 3-Clause (plus bundled permissive components)
- Package: `numpy`
- License: BSD 3-Clause (NumPy), with additional bundled components under other
  permissive licenses (see NumPy LICENSE)
- Source: https://github.com/numpy/numpy

---

## SoundFile (python-soundfile) — BSD 3-Clause
- Package: `soundfile`
- License: BSD 3-Clause
- Source: https://github.com/bastibe/python-soundfile
- Note: uses the system library `libsndfile` (commonly LGPL).

---

## PySide6 / Qt for Python — LGPLv3 / GPLv3 / Commercial
- Package: `PySide6` (Qt for Python)
- License: Qt for Python is dual-licensed (LGPL/GPL or Commercial), and includes
  additional third-party notices in Qt documentation.
- Docs: https://doc.qt.io/qtforpython-6/licenses.html
- Note: If you distribute a packaged app (e.g., with bundled Qt libraries), ensure
  LGPL compliance (e.g., allow replacement/relinking of LGPL components).

---

## FFmpeg — LGPL 2.1+ (or GPL depending on build)
- Component: `ffmpeg` (system dependency)
- License: LGPL 2.1+ by default; may become GPL if built with GPL components
- Legal: https://www.ffmpeg.org/legal.html
- Recommendation: prefer installing FFmpeg via the OS package manager instead of
  bundling your own build.

---

## System components (installed via apt)
This project may use system packages such as:
- FFmpeg
- GStreamer and plugins
- libsndfile

These are distributed under their respective licenses by your Linux distribution.
