from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Settings:
    hf_token: str = ""
    default_language: str = "auto"
    whisper_model: str = "base"
    theme: str = "Fusion Light (Blue)"
    ui_language: str = "pl"