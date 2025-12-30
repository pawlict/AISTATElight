from __future__ import annotations
from dataclasses import dataclass

# ====== App identity (used by UI + reports) ======
# NOTE: Name/version are intentionally sourced from this file.
APP_NAME: str = "AISTATElight"
APP_VERSION: str = "v2.2.2"
AUTHOR_EMAIL: str = "pawlict@proton.me"
AUTHOR_NAME: str = "pawlict"

@dataclass
class Settings:
    hf_token: str = ""
    default_language: str = "auto"
    whisper_model: str = "base"
    theme: str = "Fusion Light (Blue)"
    ui_language: str = "pl"
