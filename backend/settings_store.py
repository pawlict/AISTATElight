from __future__ import annotations
import os, json
from dataclasses import asdict
from pathlib import Path
from backend.settings import Settings

def _config_path() -> Path:
    base = Path.home() / ".config" / "AISTATElight"
    base.mkdir(parents=True, exist_ok=True)
    return base / "settings.json"

def load_settings() -> Settings:
    path = _config_path()
    s = Settings()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for k, v in data.items():
                if hasattr(s, k):
                    setattr(s, k, v)
        except Exception:
            pass
    if not s.hf_token:
        s.hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN", "") or os.environ.get("HF_TOKEN", "")
    return s

def save_settings(s: Settings) -> None:
    path = _config_path()
    path.write_text(json.dumps(asdict(s), ensure_ascii=False, indent=2), encoding="utf-8")