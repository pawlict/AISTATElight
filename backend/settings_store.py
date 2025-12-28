from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from .settings import Settings


APP_DIRNAME = "AISTATElight"
FILENAME = "settings.json"


def _legacy_config_dir() -> Path:
    """Old location (kept for backward compatibility)."""
    return Path(os.path.expanduser(f"~/.config/{APP_DIRNAME}"))


def _local_config_dir() -> Path:
    """New default location: inside the app folder (backend/.aistate/).

    Why:
      - portable installs (e.g., running from a USB stick / shared folder)
      - easy backup with the project
      - works the same on Linux & Windows (when not using system installers)

    Security note:
      This file can contain your Hugging Face token. DO NOT commit it to Git.
      Add `backend/.aistate/settings.json` to .gitignore.
    """
    # Optional override (useful for power-users / packaging)
    env_dir = (os.environ.get("AISTATE_CONFIG_DIR") or "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    # If bundled (PyInstaller, etc.), store next to the executable
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        return exe_dir / f".{APP_DIRNAME}"

    # Source run: store under backend/.aistate/
    backend_dir = Path(__file__).resolve().parent
    return backend_dir / ".aistate"


def _config_path() -> Path:
    cfg_dir = _local_config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / FILENAME


def _read_settings_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        # If the file is corrupt, don't crash the whole app.
        # User can delete it and start fresh.
        return {}


def load_settings() -> Settings:
    """Load settings.

    Priority:
      1) new local path (backend/.aistate/settings.json) or AISTATE_CONFIG_DIR
      2) legacy path (~/.config/AISTATElight/settings.json)
      3) defaults
    """
    new_path = _config_path()
    data = _read_settings_file(new_path)

    if not data:
        legacy_path = _legacy_config_dir() / FILENAME
        legacy = _read_settings_file(legacy_path)
        if legacy:
            data = legacy
            # Best effort: migrate to new location
            try:
                save_settings(Settings(**{k: legacy.get(k) for k in asdict(Settings()).keys()}))
            except Exception:
                pass

    s = Settings()
    for k, v in data.items():
        if hasattr(s, k):
            setattr(s, k, v)
    return s


def save_settings(settings: Settings) -> None:
    """Save settings to the new local path."""
    path = _config_path()
    tmp = path.with_suffix(".tmp")

    data = asdict(settings)
    # Write atomically (reduce risk of partial writes)
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
