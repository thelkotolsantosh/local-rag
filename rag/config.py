"""Load settings from config.yaml (and optional .env)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

try:  # .env is optional; only needed for gated Hugging Face models
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def load_config(path=None) -> dict:
    """Read the YAML config and turn relative paths into absolute ones."""
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg["paths"].items():
        cfg["paths"][key] = str((ROOT / value).resolve())
    return cfg
