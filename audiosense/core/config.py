"""
AudioSense Configuration and Path Management.
Ensures 100% relative and dynamic path resolution across all environments and operating systems.
"""

import os
from pathlib import Path

# Package and Project root directories resolved dynamically
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent

def get_default_models_dir() -> Path:
    """
    Dynamically locate the models directory.
    Checks:
      1. AUDIOSENSE_MODELS_DIR environment variable
      2. Path.cwd() / "models" (if running in project directory with models)
      3. PROJECT_ROOT / "models" (development repo layout)
      4. ~/.cache/audiosense/models (standard user cache for pip packages)
    """
    env_dir = os.environ.get("AUDIOSENSE_MODELS_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists():
            return p

    cwd_models = Path.cwd() / "models"
    if cwd_models.exists() and (cwd_models / "ast").exists():
        return cwd_models

    project_models = PROJECT_ROOT / "models"
    if project_models.exists() and (project_models / "ast").exists():
        return project_models

    cache_models = Path.home() / ".cache" / "audiosense" / "models"
    cache_models.mkdir(parents=True, exist_ok=True)
    return cache_models

def get_default_db_path() -> Path:
    """
    Default path for audio_sense_labels.json.
    Prefers user's current working directory, fallback to ~/.cache/audiosense/
    """
    try:
        cwd_db = Path.cwd() / "audio_sense_labels.json"
        if cwd_db.exists() or os.access(Path.cwd(), os.W_OK):
            return cwd_db
    except Exception:
        pass

    cache_dir = Path.home() / ".cache" / "audiosense"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "audio_sense_labels.json"

DEFAULT_MODELS_DIR = get_default_models_dir()
DEFAULT_DB_PATH = get_default_db_path()

# Supported language codes mapping
LANGUAGE_NAMES = {
    "en": "English",
    "hin": "Hindi",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "zh": "Chinese",
}
