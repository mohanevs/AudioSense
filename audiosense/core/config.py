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
      2. PROJECT_ROOT / "models"
      3. PACKAGE_ROOT / "models"
      4. ~/.cache/audiosense/models
    """
    env_dir = os.environ.get("AUDIOSENSE_MODELS_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists():
            return p

    project_models = PROJECT_ROOT / "models"
    if project_models.exists():
        return project_models

    pkg_models = PACKAGE_ROOT / "models"
    if pkg_models.exists():
        return pkg_models

    cache_models = Path.home() / ".cache" / "audiosense" / "models"
    cache_models.mkdir(parents=True, exist_ok=True)
    return cache_models

def get_default_db_path() -> Path:
    """
    Default path for audio_sense_labels.json.
    Prefers project root if writable, otherwise ~/.cache/audiosense/
    """
    proj_db = PROJECT_ROOT / "audio_sense_labels.json"
    if proj_db.exists() or os.access(PROJECT_ROOT, os.W_OK):
        return proj_db
    return Path.home() / ".cache" / "audiosense" / "audio_sense_labels.json"

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
