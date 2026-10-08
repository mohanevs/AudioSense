"""
AudioSense: Audio intelligence framework for machines, robots, and applications.
Version 1.0.0
"""

# Silence all background logs, oneDNN warnings, and progress bars immediately
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"
os.environ["GLOG_minloglevel"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"

from .core.silence import silence_all, disable_tqdm
disable_tqdm()

import sys
from .core.engine import AudioSense, AudioSenseResult
from .audio.conversion import convert_to_wav, is_audio_file
from .audio.preprocess import preprocess_audio, ProcessedAudio
from .labels.selected import get_label, get_labels
from .labels.all import get_all, get_all_labels
from .labels.categories import get_categories
from .labels.top import top_k
from .storage.database import view_db
from .live.loop import run_mic_loop
from .models.download import download_all_models
from .context.download_llm import download_llm_model

# Register audiosense.label alias in sys.modules so `from audiosense.label import ...` works
from . import label as _label_mod
sys.modules["audiosense.label"] = _label_mod

__version__ = "1.0.0"

def summary_db(llm_path=None, language="en") -> str:
    """Convenience top-level summary_db function."""
    engine = AudioSense(llm_path=llm_path, language=language)
    return engine.summary_db()

def loop(chunk_duration: float = 3.0, sample_rate: int = 16000, device=None, callback=None):
    """Convenience top-level audiosense.loop() function."""
    engine = AudioSense()
    return engine.loop(
        chunk_duration=chunk_duration,
        sample_rate=sample_rate,
        device=device,
        callback=callback
    )

def download_models(models_dir=None) -> bool:
    """Download all core acoustic models (AST, PANN, YAMNet)."""
    return download_all_models(models_dir=models_dir)

def download_llm(llm_dir=None):
    """Download local Phi-4-mini GGUF model."""
    return download_llm_model(llm_dir=llm_dir)

__all__ = [
    "AudioSense",
    "AudioSenseResult",
    "convert_to_wav",
    "is_audio_file",
    "preprocess_audio",
    "ProcessedAudio",
    "get_label",
    "get_labels",
    "get_all",
    "get_all_labels",
    "get_categories",
    "top_k",
    "view_db",
    "summary_db",
    "loop",
    "download_models",
    "download_llm",
    "__version__",
]
