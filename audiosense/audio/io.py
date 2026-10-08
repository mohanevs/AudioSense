"""
Audio I/O utilities for loading, saving, and inspecting audio.
"""

from pathlib import Path
from typing import Union, Tuple, Optional
import soundfile as sf
import numpy as np
import librosa

def load_audio(
    path: Union[str, Path],
    sr: Optional[int] = None,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file into a numpy float32 array.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sample_rate = librosa.load(str(path), sr=sr, mono=mono)
    return audio.astype(np.float32), sample_rate

def save_audio(
    path: Union[str, Path],
    data: np.ndarray,
    sr: int = 16000
) -> Path:
    """
    Write float32 audio data to WAV file.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, samplerate=sr, subtype="PCM_16")
    return path
