"""
Audio Preprocessing Module for AudioSense.
Internal functionality that handles dual sample-rate conditioning:
  - 32 kHz for PANN (CNN14)
  - 16 kHz for AST, YAMNet, Silero VAD, and Whisper
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional
import numpy as np
import librosa
from scipy.signal import resample_poly

from .conversion import convert_to_wav

@dataclass
class ProcessedAudio:
    """
    Standard preprocessed audio container holding dual-rate representations.
    """
    audio_16k: np.ndarray       # 16 kHz mono float32 (AST, YAMNet, Whisper)
    audio_32k: np.ndarray       # 32 kHz mono float32 (PANN)
    duration: float             # duration in seconds
    source_path: Optional[Path] = None
    wav_path: Optional[Path] = None

    @property
    def preprocess(self):
        """Allows `audio = audio.preprocess` idempotency."""
        return self

    def __repr__(self) -> str:
        return f"<ProcessedAudio duration={self.duration:.2f}s sr=[16kHz, 32kHz] source={self.source_path}>"


def preprocess_audio(
    source: Union[str, Path, np.ndarray, ProcessedAudio],
    orig_sr: Optional[int] = None
) -> ProcessedAudio:
    """
    Condition any audio input for AudioSense models.
    Converts to WAV if needed, transforms to mono float32, and generates
    both 16 kHz and 32 kHz arrays.

    Parameters
    ----------
    source : str | Path | np.ndarray | ProcessedAudio
        Audio file path or raw waveform array.
    orig_sr : int, optional
        Sample rate if source is a raw numpy array.

    Returns
    -------
    ProcessedAudio
    """
    if isinstance(source, ProcessedAudio):
        return source

    wav_file = None
    source_path = None

    if isinstance(source, (str, Path)):
        source_path = Path(source).resolve()
        # Convert non-wav formats (e.g., mp3, flac) to standard wav
        if source_path.suffix.lower() != ".wav":
            wav_file = convert_to_wav(source_path)
        else:
            wav_file = source_path

        # Load 32 kHz mono (native for PANN)
        audio_32k, sr32 = librosa.load(str(wav_file), sr=32000, mono=True)
        audio_32k = audio_32k.astype(np.float32)

        # Resample to 16 kHz mono (for AST, YAMNet, Whisper)
        audio_16k = librosa.resample(audio_32k, orig_sr=32000, target_sr=16000).astype(np.float32)

        duration = len(audio_16k) / 16000.0

    elif isinstance(source, np.ndarray):
        arr = source.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1 if arr.shape[1] < arr.shape[0] else 0)

        sr = orig_sr or 16000
        if sr == 32000:
            audio_32k = arr
            audio_16k = librosa.resample(arr, orig_sr=32000, target_sr=16000).astype(np.float32)
        elif sr == 16000:
            audio_16k = arr
            audio_32k = librosa.resample(arr, orig_sr=16000, target_sr=32000).astype(np.float32)
        else:
            audio_16k = librosa.resample(arr, orig_sr=sr, target_sr=16000).astype(np.float32)
            audio_32k = librosa.resample(arr, orig_sr=sr, target_sr=32000).astype(np.float32)

        duration = len(audio_16k) / 16000.0
    else:
        raise TypeError(f"Unsupported audio source type: {type(source)}")

    # Normalize audio levels between -1.0 and 1.0 (prevent clipping/silence distortion)
    max_16 = np.max(np.abs(audio_16k)) if len(audio_16k) > 0 else 0
    if max_16 > 1.0:
        audio_16k = audio_16k / max_16

    max_32 = np.max(np.abs(audio_32k)) if len(audio_32k) > 0 else 0
    if max_32 > 1.0:
        audio_32k = audio_32k / max_32

    return ProcessedAudio(
        audio_16k=audio_16k,
        audio_32k=audio_32k,
        duration=duration,
        source_path=source_path,
        wav_path=wav_file
    )
