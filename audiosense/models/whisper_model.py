"""
Faster-Whisper Transcription Adapter for AudioSense.
"""

from typing import Optional, Union
import numpy as np
from ..core.silence import silence_all

# Safe patch for PyAV metadata_errors compatibility
try:
    import av
    _orig_av_open = av.open
    def _safe_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)
    av.open = _safe_av_open
except Exception:
    pass

class WhisperTranscriber:
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        with silence_all():
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )

    def transcribe(self, audio_16k: np.ndarray, beam_size: int = 5) -> str:
        """
        Transcribe 16 kHz mono float32 audio.
        Returns transcribed text string.
        """
        self._load()
        audio_16k = audio_16k.astype(np.float32)
        with silence_all():
            segments, _ = self._model.transcribe(audio_16k, beam_size=beam_size)
            text = " ".join(seg.text.strip() for seg in segments)
        return text.strip()
