"""
Voice Activity Detection (Silero VAD) Module for AudioSense.
"""

from typing import List, Dict, Any, Union
from pathlib import Path
import numpy as np

class VADDetector:
    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from silero_vad import load_silero_vad
        self._model = load_silero_vad()

    def detect_speech(self, audio_16k: np.ndarray, threshold: float = 0.5) -> List[Dict[str, float]]:
        """
        Detect speech segments in 16 kHz float32 audio.
        Returns list of segment dicts with 'start' and 'end' in seconds.
        """
        self._load()
        import torch
        from silero_vad import get_speech_timestamps

        tensor = torch.tensor(audio_16k, dtype=torch.float32)
        timestamps = get_speech_timestamps(
            tensor,
            self._model,
            sampling_rate=16000,
            threshold=threshold,
            return_seconds=True
        )
        return timestamps

    def has_speech(self, audio_16k: np.ndarray, threshold: float = 0.5) -> bool:
        """Convenience method returning True if any speech is found."""
        return len(self.detect_speech(audio_16k, threshold=threshold)) > 0
