"""
PANNs (Pretrained Audio Neural Networks - CNN14) Adapter for AudioSense.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np

from ..core.config import DEFAULT_MODELS_DIR
from ..core.silence import silence_all

class PANNModel:
    def __init__(self, checkpoint_path: Optional[Union[str, Path]] = None, device: str = "auto"):
        self.checkpoint_path = Path(
            checkpoint_path or (DEFAULT_MODELS_DIR / "pann" / "Cnn14_mAP=0.431.pth")
        ).resolve()
        self.device = device
        self._tagger = None
        self._labels = None

    def _load(self):
        if self._tagger is not None:
            return

        import torch

        if self.device == "auto":
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            torch_device = self.device

        # Auto-download on first use if not present
        if not self.checkpoint_path.exists():
            from .download import download_pann
            print(f"[AudioSense] Downloading PANN weights on first use...")
            download_pann(self.checkpoint_path.parent)

        with silence_all():
            from panns_inference import AudioTagging
            self._tagger = AudioTagging(checkpoint_path=str(self.checkpoint_path), device=torch_device)
        self._labels = self._tagger.labels

    def predict(self, audio_32k: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        self._load()
        with silence_all():
            clipwise_output, _ = self._tagger.inference(audio_32k[None, :])
        scores = clipwise_output[0]

        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(self._labels[i], float(scores[i])) for i in top_idx]

    def predict_all(self, audio_32k: np.ndarray, threshold: float = 0.0) -> List[Tuple[str, float]]:
        self._load()
        with silence_all():
            clipwise_output, _ = self._tagger.inference(audio_32k[None, :])
        scores = clipwise_output[0]

        order = np.argsort(-scores)
        return [(self._labels[i], float(scores[i])) for i in order if scores[i] >= threshold]
