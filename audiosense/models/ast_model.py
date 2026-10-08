"""
AST (Audio Spectrogram Transformer) Model Adapter for AudioSense.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np

from ..core.config import DEFAULT_MODELS_DIR
from ..core.silence import silence_all

class ASTModel:
    def __init__(self, model_dir: Optional[Union[str, Path]] = None, device: str = "auto"):
        self.model_dir = Path(model_dir or (DEFAULT_MODELS_DIR / "ast")).resolve()
        self.device = device
        self._extractor = None
        self._model = None
        self._torch_device = None

    def _load(self):
        if self._model is not None:
            return

        import torch

        if self.device == "auto":
            self._torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._torch_device = self.device

        # Auto-download on first use if not present
        if not self.model_dir.exists() or not (self.model_dir / "config.json").exists():
            from .download import download_ast
            print(f"[AudioSense] Downloading AST weights on first use...")
            download_ast(self.model_dir)

        with silence_all():
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            self._extractor = AutoFeatureExtractor.from_pretrained(str(self.model_dir))
            self._model = AutoModelForAudioClassification.from_pretrained(str(self.model_dir))
            self._model.to(self._torch_device).eval()
            if self._torch_device == "cuda":
                self._model.half()

    def predict(self, audio_16k: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        self._load()
        import torch

        with silence_all():
            inputs = self._extractor(audio_16k, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}
            if self._torch_device == "cuda":
                inputs = {k: (v.half() if v.is_floating_point() else v) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.sigmoid(logits.float())[0]

        top_scores, top_idx = torch.topk(probs, k=min(top_k, probs.shape[0]))
        id2label = self._model.config.id2label
        return [(id2label[int(idx.item())], float(score.item())) for score, idx in zip(top_scores, top_idx)]

    def predict_all(self, audio_16k: np.ndarray, threshold: float = 0.0) -> List[Tuple[str, float]]:
        self._load()
        import torch

        with silence_all():
            inputs = self._extractor(audio_16k, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}
            if self._torch_device == "cuda":
                inputs = {k: (v.half() if v.is_floating_point() else v) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.sigmoid(logits.float())[0].cpu().numpy()

        id2label = self._model.config.id2label
        order = np.argsort(-probs)
        return [(id2label[i], float(probs[i])) for i in order if probs[i] >= threshold]
