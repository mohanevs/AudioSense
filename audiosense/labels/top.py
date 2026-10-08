"""
Top-K label predictions for AudioSense.
"""

from pathlib import Path
from typing import List, Tuple, Union, Optional
from collections import defaultdict
import numpy as np

from ..audio.preprocess import preprocess_audio, ProcessedAudio
from ..models.manager import ModelManager

_GLOBAL_MANAGER = None

def _get_manager():
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = ModelManager()
    return _GLOBAL_MANAGER

def fuse_predictions(
    ast_preds: List[Tuple[str, float]],
    pann_preds: List[Tuple[str, float]],
    yamnet_preds: List[Tuple[str, float]],
    k: int = 5
) -> List[Tuple[str, float]]:
    """
    Fuse predictions from AST, PANN, and YAMNet using weighted consensus.
    """
    scores = defaultdict(float)
    weights = {"ast": 0.5, "pann": 0.3, "yamnet": 0.2}

    for lbl, scr in ast_preds:
        scores[lbl] += scr * weights["ast"]
    for lbl, scr in pann_preds:
        scores[lbl] += scr * weights["pann"]
    for lbl, scr in yamnet_preds:
        scores[lbl] += scr * weights["yamnet"]

    sorted_labels = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(lbl, float(round(score, 4))) for lbl, score in sorted_labels[:k]]


def top_k(
    audio: Union[str, Path, ProcessedAudio, np.ndarray],
    k: int = 5,
    model: Optional[str] = None
) -> List[Tuple[str, float]]:
    """
    Get top-k predictions with confidence scores for an audio input.
    """
    proc = preprocess_audio(audio)
    mgr = _get_manager()

    if model:
        m = model.lower()
        if m == "ast":
            return mgr.ast.predict(proc.audio_16k, top_k=k)
        elif m in ("pann", "panns"):
            return mgr.pann.predict(proc.audio_32k, top_k=k)
        elif m == "yamnet":
            return mgr.yamnet.predict(proc.audio_16k, top_k=k)
        else:
            raise ValueError(f"Unknown model: {model}. Supported: 'ast', 'pann', 'yamnet'")

    ast_preds = mgr.ast.predict(proc.audio_16k, top_k=15)
    pann_preds = mgr.pann.predict(proc.audio_32k, top_k=15)
    yamnet_preds = mgr.yamnet.predict(proc.audio_16k, top_k=15)

    return fuse_predictions(ast_preds, pann_preds, yamnet_preds, k=k)