"""
Retrieve all predictions across AST, PANN, and YAMNet.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np

from ..audio.preprocess import preprocess_audio, ProcessedAudio
from ..models.manager import ModelManager

_GLOBAL_MANAGER = None

def _get_manager():
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = ModelManager()
    return _GLOBAL_MANAGER

def get_all(
    audio: Union[str, Path, ProcessedAudio, np.ndarray],
    threshold: float = 0.0,
    with_scores: bool = False
) -> Dict[str, Union[List[str], List[Tuple[str, float]]]]:
    """
    Run AST, PANN, and YAMNet on the audio and return all labels matching threshold.

    Parameters
    ----------
    audio : str | Path | ProcessedAudio | np.ndarray
        Audio input.
    threshold : float, default 0.0
        Minimum score threshold (0.0 to 1.0).
    with_scores : bool, default False
        If True, returns list of (label, score) tuples. Otherwise, returns list of label strings.

    Returns
    -------
    dict
        {"ast": [...], "pann": [...], "yamnet": [...]}
    """
    proc = preprocess_audio(audio)
    mgr = _get_manager()

    ast_results = mgr.ast.predict_all(proc.audio_16k, threshold=threshold)
    pann_results = mgr.pann.predict_all(proc.audio_32k, threshold=threshold)
    yamnet_results = mgr.yamnet.predict_all(proc.audio_16k, threshold=threshold)

    if with_scores:
        return {
            "ast": ast_results,
            "pann": pann_results,
            "yamnet": yamnet_results
        }

    return {
        "ast": [label for label, _ in ast_results],
        "pann": [label for label, _ in pann_results],
        "yamnet": [label for label, _ in yamnet_results]
    }

# Backwards compatibility alias
get_all_labels = get_all