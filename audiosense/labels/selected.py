"""
Primary label extraction for AudioSense.
"""

from pathlib import Path
from typing import Union, List, Optional
import numpy as np

from .top import top_k
from .taxonomy import get_category_for_label
from ..audio.preprocess import ProcessedAudio

def get_label(
    audio: Union[str, Path, ProcessedAudio, np.ndarray],
    category: Optional[str] = None,
    min_confidence: float = 0.2
) -> Optional[str]:
    """
    Predict audio and return the primary detected sound label.

    Parameters
    ----------
    audio : str | Path | ProcessedAudio | np.ndarray
        Audio input.
    category : str, optional
        Filter by category (e.g. 'Animal sounds', 'Sounds of things').
    min_confidence : float, default 0.2
        Minimum score required.

    Returns
    -------
    str or None
        The name of the most confident detected sound label.
    """
    candidates = top_k(audio, k=10)
    for lbl, score in candidates:
        if score < min_confidence:
            continue
        if category:
            cat = get_category_for_label(lbl)
            if category.lower() not in cat.lower():
                continue
        return lbl

    return candidates[0][0] if candidates else None

def get_labels(
    audio: Union[str, Path, ProcessedAudio, np.ndarray],
    top_n: int = 5,
    min_confidence: float = 0.1
) -> List[str]:
    """
    Get a list of top detected sound labels above min_confidence.
    """
    candidates = top_k(audio, k=top_n)
    return [lbl for lbl, score in candidates if score >= min_confidence]