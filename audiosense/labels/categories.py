"""
Category inspection and taxonomy mapping for AudioSense.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
from .taxonomy import CATEGORIES, get_category_for_label

def get_categories(
    labels: Optional[Union[List[str], str, Path]] = None
) -> Union[Dict[str, List[str]], Dict[str, str], str]:
    """
    Get categories of labels or the entire taxonomy hierarchy.
    Can also accept an audio file directly to return categories of detected sounds.

    Parameters
    ----------
    labels : list[str] | str | Path, optional
        If None, returns the entire category hierarchy mapping.
        If an audio file, predicts sounds and returns their categories.
        If a single label string, returns the category for that label.
        If a list of strings, returns a dictionary mapping each label to its category.

    Returns
    -------
    dict | str
    """
    if labels is None:
        return CATEGORIES

    # Check if input is an existing audio file
    if isinstance(labels, (Path, str)):
        p = Path(labels)
        if p.exists() and p.is_file():
            from .selected import get_labels
            detected = get_labels(p)
            return {lbl: get_category_for_label(lbl) for lbl in detected}

    if isinstance(labels, str):
        return get_category_for_label(labels)

    return {lbl: get_category_for_label(lbl) for lbl in labels}