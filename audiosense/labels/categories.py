"""
Category inspection and taxonomy mapping for AudioSense.
"""

from typing import Dict, List, Optional, Union
from .taxonomy import CATEGORIES, get_category_for_label

def get_categories(labels: Optional[Union[List[str], str]] = None) -> Union[Dict[str, List[str]], Dict[str, str], str]:
    """
    Get categories of labels or the entire taxonomy hierarchy.

    Parameters
    ----------
    labels : list[str] | str, optional
        If None, returns the entire category hierarchy mapping.
        If a single string, returns the category for that label.
        If a list of strings, returns a dictionary mapping each label to its category.

    Returns
    -------
    dict | str
    """
    if labels is None:
        return CATEGORIES

    if isinstance(labels, str):
        return get_category_for_label(labels)

    return {lbl: get_category_for_label(lbl) for lbl in labels}