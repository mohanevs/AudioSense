"""
Alias module so users can do `from audiosense.label import get_label, get_all, get_categories, top_k`
"""

from .labels import (
    get_label,
    get_labels,
    get_all,
    get_all_labels,
    get_categories,
    top_k,
    CATEGORIES,
    get_category_for_label,
)

__all__ = [
    "get_label",
    "get_labels",
    "get_all",
    "get_all_labels",
    "get_categories",
    "top_k",
    "CATEGORIES",
    "get_category_for_label",
]
