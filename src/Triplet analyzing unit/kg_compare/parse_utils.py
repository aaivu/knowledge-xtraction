from __future__ import annotations

import ast
from typing import Any, List, Tuple

Triplet = Tuple[str, str, str]


def _to_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    # Keep your underscore style as-is (White_board, etc.)
    return s


def parse_triplets(cell_value: Any) -> List[Triplet]:
    """
    Parses a cell like:
      "[['White_board','is_a_tool_for','Organization'], ...]"
      or "[('France','capital','Paris'), ...]"
    Returns list of (h, r, t) as strings.
    """
    if cell_value is None:
        return []
    raw = str(cell_value).strip()
    if raw == "" or raw.lower() == "nan":
        return []

    try:
        obj = ast.literal_eval(raw)
    except Exception:
        # If it is already something else, fail gracefully
        return []

    triplets: List[Triplet] = []
    if not isinstance(obj, list):
        return []

    for item in obj:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            h, r, t = item
            triplets.append((_to_str(h), _to_str(r), _to_str(t)))

    return triplets


def normalize_triplet(tri: Triplet) -> Triplet:
    """
    Light normalization for matching/display.
    (You can extend: lowercasing, replacing spaces, etc.)
    """
    h, r, t = tri
    return (h.strip(), r.strip(), t.strip())
