from __future__ import annotations
from typing import List

from .data_types import OntologyType




def format_query(header: str, cells: List[str], max_cells: int = 10) -> str:
    """Format table column as e5 query string."""
    cells_str = ", ".join(cells[:max_cells])
    return f"query: Header: {header} | Cells: {cells_str}"


def format_type(otype: OntologyType, max_parents: int = 3) -> str:
    """Format ontology type as e5 passage string."""
    text = f"passage: {otype.label}"
    if otype.description:
        text += f": {otype.description}"
    parent_labels = [
        p.get("label", "") if isinstance(p, dict) else str(p)
        for p in otype.parents[:max_parents]
        if (p.get("label") if isinstance(p, dict) else p)
    ]
    if parent_labels:
        text += f". Parent types: {', '.join(parent_labels)}"
    return text