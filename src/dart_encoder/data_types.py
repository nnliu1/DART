from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List



@dataclass
class OntologyType:
    qid: str
    label: str
    description: str
    parents: List[Dict]      # [{"qid": ..., "label": ...}, ...]



@dataclass
class TrainSample:
    anchor_header: str
    anchor_cells: List[str]
    positive_qid: str
    hard_negative_qids: List[str]
