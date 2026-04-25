from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
 
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from run_retrieve import BiEncoderInference
 
logger = logging.getLogger(__name__)
 
@dataclass
class ContextFragment:
    source: str           # "title" or "neighbour"
    header: str
    cells:  List[str]
    text:   str           # pre-formatted passage text
 
 
@dataclass
class ColumnLevelSignal:
    fragments:     List[Tuple[ContextFragment, float]]   # (fragment, weight)
    guidance_text: str
    query_source:  str


## build context fragments
def build_context_fragments(
    record:         dict,
    max_neighbours: int = 6,
) -> List[ContextFragment]:
    fragments: List[ContextFragment] = []

    # Table title
    title = (record.get("table_title") or "").strip()
    if title:
        fragments.append(ContextFragment(
            source="title", header=title, cells=[],
            text=f"passage: {title}",
        ))

    # Neighbouring columns
    neighbour_headers = record.get("neighbour_headers", [])
    neighbour_cells   = record.get("neighbour_cells", [])
    while len(neighbour_cells) < len(neighbour_headers):
        neighbour_cells.append([])

    n_added = 0
    for header, cells in zip(neighbour_headers, neighbour_cells):
        if n_added >= max_neighbours:
            break
        header = (header or "").strip()
        if not header:
            continue
        cells_clean = [c for c in cells if c and c.strip()][:3]
        text = (f"passage: {header} | {', '.join(cells_clean)}"
                if cells_clean else f"passage: {header}")
        fragments.append(ContextFragment(
            source="neighbour", header=header, cells=cells_clean, text=text,
        ))
        n_added += 1

    return fragments

def format_type_text(concept: dict) -> str:
    label   = concept.get("label", "")
    desc    = concept.get("description", "")
    chain   = concept.get("ancestor_chain", []) or []
    parents = [a for a in chain[:3] if a]
    text = f"passage: {label}"
    if desc:
        text += f": {desc[:120]}"
    if parents:
        text += f". Parent types: {', '.join(parents)}"
    return text


class ContextGuidance:
    """
    Builds column-level context signals by attending over neighbouring
    columns using a score-weighted average of the top-k candidate type
    embeddings as the attention query.
    """

    def __init__(
        self,
        model_path:  str,
        ontology:    dict,
        temperature: float = 0.1,
        batch_size:  int   = 128,
        max_length:  int   = 256,
    ):
        self.encoder     = BiEncoderInference(model_path)
        self.ontology    = ontology
        self.temperature = temperature
        self.batch_size  = batch_size
        self.max_length  = max_length



    def _get_type_text(self, candidate: dict) -> str:
        concept = self.ontology.get(candidate["uri"], {})
        if not concept:
            concept = {
                "label":          candidate.get("label", ""),
                "description":    candidate.get("description", ""),
                "ancestor_chain": candidate.get("ancestors", []),
            }
        return format_type_text(concept)

    def _candidate_query_embedding(
        self,
        candidates: List[dict],
    ) -> Optional[torch.Tensor]:
        """
            E_cand = normalize( sum_i w_i * E_{t_i} )
            w_i    = softmax(s_i / tau)

        Returns None if no valid candidates are provided.
        """
        if not candidates:
            return None

        type_texts = [self._get_type_text(c) for c in candidates]
        type_embs  = self.encoder.encode(type_texts)             # (k, H)

        scores  = torch.tensor(
            [c.get("score", 1.0) for c in candidates], dtype=torch.float32,
        )
        weights   = torch.softmax(scores / self.temperature, dim=0).unsqueeze(1)
        query_emb = F.normalize((weights * type_embs).sum(0, keepdim=True), dim=-1)
        return query_emb                                          # (1, H)

    # Column-level signal
    def build_column_level_signal(
        self,
        record:     dict,
        candidates: Optional[List[dict]] = None,
        top_k:      int = 4,
    ) -> ColumnLevelSignal:
        """
        Build column-level context guidance using candidate-aware attention.

        Args:
            record:     Preprocessed column record with neighbour info.
            candidates: List of {uri, label, score, ...} from bi-encoder.
            top_k:      Number of context fragments to select.

        Returns:
            ColumnLevelSignal with ranked fragments and guidance text.
        """
        fragments = build_context_fragments(record)

        if not fragments:
            return ColumnLevelSignal(
                fragments=[],
                guidance_text="No table context available.",
                query_source="none",
            )

        ctx_embs = self.encoder.encode([f.text for f in fragments])  # (N, H)

        # Choose query embedding
        query_emb    = self._candidate_query_embedding(candidates) if candidates else None
        query_source = "candidate_weighted_avg"

        if query_emb is None:
            query_source = "column_query_fallback"
            header    = (record.get("pk_col_header_clean") or
                         record.get("pk_col_header_raw") or "").strip()
            cells     = (record.get("cell_samples_clean") or
                         record.get("cell_samples_raw") or [])
            cells_str = ", ".join(str(c) for c in cells[:5] if c)
            query_emb = self.encoder.encode(
                [f"query: Header: {header} | Cells: {cells_str}"]
            )                                                    # (1, H)

        # Attention over context fragments
        weights     = torch.softmax(
            (query_emb @ ctx_embs.T).squeeze(0) / self.temperature, dim=0
        )                                                        # (N,)
        top_indices = weights.argsort(descending=True)[:top_k].tolist()
        top_frags   = [(fragments[i], weights[i].item()) for i in top_indices]

        return ColumnLevelSignal(
            fragments=top_frags,
            guidance_text=self._format_guidance(top_frags),
            query_source=query_source,
        )

    def _format_guidance(
        self, fragments: List[Tuple[ContextFragment, float]]
    ) -> str:
        lines = ["Most relevant context for this column (ranked by relevance):"]
        for frag, _ in fragments:
            tag = "[title]" if frag.source == "title" else "[context]"
            body = f"{frag.header} | {', '.join(frag.cells)}" if frag.cells \
                   else frag.header
            lines.append(f"  {tag} {body}")
        return "\n".join(lines)

def precompute_guidance(
    record:   dict,
    result:   dict,
    guidance: ContextGuidance,
    ontology: dict,
) -> dict:

    cand_uris   = result.get("cand_uris",   [])
    cand_labels = result.get("cand_labels", [])
    cand_scores = result.get("cand_scores", [])

    candidates = []
    for i, uri in enumerate(cand_uris):
        concept = ontology.get(uri, {})
        candidates.append({
            "uri":         uri,
            "label":       cand_labels[i] if i < len(cand_labels) else "",
            "score":       float(cand_scores[i]) if i < len(cand_scores) else 1.0,
            "description": concept.get("description", "")[:120],
            "ancestors":   (concept.get("ancestor_chain", []) or [])[:3],
        })

    col_level_text = ""
    query_source   = "none"
    try:
        col_sig        = guidance.build_column_level_signal(record, candidates=candidates)
        col_level_text = col_sig.guidance_text
        query_source   = col_sig.query_source
    except Exception as e:
        logger.warning("Column-level guidance failed: %s", e)

    return {
        "column_level": {
            "guidance_text": col_level_text,
            "query_source":  query_source,
        },
    }
