from __future__ import annotations
from typing import List, Tuple


## query rewriter

def build_rewriter_prompt(column_text: str, guidance_text: str) -> str:

    guidance_block = ""
    if guidance_text and "No table context" not in guidance_text:
        guidance_block = f"""
[Table Context]
{guidance_text}
"""

    return f"""You are helping improve a semantic search query for table column type annotation.

[Original Column — Retrieval Confidence: LOW]
{column_text}
{guidance_block}
[Task]
The original query failed to find a confident semantic type match.
Based on the column values{' and table context' if guidance_block else ''}:
1. Infer what TYPE of entities are listed in this column
2. Rewrite the header to be semantically descriptive of the entity type
3. Keep the most representative cell values

Output ONLY a valid JSON object:
{{"header": "descriptive type label", "cells": ["cell1", "cell2", "cell3"]}}

Your answer:"""



## reranker
def build_listwise_prompt(
    column_text:   str,
    candidates:    List[dict],
    guidance_text: str,
    top_k_out:     int,
    use_guidance:  bool,
) -> str:

    def anc_str(c: dict) -> str:
        return " → ".join(c["ancestors"]) if c["ancestors"] else "—"

    # Method 1: column-level guidance block (before candidates)
    guidance_block = ""
    if use_guidance and guidance_text and "No table context" not in guidance_text:
        guidance_block = f"""
[Table Context — Most Relevant Neighboring Columns]
{guidance_text}
"""

    cand_lines = []
    for c in candidates:
        cand_lines.append(
            f"  [{c['idx']}] {c['label']}\n"
            f"      Definition : {c['description']}\n"
            f"      Hierarchy  : {anc_str(c)}"
        )
    cand_block = "\n".join(cand_lines)

    guidance_instruction = (
        "- Use the table context above to understand the domain and "
        "disambiguate between similar types\n"
    ) if guidance_block else ""

    return f"""You are an expert in semantic table annotation.

[Target Column]
{column_text}
{guidance_block}
[Candidate Types]
{cand_block}

[Task]
Rank the candidates and identify the {top_k_out} best semantic types for the column.

Guidelines:
- Match all cell values to the most specific type that fits
- Use the type hierarchy to understand generality vs specificity
{guidance_instruction}- Prefer more specific types when the evidence supports them

Return ONLY a JSON object:
{{"top{top_k_out}": [best_idx, second_idx, ...]}}

Example: {{"top{top_k_out}": [3, 1, 0, 7, 2]}}

Your answer:"""



### response parsers
def parse_rewriter_response(response: dict) -> Tuple[str, List[str]]:
    """Parse query rewriter output → (header, cells)."""
    header = response.get("header", "")
    cells  = response.get("cells", [])
    if not isinstance(cells, list):
        cells = []
    cells = [str(c) for c in cells if c]
    return header, cells



def parse_listwise_response(
    response:   dict,
    candidates: List[dict],
    top_k_out:  int,
) -> List[str]:
    """Parse listwise reranker output → ordered list of URIs."""
    try:
        key     = f"top{top_k_out}"
        indices = response.get(key, response.get("top5", response.get("ranking", [])))
        valid   = [i for i in indices
                   if isinstance(i, int) and 0 <= i < len(candidates)]
        if valid:
            uris = [candidates[i]["uri"] for i in valid[:top_k_out]]
            seen = set(valid)
            for i in range(len(candidates)):
                if i not in seen:
                    uris.append(candidates[i]["uri"])
            return uris
    except Exception:
        pass
    return [c["uri"] for c in candidates]

