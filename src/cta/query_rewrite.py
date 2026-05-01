from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple

import torch
from openai import OpenAI
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
 
from run_retrieve import BiEncoderInference


from prompts import build_rewriter_prompt, parse_rewriter_response


logger = logging.getLogger(__name__)



## rewrite function

def call_rewriter(
    column_text:   str,
    guidance_text: str,
    client:        OpenAI,
    model:         str = "gpt-4o-mini",
) -> Optional[Tuple[str, List[str]]]:

    prompt = build_rewriter_prompt(column_text, guidance_text)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        parsed  = __import__("json").loads(content)
        return parse_rewriter_response(parsed)
    except Exception as e:
        logger.warning("Rewriter failed: %s", e)
        return None



def run_rewrite(
    results:      List[dict],
    concept_ids:  List[str],
    labels_map:   Dict[str, str],
    type_embs:    torch.Tensor,
    encoder:      BiEncoderInference,
    client:       OpenAI,
    low_thresh:   float = 0.45,
    top_k:        int   = 20,
    use_guidance: bool  = True,
    model:        str   = "gpt-4o-mini",
) -> List[dict]:

    n_total = n_ok = n_fail = n_skip = 0

    for i, r in enumerate(results):
        top1_score = r.get("top1_score", 0.0)

        if top1_score >= low_thresh:
            n_skip += 1
            continue

        n_total    += 1
        col_text    = r.get("column_text", "")
        g_text      = (r.get("guidance", {})
                        .get("column_level", {})
                        .get("guidance_text", "")) if use_guidance else ""

        result = call_rewriter(col_text, g_text, client, model)
        if result is None:
            n_fail += 1
            continue

        rw_header, rw_cells = result
        if not rw_header or not rw_cells:
            n_fail += 1
            continue

        # Re-retrieve with rewritten query
        query_text   = f"query: Header: {rw_header} | Cells: {', '.join(rw_cells[:5])}"
        q_emb        = encoder.encode([query_text],batch_size=128)                  # (1, H)
        scores       = (q_emb @ type_embs.T).squeeze(0)             # (N,)
        top_idx      = scores.argsort(descending=True)[:top_k].tolist()
        new_uris     = [concept_ids[j]    for j in top_idx]
        new_scores   = [scores[j].item()  for j in top_idx]
        new_top1     = new_scores[0] if new_scores else 0.0

        if new_top1 >= low_thresh:
            r["rewritten_cand_uris"]   = new_uris
            r["rewritten_cand_labels"] = [labels_map.get(u, "") for u in new_uris]
            r["rewritten_cand_scores"] = new_scores
            r["rewritten_top1_score"]  = new_top1
            r["rewritten_query"]       = query_text
            n_ok += 1
        else:
            n_fail += 1

        if (i + 1) % 50 == 0:
            logger.info("Rewrite progress: %d / %d  (ok=%d  fail=%d  skip=%d)",
                        i + 1, len(results), n_ok, n_fail, n_skip)

    logger.info("Rewrite complete: total=%d  ok=%d  fail=%d  skip=%d",
                n_total, n_ok, n_fail, n_skip)
    return results
