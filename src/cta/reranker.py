from __future__ import annotations
import argparse
import logging
from typing import List, Optional
import json


from openai import OpenAI
from prompts import build_listwise_prompt, parse_listwise_response

logger = logging.getLogger(__name__)

GENERIC_ANCESTORS = {"thing", "agent", "species", "eukaryote"}


##  candidate builder
def build_candidates(
    cand_uris:   List[str],
    cand_labels: List[str],
    ontology:    dict,
    top_k:       int,
) -> List[dict]:
    candidates = []
    for i, (uri, label) in enumerate(zip(cand_uris[:top_k], cand_labels[:top_k])):
        concept = ontology.get(uri, {})
        desc    = concept.get("description") or label
        chain   = concept.get("ancestor_chain") or []
        anc     = [a for a in chain[:3] if a.lower() not in GENERIC_ANCESTORS]
        candidates.append({
            "idx":         i,
            "uri":         uri,
            "label":       label,
            "description": desc[:120],
            "ancestors":   anc,
        })
    return candidates



def call_reranker(
    column_text:   str,
    candidates:    List[dict],
    guidance_text: str,
    client:        OpenAI,
    top_k_out:     int  = 5,
    use_guidance:  bool = True,
    model:         str  = "gpt-4o-mini",
) -> Optional[List[str]]:
    prompt = build_listwise_prompt(
        column_text, candidates, guidance_text, top_k_out, use_guidance
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        parsed  = json.loads(content)
        return parse_listwise_response(parsed, candidates, top_k_out)
    except Exception as e:
        logger.warning("Reranker failed: %s", e)
        return None

    

def run_rerank(
    results:      List[dict],
    ontology:     dict,
    client:       OpenAI,
    top_k_llm:    int  = 10,
    top_k_out:    int  = 5,
    use_guidance: bool = True,
    use_rewrite:  bool = True,
    model:        str  = "gpt-4o-mini",
) -> List[dict]:
    n_ok = n_fail = n_rewritten = n_original = 0
 
    for i, r in enumerate(results):
        col_text  = r.get("column_text", "")
        g_text    = (r.get("guidance", {})
                      .get("column_level", {})
                      .get("guidance_text", "")) if use_guidance else ""
 
        # Choose candidate pool
        if use_rewrite and "rewritten_cand_uris" in r:
            cand_uris = r["rewritten_cand_uris"]
            cand_labs = r.get("rewritten_cand_labels", [])
            n_rewritten += 1
        else:
            cand_uris = r["cand_uris"]
            cand_labs = r.get("cand_labels", [])
            n_original += 1
 
        candidates = build_candidates(cand_uris, cand_labs, ontology, top_k_llm)
        pred_uris  = call_reranker(
            col_text, candidates, g_text, client,
            top_k_out=top_k_out, use_guidance=use_guidance, model=model,
        )
 
        if pred_uris is not None:
            r["pred_uris"] = pred_uris
            r["pred_uri"]  = pred_uris[0] if pred_uris else ""
            n_ok += 1
        else:
            r["pred_uris"] = cand_uris   # fallback to original order
            r["pred_uri"]  = cand_uris[0] if cand_uris else ""
            n_fail += 1
 
        if (i + 1) % 50 == 0:
            logger.info("Rerank progress: %d / %d  (ok=%d  fail=%d)",
                        i + 1, len(results), n_ok, n_fail)
 
    logger.info("Rerank complete: ok=%d  fail=%d  rewritten=%d  original=%d",
                n_ok, n_fail, n_rewritten, n_original)
    return results
