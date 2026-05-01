from __future__ import annotations
 
import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import List
 
from openai import OpenAI
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))
 

from run_retrieve import BiEncoderInference
from guidance import ContextGuidance
from query_rewrite import run_rewrite
from reranker import run_rerank
 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
 
STEPS = ["rerank"]


def parse_args():
    p = argparse.ArgumentParser(description="DART pipeline")
    # Data
    p.add_argument("--ontology",         required=True)
    p.add_argument("--retrieval_pkl",    required=True)
    p.add_argument("--preprocessed_dir", default=None)
    p.add_argument("--output_dir",       required=True)
    # Model
    p.add_argument("--model_path",       required=True)
    p.add_argument("--llm_model",        default="gpt-4o-mini")
    # Guidance
    p.add_argument("--temperature",      type=float, default=0.1)
    # Rewrite
    p.add_argument("--low_thresh",       type=float, default=0.45)
    # Rerank
    p.add_argument("--top_k_llm",        type=int, default=10)
    p.add_argument("--top_k_out",        type=int, default=5)
    # Optional steps
    p.add_argument("--no_guidance",      action="store_true")
    p.add_argument("--no_rewrite",       action="store_true")
    p.add_argument("--batch_size",       type=int, default=128)
    return p.parse_args()

 


## pkl helper
def save_pkl(data: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.info("Saved → %s", path)


def load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


## guidance generation
def step_build_guidance(args, results: List[dict], ontology_raw: dict) -> List[dict]:

    logger.info("=" * 60)
    logger.info("STEP 2: Build Guidance")
    logger.info("=" * 60)

    if args.preprocessed_dir is None:
        logger.warning("--preprocessed_dir not set. Skipping guidance.")
        for r in results:
            r["guidance"] = {"column_level": {"guidance_text": ""}}
        return results

    preprocessed_dir = Path(args.preprocessed_dir)
    guidance_module  = ContextGuidance(
        model_path=args.model_path,
        ontology=ontology_raw,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )

    n_ok = n_no_record = n_no_context = n_error = 0

    for r in results:
        table_id    = r.get("table_id", "")
        record_path = preprocessed_dir / f"{table_id}.json"
        if not record_path.exists():
            matches     = sorted(preprocessed_dir.glob(f"{table_id}_col*.json"))
            record_path = matches[0] if matches else None

        if record_path is None or not record_path.exists():
            r["guidance"] = {"column_level": {"guidance_text": ""}}
            n_no_record  += 1
            continue

        with open(record_path) as f:
            record = json.load(f)
        record["top1_score"] = r.get("top1_score", 0.0)

        try:
            col_sig = guidance_module.build_column_level_signal(record)
            r["guidance"] = {
                "column_level": {"guidance_text": col_sig.guidance_text}
            }
            if col_sig.guidance_text and "No table context" not in col_sig.guidance_text:
                n_ok += 1
            else:
                n_no_context += 1
        except Exception as e:
            logger.warning("Guidance failed for %s: %s", table_id, e)
            r["guidance"] = {"column_level": {"guidance_text": ""}}
            n_error += 1

    logger.info("Guidance done: ok=%d  no_record=%d  no_context=%d  error=%d",
                n_ok, n_no_record, n_no_context, n_error)
    return results
 
## Query rewriting with LLM
def step_rewrite(args, results: List[dict], ontology_raw: dict,
                 client: OpenAI) -> List[dict]:
    """
    Query rewriting for low-confidence columns via LLM.
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Query Rewrite")
    logger.info("=" * 60)

    # Build type index for re-retrieval
    encoder     = BiEncoderInference(args.model_path)
    concept_ids = list(ontology_raw.keys())
    labels_map  = {uri: ontology_raw[uri].get("label", "") for uri in concept_ids}
    type_texts  = []
    for uri in concept_ids:
        c     = ontology_raw[uri]
        label = c.get("label", "")
        desc  = c.get("description", "")
        chain = c.get("ancestor_chain", []) or []
        pars  = [a for a in chain[:3] if a]
        t     = f"passage: {label}"
        if desc: t += f": {desc[:120]}"
        if pars: t += f". Parent types: {', '.join(pars)}"
        type_texts.append(t)

    logger.info("Encoding %d types for re-retrieval...", len(type_texts))
    type_embs = encoder.encode(type_texts,batch_size=args.batch_size)

    results = run_rewrite(
        results=results,
        concept_ids=concept_ids,
        labels_map=labels_map,
        type_embs=type_embs,
        encoder=encoder,
        client=client,
        low_thresh=args.low_thresh,
        top_k=20,
        use_guidance=not args.no_guidance,
        model=args.llm_model,
    )

    return results

 
## Reranking with LLM
def step_rerank(args, results: List[dict], ontology_raw: dict,
                client: OpenAI) -> List[dict]:

    logger.info("=" * 60)
    logger.info("STEP 4: Rerank")
    logger.info("=" * 60)

    results = run_rerank(
        results=results,
        ontology=ontology_raw,
        client=client,
        top_k_llm=args.top_k_llm,
        top_k_out=args.top_k_out,
        use_guidance=not args.no_guidance,
        use_rewrite=not args.no_rewrite,
        model=args.llm_model,
    )

    save_pkl({"results": results},
             os.path.join(args.output_dir, "final_results.pkl"))
    return results

 
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
 
    # Load ontology once, shared across all steps
    with open(args.ontology) as f:
        raw = json.load(f)
    ontology_raw = raw.get("concepts", raw)
 
    # OpenAI client created once, shared across rewrite and rerank
    client    = OpenAI()
 
    # Step 1: Retrieve
    logger.info("Loading retrieval results from %s", args.retrieval_pkl)
    results = load_pkl(args.retrieval_pkl)["results"]
 
    # Step 2: Guidance
    if not args.no_guidance:
        results = step_build_guidance(args, results, ontology_raw)
    else:
        logger.info("Guidance disabled.")
 
    # Step 3: Query rewrite
    if not args.no_rewrite:
        results = step_rewrite(args, results, ontology_raw, client)
    else:
        logger.info("Query rewrite disabled.")
 
    # Step 4: Rerank
    results = step_rerank(args, results, ontology_raw, client)
 
    logger.info("Pipeline complete. Results saved to %s", args.output_dir)
 
 
if __name__ == "__main__":
    main()
 
