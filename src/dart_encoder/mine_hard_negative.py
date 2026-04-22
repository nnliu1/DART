from __future__ import annotations

import json
import argparse
import logging
from typing import Dict, List

import torch
from transformers import AutoTokenizer
from input_format import format_query, format_type
from model import BiEncoder
from data_types import OntologyType


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("hard_neg_mining")


## load ontology 
def load_ontology(ontology_path: str) -> Dict[str, OntologyType]:
    ontology: Dict[str, OntologyType] = {}
    with open(ontology_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # Fix double-quoted labels from label_db serialization
            for field in ["label", "description"]:
                v = d.get(field, "")
                if isinstance(v, str) and v.startswith('"') and v.endswith('"'):
                    d[field] = v[1:-1]
            ontology[d["qid"]] = OntologyType(
                qid=d["qid"],
                label=d.get("label", ""),
                description=d.get("description", ""),
                aliases=d.get("aliases", []),
                parents=d.get("parents", []),
                examples=d.get("examples", []),
            )
    return ontology

## main function
def mine_hard_negatives(
    model_path: str,
    train_path: str,
    ontology_path: str,
    output_path: str,
    top_k: int = 50,          # retrieve top-K candidates
    n_hard_negs: int = 5,     # how many hard negatives to keep per sample
    min_rank: int = 2,        # start from rank 2 (rank 1 might be GT)
    max_rank: int = 50,       # don't use very distant negatives
    min_score: float = 0.1,   # minimum similarity score to be a hard negative
    batch_size: int = 256,
    max_length: int = 256,
    max_cells: int = 10,
    max_parents: int = 3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
 
    # Load ontology
    ontology = load_ontology(ontology_path)
    logger.info("Loaded %d ontology types", len(ontology))
 
    # Load training samples
    train_records = []
    with open(train_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["positive_type_qid"] in ontology:
                train_records.append(d)
    logger.info("Loaded %d training samples", len(train_records))
 
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = BiEncoder(model_path).to(device)
    model.eval()
    logger.info("Model loaded on %s", device)
 
    def encode(texts: List[str]) -> torch.Tensor:
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tokenizer(
                    texts[i : i + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                enc  = {k: v.to(device) for k, v in enc.items()}
                emb  = model.encode(enc) 
                all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0)
 
    # Build type index
    all_qids  = list(ontology.keys())
    all_texts = [
        format_type(ontology[q], max_parents)  # reuse formatting.py
        for q in all_qids
    ]
    logger.info("Encoding %d types...", len(all_texts))
    type_embs  = encode(all_texts)              # (N_types, H)
    qid_to_idx = {q: i for i, q in enumerate(all_qids)}
 
    # Encode queries
    query_texts = [
        format_query(   
            r["anchor_header"],
            r["anchor_cells"],
            max_cells,
        )
        for r in train_records
    ]
    logger.info("Encoding %d queries...", len(query_texts))
    query_embs = encode(query_texts)            # (M, H)
 
    scores = query_embs @ type_embs.T          # (M, N_types)
 
    # Mine hard negatives
    mined: Dict[str, List[str]] = {}
    stats = {"total": 0, "has_mined": 0, "fallback": 0}
 
    for i, r in enumerate(train_records):
        pos_qid  = r["positive_type_qid"]
        table_id = r.get("table_id", "")
        col_idx  = r.get("col_index", -1)
        key      = f"{table_id}___{col_idx}"
 
        pos_idx = qid_to_idx.get(pos_qid)
        if pos_idx is None:
            continue
 
        # Select hard negatives: not GT, within rank range, above score threshold
        ranked     = scores[i].argsort(descending=True).tolist()
        row_scores = scores[i]
 
        hard_negs = []
        for rank_idx, type_idx in enumerate(ranked[:max_rank], start=1):
            if len(hard_negs) >= n_hard_negs:
                break
            if type_idx == pos_idx:
                continue
            if rank_idx < min_rank:
                continue
            if row_scores[type_idx].item() < min_score:
                continue
            hard_negs.append(all_qids[type_idx])
 
        stats["total"] += 1
 
        if len(hard_negs) >= 3:
            # Good mined negatives
            mined[key] = hard_negs
            stats["has_mined"] += 1
        else:
            # Fall back to original hard negatives from training data
            orig_negs = [
                q for q in r.get("hard_negative_type_qids", [])
                if q in ontology
            ]
            mined[key] = (hard_negs + orig_negs)[:n_hard_negs]
            stats["fallback"] += 1
 
    logger.info(
        "Mining complete: %d total, %d mined, %d fallback",
        stats["total"], stats["has_mined"], stats["fallback"],
    )
 
    # Save
    with open(output_path, "w") as f:
        json.dump(mined, f)
    logger.info("Saved mined hard negatives to %s", output_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Mine hard negatives using trained model")
    p.add_argument("--model_path",    required=True,
                   help="Path to trained best_model checkpoint")
    p.add_argument("--train_path",    required=True,
                   help="Path to train.jsonl")
    p.add_argument("--ontology_path", required=True,
                   help="Path to ontology_types_fixed.jsonl")
    p.add_argument("--output_path",   required=True,
                   help="Output path for mined hard negatives JSON")
    p.add_argument("--top_k",         type=int,   default=50)
    p.add_argument("--n_hard_negs",   type=int,   default=5)
    p.add_argument("--min_rank",      type=int,   default=2)
    p.add_argument("--max_rank",      type=int,   default=50)
    p.add_argument("--min_score",     type=float, default=0.1)
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--max_cells",     type=int,   default=10)
    p.add_argument("--max_parents",   type=int,   default=3)
    args = p.parse_args()

    mine_hard_negatives(
        model_path=args.model_path,
        train_path=args.train_path,
        ontology_path=args.ontology_path,
        output_path=args.output_path,
        top_k=args.top_k,
        n_hard_negs=args.n_hard_negs,
        min_rank=args.min_rank,
        max_rank=args.max_rank,
        min_score=args.min_score,
        batch_size=args.batch_size,
        max_cells=args.max_cells,
        max_parents=args.max_parents,
    )