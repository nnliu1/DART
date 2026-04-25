from __future__ import annotations
import argparse
import logging
import pickle
from pathlib import Path
from typing import List, Tuple
import json

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_query(header: str, cells: List[str], max_cells: int = 10) -> str:
    cells_str = ", ".join(cells[:max_cells])
    return f"query: Header: {header} | Cells: {cells_str}"


def format_type(label: str, description: str, parents: str,
                max_parents: int = 3) -> str:
    text = f"passage: {label}"
    if description:
        text += f": {description}"
    if parents:
        if isinstance(parents, list):
            parent_str = ", ".join(parents[:max_parents])
        else:
            parent_str = parents
        if parent_str:
            text += f". Parent types: {parent_str}"
    return text


class BiEncoderInference:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        logger.info("Model loaded from %s on %s", model_path, self.device)

    def mean_pool(self, model_output, attention_mask) -> torch.Tensor:
        token_emb = model_output.last_hidden_state
        mask      = attention_mask.unsqueeze(-1).float()
        return (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 128) -> torch.Tensor:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc   = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=256, return_tensors="pt",
            )
            enc  = {k: v.to(self.device) for k, v in enc.items()}
            out  = self.model(**enc)
            emb  = self.mean_pool(out, enc["attention_mask"])
            emb  = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0)



def load_ontology(
    ontology_path: str,
    max_parents:   int = 3,
) -> Tuple[List[str], List[str], List[str]]:
    with open(ontology_path) as f:
        data = json.load(f)

    concepts    = data["concepts"]
    concept_ids = []
    labels      = []
    type_texts  = []

    for uri, info in concepts.items():
        label       = info.get("label", "")
        description = info.get("description", "")
        parents     = info.get("parents", "")
        if not parents:
            chain   = info.get("ancestor_chain", [])
            parents = chain[:max_parents] if chain else []

        concept_ids.append(uri)
        labels.append(label)
        type_texts.append(format_type(label, description, parents, max_parents))

    logger.info("Loaded %d DBpedia types", len(concept_ids))
    return concept_ids, labels, type_texts


def load_dataset(data_dir: str) -> List[dict]:
    records = []
    for f in sorted(Path(data_dir).glob("*.json")):
        with open(f) as fp:
            records.append(json.load(fp))
    logger.info("Loaded %d T2Dv2 columns", len(records))
    return records

def run_retrieval(
    model_path: str,
    ontology_path: str,
    data_dir: str,
    output_pkl: str,
    top_k: int = 20,
    max_cells: int = 10,
    max_parents: int = 3,
    batch_size: int = 128,
    device: str = "cuda",
):
    # Build model and type index
    model = BiEncoderInference(model_path, device=device)
    concept_ids, labels, type_texts = load_ontology(
        ontology_path, max_parents=max_parents
    )
    logger.info("Encoding %d ontology types...", len(type_texts))
    type_embs  = model.encode(type_texts, batch_size=batch_size)   # (N, H)
    uri_to_idx = {uri: i for i, uri in enumerate(concept_ids)}
 
    # Load data and encode queries
    records     = load_dataset(data_dir)
    query_texts = [
        format_query(
            r.get("pk_col_header_clean") or r.get("pk_col_header_raw", ""),
            r.get("cell_samples_clean")  or r.get("cell_samples_raw", []),
            max_cells,
        )
        for r in records
    ]
    logger.info("Encoding %d queries...", len(query_texts))
    query_embs = model.encode(query_texts, batch_size=batch_size)  # (M, H)
 
    scores = query_embs @ type_embs.T                              # (M, N)

    all_results = []
    for i, r in enumerate(records):
        gt_uri  = r.get("gt_uri", "")
        gt_idx  = uri_to_idx.get(gt_uri)
        gt_type = r.get("gt_ontology", "")
 
        ranked       = scores[i].argsort(descending=True).tolist()
        gt_rank      = ranked.index(gt_idx) + 1 if gt_idx is not None else top_k + 1
 
        top_k_idx    = ranked[:top_k]
        top_k_uris   = [concept_ids[j]     for j in top_k_idx]
        top_k_labels = [labels[j]           for j in top_k_idx]
        top_k_scores = [scores[i][j].item() for j in top_k_idx]
 
        top1_score = top_k_scores[0] if top_k_scores else 0.0
        top2_score = top_k_scores[1] if len(top_k_scores) > 1 else 0.0
 
        all_results.append({
            "table_id":      r.get("table_id", ""),
            "column_text":   query_texts[i],
            "gold_uri":      gt_uri,
            "gold_ontology": gt_type,
            "cand_uris":     top_k_uris,
            "cand_labels":   top_k_labels,
            "cand_scores":   top_k_scores,
            "top1_score":    top1_score,
            "top2_score":    top2_score,
            "margin":        top1_score - top2_score,
            "gold_rank":     gt_rank,
            "gold_in_topk":  gt_rank <= top_k,
        })
 
    # Save PKL
    Path(output_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump({"results": all_results}, f)
    logger.info("Saved %d results → %s", len(all_results), output_pkl)

# Entry point
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run CTA retrieval and save results as PKL")
    p.add_argument("--model_path",    required=True, help="Path to trained model checkpoint")
    p.add_argument("--ontology_path", required=True, help="Path to ontology JSON")
    p.add_argument("--data_dir",      required=True, help="Directory with column JSON files")
    p.add_argument("--output_pkl",    required=True, help="Output PKL path")
    p.add_argument("--top_k",         type=int, default=20)
    p.add_argument("--max_cells",     type=int, default=10)
    p.add_argument("--max_parents",   type=int, default=3)
    p.add_argument("--batch_size",    type=int, default=128)
    p.add_argument("--device",        default="cuda")
    args = p.parse_args()
 
    run_retrieval(
        model_path=args.model_path,
        ontology_path=args.ontology_path,
        data_dir=args.data_dir,
        output_pkl=args.output_pkl,
        top_k=args.top_k,
        max_cells=args.max_cells,
        max_parents=args.max_parents,
        batch_size=args.batch_size,
        device=args.device,
    )
 
