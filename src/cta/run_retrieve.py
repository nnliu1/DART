from __future__ import annotations
import argparse
import logging
import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
 
from dart_encoder.input_format import format_query
from dart_encoder.retrieve import load_dataset, load_ontology, BiEncoderInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
 
