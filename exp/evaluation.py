from __future__ import annotations
import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_pkl", required=True,
                   help="final_results.pkl from llm_rerank eval mode")
    p.add_argument("--ontology",    required=True,
                   help="Ontology JSON with 'concepts' key")
    p.add_argument("--output_json",    required=True,
                   help="Output JSON file path")
    p.add_argument("--top_n",       type=int, default=10,
                   help="Show top N worst/best performing types")
    return p.parse_args()



## ontology helper
def get_label(uri: str, ontology: dict) -> str:
    concept = ontology.get(uri) or {}
    return concept.get("label") or uri.split("/")[-1]
 
 
def get_ancestor_chain(uri: str, ontology: dict) -> list:
    """Return ancestor labels ordered from direct parent outward."""
    concept = ontology.get(uri) or {}
    return [a.lower() for a in (concept.get("ancestor_chain") or [])]



## evaluate criteria
def is_exact(pred_uri: str, gold_uri: str) -> bool:
    return pred_uri == gold_uri
 
 
def is_approximate(pred_uri: str, gold_uri: str, ontology: dict) -> bool:
    """
    Approximate correctness: exact match OR direct child OR direct parent.
 
    Direct child  (over-specific): pred's direct parent == gold
    Direct parent (over-general):  gold's direct parent == pred
    """
    if pred_uri == gold_uri:
        return True
 
    gold_label = get_label(gold_uri, ontology).lower()
    pred_label = get_label(pred_uri, ontology).lower()
 
    # pred is a DIRECT CHILD of gold (pred's parent == gold)
    pred_ancestors = get_ancestor_chain(pred_uri, ontology)
    if pred_ancestors and pred_ancestors[0] == gold_label:
        return True
 
    # pred is a DIRECT PARENT of gold (gold's parent == pred)
    gold_ancestors = get_ancestor_chain(gold_uri, ontology)
    if gold_ancestors and gold_ancestors[0] == pred_label:
        return True
 
    return False

## F1 helper
def compute_f1_rows(tp_dict, fp_dict, fn_dict, ontology) -> list:
    rows = []
    for uri in set(list(tp_dict) + list(fn_dict)):
        tp = tp_dict[uri]; fp = fp_dict[uri]; fn = fn_dict[uri]
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        rows.append({
            "label":   get_label(uri, ontology),
            "p": p, "r": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
            "support": tp + fn,
        })
    return rows
 
 
def aggregate_f1(rows, tp_dict, fp_dict, fn_dict):
    macro_f1 = sum(x["f1"] for x in rows) / len(rows) if rows else 0.0
    s_tp = sum(tp_dict.values())
    s_fp = sum(fp_dict.values())
    s_fn = sum(fn_dict.values())
    micro_p  = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
    micro_r  = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    return micro_f1, micro_p, micro_r, macro_f1

## main evaluation function
def compute_metrics(
    results_pkl: str,
    ontology:    dict,
    top_n:       int,
    output_json: str = None,
):
    with open(results_pkl, "rb") as f:
        saved = pickle.load(f)
    results  = saved["results"]
    k_values = [1, 10]
 
    exact_hits     = {k: 0 for k in k_values}
    approx_hits    = {k: 0 for k in k_values}
    exact_mrr_sum  = 0.0
    approx_mrr_sum = 0.0
 
    # Exact F1
    ex_tp = defaultdict(int)
    ex_fp = defaultdict(int)
    ex_fn = defaultdict(int)
 
    # Approximate F1
    ah_tp = defaultdict(int)
    ah_fp = defaultdict(int)
    ah_fn = defaultdict(int)
 
    approx_corrections = []
    total = 0
 
    for r in results:
        gold_uri = r.get("gold_uri", "")
        if not gold_uri:
            continue
 
        pred_uris = r.get("_pred_uris") or r.get("cand_uris", [])
        pred_top1 = pred_uris[0] if pred_uris else ""
        total    += 1
 
        # Ranking: Hit@k and MRR
        exact_rank = next(
            (rank for rank, uri in enumerate(pred_uris, 1) if is_exact(uri, gold_uri)),
            None,
        )
        approx_rank = next(
            (rank for rank, uri in enumerate(pred_uris, 1)
             if is_approximate(uri, gold_uri, ontology)),
            None,
        )
        for k in k_values:
            exact_hits[k]  += int(exact_rank  is not None and exact_rank  <= k)
            approx_hits[k] += int(approx_rank is not None and approx_rank <= k)
        exact_mrr_sum  += (1.0 / exact_rank)  if exact_rank  else 0.0
        approx_mrr_sum += (1.0 / approx_rank) if approx_rank else 0.0
 
        # Exact F1
        if pred_top1 == gold_uri:
            ex_tp[gold_uri] += 1
        else:
            ex_fn[gold_uri] += 1
            if pred_top1:
                ex_fp[pred_top1] += 1
 
        # Approximate F1
        if pred_top1 and is_approximate(pred_top1, gold_uri, ontology):
            ah_tp[gold_uri] += 1
        else:
            ah_fn[gold_uri] += 1
            if pred_top1:
                ah_fp[pred_top1] += 1
 
        # Approx correction cases
        if pred_top1 and pred_top1 != gold_uri and \
                is_approximate(pred_top1, gold_uri, ontology):
            pred_ancestors = get_ancestor_chain(pred_top1, ontology)
            gold_label     = get_label(gold_uri, ontology).lower()
            direction = (
                "over-specific (child)"
                if pred_ancestors and pred_ancestors[0] == gold_label
                else "over-general (parent)"
            )
            approx_corrections.append({
                "gold":      get_label(gold_uri, ontology),
                "pred":      get_label(pred_top1, ontology),
                "direction": direction,
            })
 
    # Aggregate
    def safe(x, n):
        return round(x / n, 4) if n > 0 else 0.0
 
    ex_rows = compute_f1_rows(ex_tp, ex_fp, ex_fn, ontology)
    ah_rows = compute_f1_rows(ah_tp, ah_fp, ah_fn, ontology)
 
    micro_f1,    micro_p,    micro_r,    macro_f1    = aggregate_f1(ex_rows, ex_tp, ex_fp, ex_fn)
    ah_micro_f1, ah_micro_p, ah_micro_r, ah_macro_f1 = aggregate_f1(ah_rows, ah_tp, ah_fp, ah_fn)
 

    output = {
        "total": total,
        "exact": {
            "Hit@1":    safe(exact_hits[1],   total),
            "Hit@10":   safe(exact_hits[10],  total),
            "MRR":      safe(exact_mrr_sum,   total),
            "micro_f1": round(micro_f1,    4),
            "micro_p":  round(micro_p,     4),
            "micro_r":  round(micro_r,     4),
        },
        "approximate_hierarchical": {
            "AH-Hit@1":    safe(approx_hits[1],  total),
            "AH-Hit@10":   safe(approx_hits[10], total),
            "AH-MRR":      safe(approx_mrr_sum,  total),
            "micro_f1":    round(ah_micro_f1, 4),
            "micro_p":     round(ah_micro_p,  4),
            "micro_r":     round(ah_micro_r,  4),
            "n_corrections": len(approx_corrections),
        },
        "per_type_exact": sorted(
            [{"label": r["label"], "f1": round(r["f1"], 4),
              "p": round(r["p"], 4), "r": round(r["r"], 4),
              "support": r["support"]}
             for r in ex_rows],
            key=lambda x: -x["f1"],
        ),
    }
 
    # Save JSON
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nMetrics saved → {output_json}")
 
    return output
 


## entry point
if __name__ == "__main__":
    args = parse_args()
    with open(args.ontology) as f:
        raw = json.load(f)
    ontology = raw.get("concepts", raw)
    compute_metrics(
        results_pkl=args.results_pkl,
        ontology=ontology,
        top_n=args.top_n,
        output_json=args.output_json,
    )
