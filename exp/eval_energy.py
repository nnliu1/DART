from __future__ import annotations
import argparse
import csv
import json
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_pkl", required=True)
    p.add_argument("--gt_csv",      required=True)
    p.add_argument("--ontology",    required=True)
    p.add_argument("--output_json", default=None,
                   help="Path to save metrics JSON. Defaults to same "
                        "directory as results_pkl as metrics.json")
    p.add_argument("--top_n",       type=int, default=10)
    return p.parse_args()



# Ontology helpers
def get_label(uri: str, ontology: dict) -> str:
    concept = ontology.get(uri) or {}
    return concept.get("label") or uri.split("#")[-1]


def get_ancestor_chain(uri: str, ontology: dict) -> list:
    """Return ancestor labels (lower-cased), ordered direct parent first."""
    concept = ontology.get(uri) or {}
    return [a.lower() for a in (concept.get("ancestor_chain") or [])]


# GT loading
def load_gt(gt_csv: str, ontology: dict) -> tuple:
    label_to_uri = {}
    for uri, concept in ontology.items():
        label = (concept.get("label") or "").strip()
        if label:
            label_lower = label.lower()
            label_to_uri[label_lower] = uri
            # Also index underscore variant (e.g. "physical object" → "physical_object")
            label_to_uri.setdefault(label_lower.replace(" ", "_"), uri)
            # Also index space variant (e.g. "physical_object" → "physical object")
            label_to_uri.setdefault(label_lower.replace("_", " "), uri)
        local = uri.split("#")[-1] if "#" in uri else uri.rstrip("/").split("/")[-1]
        if local:
            local_lower = local.lower()
            label_to_uri.setdefault(local_lower, uri)
            label_to_uri.setdefault(local_lower.replace("_", " "), uri)
            label_to_uri.setdefault(local_lower.replace(" ", "_"), uri)

    def resolve_uri(label: str) -> str:
        if not label or label.strip() == "-":
            return ""
        key = label.strip().lower()
        if key in label_to_uri:
            return label_to_uri[key]
        # Try underscore/space variants
        key_underscore = key.replace(" ", "_")
        key_space      = key.replace("_", " ")
        if key_underscore in label_to_uri:
            return label_to_uri[key_underscore]
        if key_space in label_to_uri:
            return label_to_uri[key_space]
        # Full normalisation (remove all separators)
        norm = key.replace("_", "").replace(" ", "")
        for k, v in label_to_uri.items():
            if k.replace("_", "").replace(" ", "") == norm:
                return v
        return ""

    gt_map        = {}
    gt_labels_map = {}
    table_uri_map = {}
    n_unresolved  = 0

    with open(gt_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            if row[0].strip().lower() in ("table_id", "tableid", "file"):
                continue

            table_id = row[0].strip()
            try:
                col_idx = int(float(row[1].strip()))
            except ValueError:
                continue

            col_name   = row[2].strip()
            raw_labels = [x.strip() for x in row[3:]
                          if x.strip() and x.strip() != "-"]
            correct_uris   = set()
            correct_labels = set()
            for label in raw_labels:
                uri = resolve_uri(label)
                if uri:
                    correct_uris.add(uri)
                    correct_labels.add(label)
                else:
                    n_unresolved += 1

            key = f"{table_id}_col{col_idx}"
            if correct_uris:
                gt_map[key]        = correct_uris
                gt_labels_map[key] = correct_labels
                if table_id not in table_uri_map:
                    table_uri_map[table_id] = []
                table_uri_map[table_id].append(
                    (col_idx, col_name, correct_uris, correct_labels)
                )

    print(f"GT entries loaded : {len(gt_map)}")
    if n_unresolved > 0:
        print(f"Unresolved labels : {n_unresolved}")
    return gt_map, gt_labels_map, table_uri_map



def is_exact(pred_uri: str, correct_uris: set) -> bool:
    return pred_uri in correct_uris


def is_approximate(
    pred_uri:     str,
    correct_uris: set,
    ontology:     dict,
) -> bool:

    if pred_uri in correct_uris:
        return True

    pred_ancestors = get_ancestor_chain(pred_uri, ontology)
    pred_label     = get_label(pred_uri, ontology).lower()

    for gt_uri in correct_uris:
        gt_label      = get_label(gt_uri, ontology).lower()
        gt_ancestors  = get_ancestor_chain(gt_uri, ontology)

        # Case 1: pred is a DIRECT CHILD of GT (pred's direct parent == GT)
        if pred_ancestors and pred_ancestors[0] == gt_label:
            return True

        # Case 2: pred is a DIRECT PARENT of GT (GT's direct parent == pred)
        if gt_ancestors and gt_ancestors[0] == pred_label:
            return True

    return False


# Main evaluation
def evaluate(args):
    # Load ontology
    with open(args.ontology) as f:
        ontology = json.load(f)["concepts"]
 
    # Load GT
    gt_map, gt_labels_map, table_uri_map = load_gt(args.gt_csv, ontology)
 
    # Load results
    with open(args.results_pkl, "rb") as f:
        saved = pickle.load(f)
    results = saved["results"]
 
    k_values = [1, 5, 10, 20]
 
    exact_hits  = {k: 0 for k in k_values}
    approx_hits = {k: 0 for k in k_values}
    exact_mrr_sum  = 0.0
    approx_mrr_sum = 0.0
 
    # Per-type F1 (multi-label: credit all GT URIs)
    ex_tp = defaultdict(int)
    ex_fp = defaultdict(int)
    ex_fn = defaultdict(int)
 
    ah_tp = defaultdict(int)
    ah_fp = defaultdict(int)
    ah_fn = defaultdict(int)
 
    approx_corrections = []
    total = 0
    no_gt = 0
 
    for r in results:
        table_id  = r.get("table_id", "")
        pred_uris = r.get("_pred_uris") or r.get("cand_uris", [])
 
        # Resolve GT by header name, fallback to gold_uri
        header = ""
        m = re.search(r"Header:\s*([^|]+)", r.get("column_text", ""))
        if m:
            header = m.group(1).strip().lower()
 
        correct_uris   = None
        correct_labels = None
 
        if table_id in table_uri_map:
            for entry in table_uri_map[table_id]:
                col_idx, col_name, c_uris, c_labels = entry
                if header and col_name.strip().lower() == header:
                    correct_uris, correct_labels = c_uris, c_labels
                    break
            if correct_uris is None:
                gold_uri = r.get("gold_uri", "")
                if gold_uri:
                    for entry in table_uri_map[table_id]:
                        col_idx, col_name, c_uris, c_labels = entry
                        if gold_uri in c_uris:
                            correct_uris, correct_labels = c_uris, c_labels
                            break
 
        if correct_uris is None:
            no_gt += 1
            continue
 
        total    += 1
        pred_top1 = pred_uris[0] if pred_uris else ""
 
        # Hit@k and MRR
        exact_rank = next(
            (rank for rank, uri in enumerate(pred_uris, 1)
             if is_exact(uri, correct_uris)), None,
        )
        approx_rank = next(
            (rank for rank, uri in enumerate(pred_uris, 1)
             if is_approximate(uri, correct_uris, ontology)), None,
        )
        for k in k_values:
            if exact_rank  is not None and exact_rank  <= k: exact_hits[k]  += 1
            if approx_rank is not None and approx_rank <= k: approx_hits[k] += 1
        exact_mrr_sum  += (1.0 / exact_rank)  if exact_rank  else 0.0
        approx_mrr_sum += (1.0 / approx_rank) if approx_rank else 0.0
 
        # Per-type F1: credit all GT URIs
        matched_exact  = is_exact(pred_top1, correct_uris)
        matched_approx = is_approximate(pred_top1, correct_uris, ontology) \
                         if pred_top1 else False
 
        for gt_uri in correct_uris:
            if matched_exact:  ex_tp[gt_uri] += 1
            else:              ex_fn[gt_uri] += 1
            if matched_approx: ah_tp[gt_uri] += 1
            else:              ah_fn[gt_uri] += 1
 
        if not matched_exact  and pred_top1: ex_fp[pred_top1] += 1
        if not matched_approx and pred_top1: ah_fp[pred_top1] += 1
 
        # Approx correction tracking
        if pred_top1 and not matched_exact and matched_approx:
            pred_ancestors = get_ancestor_chain(pred_top1, ontology)
            pred_label     = get_label(pred_top1, ontology).lower()
            direction      = "unknown"
            for gt_uri in correct_uris:
                gt_label     = get_label(gt_uri, ontology).lower()
                gt_ancestors = get_ancestor_chain(gt_uri, ontology)
                if pred_ancestors and pred_ancestors[0] == gt_label:
                    direction = "over-specific (child)"; break
                if gt_ancestors and gt_ancestors[0] == pred_label:
                    direction = "over-general (parent)"; break
            approx_corrections.append({
                "gold":      sorted(correct_labels),
                "pred":      get_label(pred_top1, ontology),
                "direction": direction,
            })
 
    # Aggregate F1
    def f1_rows(tp_d, fp_d, fn_d):
        rows = []
        for uri in set(list(tp_d) + list(fn_d)):
            tp = tp_d[uri]; fp = fp_d[uri]; fn = fn_d[uri]
            p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            rows.append({"label": get_label(uri, ontology),
                         "f1": f1, "p": p, "r": r, "support": tp + fn})
        return rows
 
    def aggregate(rows, tp_d, fp_d, fn_d):
        macro = sum(x["f1"] for x in rows) / len(rows) if rows else 0.0
        s_tp  = sum(tp_d.values())
        s_fp  = sum(fp_d.values())
        s_fn  = sum(fn_d.values())
        mp    = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
        mr    = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0
        mf1   = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0
        return mf1, mp, mr, macro
 
    ex_rows = f1_rows(ex_tp, ex_fp, ex_fn)
    ah_rows = f1_rows(ah_tp, ah_fp, ah_fn)
    micro_f1,    micro_p,    micro_r,    macro_f1    = aggregate(ex_rows, ex_tp, ex_fp, ex_fn)
    ah_micro_f1, ah_micro_p, ah_micro_r, ah_macro_f1 = aggregate(ah_rows, ah_tp, ah_fp, ah_fn)
 
    def safe(x):
        return round(x / total, 4) if total > 0 else 0.0
 
    metrics = {
        # Exact
        **{f"Hit@{k}":    safe(exact_hits[k])  for k in k_values},
        "MRR":            safe(exact_mrr_sum),
        "micro_f1":       round(micro_f1, 4),
        "macro_f1":       round(macro_f1, 4),
        "micro_p":        round(micro_p,  4),
        "micro_r":        round(micro_r,  4),
        # Approximate hierarchical
        **{f"AH-Hit@{k}": safe(approx_hits[k]) for k in k_values},
        "AH-MRR":         safe(approx_mrr_sum),
        "AH-micro_f1":    round(ah_micro_f1, 4),
        "AH-macro_f1":    round(ah_macro_f1, 4),
        "AH-micro_p":     round(ah_micro_p,  4),
        "AH-micro_r":     round(ah_micro_r,  4),
        "n_corrections":  len(approx_corrections),
        "n_over_specific": len([c for c in approx_corrections
                                if "child"  in c["direction"]]),
        "n_over_general":  len([c for c in approx_corrections
                                if "parent" in c["direction"]]),
        "per_type_exact": sorted(
            [{"label": r["label"], "f1": round(r["f1"], 4),
              "p": round(r["p"], 4), "r": round(r["r"], 4),
              "support": r["support"]}
             for r in ex_rows],
            key=lambda x: -x["f1"],
        ),
    }
 
    # Save JSON
    out_path = args.output_json or os.path.join(
        os.path.dirname(os.path.abspath(args.results_pkl)), "metrics.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved → {out_path}")
 
    return metrics
 



# Entry point
if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
