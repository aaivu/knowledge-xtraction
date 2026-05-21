from __future__ import annotations

import argparse
import json

from kg_compare.io_utils import read_excel, write_excel
from kg_compare.parse_utils import parse_triplets
from kg_compare.compare import compare_kgs_three_class, CompareConfig


def _safe_json(obj) -> str:
    """Store lists/dicts nicely inside Excel cells."""
    return json.dumps(obj, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(
        description="Triplet Analysis: aligned by whole-triplet similarity, else cosine split (relation vs entity diff)"
    )

    # Paths
    ap.add_argument("--in", dest="inp", required=True, help="Input Excel path")
    ap.add_argument("--out", dest="out", required=True, help="Output Excel path")
    ap.add_argument("--sheet", dest="sheet", default=0, help="Sheet name or index (default 0)")

    # Input columns
    ap.add_argument("--col_gold", default="kg_gold", help="Column name for gold triplets")
    ap.add_argument("--col_llm", default="kg_llm", help="Column name for LLM triplets")

    # Stage 1: aligned threshold (whole triplet)
    ap.add_argument("--aligned_threshold", type=float, default=0.73,
                    help="Whole-triplet similarity >= this => aligned")

    # Stage 2: cosine thresholds (only used if NOT aligned)
    ap.add_argument("--entity_cos_th", type=float, default=0.82,
                    help="Cosine threshold for head and tail to be considered matching entities")
    ap.add_argument("--relation_cos_th", type=float, default=0.75,
                    help="Cosine threshold for relation to be considered matching")

    # Embedding model
    ap.add_argument("--model_name", type=str, default="paraphrase-MiniLM-L6-v2",
                    help="SentenceTransformer model name")

    args = ap.parse_args()

    # Read input
    df = read_excel(args.inp, sheet_name=args.sheet)

    # Config
    cfg = CompareConfig(
        aligned_threshold=args.aligned_threshold,
        entity_cos_th=args.entity_cos_th,
        relation_cos_th=args.relation_cos_th,
        model_name=args.model_name,
    )

    aligned_col = []
    rel_diff_col = []
    ent_diff_col = []
    scored_col = []

    # Row-by-row analysis
    for _, row in df.iterrows():
        gold_trips = parse_triplets(row.get(args.col_gold))
        llm_trips = parse_triplets(row.get(args.col_llm))

        res = compare_kgs_three_class(gold_trips, llm_trips, cfg)

        aligned_col.append(_safe_json(res["aligned"]))
        rel_diff_col.append(_safe_json(res["relation_different"]))
        ent_diff_col.append(_safe_json(res["entity_different"]))
        scored_col.append(_safe_json(res["matches_scored"]))

    # Append outputs (same rows as input)
    df["aligned_triplets"] = aligned_col
    df["relation_different_triplets"] = rel_diff_col
    df["entity_different_triplets"] = ent_diff_col

    # Trace/debug output (gold → best LLM + scores + decision)
    df["best_match_scores"] = scored_col

    # Write output Excel
    write_excel(df, args.out)


if __name__ == "__main__":
    main()
