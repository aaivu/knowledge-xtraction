from __future__ import annotations

import argparse
import json

from kg_compare.io_utils import read_excel, write_excel
from kg_compare.parse_utils import parse_triplets
from kg_compare.compare_kea import compare_kgs_kea, KEAConfig


def safe_json_for_excel(obj) -> str:
    """Convert object to JSON string for Excel cell storage."""
    return json.dumps(obj, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description="KEA Triplet Analysis (head/rel/tail cosine; th=0.76)")

    ap.add_argument("--in", dest="inp", required=True, help="Input Excel path")
    ap.add_argument("--out", dest="out", required=True, help="Output Excel path")
    ap.add_argument("--sheet", dest="sheet", default=0, help="Sheet name or index (default 0)")

    ap.add_argument("--col_gold", default="gold_kg", help="Column name for gold triplets")
    ap.add_argument("--col_llm", default="llm_kg", help="Column name for LLM triplets")

    # KEA thresholds
    ap.add_argument("--cos_th", type=float, default=0.76, help="Cosine threshold for head/rel/tail")
    ap.add_argument("--match_floor", type=float, default=0.50, help="Below this overall => mark as missing")
    ap.add_argument("--model_name", type=str, default="paraphrase-MiniLM-L6-v2", help="SentenceTransformer model")

    args = ap.parse_args()

    df = read_excel(args.inp, sheet_name=args.sheet)

    cfg = KEAConfig(
        cos_th=args.cos_th,
        match_floor=args.match_floor,
        model_name=args.model_name,
    )

    aligned_col = []
    rel_sub_col = []
    ent_sub_col = []
    missing_col = []
    extra_col = []
    scored_col = []

    for _, row in df.iterrows():
        gold_trips = parse_triplets(row.get(args.col_gold))
        llm_trips = parse_triplets(row.get(args.col_llm))

        res = compare_kgs_kea(gold_trips, llm_trips, cfg)

        aligned_col.append(safe_json_for_excel(res["aligned"]))
        rel_sub_col.append(safe_json_for_excel(res["relation_substitution"]))
        ent_sub_col.append(safe_json_for_excel(res["entity_substitution"]))
        missing_col.append(safe_json_for_excel(res["missing"]))
        extra_col.append(safe_json_for_excel(res["extra"]))
        scored_col.append(safe_json_for_excel(res["matches_scored"]))

    df["aligned_triplets"] = aligned_col
    df["relation_substitution_triplets"] = rel_sub_col
    df["entity_substitution_triplets"] = ent_sub_col
    df["missing_triplets"] = missing_col
    df["extra_triplets"] = extra_col
    df["best_match_scores"] = scored_col

    write_excel(df, args.out)


if __name__ == "__main__":
    main()
