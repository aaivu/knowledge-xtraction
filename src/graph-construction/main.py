import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import extractor
from llms import LLMFactorySelector

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_processed_ids(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    with open(output_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {
            row["id"] for row in reader
            if any(row.get(k) and row[k] != "[]" for k in row if k.startswith("kg_"))
        }


def get_fieldnames(row: dict) -> list:
    fields = ["id"]
    i = 1
    while f"paragraph_{i}" in row:
        fields += [f"paragraph_{i}", f"kg_{i}"]
        i += 1
    return fields


def append_row(output_path: Path, row: dict, graphs: dict, fieldnames: list):
    out = {"id": row.get("id", "")}
    i = 1
    while f"paragraph_{i}" in row:
        out[f"paragraph_{i}"] = row.get(f"paragraph_{i}", "")
        out[f"kg_{i}"] = json.dumps(graphs.get(i, []), ensure_ascii=False)
        i += 1
    with open(output_path, "a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(out)


def main():
    ap = argparse.ArgumentParser(description="Knowledge graph construction pipeline")
    ap.add_argument("--input_csv", required=True, help="Input CSV: id, paragraph_1, paragraph_2, ...")
    ap.add_argument("--output_csv", default="output/kg_results.csv")
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    args = ap.parse_args()

    llm = LLMFactorySelector.get_factory(args.model)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        log.warning("No rows in input file.")
        return

    processed_ids = load_processed_ids(output_path)
    fieldnames = get_fieldnames(rows[0])

    if not output_path.exists():
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for row in rows:
        row_id = row.get("id", "")
        if row_id in processed_ids:
            continue

        paragraphs = []
        i = 1
        while f"paragraph_{i}" in row and row.get(f"paragraph_{i}") and row[f"paragraph_{i}"].strip():
            paragraphs.append(row[f"paragraph_{i}"].strip())
            i += 1

        log.info(f"Processing id={row_id} ({len(paragraphs)} paragraphs)")
        graphs = extractor.extract(paragraphs, llm) if paragraphs else {}
        append_row(output_path, row, graphs, fieldnames)

    log.info(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()