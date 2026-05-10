import ast
import pandas as pd
import numpy as np

# ---------------- CONFIG ----------------
CSV_PATH = "Annotations - triplet_analysis.csv"   # <-- change
ROW_ID_COL = "row_id"
ANN_COLS = ["aligned_triplets_1", "aligned_triplets_2", "aligned_triplets_3"]

OUT_ITEMS_CSV = "agreement_items.csv"
# ----------------------------------------


def safe_eval(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return []
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def norm_text(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def canon_triplet(tri):
    if not isinstance(tri, (list, tuple)) or len(tri) != 3:
        return None
    h, r, t = tri
    return f"{norm_text(h)}|{norm_text(r)}|{norm_text(t)}"


def parse_aligned_pairs(cell):
    """
    Each cell is a list like:
      [
        [[gt_h,gt_r,gt_t], [llm_h,llm_r,llm_t], 1.0],
        ...
      ]
    Return set of canonical pair keys: "gt||llm"
    """
    obj = safe_eval(cell)
    pairs = set()
    if not isinstance(obj, list):
        return pairs

    for item in obj:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        gt = item[0]
        llm = item[1]
        cgt = canon_triplet(gt)
        cll = canon_triplet(llm)
        if cgt and cll:
            pairs.add(f"{cgt}||{cll}")
    return pairs


def fleiss_kappa(M):
    M = np.asarray(M, dtype=float)
    m, k = M.shape
    n = np.sum(M[0, :])  # number of raters per item (assumed constant)
    if m == 0 or n <= 1:
        return np.nan

    p = np.sum(M, axis=0) / (m * n)
    P = (np.sum(M * M, axis=1) - n) / (n * (n - 1))
    Pbar = np.mean(P)
    Pe = np.sum(p * p)
    return (Pbar - Pe) / (1 - Pe) if (1 - Pe) != 0 else np.nan


def pairwise_f1(a_set, b_set):
    inter = len(a_set & b_set)
    if len(a_set) == 0 and len(b_set) == 0:
        return 1.0
    if len(a_set) == 0 or len(b_set) == 0:
        return 0.0
    prec = inter / len(a_set)
    rec = inter / len(b_set)
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def jaccard(a_set, b_set):
    if len(a_set) == 0 and len(b_set) == 0:
        return 1.0
    union = len(a_set | b_set)
    return (len(a_set & b_set) / union) if union else 0.0


def main():
    # Read CSV safely
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)

    # Collect per-row sets for each annotator
    ann_sets_by_row = []
    for _, row in df.iterrows():
        rid = row.get(ROW_ID_COL)
        row_sets = [parse_aligned_pairs(row.get(col)) for col in ANN_COLS]
        ann_sets_by_row.append((rid, row_sets))

    # Candidate items = union of all aligned pairs any annotator selected
    all_items = []
    for rid, sets in ann_sets_by_row:
        union_pairs = set().union(*sets)
        for pk in union_pairs:
            all_items.append((rid, pk))

    item_records = []
    M = []  # [count_not_aligned, count_aligned]
    agree_count = 0

    # Build a quick lookup from row_id -> annotator sets
    row_lookup = {rid: sets for rid, sets in ann_sets_by_row}

    for rid, pk in all_items:
        sets = row_lookup[rid]
        labels = [1 if pk in sets[i] else 0 for i in range(3)]
        aligned_count = sum(labels)
        not_count = 3 - aligned_count

        M.append([not_count, aligned_count])
        if aligned_count in (0, 3):
            agree_count += 1

        item_records.append({
            "row_id": rid,
            "pair_key": pk,
            "ann1": labels[0],
            "ann2": labels[1],
            "ann3": labels[2],
            "aligned_votes": aligned_count,
        })

    m = len(M)
    percent_agreement = agree_count / m if m else np.nan
    kappa = fleiss_kappa(M) if m else np.nan

    # Pairwise agreement on pooled sets
    pooled = []
    for i in range(3):
        s = set()
        for _, sets in ann_sets_by_row:
            s |= sets[i]
        pooled.append(s)

    f12 = pairwise_f1(pooled[0], pooled[1])
    f13 = pairwise_f1(pooled[0], pooled[2])
    f23 = pairwise_f1(pooled[1], pooled[2])

    j12 = jaccard(pooled[0], pooled[1])
    j13 = jaccard(pooled[0], pooled[2])
    j23 = jaccard(pooled[1], pooled[2])

    pd.DataFrame(item_records).to_csv(OUT_ITEMS_CSV, index=False)

    print("Items (candidate aligned pairs):", m)
    print("Percent agreement (all 3 agree):", round(float(percent_agreement), 4))
    print("Fleiss' kappa:", round(float(kappa), 4))
    print("Pairwise F1:  1-2=", round(f12, 4), " 1-3=", round(f13, 4), " 2-3=", round(f23, 4))
    print("Pairwise Jaccard: 1-2=", round(j12, 4), " 1-3=", round(j13, 4), " 2-3=", round(j23, 4))
    print("Saved:", OUT_ITEMS_CSV)


if __name__ == "__main__":
    main()