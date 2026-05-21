# import ast
# import pandas as pd


# def parse_triplet_list(cell):
#     """
#     cell can be:
#       - a Python list string: [[...],[...]]
#       - empty / NaN
#     returns list of triplets: [(h,r,t), ...]
#     """
#     if pd.isna(cell):
#         return []
#     s = str(cell).strip()
#     if not s:
#         return []
#     try:
#         obj = ast.literal_eval(s)
#     except Exception:
#         return []

#     # Your aligned_triplets sometimes contains tuples like:
#     # [(gold_triplet, llm_triplet, score), ...]
#     # or direct triplets [[h,r,t], ...]
#     triplets = []

#     if isinstance(obj, list):
#         for item in obj:
#             # case 1: direct triplet
#             if isinstance(item, (list, tuple)) and len(item) == 3 and all(isinstance(x, str) for x in item):
#                 triplets.append(tuple(item))
#             # case 2: (gt, llm, score) where gt is [h,r,t]
#             elif isinstance(item, (list, tuple)) and len(item) >= 2:
#                 gt = item[0]
#                 if isinstance(gt, (list, tuple)) and len(gt) == 3:
#                     triplets.append(tuple(gt))
#     return triplets


# def canon(tri):
#     h, r, t = tri
#     def norm(x):
#         return " ".join(str(x).strip().lower().split())
#     return f"{norm(h)}|{norm(r)}|{norm(t)}"


# def safe_div(a, b, when_zero=0.0):
#     return a / b if b != 0 else when_zero


# def main(
#     gt_path,
#     pred_path,
#     gt_sheet=0,
#     pred_sheet=0,
#     key_cols=("row_id",),  # use ("row_id",) if both files share row_id
#     gt_aligned_col="aligned_triplets",
#     pred_aligned_col="aligned_triplets",
#     out_path="aligned_eval_report.xlsx",
# ):
#     gt_df = pd.read_excel(gt_path, sheet_name=gt_sheet, engine="openpyxl")
#     pr_df = pd.read_excel(pred_path, sheet_name=pred_sheet, engine="openpyxl")

#     # Merge rows by row_id (recommended). If you don't have row_id, change key_cols to something stable.
#     merged = gt_df.merge(pr_df, on=list(key_cols), suffixes=("_gt", "_pred"), how="inner")

#     per_row_records = []

#     TP_total = FP_total = FN_total = 0

#     for _, row in merged.iterrows():
#         gt_trips = parse_triplet_list(row[f"{gt_aligned_col}_gt"])
#         pr_trips = parse_triplet_list(row[f"{pred_aligned_col}_pred"])

#         gt_set = set(canon(t) for t in gt_trips)
#         pr_set = set(canon(t) for t in pr_trips)

#         tp = len(gt_set & pr_set)
#         fp = len(pr_set - gt_set)
#         fn = len(gt_set - pr_set)

#         TP_total += tp
#         FP_total += fp
#         FN_total += fn

#         precision = safe_div(tp, tp + fp, when_zero=1.0)  # if predicted nothing, precision=1
#         recall = safe_div(tp, tp + fn, when_zero=1.0)      # if no GT aligned, recall=1
#         f1 = safe_div(2 * precision * recall, precision + recall, when_zero=0.0)

#         per_row_records.append({
#             **{k: row[k] for k in key_cols},
#             "tp": tp, "fp": fp, "fn": fn,
#             "precision": round(precision, 4),
#             "recall": round(recall, 4),
#             "f1": round(f1, 4),
#             "gt_count": len(gt_set),
#             "pred_count": len(pr_set),
#         })

#     # Micro metrics
#     micro_precision = safe_div(TP_total, TP_total + FP_total, when_zero=1.0)
#     micro_recall = safe_div(TP_total, TP_total + FN_total, when_zero=1.0)
#     micro_f1 = safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall, when_zero=0.0)

#     # Macro metrics
#     per_row_df = pd.DataFrame(per_row_records)
#     macro_precision = per_row_df["precision"].mean() if len(per_row_df) else 0.0
#     macro_recall = per_row_df["recall"].mean() if len(per_row_df) else 0.0
#     macro_f1 = per_row_df["f1"].mean() if len(per_row_df) else 0.0

#     summary_df = pd.DataFrame([{
#         "TP_total": TP_total,
#         "FP_total": FP_total,
#         "FN_total": FN_total,
#         "micro_precision": round(micro_precision, 4),
#         "micro_recall": round(micro_recall, 4),
#         "micro_f1": round(micro_f1, 4),
#         "macro_precision": round(macro_precision, 4),
#         "macro_recall": round(macro_recall, 4),
#         "macro_f1": round(macro_f1, 4),
#         "rows_evaluated": len(per_row_df),
#     }])

#     with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
#         summary_df.to_excel(writer, index=False, sheet_name="summary")
#         per_row_df.to_excel(writer, index=False, sheet_name="per_row")

#     print("Saved:", out_path)
#     print(summary_df.to_string(index=False))


# if __name__ == "__main__":
#     # Example usage (edit paths + columns)
#     main(
#         gt_path="ground_truth_2.xlsx",
#         pred_path="kg_compare/output_5.xlsx",
#         key_cols=("row_id",),
#         gt_aligned_col="aligned_triplets",
#         pred_aligned_col="aligned_triplets",
#         out_path="aligned_eval_report.xlsx",
#     )




import ast
import pandas as pd


def parse_triplet_list(cell):
    """
    cell can be:
      - a Python list string: [[...],[...]]
      - empty / NaN
    returns list of triplets: [(h,r,t), ...]
    """
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    try:
        obj = ast.literal_eval(s)
    except Exception:
        return []

    # aligned_triplets may contain:
    # 1) direct triplets: [[h,r,t], ...] or [(h,r,t), ...]
    # 2) match tuples: [(gold_triplet, llm_triplet, score), ...]
    triplets = []

    if isinstance(obj, list):
        for item in obj:
            # case 1: direct triplet
            if isinstance(item, (list, tuple)) and len(item) == 3 and all(isinstance(x, str) for x in item):
                triplets.append(tuple(item))
            # case 2: (gt, llm, score) where gt is [h,r,t]
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                gt = item[0]
                if isinstance(gt, (list, tuple)) and len(gt) == 3:
                    triplets.append(tuple(gt))
    return triplets


def canon(tri):
    h, r, t = tri

    def norm(x):
        return " ".join(str(x).strip().lower().split())

    return f"{norm(h)}|{norm(r)}|{norm(t)}"


def safe_div(a, b, when_zero=0.0):
    return a / b if b != 0 else when_zero


def main(
    gt_path,
    pred_path,
    gt_sheet=0,
    pred_sheet=0,
    key_cols=("row_id",),
    gt_aligned_col="aligned_triplets",
    pred_aligned_col="aligned_triplets",
    out_path="aligned_eval_report.xlsx",
    max_rows=50,  # <--- ONLY evaluate first N rows after merge
):
    gt_df = pd.read_excel(gt_path, sheet_name=gt_sheet, engine="openpyxl")
    pr_df = pd.read_excel(pred_path, sheet_name=pred_sheet, engine="openpyxl")

    # Merge by keys
    merged = gt_df.merge(pr_df, on=list(key_cols), suffixes=("_gt", "_pred"), how="inner")

    # Keep only first max_rows (based on merged order)
    if max_rows is not None:
        merged = merged.head(int(max_rows))

    per_row_records = []
    TP_total = FP_total = FN_total = 0

    for _, row in merged.iterrows():
        gt_trips = parse_triplet_list(row[f"{gt_aligned_col}_gt"])
        pr_trips = parse_triplet_list(row[f"{pred_aligned_col}_pred"])

        gt_set = set(canon(t) for t in gt_trips)
        pr_set = set(canon(t) for t in pr_trips)

        tp = len(gt_set & pr_set)
        fp = len(pr_set - gt_set)
        fn = len(gt_set - pr_set)

        TP_total += tp
        FP_total += fp
        FN_total += fn

        precision = safe_div(tp, tp + fp, when_zero=1.0)  # if predicted nothing, precision=1
        recall = safe_div(tp, tp + fn, when_zero=1.0)      # if no GT aligned, recall=1
        f1 = safe_div(2 * precision * recall, precision + recall, when_zero=0.0)

        per_row_records.append({
            **{k: row[k] for k in key_cols},
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "gt_count": len(gt_set),
            "pred_count": len(pr_set),
        })

    # Micro metrics
    micro_precision = safe_div(TP_total, TP_total + FP_total, when_zero=1.0)
    micro_recall = safe_div(TP_total, TP_total + FN_total, when_zero=1.0)
    micro_f1 = safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall, when_zero=0.0)

    # Macro metrics
    per_row_df = pd.DataFrame(per_row_records)
    macro_precision = per_row_df["precision"].mean() if len(per_row_df) else 0.0
    macro_recall = per_row_df["recall"].mean() if len(per_row_df) else 0.0
    macro_f1 = per_row_df["f1"].mean() if len(per_row_df) else 0.0

    summary_df = pd.DataFrame([{
        "TP_total": TP_total,
        "FP_total": FP_total,
        "FN_total": FN_total,
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "rows_evaluated": len(per_row_df),
        "max_rows_setting": max_rows,
    }])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="summary")
        per_row_df.to_excel(writer, index=False, sheet_name="per_row")

    print("Saved:", out_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main(
        gt_path="ground_truth_2.xlsx",
        pred_path="kg_compare/output_6.xlsx",
        key_cols=("row_id",),
        gt_aligned_col="aligned_triplets",
        pred_aligned_col="aligned_triplets",
        out_path="aligned_eval_report.xlsx",
        max_rows=50,  # <--- evaluate only first 50 merged rows
    )
