"""
S3KG temperature analysis for LLM QA knowledge graphs.

  GoldSim(q) = S3KG(KG_LLM(q), KG_gold(q))
  CtxSim(q)  = S3KG(KG_LLM(q), KG_ctx(q))
  CUS(q)     = 2 * GoldSim(q) * CtxSim(q) / (GoldSim(q) + CtxSim(q))
"""

import argparse
import os
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "LLM_Evaluation", "Results_KGs_With_Temps")
OUT_DIR = os.path.join(RESULTS_DIR, "Analysis_plot_temps")

COL_GOLD = "s3kg_gold_llm"
COL_CTX = "s3kg_ctx_llm"

MODEL_LABELS = {
    "Llama-2-7b-chat-hf": "Llama-2-7B",
    "Mistral-7B-Instruct-v0.2": "Mistral-7B",
    "gemma-7b-it": "Gemma-7B",
    "falcon-7b-instruct": "Falcon-7B",
}

MODEL_FULL_NAMES = {
    "Llama-2-7B": "meta-llama/Llama-2-7b-chat-hf",
    "Gemma-7B": "google/gemma-7b-it",
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.2",
    "Falcon-7B": "tiiuae/falcon-7b-instruct",
}

DATASET_LABELS = {
    "mesaqa": "MesaQA",
    "pubmedqa": "PubMedQA",
}

MODELS = ["Llama-2-7B", "Gemma-7B", "Mistral-7B", "Falcon-7B"]
DATASETS = ["MesaQA", "PubMedQA"]
TEMPERATURES = [0.0, 0.3, 0.7, 1.0]

COLORS = {
    "Llama-2-7B": "#2E86AB",
    "Gemma-7B": "#6A994E",
    "Mistral-7B": "#A23B72",
    "Falcon-7B": "#E07B39",
}


def infer_names(path):
    filename = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(path))
    lower = filename.lower()

    dataset = next((label for key, label in DATASET_LABELS.items() if key in lower), None)
    model = next((label for key, label in MODEL_LABELS.items() if key in filename), None)

    try:
        temperature = float(parent)
    except ValueError:
        temperature = 0.0

    return dataset, model, temperature


def load_results(results_dir):
    rows = []
    for root, _, files in os.walk(results_dir):
        if os.path.basename(root) == "Analysis_plot_temps":
            continue
        for filename in files:
            if not filename.endswith(".csv"):
                continue
            path = os.path.join(root, filename)
            dataset, model, temperature = infer_names(path)
            if not dataset or not model:
                continue

            df = pd.read_csv(path)
            if COL_GOLD not in df.columns or COL_CTX not in df.columns:
                continue

            for _, row in df.iterrows():
                gold = pd.to_numeric(row.get(COL_GOLD), errors="coerce")
                ctx = pd.to_numeric(row.get(COL_CTX), errors="coerce")
                gold = 0.0 if pd.isna(gold) else float(gold)
                ctx = 0.0 if pd.isna(ctx) else float(ctx)
                cus = (2 * gold * ctx / (gold + ctx)) if (gold + ctx) > 0 else 0.0
                rows.append(
                    {
                        "id": str(row.get("id", "")),
                        "dataset": dataset,
                        "model": model,
                        "temperature": temperature,
                        "gold_similarity": gold,
                        "context_similarity": ctx,
                        "cus": cus,
                        "source_file": path,
                    }
                )

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} scored rows from {results_dir}")
    if not df.empty:
        print(f"Datasets: {sorted(df['dataset'].unique())}")
        print(f"Models: {sorted(df['model'].unique())}")
        print(f"Temperatures: {sorted(df['temperature'].unique())}\n")
    return df


def metric_values(df, dataset, model, metric, temperature=None):
    sub = df[(df["dataset"] == dataset) & (df["model"] == model)]
    if temperature is not None:
        sub = sub[np.isclose(sub["temperature"], temperature)]
    return sub[metric].dropna().to_numpy()


def save(fig, out_dir, filename):
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def table_figure(rows, cols, title, out_dir, filename, figsize=None):
    height = 0.35 * max(len(rows), 1) + 1.2
    fig, ax = plt.subplots(figsize=figsize or (14, height))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(cols))))
    for col in range(len(cols)):
        tbl[0, col].set_facecolor("#2E86AB")
        tbl[0, col].set_text_props(color="white", fontweight="bold")
    for row_idx in range(1, len(rows) + 1):
        bg = "#f0f4f8" if row_idx % 2 == 0 else "white"
        for col in range(len(cols)):
            tbl[row_idx, col].set_facecolor(bg)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    save(fig, out_dir, filename)


def print_table(rows, cols, title):
    print(title)
    widths = [len(col) for col in cols]
    for row in rows:
        widths = [max(widths[i], len(str(row[i]))) for i in range(len(cols))]
    print("  ".join(cols[i].ljust(widths[i]) for i in range(len(cols))))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(row[i]).ljust(widths[i]) for i in range(len(cols))))
    print()


def p_text(p_value):
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    return f"{p_value:.4e} {sig}"


def format_score(value):
    return f"{value:.4f}"


def write_csv(path, rows, cols):
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    print(f"  Saved: {path}")


def save_mean_score_csvs(df, out_dir):
    model_rows = []
    for dataset in DATASETS:
        for model in MODELS:
            gold = metric_values(df, dataset, model, "gold_similarity")
            ctx = metric_values(df, dataset, model, "context_similarity")
            cus = metric_values(df, dataset, model, "cus")
            if len(gold):
                model_rows.append(
                    [
                        dataset,
                        model,
                        format_score(gold.mean()),
                        format_score(ctx.mean()),
                        format_score(cus.mean()),
                    ]
                )
    cols = ["Dataset", "Model", "Mean GoldSim", "Mean CtxSim", "Mean CUS"]
    write_csv(os.path.join(out_dir, "mean_scores_by_model.csv"), model_rows, cols)

    temperature_rows = []
    for dataset in DATASETS:
        for temp in TEMPERATURES:
            sub = df[(df["dataset"] == dataset) & np.isclose(df["temperature"], temp)]
            if not sub.empty:
                temperature_rows.append(
                    [
                        dataset,
                        str(temp),
                        format_score(sub["gold_similarity"].mean()),
                        format_score(sub["context_similarity"].mean()),
                        format_score(sub["cus"].mean()),
                    ]
                )
    cols = ["Dataset", "Temperature", "Mean GoldSim", "Mean CtxSim", "Mean CUS"]
    write_csv(os.path.join(out_dir, "mean_scores_by_temperature.csv"), temperature_rows, cols)


def plot_summary_stats(df, out_dir):
    cols = [
        "Dataset",
        "Model",
        "Average (%)",
        "Median (%)",
        "Std Dev (%)",
        "Max (%)",
        "Min (%)",
        "Perfect (>=0.99)",
        "High (>=0.8)",
    ]

    for metric, title, filename in [
        ("gold_similarity", "Statistics of Gold Similarity", "0a_summary_gold.png"),
        ("context_similarity", "Statistics of Context Similarity", "0b_summary_context.png"),
    ]:
        rows = []
        for dataset in DATASETS:
            for model in MODELS:
                vals = metric_values(df, dataset, model, metric)
                if len(vals) == 0:
                    continue
                perfect = int((vals >= 0.99).sum())
                high = int((vals >= 0.80).sum())
                rows.append(
                    [
                        dataset,
                        MODEL_FULL_NAMES.get(model, model),
                        f"{vals.mean() * 100:.2f}",
                        f"{np.median(vals) * 100:.2f}",
                        f"{vals.std() * 100:.2f}",
                        f"{vals.max() * 100:.2f}",
                        f"{vals.min() * 100:.2f}",
                        f"{perfect}/{len(vals)} ({100 * perfect / len(vals):.1f}%)",
                        f"{high}/{len(vals)} ({100 * high / len(vals):.1f}%)",
                    ]
                )
        print_table(rows, cols, title)
        table_figure(rows, cols, title, out_dir, filename, figsize=(20, 0.45 * len(rows) + 1.2))


def plot_comparison(df, out_dir):
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(6 * len(DATASETS), 10))
    if len(DATASETS) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, dataset in enumerate(DATASETS):
        ax_bar = axes[0][col]
        ax_box = axes[1][col]
        x = np.arange(len(MODELS))
        colors = [COLORS[model] for model in MODELS]
        data = [metric_values(df, dataset, model, "gold_similarity") for model in MODELS]
        means = [vals.mean() if len(vals) else 0 for vals in data]
        stds = [vals.std() if len(vals) else 0 for vals in data]

        bars = ax_bar.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.75, edgecolor="black")
        for bar, mean in zip(bars, means):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                mean + 0.005,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        ax_bar.set_title(f"{dataset}\nMean GoldSim (+/- std)", fontweight="bold")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(MODELS, fontsize=9)
        ax_bar.set_ylim(0, 1.05)
        ax_bar.set_ylabel("GoldSim")
        ax_bar.grid(axis="y", linestyle="--", alpha=0.35)

        bp = ax_box.boxplot(data, patch_artist=True, tick_labels=MODELS, widths=0.5)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax_box.set_title(f"{dataset}\nGoldSim Distribution", fontweight="bold")
        ax_box.set_ylim(0, 1.05)
        ax_box.set_ylabel("GoldSim")
        ax_box.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("LLM KG Quality - S3KG Gold Similarity", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, out_dir, "1_comparison.png")


def plot_primary_table(df, out_dir):
    rows = []
    for model in MODELS:
        row = [model]
        for dataset in DATASETS:
            gold = metric_values(df, dataset, model, "gold_similarity")
            ctx = metric_values(df, dataset, model, "context_similarity")
            row.extend(
                [
                    format_score(gold.mean()) if len(gold) else "N/A",
                    format_score(ctx.mean()) if len(ctx) else "N/A",
                ]
            )
        rows.append(row)
    cols = ["Model"] + [f"{dataset} {metric}" for dataset in DATASETS for metric in ["GoldSim", "CtxSim"]]
    print_table(rows, cols, "Primary Benchmark Table")
    table_figure(rows, cols, "Primary Benchmark Table - Mean Similarity", out_dir, "2_primary_table.png")


def plot_cus_table(df, out_dir):
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            gold = metric_values(df, dataset, model, "gold_similarity")
            ctx = metric_values(df, dataset, model, "context_similarity")
            cus = metric_values(df, dataset, model, "cus")
            if len(cus):
                rows.append([dataset, model, format_score(gold.mean()), format_score(ctx.mean()), format_score(cus.mean())])
    rows.sort(key=lambda row: (row[0], -float(row[4])))
    cols = ["Dataset", "Model", "GoldSim", "CtxSim", "CUS"]
    print_table(rows, cols, "Contextual Understanding Score (CUS)")
    write_csv(os.path.join(out_dir, "table_llm_evaluation_results.csv"), rows, cols)
    table_figure(rows, cols, "Contextual Understanding Score (CUS)", out_dir, "3_cus_leaderboard.png")


def plot_rmse(df, out_dir):
    x = np.arange(len(DATASETS))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, model in enumerate(MODELS):
        vals = []
        for dataset in DATASETS:
            gold = metric_values(df, dataset, model, "gold_similarity")
            vals.append(np.sqrt(np.mean((1 - gold) ** 2)) if len(gold) else 0)
        bars = ax.bar(
            x + idx * width,
            vals,
            width,
            label=model,
            color=COLORS[model],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.7,
        )
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("RMSE against ideal GoldSim=1 (lower is better)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("RMSE Against Gold Answer", fontweight="bold")
    plt.tight_layout()
    save(fig, out_dir, "4_rmse.png")


def plot_grouped_bar(df, out_dir):
    metrics = [
        ("gold_similarity", "GoldSim"),
        ("context_similarity", "CtxSim"),
        ("cus", "CUS"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    width = 0.2
    x = np.arange(len(DATASETS))
    for ax, (metric, title) in zip(axes, metrics):
        for idx, model in enumerate(MODELS):
            means = []
            for dataset in DATASETS:
                vals = metric_values(df, dataset, model, metric)
                means.append(vals.mean() if len(vals) else 0)
            ax.bar(x + idx * width, means, width, label=model, color=COLORS[model], alpha=0.88)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(DATASETS)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("LLM Performance by Dataset and Metric", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, out_dir, "5_grouped_bar.png")


def plot_winrate(df, out_dir):
    rows = []
    for dataset in DATASETS:
        sub = df[df["dataset"] == dataset]
        for model_a, model_b in combinations(MODELS, 2):
            left = sub[sub["model"] == model_a][["id", "temperature", "gold_similarity"]]
            right = sub[sub["model"] == model_b][["id", "temperature", "gold_similarity"]]
            merged = left.merge(right, on=["id", "temperature"], suffixes=("_a", "_b"))
            if merged.empty:
                continue
            a = merged["gold_similarity_a"].to_numpy()
            b = merged["gold_similarity_b"].to_numpy()
            rows.append(
                [
                    dataset,
                    model_a,
                    model_b,
                    f"{np.mean(a > b):.3f}",
                    f"{np.mean(b > a):.3f}",
                    f"{np.mean(a == b):.3f}",
                    str(len(merged)),
                ]
            )
    cols = ["Dataset", "Model A", "Model B", "Win Rate A", "Win Rate B", "Tie Rate", "N"]
    print_table(rows, cols, "Pairwise Win-Rate - GoldSim")
    table_figure(rows, cols, "Pairwise Win-Rate - GoldSim", out_dir, "6_winrate.png", figsize=(14, 0.48 * len(rows) + 1.2))


def plot_distributions(df, out_dir):
    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        data = [metric_values(df, dataset, model, "gold_similarity") for model in MODELS]
        for model, vals in zip(MODELS, data):
            if len(vals):
                axes[0].hist(vals, bins=20, alpha=0.55, label=model, color=COLORS[model])
        axes[0].set_title("Histogram")
        axes[0].set_xlabel("GoldSim")
        axes[0].set_ylabel("Count")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        bp = axes[1].boxplot(data, patch_artist=True, tick_labels=MODELS, widths=0.5)
        for patch, model in zip(bp["boxes"], MODELS):
            patch.set_facecolor(COLORS[model])
            patch.set_alpha(0.75)
        axes[1].set_title("Boxplot")
        axes[1].set_ylabel("GoldSim")
        axes[1].grid(axis="y", alpha=0.3)

        parts = axes[2].violinplot([vals if len(vals) else [0] for vals in data], positions=range(len(MODELS)), showmedians=True)
        for body, model in zip(parts["bodies"], MODELS):
            body.set_facecolor(COLORS[model])
            body.set_alpha(0.75)
        axes[2].set_xticks(range(len(MODELS)))
        axes[2].set_xticklabels(MODELS)
        axes[2].set_title("Violin Plot")
        axes[2].set_ylabel("GoldSim")
        axes[2].grid(axis="y", alpha=0.3)

        fig.suptitle(f"Distribution Analysis - {dataset} (GoldSim)", fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save(fig, out_dir, f"7_distribution_{dataset.lower()}.png")

    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            vals = metric_values(df, dataset, model, "gold_similarity")
            if len(vals):
                rows.append(
                    [
                        dataset,
                        model,
                        format_score(vals.mean()),
                        format_score(np.median(vals)),
                        format_score(vals.std()),
                        format_score(vals.min()),
                        format_score(vals.max()),
                        str(len(vals)),
                    ]
                )
    cols = ["Dataset", "Model", "Mean", "Median", "Std", "Min", "Max", "N"]
    print_table(rows, cols, "Distribution Summary Statistics - GoldSim")
    table_figure(rows, cols, "Distribution Summary Statistics - GoldSim", out_dir, "7_distribution_stats.png")


def plot_scatter(df, out_dir):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(13, 5))
    if len(DATASETS) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, DATASETS):
        for model in MODELS:
            sub = df[(df["dataset"] == dataset) & (df["model"] == model)]
            if not sub.empty:
                ax.scatter(
                    sub["gold_similarity"],
                    sub["context_similarity"],
                    label=model,
                    color=COLORS[model],
                    alpha=0.45,
                    s=18,
                    edgecolors="none",
                )
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("GoldSim (factual accuracy)")
        ax.set_ylabel("CtxSim (contextual faithfulness)")
        ax.set_title(dataset, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("GoldSim vs CtxSim per Model", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, out_dir, "8_scatter_gold_vs_context.png")


def plot_significance(df, out_dir):
    rows = []
    for dataset in DATASETS:
        sub = df[df["dataset"] == dataset]
        for model_a, model_b in combinations(MODELS, 2):
            left = sub[sub["model"] == model_a][["id", "temperature", "gold_similarity"]]
            right = sub[sub["model"] == model_b][["id", "temperature", "gold_similarity"]]
            merged = left.merge(right, on=["id", "temperature"], suffixes=("_a", "_b"))
            if len(merged) < 5:
                continue
            a = merged["gold_similarity_a"].to_numpy()
            b = merged["gold_similarity_b"].to_numpy()
            try:
                _, wilcoxon_p = stats.wilcoxon(a, b)
                wilcoxon_text = p_text(wilcoxon_p)
            except Exception:
                wilcoxon_text = "N/A"
            _, mann_p = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append([dataset, model_a, model_b, format_score(a.mean()), format_score(b.mean()), wilcoxon_text, p_text(mann_p)])
    cols = ["Dataset", "Model A", "Model B", "Mean A", "Mean B", "Wilcoxon p", "Mann-Whitney p"]
    print_table(rows, cols, "Statistical Significance Tests")
    table_figure(rows, cols, "Statistical Significance Tests", out_dir, "9_significance.png", figsize=(17, 0.62 * len(rows) + 1.8))


def plot_radar(df, out_dir):
    categories = ["Gold\nMesaQA", "Ctx\nMesaQA", "Gold\nPubMedQA", "Ctx\nPubMedQA"]
    angles = [idx / len(categories) * 2 * np.pi for idx in range(len(categories))]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    for model in MODELS:
        vals = []
        for dataset in DATASETS:
            gold = metric_values(df, dataset, model, "gold_similarity")
            ctx = metric_values(df, dataset, model, "context_similarity")
            vals.extend([gold.mean() if len(gold) else 0, ctx.mean() if len(ctx) else 0])
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=model, color=COLORS[model])
        ax.fill(angles, vals, alpha=0.12, color=COLORS[model])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Model Capability Radar", fontweight="bold", pad=15)
    plt.tight_layout()
    save(fig, out_dir, "10_radar.png")


def plot_heatmap(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for ax, metric in zip(axes, ["gold_similarity", "context_similarity", "cus"]):
        values = {}
        for dataset in DATASETS:
            values[dataset] = {}
            for model in MODELS:
                vals = metric_values(df, dataset, model, metric)
                values[dataset][model] = vals.mean() if len(vals) else np.nan
        pivot = pd.DataFrame(values).reindex(MODELS)
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1, linewidths=0.5)
        ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Performance Heatmap - Model x Dataset", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, out_dir, "11_heatmap.png")


def plot_temperature_trends(df, out_dir):
    for metric, label, tag in [
        ("gold_similarity", "GoldSim", "gold"),
        ("context_similarity", "CtxSim", "context"),
        ("cus", "CUS", "cus"),
    ]:
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(7 * len(DATASETS), 5), sharey=True)
        if len(DATASETS) == 1:
            axes = [axes]
        for ax, dataset in zip(axes, DATASETS):
            for model in MODELS:
                means = []
                for temp in TEMPERATURES:
                    vals = metric_values(df, dataset, model, metric, temperature=temp)
                    means.append(vals.mean() if len(vals) else np.nan)
                if np.isfinite(means).any():
                    ax.plot(TEMPERATURES, means, marker="o", linewidth=2, label=model, color=COLORS[model])
                    for temp, mean in zip(TEMPERATURES, means):
                        if np.isfinite(mean):
                            ax.annotate(f"{mean:.3f}", (temp, mean), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
            ax.set_title(dataset, fontweight="bold")
            ax.set_xlabel("Temperature")
            ax.set_ylabel(label)
            ax.set_xticks(TEMPERATURES)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.suptitle(f"Effect of Temperature on {label}", fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save(fig, out_dir, f"12_temperature_trend_{tag}.png")

    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            for temp in TEMPERATURES:
                gold = metric_values(df, dataset, model, "gold_similarity", temperature=temp)
                ctx = metric_values(df, dataset, model, "context_similarity", temperature=temp)
                cus = metric_values(df, dataset, model, "cus", temperature=temp)
                if len(gold):
                    rows.append([dataset, model, str(temp), format_score(gold.mean()), format_score(ctx.mean()), format_score(cus.mean())])
    cols = ["Dataset", "Model", "Temperature", "Mean GoldSim", "Mean CtxSim", "Mean CUS"]
    print_table(rows, cols, "Temperature Effect - Mean Scores")
    write_csv(os.path.join(out_dir, "mean_scores_temperature_summary.csv"), rows, cols)
    table_figure(rows, cols, "Effect of Temperature on S3KG Scores", out_dir, "12_temperature_table.png", figsize=(16, 0.4 * len(rows) + 1.0))


def save_paper_tables(df, out_dir, temperature=0.0):
    main_rows = []
    for dataset in DATASETS:
        means = {}
        for model in MODELS:
            sub = df[(df["dataset"] == dataset) & (df["model"] == model) & np.isclose(df["temperature"], temperature)]
            if not sub.empty:
                means[model] = (
                    sub["gold_similarity"].mean(),
                    sub["context_similarity"].mean(),
                    sub["cus"].mean(),
                )
        for model in MODELS:
            if model not in means:
                continue
            gold, ctx, cus = means[model]
            main_rows.append([dataset, model, format_score(gold), format_score(ctx), format_score(cus)])

    cols = ["Dataset", "Model", "GoldSim", "CtxSim", "CUS"]
    write_csv(os.path.join(out_dir, "table_llm_evaluation_results.csv"), main_rows, cols)
    write_latex_table(os.path.join(out_dir, "table_llm_evaluation_results.tex"), main_rows, cols, "tab:llm")

    for metric, metric_label, csv_name, tex_name, label in [
        ("gold_similarity", "GoldSim", "table_goldsim_by_temperature.csv", "table_goldsim_by_temperature.tex", "tab:temp_gold"),
        ("context_similarity", "ContextSim", "table_ctxsim_by_temperature.csv", "table_ctxsim_by_temperature.tex", "tab:temp_ctx"),
    ]:
        rows = []
        for dataset in DATASETS:
            for model in MODELS:
                row = [dataset, model]
                for temp in TEMPERATURES:
                    vals = metric_values(df, dataset, model, metric, temperature=temp)
                    row.append(format_score(vals.mean()) if len(vals) else "N/A")
                rows.append(row)
        temp_cols = ["Dataset", "Model"] + [f"T = {temp:.1f}" for temp in TEMPERATURES]
        write_csv(os.path.join(out_dir, csv_name), rows, temp_cols)
        write_latex_table(os.path.join(out_dir, tex_name), rows, temp_cols, label, caption=f"Mean {metric_label} per model across temperatures.")


def write_latex_table(path, rows, cols, label, caption=None):
    caption = caption or "LLM evaluation results."
    with open(path, "w", encoding="utf-8") as file:
        file.write("\\begin{table}[t]\n")
        file.write("    \\centering\n")
        file.write(f"    \\caption{{{caption}}}\n")
        file.write(f"    \\label{{{label}}}\n")
        file.write("    \\resizebox{\\columnwidth}{!}{%\n")
        file.write("    \\begin{tabular}{" + " ".join(["l"] * len(cols)) + "}\n")
        file.write("        \\toprule\n")
        file.write("        " + " & ".join(f"\\textbf{{{col}}}" for col in cols) + " \\\\\n")
        file.write("        \\midrule\n")
        for row in rows:
            file.write("        " + " & ".join(map(str, row)) + " \\\\\n")
        file.write("        \\bottomrule\n")
        file.write("    \\end{tabular}%\n")
        file.write("    }\n")
        file.write("\\end{table}\n")
    print(f"  Saved: {path}")


def plot_eval_table(df, out_dir, temperature=0.0):
    rows = []
    for dataset in DATASETS:
        means = {}
        for model in MODELS:
            sub = df[(df["dataset"] == dataset) & (df["model"] == model) & np.isclose(df["temperature"], temperature)]
            if not sub.empty:
                means[model] = (
                    sub["gold_similarity"].mean(),
                    sub["context_similarity"].mean(),
                    sub["cus"].mean(),
                )
        if not means:
            continue
        best_gold = max(vals[0] for vals in means.values())
        best_ctx = max(vals[1] for vals in means.values())
        best_cus = max(vals[2] for vals in means.values())
        for model in MODELS:
            if model not in means:
                continue
            gold, ctx, cus = means[model]
            rows.append(
                [
                    dataset,
                    model,
                    f"{'*' if np.isclose(gold, best_gold) else ''}{gold:.4f}",
                    f"{'*' if np.isclose(ctx, best_ctx) else ''}{ctx:.4f}",
                    f"{'*' if np.isclose(cus, best_cus) else ''}{cus:.4f}",
                ]
            )
    cols = ["Dataset", "Model", "GoldSim", "CtxSim", "CUS"]
    title = f"LLM Evaluation Results - Mean per Model per Dataset (T={temperature})"
    print_table(rows, cols, title)
    table_figure(rows, cols, title + "\n* = best per metric per dataset", out_dir, f"13_eval_table_t{temperature}.png", figsize=(10, 0.4 * len(rows) + 1.2))


def main():
    parser = argparse.ArgumentParser(description="Analyze S3KG GoldSim, CtxSim, and CUS over temperatures")
    parser.add_argument("--input-dir", default=RESULTS_DIR, help="Directory containing temperature CSV folders")
    parser.add_argument("--outdir", default=OUT_DIR, help="Directory to write plot images and tables")
    parser.add_argument("--eval-temp", type=float, default=0.0, help="Temperature for the final evaluation table")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_results(args.input_dir)
    if df.empty:
        raise SystemExit("No scored CSV rows found. Check --input-dir and score column names.")

    save_mean_score_csvs(df, args.outdir)
    plot_summary_stats(df, args.outdir)
    plot_comparison(df, args.outdir)
    plot_primary_table(df, args.outdir)
    plot_cus_table(df, args.outdir)
    plot_rmse(df, args.outdir)
    plot_grouped_bar(df, args.outdir)
    plot_winrate(df, args.outdir)
    plot_distributions(df, args.outdir)
    plot_scatter(df, args.outdir)
    plot_significance(df, args.outdir)
    plot_radar(df, args.outdir)
    plot_heatmap(df, args.outdir)
    plot_temperature_trends(df, args.outdir)
    save_paper_tables(df, args.outdir, temperature=args.eval_temp)
    plot_eval_table(df, args.outdir, temperature=args.eval_temp)

    print(f"\nAll outputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()
