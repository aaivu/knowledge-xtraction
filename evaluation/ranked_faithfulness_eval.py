"""
ranked_faithfulness_eval.py

Evaluates semantic similarity methods on the PubMedQA Ranked Faithfulness
benchmark using ranking metrics (Kendall's tau, Spearman correlation).

Dataset structure
-----------------
  100 questions × 4 candidates = 400 rows.
  Each candidate has a faithfulness_level (4 = best, 1 = worst).
  A good metric should score them in descending order: 4 > 3 > 2 > 1.

Evaluation metrics
------------------
  Mean Kendall's tau   — primary ranking quality measure
  Mean Spearman ρ      — monotonic correlation with expected levels
  Perfect rank %       — % of questions with all 4 candidates in correct order
  Boundary accuracy    — P(score_L > score_{L-1}) for each adjacent pair,
                         especially the critical 3→2 boundary
                         (numerical/entity distortion vs relation inversion)

Methods evaluated
-----------------
  AA-KEA (KG-based)     — loaded from pre-computed CSV
  ROUGE-L               — n-gram overlap (surface)
  BLEU                  — n-gram precision (surface)
  BERTScore F1          — contextual embedding alignment
  Sentence-BERT cosine  — dense embedding similarity

Input files
-----------
  datasets/pubmedqa_ranked_faithfulness_400.csv
  datasets/pubmedqa_ranked_faithfulness_aa_kea_results.csv  (optional)

Output
------
  output/pubmedqa_ranked_faithfulness/ranking_results.csv
  output/pubmedqa_ranked_faithfulness/boundary_accuracy.csv
  output/pubmedqa_ranked_faithfulness/kendall_tau_comparison.png
  output/pubmedqa_ranked_faithfulness/score_by_level.png
  output/pubmedqa_ranked_faithfulness/boundary_heatmap.png
  output/pubmedqa_ranked_faithfulness/tau_distribution.png
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams.update({'figure.dpi': 130, 'font.size': 11})

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATASET_FILE   = 'datasets/pubmedqa_ranked_faithfulness_400.csv'
AA_KEA_FILE    = 'datasets/pubmedqa_ranked_faithfulness_aa_kea_results.csv'
S3KG_FILE = 'datasets/pubmedqa_ranked_faithfulness_400_s3kg_results.csv'
OUTPUT_DIR     = Path('output/pubmedqa_ranked_faithfulness')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

EXPECTED_LEVELS = [4, 3, 2, 1]     # canonical faithfulness order

LEVEL_COLORS = {4: '#2ecc71', 3: '#f39c12', 2: '#e74c3c', 1: '#8e44ad'}
LEVEL_LABELS = {
    4: 'Level 4\nFaithful paraphrase',
    3: 'Level 3\nFactual distortion',
    2: 'Level 2\nRelation inversion',
    1: 'Level 1\nContext ignoring',
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    df = pd.read_csv(DATASET_FILE)
    print(f"✓ Dataset  : {len(df)} rows, {df['question_id'].nunique()} questions")

    # Merge AA-KEA scores if available
    if Path(AA_KEA_FILE).exists():
        aa = pd.read_csv(AA_KEA_FILE)
        score_col = next(
            (c for c in aa.columns if 'similarity' in c.lower() or 'score' in c.lower()),
            None
        )
        if score_col:
            aa = aa.rename(columns={score_col: 'aa_kea_score'})
            df = df.merge(aa[['pair_id', 'aa_kea_score']], on='pair_id', how='left')
            missing = df['aa_kea_score'].isna().sum()
            if missing:
                print(f"  ⚠ {missing} AA-KEA scores missing → filled with 0")
                df['aa_kea_score'] = df['aa_kea_score'].fillna(0.0)
            print(f"✓ AA-KEA   : scores merged ({score_col})")
        else:
            print("⚠ AA-KEA file found but no similarity column detected — skipping")
    else:
        print(f"⚠ AA-KEA results not found ({AA_KEA_FILE}) — skipping")

    # Merge S3KG scores if available
    if Path(S3KG_FILE).exists():
        snea = pd.read_csv(S3KG_FILE)
        # Deduplicate on 'id' — same issue as other datasets
        if snea['id'].duplicated().any():
            n_before = len(snea)
            snea = snea.drop_duplicates(subset='id', keep='first')
            print(f"  ⚠ Dropped {n_before - len(snea)} duplicate id(s) in S3KG file")
        # Cross-key merge: results use 'id', dataset uses 'pair_id'
        snea = snea.rename(columns={'s3kg_similarity': 's3kg_score'})
        df = df.merge(snea[['id', 's3kg_score']],
                      left_on='pair_id', right_on='id', how='left')
        df = df.drop(columns=['id'], errors='ignore')
        missing = df['s3kg_score'].isna().sum()
        if missing:
            print(f"  ⚠ {missing} S3KG scores missing → filled with 0")
            df['s3kg_score'] = df['s3kg_score'].fillna(0.0)
        print(f"✓ S3KG: scores merged ({len(snea)} unique rows)")
    else:
        print(f"⚠ S3KG results not found ({S3KG_FILE}) — skipping")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Baseline computation
# (compare gt_answer vs candidate_answer for all 400 rows)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rouge(df: pd.DataFrame) -> pd.DataFrame:
    from rouge_score import rouge_scorer as rs
    print("\nComputing ROUGE-L …")
    scorer = rs.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [
        scorer.score(row['gt_answer'], row['candidate_answer'])['rougeL'].fmeasure
        for _, row in tqdm(df.iterrows(), total=len(df), desc='ROUGE-L', leave=False)
    ]
    df['rougeL_score'] = scores
    print("✓ ROUGE-L done")
    return df


def compute_bleu(df: pd.DataFrame) -> pd.DataFrame:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    for pkg in ('punkt', 'punkt_tab'):
        nltk.download(pkg, quiet=True)
    print("\nComputing BLEU …")
    smooth = SmoothingFunction().method1
    scores = [
        sentence_bleu(
            [row['gt_answer'].lower().split()],
            row['candidate_answer'].lower().split(),
            smoothing_function=smooth,
        )
        for _, row in tqdm(df.iterrows(), total=len(df), desc='BLEU', leave=False)
    ]
    df['bleu_score'] = scores
    print("✓ BLEU done")
    return df


def compute_bertscore(df: pd.DataFrame, batch_size: int = 16) -> pd.DataFrame:
    from bert_score import score as bscore
    print("\nComputing BERTScore (roberta-large) …")
    try:
        all_f1 = []
        for i in tqdm(range(0, len(df), batch_size), desc='BERTScore', leave=False):
            batch = df.iloc[i:i + batch_size]
            refs  = [str(t)[:2000] for t in batch['gt_answer']]
            cands = [str(t)[:2000] for t in batch['candidate_answer']]
            _, _, F1 = bscore(
                cands, refs,
                lang='en', model_type='roberta-large',
                verbose=False, batch_size=batch_size,
            )
            all_f1.extend(F1.tolist())
        df['bertscore_f1'] = all_f1
        print("✓ BERTScore done")
    except Exception as e:
        print(f"⚠ BERTScore failed: {e} — setting to 0")
        df['bertscore_f1'] = 0.0
    return df


def compute_sbert(df: pd.DataFrame) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    print(f"\nComputing Sentence-BERT cosine ({EMBEDDING_MODEL}) …")
    model = SentenceTransformer(EMBEDDING_MODEL)
    scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='SBERT', leave=False):
        e1 = model.encode([row['gt_answer']])[0]
        e2 = model.encode([row['candidate_answer']])[0]
        scores.append(float(cos_sim([e1], [e2])[0][0]))
    df['sbert_score'] = scores
    print("✓ SBERT done")
    return df


def compute_all_baselines(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("COMPUTING BASELINE METHODS")
    print("=" * 70)
    df = compute_rouge(df)
    df = compute_bleu(df)
    df = compute_bertscore(df)
    df = compute_sbert(df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Ranking evaluation
# ─────────────────────────────────────────────────────────────────────────────

def get_methods(df: pd.DataFrame) -> dict[str, str]:
    """Return {display_name: column_name} for all available methods."""
    method_map = {}
    if 'aa_kea_score' in df.columns:
        method_map['AA-KEA'] = 'aa_kea_score'
    if 's3kg_score' in df.columns:
        method_map['S3KG'] = 's3kg_score'
    method_map.update({
        'ROUGE-L':       'rougeL_score',
        'BLEU':          'bleu_score',
        'BERTScore F1':  'bertscore_f1',
        'Sentence-BERT': 'sbert_score',
    })
    return {k: v for k, v in method_map.items() if v in df.columns}


def evaluate_ranking(df: pd.DataFrame, methods: dict[str, str]) -> dict:
    """
    For each question (question_id), rank the 4 candidates by each method's score
    and compute Kendall's tau vs expected [4, 3, 2, 1].

    Returns a dict with per-question taus and aggregate stats.
    """
    results = {name: {'taus': [], 'spearmans': []} for name in methods}

    for qid, group in df.groupby('question_id'):
        # Sort by faithfulness_level to get consistent ordering
        group = group.sort_values('faithfulness_level', ascending=False)
        expected = group['faithfulness_level'].tolist()   # [4, 3, 2, 1]

        for name, col in methods.items():
            scores = group[col].tolist()
            tau, _  = kendalltau(expected, scores)
            rho, _  = spearmanr(expected, scores)
            results[name]['taus'].append(tau)
            results[name]['spearmans'].append(rho)

    return results


def compute_boundary_accuracy(df: pd.DataFrame, methods: dict[str, str]) -> pd.DataFrame:
    """
    For each adjacent level pair (4>3, 3>2, 2>1), compute the fraction of
    questions where the higher-level candidate scores higher.
    The 3>2 boundary is the critical KG-specific discriminator.
    """
    boundaries = [(4, 3), (3, 2), (2, 1)]
    rows = []

    for name, col in methods.items():
        for (hi, lo) in boundaries:
            correct = 0
            total   = 0
            for _, group in df.groupby('question_id'):
                hi_score = group.loc[group['faithfulness_level'] == hi, col]
                lo_score = group.loc[group['faithfulness_level'] == lo, col]
                if hi_score.empty or lo_score.empty:
                    continue
                if hi_score.values[0] > lo_score.values[0]:
                    correct += 1
                total += 1
            rows.append({
                'method':   name,
                'boundary': f'L{hi} > L{lo}',
                'accuracy': correct / total if total else 0.0,
                'correct':  correct,
                'total':    total,
            })

    return pd.DataFrame(rows)


def compute_perfect_rank(df: pd.DataFrame, methods: dict[str, str]) -> dict[str, float]:
    """% of questions where all 4 candidates are ranked in exact correct order."""
    perfect = {name: 0 for name in methods}
    total   = df['question_id'].nunique()

    for _, group in df.groupby('question_id'):
        group = group.sort_values('faithfulness_level', ascending=False)
        expected = group['faithfulness_level'].tolist()
        for name, col in methods.items():
            predicted_order = group.sort_values(col, ascending=False)['faithfulness_level'].tolist()
            if predicted_order == expected:
                perfect[name] += 1

    return {name: v / total for name, v in perfect.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(
    rank_results: dict,
    boundary_df: pd.DataFrame,
    perfect_rank: dict[str, float],
    methods: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for name in methods:
        taus   = rank_results[name]['taus']
        spears = rank_results[name]['spearmans']
        bd     = boundary_df[boundary_df['method'] == name].set_index('boundary')['accuracy']
        rows.append({
            'Method':            name,
            'Mean Kendall τ':    round(np.mean(taus),   4),
            'Std Kendall τ':     round(np.std(taus),    4),
            'Mean Spearman ρ':   round(np.mean(spears), 4),
            'Perfect rank %':    round(perfect_rank.get(name, 0) * 100, 1),
            'L4>L3 acc':         round(bd.get('L4 > L3', 0), 3),
            'L3>L2 acc (crit)':  round(bd.get('L3 > L2', 0), 3),
            'L2>L1 acc':         round(bd.get('L2 > L1', 0), 3),
        })
    return pd.DataFrame(rows).sort_values('Mean Kendall τ', ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_tau_comparison(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ['#2ecc71' if m in ('AA-KEA', 'S3KG') else '#3498db' for m in summary['Method']]
    bars    = ax.bar(summary['Method'], summary['Mean Kendall τ'], color=colors,
                     edgecolor='white', linewidth=0.8, zorder=3)
    ax.errorbar(
        range(len(summary)), summary['Mean Kendall τ'],
        yerr=summary['Std Kendall τ'],
        fmt='none', color='#2c3e50', capsize=5, linewidth=1.5, zorder=4,
    )
    for bar, val in zip(bars, summary['Mean Kendall τ']):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel("Mean Kendall's τ", fontsize=12)
    ax.set_title("Ranking Quality — Mean Kendall's τ\n(higher = better faithfulness ordering)",
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.grid(axis='y', alpha=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'kendall_tau_comparison.png', bbox_inches='tight')
    plt.close()
    print("  ✓ kendall_tau_comparison.png")


def plot_score_by_level(df: pd.DataFrame, methods: dict[str, str]) -> None:
    n = len(methods)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows))
    axes = np.array(axes).flatten()

    palette = [LEVEL_COLORS[l] for l in [4, 3, 2, 1]]

    for ax, (name, col) in zip(axes, methods.items()):
        data_by_level = [
            df[df['faithfulness_level'] == lvl][col].values
            for lvl in [4, 3, 2, 1]
        ]
        vp = ax.violinplot(data_by_level, positions=[4, 3, 2, 1],
                           showmedians=True, showextrema=True)
        for patch, color in zip(vp['bodies'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for part in ('cmedians', 'cbars', 'cmins', 'cmaxes'):
            if part in vp:
                vp[part].set_color('#2c3e50')
                vp[part].set_linewidth(1.2)
        ax.set_xticks([4, 3, 2, 1])
        ax.set_xticklabels(['L4\nParaphrase', 'L3\nDistortion',
                            'L2\nInversion', 'L1\nOff-topic'], fontsize=9)
        ax.set_ylabel('Similarity score', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    # Hide unused subplots
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle('Score Distribution by Faithfulness Level\n(scores should decrease left → right)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'score_by_level.png', bbox_inches='tight')
    plt.close()
    print("  ✓ score_by_level.png")


def plot_boundary_heatmap(boundary_df: pd.DataFrame) -> None:
    pivot = boundary_df.pivot(index='method', columns='boundary', values='accuracy')
    # Reorder columns to show critical boundary in middle
    col_order = ['L4 > L3', 'L3 > L2', 'L2 > L1']
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        pivot, annot=True, fmt='.3f', cmap='RdYlGn',
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
        cbar_kws={'label': 'Accuracy'},
    )
    ax.set_title(
        'Boundary Accuracy — P(higher level scores higher)\n'
        '★ L3 > L2 is the critical KG-specific boundary',
        fontsize=12, fontweight='bold',
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    # Highlight critical column
    ax.add_patch(plt.Rectangle((1, 0), 1, len(pivot), fill=False,
                                edgecolor='#2c3e50', lw=3))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'boundary_heatmap.png', bbox_inches='tight')
    plt.close()
    print("  ✓ boundary_heatmap.png")


def plot_tau_distribution(rank_results: dict, methods: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    method_names = list(methods.keys())
    colors = ['#2ecc71' if m in ('AA-KEA', 'S3KG') else '#3498db' for m in method_names]

    positions = range(1, len(method_names) + 1)
    bp = ax.boxplot(
        [rank_results[m]['taus'] for m in method_names],
        positions=list(positions),
        patch_artist=True,
        medianprops={'color': '#2c3e50', 'linewidth': 2},
        whiskerprops={'linewidth': 1.2},
        capprops={'linewidth': 1.2},
        flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.5},
        widths=0.5,
    )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(method_names, fontsize=11)
    ax.set_ylabel("Kendall's τ per question", fontsize=12)
    ax.set_title("Distribution of Per-Question Kendall's τ\n(each dot = one question)",
                 fontsize=13, fontweight='bold')
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tau_distribution.png', bbox_inches='tight')
    plt.close()
    print("  ✓ tau_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("PUBMEDQA RANKED FAITHFULNESS — EVALUATION")
    print("=" * 70)
    print("Metric : Kendall's tau (ranking quality vs expected [4>3>2>1])")
    print("Critical boundary : L3 > L2  (factual distortion vs relation inversion)")

    # 1. Load
    df = load_data()

    # 2. Compute baselines
    df = compute_all_baselines(df)

    # 3. Identify available methods
    methods = get_methods(df)
    print(f"\nMethods to evaluate: {list(methods.keys())}")

    # 4. Ranking metrics
    print("\n" + "=" * 70)
    print("COMPUTING RANKING METRICS")
    print("=" * 70)
    rank_results  = evaluate_ranking(df, methods)
    boundary_df   = compute_boundary_accuracy(df, methods)
    perfect_rank  = compute_perfect_rank(df, methods)

    # 5. Summary table
    summary = build_summary(rank_results, boundary_df, perfect_rank, methods)

    print("\n" + "─" * 70)
    print("RESULTS SUMMARY")
    print("─" * 70)
    print(summary.to_string(index=False))

    # 6. Plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    plot_tau_comparison(summary)
    plot_score_by_level(df, methods)
    plot_boundary_heatmap(boundary_df)
    plot_tau_distribution(rank_results, methods)

    # 7. Save CSVs
    summary.to_csv(OUTPUT_DIR / 'ranking_results.csv', index=False)
    boundary_df.to_csv(OUTPUT_DIR / 'boundary_accuracy.csv', index=False)

    # Save per-question taus for inspection
    per_q_rows = []
    for qid, group in df.groupby('question_id'):
        group = group.sort_values('faithfulness_level', ascending=False)
        expected = group['faithfulness_level'].tolist()
        row = {'question_id': qid}
        for name, col in methods.items():
            tau, _ = kendalltau(expected, group[col].tolist())
            row[f'{name}_tau'] = round(tau, 4)
        per_q_rows.append(row)
    pd.DataFrame(per_q_rows).to_csv(OUTPUT_DIR / 'per_question_taus.csv', index=False)

    print("\n" + "=" * 70)
    print(f"✓ All outputs saved → {OUTPUT_DIR}/")
    print("\nKey finding to check:")
    our_methods = [m for m in methods if m in ('AA-KEA', 'S3KG')]
    baselines   = [m for m in methods if m not in ('AA-KEA', 'S3KG')]
    for our in our_methods:
        our_l32 = boundary_df.query(f"method == '{our}' and boundary == 'L3 > L2'")['accuracy']
        for name in baselines:
            bl_l32 = boundary_df.query(f"method == '{name}' and boundary == 'L3 > L2'")['accuracy']
            if not our_l32.empty and not bl_l32.empty:
                diff = our_l32.values[0] - bl_l32.values[0]
                print(f"  {our} L3>L2 acc = {our_l32.values[0]:.3f}  vs  "
                      f"{name} = {bl_l32.values[0]:.3f}  (Δ = {diff:+.3f})")


if __name__ == '__main__':
    main()