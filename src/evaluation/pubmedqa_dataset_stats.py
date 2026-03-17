"""
pubmedqa_dataset_stats.py

Dataset-level statistical validation for the PubMedQA Ranked Faithfulness
benchmark BEFORE interpreting ranking metrics.

Analyses
--------
1. Descriptive stats (mean ± std, min/max) per faithfulness level per method
2. One-way ANOVA across 4 levels per method
3. Pairwise Welch t-tests between adjacent levels (L4↔L3, L3↔L2, L2↔L1)
4. Cohen's d effect size between adjacent levels
5. Score-inversion rate per boundary per method
   (% of questions where upper level scores LOWER — explains 0% perfect rank)
6. Level separability score (are levels cleanly ordered?)
7. Visualisations:
   - Violin + boxplot: score distributions per level per method
   - Heatmap: Cohen's d across methods × boundaries
   - Bar chart: inversion rate per boundary per method
   - Summary stats table (PNG)

Output
------
  output/pubmedqa_ranked_faithfulness/dataset_stats/
      stats_summary.csv
      stats_table.png
      score_distributions.png
      cohens_d_heatmap.png
      inversion_rate_bar.png
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams.update({'figure.dpi': 130, 'font.size': 11})

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_HERE        = Path(__file__).parent
DATASET_FILE = _HERE / 'datasets/pubmedqa_ranked_faithfulness_400.csv'
SNEA_FILE    = _HERE / 'datasets/pubmedqa_ranked_faithfulness_400_snea_bert_results.csv'
OUTPUT_DIR   = _HERE / 'output/pubmedqa_ranked_faithfulness/dataset_stats'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEVEL_LABELS = {
    4: 'L4 – Faithful\nParaphrase',
    3: 'L3 – Factual\nDistortion',
    2: 'L2 – Relation\nInversion',
    1: 'L1 – Context\nIgnoring',
}
LEVEL_COLORS = {4: '#2ecc71', 3: '#f39c12', 2: '#e74c3c', 1: '#8e44ad'}
ADJACENCIES  = [(4, 3), (3, 2), (2, 1)]
ADJ_LABELS   = ['L4>L3', 'L3>L2', 'L2>L1']


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load + compute baseline scores
# ─────────────────────────────────────────────────────────────────────────────

def load_scores() -> pd.DataFrame:
    df = pd.read_csv(DATASET_FILE)
    print(f"Dataset: {len(df)} rows, {df['question_id'].nunique()} questions")

    # SNEA-BERT
    snea = pd.read_csv(SNEA_FILE)
    snea = snea.drop_duplicates(subset='id', keep='first')
    snea = snea.rename(columns={'snea_bert_similarity': 'SNEA-BERT'})
    df   = df.merge(snea[['id', 'SNEA-BERT']], left_on='pair_id', right_on='id', how='left')
    df   = df.drop(columns=['id'], errors='ignore')

    # ROUGE-L
    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(['rougeL'], use_stemmer=True)
    df['ROUGE-L'] = [
        scorer.score(r['gt_answer'], r['candidate_answer'])['rougeL'].fmeasure
        for _, r in df.iterrows()
    ]
    print("✓ ROUGE-L computed")

    # BLEU
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    for pkg in ('punkt', 'punkt_tab'):
        nltk.download(pkg, quiet=True)
    smooth = SmoothingFunction().method1
    df['BLEU'] = [
        sentence_bleu(
            [r['gt_answer'].lower().split()],
            r['candidate_answer'].lower().split(),
            smoothing_function=smooth,
        )
        for _, r in df.iterrows()
    ]
    print("✓ BLEU computed")

    # BERTScore
    try:
        from bert_score import score as bscore
        _, _, F1 = bscore(
            df['candidate_answer'].tolist(), df['gt_answer'].tolist(),
            lang='en', model_type='roberta-large', verbose=False, batch_size=16,
        )
        df['BERTScore'] = F1.tolist()
        print("✓ BERTScore computed")
    except Exception as e:
        print(f"⚠ BERTScore skipped: {e}")
        df['BERTScore'] = np.nan

    # Sentence-BERT
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb_gt   = model.encode(df['gt_answer'].tolist(),        convert_to_tensor=True, show_progress_bar=False)
        emb_cand = model.encode(df['candidate_answer'].tolist(), convert_to_tensor=True, show_progress_bar=False)
        df['Sentence-BERT'] = util.cos_sim(emb_gt, emb_cand).diagonal().cpu().numpy()
        print("✓ Sentence-BERT computed")
    except Exception as e:
        print(f"⚠ Sentence-BERT skipped: {e}")
        df['Sentence-BERT'] = np.nan

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Statistical analyses
# ─────────────────────────────────────────────────────────────────────────────

def cohens_d(a, b):
    """Pooled Cohen's d (unsigned)."""
    a, b = np.asarray(a), np.asarray(b)
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0


def run_stats(df: pd.DataFrame, methods: list[str]) -> dict:
    results = {}
    for method in methods:
        sub = df[['question_id', 'faithfulness_level', method]].dropna()
        levels_data = {lv: sub[sub['faithfulness_level'] == lv][method].values
                       for lv in [4, 3, 2, 1]}

        # Descriptive
        desc = {}
        for lv in [4, 3, 2, 1]:
            d = levels_data[lv]
            desc[lv] = {
                'n':    len(d),
                'mean': round(float(d.mean()), 4),
                'std':  round(float(d.std()),  4),
                'min':  round(float(d.min()),  4),
                'max':  round(float(d.max()),  4),
            }

        # One-way ANOVA
        f_stat, p_anova = scipy_stats.f_oneway(*[levels_data[lv] for lv in [4, 3, 2, 1]])

        # Adjacent pairwise Welch t-tests + Cohen's d + inversion rate
        pairwise = {}
        for hi, lo in ADJACENCIES:
            a, b = levels_data[hi], levels_data[lo]
            t, p = scipy_stats.ttest_ind(a, b, equal_var=False)
            d    = cohens_d(a, b)
            # Inversion rate: % of questions where lower-level scores HIGHER
            # (requires per-question comparison)
            q_ids = sub['question_id'].unique()
            inversions = 0
            for qid in q_ids:
                q = sub[sub['question_id'] == qid]
                hi_score = q[q['faithfulness_level'] == hi][method].values
                lo_score = q[q['faithfulness_level'] == lo][method].values
                if len(hi_score) > 0 and len(lo_score) > 0:
                    if hi_score[0] <= lo_score[0]:   # expected hi > lo; if not → inversion
                        inversions += 1
            inversion_rate = round(inversions / len(q_ids), 3)

            pairwise[f'L{hi}>L{lo}'] = {
                't':              round(float(t), 3),
                'p':              round(float(p), 4),
                'cohens_d':       round(float(d), 3),
                'inversion_rate': inversion_rate,
                'significant':    p < 0.05,
            }

        results[method] = {
            'descriptive': desc,
            'anova_f':     round(float(f_stat), 3),
            'anova_p':     round(float(p_anova), 6),
            'pairwise':    pairwise,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_score_distributions(df: pd.DataFrame, methods: list[str]):
    """Violin + strip for each method — one subplot per method."""
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(n * 4.5, 6), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        sub = df[['faithfulness_level', method]].dropna()
        sub = sub.copy()
        sub['Level'] = sub['faithfulness_level'].map(
            {4: 'L4', 3: 'L3', 2: 'L2', 1: 'L1'})
        palette = {'L4': LEVEL_COLORS[4], 'L3': LEVEL_COLORS[3],
                   'L2': LEVEL_COLORS[2], 'L1': LEVEL_COLORS[1]}
        order = ['L4', 'L3', 'L2', 'L1']

        sns.violinplot(data=sub, x='Level', y=method, order=order,
                       palette=palette, inner=None, alpha=0.6, ax=ax)
        sns.boxplot(data=sub, x='Level', y=method, order=order,
                    palette=palette, width=0.25, fliersize=2,
                    linewidth=1.2, ax=ax)

        ax.set_title(method, fontsize=12, fontweight='bold')
        ax.set_xlabel('Faithfulness Level', fontsize=10)
        ax.set_ylabel('Similarity Score',   fontsize=10)

        # Add mean labels
        for i, lv_label in enumerate(order):
            lv = int(lv_label[1])
            mean_val = sub[sub['Level'] == lv_label][method].mean()
            ax.text(i, mean_val, f'{mean_val:.3f}',
                    ha='center', va='bottom', fontsize=8,
                    fontweight='bold', color='black')

    patches = [mpatches.Patch(color=LEVEL_COLORS[lv],
                               label=LEVEL_LABELS[lv].replace('\n', ' '))
               for lv in [4, 3, 2, 1]]
    fig.legend(handles=patches, loc='lower center', ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    plt.suptitle('Score Distributions by Faithfulness Level per Method',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = OUTPUT_DIR / 'score_distributions.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out}')


def plot_cohens_d_heatmap(stats: dict, methods: list[str]):
    """Heatmap of Cohen's d per method × boundary."""
    data = []
    for method in methods:
        row = [stats[method]['pairwise'][adj]['cohens_d'] for adj in ADJ_LABELS]
        data.append(row)

    display_labels = ['L4 > L3', 'L3 > L2', 'L2 > L1']
    pivot = pd.DataFrame(data, index=methods, columns=display_labels)

    fig, ax = plt.subplots(figsize=(7, max(len(methods) * 0.7 + 1.5, 4)))
    sns.heatmap(pivot, ax=ax,
                annot=True, fmt='.2f',
                annot_kws={'size': 11, 'weight': 'bold'},
                cmap='RdYlGn',
                center=0, vmin=-3, vmax=3,
                linewidths=0.5, linecolor='#cccccc',
                cbar_kws={'label': "Cohen's d  (positive = correct direction)"})
    ax.set_title("Cohen's d between adjacent faithfulness levels\n"
                 "(positive = upper level scores higher as expected)",
                 fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Boundary', fontsize=11)
    ax.set_ylabel('Method',   fontsize=11)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    out = OUTPUT_DIR / 'cohens_d_heatmap.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out}')


def plot_inversion_rate(stats: dict, methods: list[str]):
    """Bar chart: % of questions where the boundary is INVERTED per method."""
    rows = []
    disp = {'L4>L3': 'L4 > L3', 'L3>L2': 'L3 > L2', 'L2>L1': 'L2 > L1'}
    for method in methods:
        for adj in ADJ_LABELS:
            rows.append({
                'Method':          method,
                'Boundary':        disp[adj],
                'Inversion Rate %': stats[method]['pairwise'][adj]['inversion_rate'] * 100,
            })
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = {'L4 > L3': '#e74c3c', 'L3 > L2': '#f39c12', 'L2 > L1': '#2ecc71'}
    sns.barplot(data=plot_df, x='Method', y='Inversion Rate %',
                hue='Boundary', palette=palette, ax=ax)
    ax.axhline(50, ls='--', color='grey', lw=1.2, label='50% (random)')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Inversion Rate (%) — lower is better', fontsize=11)
    ax.set_xlabel('Method', fontsize=11)
    ax.set_title('Boundary Inversion Rate per Method\n'
                 '(% of questions where lower-level candidate scores HIGHER than upper-level)',
                 fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.legend(title='Boundary', fontsize=9)
    plt.tight_layout()
    out = OUTPUT_DIR / 'inversion_rate_bar.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out}')


def plot_stats_table(stats: dict, methods: list[str]):
    """Compact summary table PNG: mean scores per level + ANOVA p + inversion rates."""
    rows = []
    for method in methods:
        desc = stats[method]['descriptive']
        pw   = stats[method]['pairwise']
        rows.append({
            'Method':         method,
            'L4 mean':        desc[4]['mean'],
            'L3 mean':        desc[3]['mean'],
            'L2 mean':        desc[2]['mean'],
            'L1 mean':        desc[1]['mean'],
            'ANOVA p':        stats[method]['anova_p'],
            'Inv L4>L3 %':    int(pw['L4>L3']['inversion_rate'] * 100),
            'Inv L3>L2 %':    int(pw['L3>L2']['inversion_rate'] * 100),
            'Inv L2>L1 %':    int(pw['L2>L1']['inversion_rate'] * 100),
            'd(L4-L3)':       round(pw['L4>L3']['cohens_d'], 3),
            'd(L3-L2)':       round(pw['L3>L2']['cohens_d'], 3),
            'd(L2-L1)':       round(pw['L2>L1']['cohens_d'], 3),
        })
    tbl = pd.DataFrame(rows)

    n   = len(tbl)
    fig, ax = plt.subplots(figsize=(22, n * 0.7 + 2.5))
    ax.axis('off')
    col_keys   = list(tbl.columns)
    col_widths = [0.12, 0.07, 0.07, 0.07, 0.07, 0.07,
                  0.08, 0.08, 0.08, 0.08, 0.08, 0.08]

    table = ax.table(
        cellText=tbl.values.tolist(),
        colLabels=col_keys,
        cellLoc='center', loc='center',
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    # Header
    for j in range(len(col_keys)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Colour inversion-rate cells
    inv_cols = ['Inv L4>L3 %', 'Inv L3>L2 %', 'Inv L2>L1 %']
    for i, row in tbl.iterrows():
        for col in inv_cols:
            val = row[col]
            j   = col_keys.index(col)
            # high inversion = red, low = green
            if val >= 80:
                bg = '#f8d7da'
            elif val >= 50:
                bg = '#fde8d8'
            elif val <= 20:
                bg = '#d4edda'
            else:
                bg = '#fff3cd'
            table[i + 1, j].set_facecolor(bg)
            table[i + 1, j].set_text_props(fontweight='bold')
        # Alternate row colour for rest
        for j, col in enumerate(col_keys):
            if col not in inv_cols:
                table[i + 1, j].set_facecolor('#f9f9f9' if i % 2 == 0 else 'white')
            table[i + 1, j].set_edgecolor('#cccccc')

    plt.suptitle('PubMedQA Ranked Faithfulness — Dataset Statistical Summary\n'
                 '(Inversion Rate = % of questions where a lower-level candidate '
                 'scores HIGHER than the upper-level candidate)',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / 'stats_table.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ {out}')


# ─────────────────────────────────────────────────────────────────────────────
# 4. Print + save text summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(stats: dict, methods: list[str]):
    print('\n' + '=' * 70)
    print('DATASET STATISTICAL SUMMARY')
    print('=' * 70)
    for method in methods:
        s  = stats[method]
        pw = s['pairwise']
        print(f'\n── {method}')
        print(f"   ANOVA: F={s['anova_f']}, p={s['anova_p']}"
              f"  {'(significant)' if s['anova_p'] < 0.05 else '(NOT significant)'}")
        desc = s['descriptive']
        print(f"   Mean scores: L4={desc[4]['mean']:.3f}  L3={desc[3]['mean']:.3f}"
              f"  L2={desc[2]['mean']:.3f}  L1={desc[1]['mean']:.3f}")
        for adj in ADJ_LABELS:
            p = pw[adj]
            print(f"   {adj}:  inversion={p['inversion_rate']*100:.1f}%"
                  f"  Cohen's d={p['cohens_d']:.2f}"
                  f"  t={p['t']:.2f}, p={p['p']:.4f}"
                  f"  {'✓' if p['significant'] else '✗'}")


def save_stats_csv(stats: dict, methods: list[str]):
    rows = []
    for method in methods:
        s  = stats[method]
        pw = s['pairwise']
        desc = s['descriptive']
        base = {
            'Method':    method,
            'ANOVA_F':   s['anova_f'],
            'ANOVA_p':   s['anova_p'],
        }
        for lv in [4, 3, 2, 1]:
            for k, v in desc[lv].items():
                base[f'L{lv}_{k}'] = v
        for adj in ADJ_LABELS:
            key = adj.replace(' ', '').replace('>', '_gt_')
            for k, v in pw[adj].items():
                base[f'{key}_{k}'] = v
        rows.append(base)
    out = OUTPUT_DIR / 'stats_summary.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'  ✓ {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('=' * 70)
    print('PubMedQA Ranked Faithfulness — Dataset Statistics')
    print('=' * 70)

    df = load_scores()

    methods = [m for m in ['SNEA-BERT', 'ROUGE-L', 'BLEU', 'BERTScore', 'Sentence-BERT']
               if m in df.columns and df[m].notna().any()]
    print(f'\nMethods available: {methods}')

    print('\nRunning statistical analyses …')
    stats = run_stats(df, methods)

    print('\nGenerating plots …')
    plot_score_distributions(df, methods)
    plot_cohens_d_heatmap(stats, methods)
    plot_inversion_rate(stats, methods)
    plot_stats_table(stats, methods)

    print('\nSaving CSV …')
    save_stats_csv(stats, methods)

    print_summary(stats, methods)
    print(f'\n✓ All outputs saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    import os
    os.chdir(_HERE)
    main()
