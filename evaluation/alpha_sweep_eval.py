"""
alpha_sweep_eval.py
===================
Evaluates SNEA-BERT at each alpha value (0.0 → 1.0, step 0.1) across all
datasets. Baselines (ROUGE-L, BERTScore, MiniLM, sentence-T5) are computed
once per dataset to avoid redundant heavy computation.

Produces:
  output/alpha_sweep/
    alpha_sweep_f1.csv          — F1  per dataset × alpha
    alpha_sweep_auc.csv         — AUC per dataset × alpha
    alpha_sweep_f1_lines.png    — Line plot: F1  vs alpha, one line per dataset
    alpha_sweep_auc_lines.png   — Line plot: AUC vs alpha, one line per dataset
    alpha_sweep_f1_heatmap.png  — Heatmap: datasets × alpha, coloured by F1
    alpha_sweep_auc_heatmap.png — Heatmap: datasets × alpha, coloured by AUC
    baselines_summary.csv       — Baseline F1 + AUC per dataset (reference)

Run from src/evaluation/:
    python alpha_sweep_eval.py
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

from rouge_score import rouge_scorer as rouge_lib
import bert_score as bs_lib
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
DATASETS_DIR = HERE / 'datasets'
OUT_DIR = HERE / 'output' / 'alpha_sweep'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Alpha values ──────────────────────────────────────────────────────────────
ALPHAS = ['0p0', '0p1', '0p2', '0p3', '0p4', '0p5', '0p6', '0p7', '0p8', '0p9', '1p0']
ALPHA_FLOAT = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# ── Dataset registry ──────────────────────────────────────────────────────────
# base_csv  : has pair_id, response1, response2, label
# score_prefix : prefix for <score_prefix>_snea_alpha_<X>_results.csv
DATASETS = [
    {'name': 'mrpc_400',                         'label': 'MRPC',            'base_csv': 'mrpc_400.csv'},
    {'name': 'paws_wiki_400',                    'label': 'PAWS-Wiki',       'base_csv': 'paws_wiki_400.csv'},
    {'name': 'sts12_400',                        'label': 'STS12',           'base_csv': 'sts12_400.csv'},
    {'name': 'semantic_kg_combined_400',         'label': 'SK-Combined',     'base_csv': 'semantic_kg_combined_400.csv'},
    {'name': 'semantic_kg_codex_400',            'label': 'SK-Codex',        'base_csv': 'semantic_kg_codex_400.csv'},
    {'name': 'semantic_kg_findkg_400',           'label': 'SK-FindKG',       'base_csv': 'semantic_kg_findkg_400.csv'},
    {'name': 'semantic_kg_globi_400',            'label': 'SK-GloBI',        'base_csv': 'semantic_kg_globi_400.csv'},
    {'name': 'semantic_kg_oregano_400',          'label': 'SK-Oregano',      'base_csv': 'semantic_kg_oregano_400.csv'},
    {'name': 'wikipedia_entity_swap_400',        'label': 'Wiki Swap',       'base_csv': 'wikipedia_entity_swap_400.csv'},
]

# ── Baseline scorers ──────────────────────────────────────────────────────────
_rouge = rouge_lib.RougeScorer(['rougeL'], use_stemmer=False)

def rouge_l_scores(t1, t2):
    return [_rouge.score(a, b)['rougeL'].fmeasure for a, b in zip(t1, t2)]

def bertscore_scores(t1, t2, batch_size=16):
    print('    Computing BERTScore...')
    _, _, F1 = bs_lib.score(t2, t1, model_type='roberta-large', lang='en',
                             verbose=False, batch_size=batch_size, device='cpu')
    return F1.tolist()

_minilm = None
_t5 = None

def get_minilm():
    global _minilm
    if _minilm is None:
        print('    Loading MiniLM...')
        _minilm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _minilm

def get_t5():
    global _t5
    if _t5 is None:
        print('    Loading sentence-T5-base...')
        _t5 = SentenceTransformer('sentence-transformers/sentence-t5-base')
    return _t5

def cos_sim_scores(model, t1, t2, batch_size=32):
    all_texts = t1 + t2
    embs = model.encode(all_texts, batch_size=batch_size,
                        show_progress_bar=False, convert_to_numpy=True)
    n = len(t1)
    e1, e2 = embs[:n], embs[n:]
    norms = (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)).clip(min=1e-9)
    return (np.einsum('ij,ij->i', e1, e2) / norms).tolist()

# ── Metric helpers ────────────────────────────────────────────────────────────
def best_f1(labels, scores):
    thresholds = np.unique(scores)
    best = 0.0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f = f1_score(labels, preds, zero_division=0)
        if f > best:
            best = f
    return best

def safe_auc(labels, scores):
    if len(np.unique(labels)) < 2:
        return float('nan')
    return roc_auc_score(labels, scores)

# ── Main evaluation ───────────────────────────────────────────────────────────
def evaluate_dataset(spec):
    base_path = DATASETS_DIR / spec['base_csv']
    if not base_path.exists():
        print(f'  ⚠ Base CSV not found: {base_path} — skipping.')
        return None, None

    base_df = pd.read_csv(base_path)
    base_df = base_df[['pair_id', 'response1', 'response2', 'label']].dropna(subset=['label'])
    t1 = base_df['response1'].tolist()
    t2 = base_df['response2'].tolist()
    labels = base_df['label'].values

    # ── Compute baselines once ────────────────────────────────────────────
    print('  Computing baselines...')
    baseline_scores = {
        'ROUGE-L':   np.array(rouge_l_scores(t1, t2)),
        'BERTScore': np.array(bertscore_scores(t1, t2)),
        'MiniLM':    np.array(cos_sim_scores(get_minilm(), t1, t2)),
        'sentence-T5': np.array(cos_sim_scores(get_t5(), t1, t2)),
    }
    baselines = {}
    for name, scores in baseline_scores.items():
        baselines[name] = {
            'f1':  best_f1(labels, scores),
            'auc': safe_auc(labels, scores),
        }

    # ── Sweep alphas ──────────────────────────────────────────────────────
    alpha_results = {}
    for alpha_label, alpha_val in zip(ALPHAS, ALPHA_FLOAT):
        score_path = DATASETS_DIR / f'{spec["name"]}_snea_alpha_{alpha_label}_results.csv'
        if not score_path.exists():
            print(f'  ⚠ Missing: {score_path.name} — skipping α={alpha_val}')
            alpha_results[alpha_val] = {'f1': float('nan'), 'auc': float('nan')}
            continue

        scores_df = pd.read_csv(score_path)
        merged = base_df[['pair_id', 'label']].merge(
            scores_df[['pair_id', 'snea_bert_similarity']],
            on='pair_id', how='inner'
        ).dropna()

        s = merged['snea_bert_similarity'].values
        l = merged['label'].values
        alpha_results[alpha_val] = {
            'f1':  best_f1(l, s),
            'auc': safe_auc(l, s),
            'n':   len(merged),
        }

    return alpha_results, baselines


def main():
    all_f1  = {}   # dataset_label -> {alpha -> f1}
    all_auc = {}   # dataset_label -> {alpha -> auc}
    all_baselines = {}   # dataset_label -> {method -> {f1, auc}}

    for spec in DATASETS:
        print(f'\n{"="*60}')
        print(f'Dataset: {spec["label"]}')
        print('='*60)

        alpha_res, baselines = evaluate_dataset(spec)
        if alpha_res is None:
            continue

        all_f1[spec['label']]  = {a: alpha_res[a]['f1']  for a in ALPHA_FLOAT}
        all_auc[spec['label']] = {a: alpha_res[a]['auc'] for a in ALPHA_FLOAT}
        all_baselines[spec['label']] = baselines

        # Print alpha sweep summary
        print(f'  {"Alpha":>6}  {"F1":>7}  {"AUC":>7}  {"N":>5}')
        for a_lbl, a_val in zip(ALPHAS, ALPHA_FLOAT):
            r = alpha_res[a_val]
            n = r.get('n', '-')
            print(f'  {a_val:>6.1f}  {r["f1"]:>7.4f}  {r["auc"]:>7.4f}  {n!s:>5}')

    # ── Save CSVs ─────────────────────────────────────────────────────────
    f1_df  = pd.DataFrame(all_f1).T
    f1_df.columns = [f'alpha_{a}' for a in ALPHA_FLOAT]
    f1_df.index.name = 'dataset'
    f1_df.to_csv(OUT_DIR / 'alpha_sweep_f1.csv')

    auc_df = pd.DataFrame(all_auc).T
    auc_df.columns = [f'alpha_{a}' for a in ALPHA_FLOAT]
    auc_df.index.name = 'dataset'
    auc_df.to_csv(OUT_DIR / 'alpha_sweep_auc.csv')

    # Baselines summary
    rows = []
    for ds_label, bl in all_baselines.items():
        for method, vals in bl.items():
            rows.append({'dataset': ds_label, 'method': method,
                         'f1': vals['f1'], 'auc': vals['auc']})
    pd.DataFrame(rows).to_csv(OUT_DIR / 'baselines_summary.csv', index=False)

    print(f'\n✓ CSVs saved to {OUT_DIR}')

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_lines(f1_df,  'F1',  'alpha_sweep_f1_lines.png',  all_baselines)
    plot_lines(auc_df, 'AUC', 'alpha_sweep_auc_lines.png', all_baselines)
    plot_heatmap(f1_df,  'F1 Score',  'alpha_sweep_f1_heatmap.png')
    plot_heatmap(auc_df, 'ROC-AUC',   'alpha_sweep_auc_heatmap.png')

    print(f'✓ Plots saved to {OUT_DIR}')


# ── Plotting ──────────────────────────────────────────────────────────────────
DATASET_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
]

def plot_lines(df, metric_name, filename, all_baselines):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (ds_label, row) in enumerate(df.iterrows()):
        color = DATASET_COLORS[i % len(DATASET_COLORS)]
        ax.plot(ALPHA_FLOAT, row.values, marker='o', markersize=5,
                lw=2, color=color, label=ds_label)

    ax.set_xlabel('Alpha (KG weight)', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'SNEA-BERT {metric_name} vs Alpha — All Datasets', fontsize=13, fontweight='bold')
    ax.set_xticks(ALPHA_FLOAT)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9, ncol=2)

    plt.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  ✓ {filename}')


def plot_heatmap(df, metric_name, filename):
    fig, ax = plt.subplots(figsize=(13, 5))

    data = df.values.astype(float)
    col_labels = [f'{a:.1f}' for a in ALPHA_FLOAT]

    im = ax.imshow(data, aspect='auto', cmap='RdBu', vmin=0.4, vmax=1.0)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index.tolist(), fontsize=10)
    ax.set_xlabel('Alpha (KG weight →  BERT weight)', fontsize=11)
    ax.set_title(f'SNEA-BERT {metric_name} by Alpha and Dataset', fontsize=13, fontweight='bold')

    # Annotate cells
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = data[r, c]
            if not np.isnan(v):
                ax.text(c, r, f'{v:.3f}', ha='center', va='center',
                        fontsize=7.5, color='black' if 0.5 < v < 0.9 else 'white')

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label=metric_name)
    plt.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  ✓ {filename}')


if __name__ == '__main__':
    main()