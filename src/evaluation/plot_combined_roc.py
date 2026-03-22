"""
plot_combined_roc.py
====================
Generates a multi-panel ROC curve figure showing:
  - AA-KEA          (pure KG structural)
  - SNEA-BERT       (pure KG + BERT embeddings, α=1.0)
  - SNEA-BERT α=0.3 (hybrid: 30% KG + 70% sentence-transformer)
  - ROUGE-L, BERTScore, MiniLM, sentence-T5-base  (baselines)

One panel per dataset (3×3 grid), saved to:
  output/cross_dataset/combined_roc_curves.png

Run from src/evaluation/:
  python plot_combined_roc.py
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics import roc_curve, auc as sk_auc
from tqdm import tqdm

# ── optional heavy imports loaded lazily ────────────────────────────────────
from rouge_score import rouge_scorer as rouge_lib
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import bert_score as bs_lib
from sentence_transformers import SentenceTransformer
import torch

HERE = Path(__file__).parent

# ── Dataset registry ─────────────────────────────────────────────────────────
# Each entry:
#   base_csv  : path to file with pair_id, response1, response2, label
#   score_csv : path to *_KGs_results_snea_0.3.csv  (has all 3 method scores)
#   label     : display name for the subplot title
DATASETS = [
    {
        'name':      'mrpc',
        'base_csv':  HERE / 'datasets/mrpc_400.csv',
        'score_csv': HERE / 'datasets/mrpc_400_KGs_results_snea_0.3.csv',
        'label':     'MRPC',
    },
    {
        'name':      'paws_wiki',
        'base_csv':  HERE / 'datasets/paws_wiki_400.csv',
        'score_csv': HERE / 'datasets/paws_wiki_400_KGs_results_snea_0.3.csv',
        'label':     'PAWS-Wiki',
    },
    {
        'name':      'sts12',
        'base_csv':  HERE / 'datasets/sts12_400.csv',
        'score_csv': HERE / 'datasets/sts12_400_KGs_results_snea_0.3.csv',
        'label':     'STS12',
    },
    {
        'name':      'semantic_kg_combined',
        'base_csv':  HERE / 'datasets/semantic_kg_combined_400.csv',
        'score_csv': HERE / 'datasets/semantic_kg_combined_400_KGs_results_snea_0.3.csv',
        'label':     'SK-Combined',
    },
    {
        'name':      'semantic_kg_codex_400',
        'base_csv':  HERE / 'datasets/semantic_kg_codex_400.csv',
        'score_csv': HERE / 'datasets/semantic_kg_codex_400_KGs_results_snea_0.3.csv',
        'label':     'SK-Codex 400',
    },
    {
        'name':      'semantic_kg_findkg',
        'base_csv':  HERE / 'datasets/semantic_kg_findkg_400.csv',
        'score_csv': HERE / 'datasets/semantic_kg_findkg_400_KGs_results_snea_0.3.csv',
        'label':     'SK-FindKG',
    },
    {
        'name':      'semantic_kg_globi',
        'base_csv':  HERE / 'datasets/semantic_kg_globi_400.csv',
        'score_csv': HERE / 'datasets/semantic_kg_globi_400_KGs_results_snea_0.3.csv',
        'label':     'SK-GloBI',
    },
    {
        'name':      'semantic_kg_oregano',
        'base_csv':  HERE / 'datasets/semantic_kg_oregano_400.csv',
        'score_csv': HERE / 'datasets/semantic_kg_oregano_400_KGs_results_snea_0.3.csv',
        'label':     'SK-Oregano',
    },
    {
        'name':      'wikipedia_entity_swap',
        'base_csv':  HERE / 'datasets/wikipedia_entity_swap_400.csv',
        'score_csv': HERE / 'datasets/wikipedia_entity_swap_400_KGs_results_snea_0.3.csv',
        'label':     'Wiki Swap',
        # Per-method overrides for datasets with partial snea_0.3 coverage
        'method_overrides': {
            'AA-KEA': {
                'csv': HERE / 'datasets/wikipedia_entity_swap_aa_kea_results.csv',
                'col': 'aa_kea_similarity',
            },
            'SNEA-BERT': {
                'csv': HERE / 'datasets/wikipedia_entity_swap_400_snea_bert_results.csv',
                'col': 'snea_bert_similarity',
            },
        },
    },
]

# ── Colours & styles ─────────────────────────────────────────────────────────
METHOD_STYLE = {
    'AA-KEA':           {'color': '#1a6e2e', 'lw': 2.5, 'ls': '-',  'zorder': 10},
    'SNEA-BERT':        {'color': '#2166ac', 'lw': 2.5, 'ls': '-',  'zorder': 9},
    'SNEA-BERT α=0.3':  {'color': '#e07b00', 'lw': 2.5, 'ls': '--', 'zorder': 8},
    'ROUGE-L':          {'color': '#888888', 'lw': 1.2, 'ls': '-',  'zorder': 4},
    'BERTScore':        {'color': '#aaaaaa', 'lw': 1.2, 'ls': '--', 'zorder': 3},
    'MiniLM':           {'color': '#bbbbbb', 'lw': 1.2, 'ls': ':',  'zorder': 2},
    'sentence-T5':      {'color': '#cccccc', 'lw': 1.2, 'ls': '-.', 'zorder': 1},
}

# ── Load data ─────────────────────────────────────────────────────────────────
def load_dataset(spec):
    base = pd.read_csv(spec['base_csv'])
    scores = pd.read_csv(spec['score_csv'])

    # normalise join key
    if 'id' in scores.columns:
        scores = scores.rename(columns={'id': 'pair_id'})

    # drop duplicate pair_ids in scores
    scores = scores.drop_duplicates('pair_id')

    merged = base[['pair_id', 'response1', 'response2', 'label']].merge(
        scores[['pair_id', 'aa_kea_similarity',
                'snea_bert_alpha_1.0_SNEA_alone',
                'snea_bert_alpha_0.3']],
        on='pair_id', how='inner'
    )
    merged = merged.dropna(subset=['label'])
    return merged


# ── Baseline scorers ─────────────────────────────────────────────────────────
_rouge = rouge_lib.RougeScorer(['rougeL'], use_stemmer=False)
_smooth = SmoothingFunction().method1

def rouge_l_scores(texts1, texts2):
    return [_rouge.score(t1, t2)['rougeL'].fmeasure
            for t1, t2 in zip(texts1, texts2)]

def bleu_scores(texts1, texts2):
    scores = []
    for t1, t2 in zip(texts1, texts2):
        ref = t1.split()
        hyp = t2.split()
        scores.append(sentence_bleu([ref], hyp, smoothing_function=_smooth)
                       if hyp else 0.0)
    return scores

_bert_scorer = None
def get_bert_scorer():
    global _bert_scorer
    if _bert_scorer is None:
        print('  Loading BERTScore model (roberta-large)...')
        _bert_scorer = True          # sentinel
    return None                      # we call bert_score directly

def bertscore_scores(texts1, texts2, batch_size=16):
    print('  Computing BERTScore...')
    P, R, F1 = bs_lib.score(
        texts2, texts1,
        model_type='roberta-large',
        lang='en',
        verbose=False,
        batch_size=batch_size,
        device='cpu',
    )
    return F1.tolist()

_minilm = None
_t5 = None

def get_minilm():
    global _minilm
    if _minilm is None:
        print('  Loading MiniLM...')
        _minilm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _minilm

def get_t5():
    global _t5
    if _t5 is None:
        print('  Loading sentence-T5-base...')
        _t5 = SentenceTransformer('sentence-transformers/sentence-t5-base')
    return _t5

def cos_sim_scores(model, texts1, texts2, batch_size=32):
    all_texts = texts1 + texts2
    embs = model.encode(all_texts, batch_size=batch_size,
                        show_progress_bar=False, convert_to_numpy=True)
    n = len(texts1)
    e1, e2 = embs[:n], embs[n:]
    norms = (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)).clip(min=1e-9)
    return (np.einsum('ij,ij->i', e1, e2) / norms).tolist()


# ── Compute all scores for one dataset ────────────────────────────────────────
def compute_scores(df, base_df, spec):
    """
    Returns dict: method -> (labels_array, scores_array)
    Baselines use all pairs in df. Our methods may use per-method override files.
    """
    t1 = df['response1'].tolist()
    t2 = df['response2'].tolist()
    labels = df['label'].values
    overrides = spec.get('method_overrides', {})

    out = {}  # method -> (labels_arr, scores_arr)

    def _our(name, scores_arr, lbls=None):
        out[name] = (lbls if lbls is not None else labels, np.array(scores_arr))

    # ── Our methods ────────────────────────────────────────────────────────
    for method, col, alpha_col in [
        ('AA-KEA',          'aa_kea_similarity',              None),
        ('SNEA-BERT',       'snea_bert_alpha_1.0_SNEA_alone', None),
        ('SNEA-BERT α=0.3', None,                             'snea_bert_alpha_0.3'),
    ]:
        if method in overrides:
            ov = overrides[method]
            ov_df = pd.read_csv(ov['csv'])
            if 'id' in ov_df.columns:
                ov_df = ov_df.rename(columns={'id': 'pair_id'})
            merged = base_df[['pair_id', 'label']].merge(
                ov_df[['pair_id', ov['col']]], on='pair_id', how='inner'
            ).dropna()
            _our(method, merged[ov['col']].fillna(0).values, merged['label'].values)
        else:
            src_col = alpha_col if col is None else col
            if src_col in df.columns:
                _our(method, df[src_col].fillna(0).values)
            else:
                print(f'  ⚠ Column {src_col} not found — skipping {method}')

    # ── Baselines (computed on df pairs) ──────────────────────────────────
    print('  Computing ROUGE-L...')
    out['ROUGE-L'] = (labels, np.array(rouge_l_scores(t1, t2)))

    out['BERTScore'] = (labels, np.array(bertscore_scores(t1, t2)))

    print('  Computing MiniLM...')
    out['MiniLM'] = (labels, np.array(cos_sim_scores(get_minilm(), t1, t2)))

    print('  Computing sentence-T5...')
    out['sentence-T5'] = (labels, np.array(cos_sim_scores(get_t5(), t1, t2)))

    return out


# ── Plot ──────────────────────────────────────────────────────────────────────
def make_roc_panel(ax, scores_dict, title):
    """scores_dict: method -> (labels_array, scores_array)"""
    for method, (lbls, scores) in scores_dict.items():
        s = np.asarray(scores, dtype=float)
        l = np.asarray(lbls)
        if np.isnan(s).all() or len(np.unique(s)) < 2:
            continue
        fpr, tpr, _ = roc_curve(l, s)
        area = sk_auc(fpr, tpr)
        sty = METHOD_STYLE.get(method, {'color': 'gray', 'lw': 1.0, 'ls': '-', 'zorder': 1})
        n_label = f' [N={len(l)}]' if len(l) < 380 else ''
        ax.plot(fpr, tpr,
                color=sty['color'], lw=sty['lw'], ls=sty['ls'],
                zorder=sty['zorder'],
                label=f'{method}{n_label} ({area:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_title(title, fontsize=11, fontweight='bold', pad=4)
    ax.set_xlabel('FPR', fontsize=8)
    ax.set_ylabel('TPR', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, lw=0.5)

    # Legend inside panel — small font
    leg = ax.legend(fontsize=6.5, loc='lower right',
                    framealpha=0.9, edgecolor='#cccccc',
                    handlelength=1.5, handletextpad=0.4)
    return ax


def make_legend_handles():
    handles = []
    for method, sty in METHOD_STYLE.items():
        handles.append(mlines.Line2D(
            [], [],
            color=sty['color'], lw=sty['lw'], ls=sty['ls'],
            label=method
        ))
    return handles


def plot_all(all_data):
    n = len(all_data)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    for i, (spec, scores_dict) in enumerate(all_data):
        make_roc_panel(axes[i], scores_dict, spec['label'])

    # Hide unused panels
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Shared super-legend below the grid
    fig.legend(
        handles=make_legend_handles(),
        loc='lower center',
        ncol=len(METHOD_STYLE),
        fontsize=9,
        framealpha=0.95,
        edgecolor='#aaaaaa',
        title='Methods  (AUC shown per panel)',
        title_fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        'ROC Curves — AA-KEA · SNEA-BERT · SNEA-BERT α=0.3  vs  Baselines\n'
        'Across All Benchmark Datasets',
        fontsize=13, fontweight='bold', y=1.01
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out_dir = HERE / 'output/cross_dataset'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'combined_roc_curves.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'\n✓ Saved: {out_path}')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    all_data = []
    for spec in DATASETS:
        print(f'\n{"="*60}')
        print(f'Dataset: {spec["label"]}')
        print('='*60)

        if not spec['base_csv'].exists():
            print(f'  ⚠ Base CSV not found: {spec["base_csv"]} — skipping.')
            continue
        if not spec['score_csv'].exists():
            print(f'  ⚠ Score CSV not found: {spec["score_csv"]} — skipping.')
            continue

        base_df = pd.read_csv(spec['base_csv'])
        df = load_dataset(spec)
        print(f'  Loaded {len(df)} pairs (snea_0.3 join).')

        scores = compute_scores(df, base_df, spec)
        all_data.append((spec, scores))

    print('\nPlotting combined ROC figure...')
    plot_all(all_data)


if __name__ == '__main__':
    main()
