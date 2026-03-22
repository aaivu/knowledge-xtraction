"""
Cross-dataset evaluation plotting and summary export.

Structure of all_methods_confusion.csv:
  - "Dataset" column encodes both the benchmark dataset AND which variant of
    our methodology was used, e.g. "MRPC (WL)", "Wiki Swap (SNEA-BERT)".
  - "Method" column contains the scoring method: our variants are labelled
    "AA-KEA (Our Method)" / "SNEA-BERT (Our Method)"; the rest are baselines.

The 5 benchmark datasets are:
    MRPC, PAWS-Wiki, Semantic-KG Codex, Semantic-KG Combined, Wiki Swap

Our methodology variants are:
    AA-KEA, SNEA-BERT, WL, WL Accurate, WL Clean, KEA Enhanced, KEA BERT, GNN

Baselines are:
    ROUGE-1, ROUGE-2, ROUGE-L, BLEU, BERTScore, MiniLM, T5-base

Outputs:
  1. all_methods_heatmap.png      — F1 + AUC, all variants + baselines × datasets
  2. variants_comparison_heatmap.png — F1 + AUC, our variants only × datasets
  3. all_methods_ranks.png        — mean rank by F1 and AUC
  4. cross_dataset_summary.csv    — flat summary table
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

sns.set_style('whitegrid')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE         = Path(__file__).parent
OUTPUT_DIR    = _HERE / 'output'
CROSS_DIR     = OUTPUT_DIR / 'cross_dataset'
CROSS_DIR.mkdir(parents=True, exist_ok=True)
CONFUSION_CSV = CROSS_DIR / 'all_methods_confusion.csv'

# ---------------------------------------------------------------------------
# Method / dataset configuration
# ---------------------------------------------------------------------------

BASELINE_METHODS = [
    'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU',
    'BERTScore', 'all-MiniLM-L6-v2', 'sentence-t5-base',
]

OUR_METHOD_LABELS = {'AA-KEA (Our Method)', 'SNEA-BERT (Our Method)', 'SNEA-BERT α=0.3 (Our Method)', 'SNEA-BERT α=0.0 (Our Method)', 'SNEA-BERT α=0.1 (Our Method)', 'SNEA-BERT α=0.2 (Our Method)', 'SNEA-BERT α=0.3 (Our Method)', 'SNEA-BERT α=0.4 (Our Method)', 'SNEA-BERT α=0.5 (Our Method)', 'SNEA-BERT α=0.6 (Our Method)', 'SNEA-BERT α=0.7 (Our Method)', 'SNEA-BERT α=0.8 (Our Method)', 'SNEA-BERT α=0.9 (Our Method)', 'SNEA-BERT α=1.0 (Our Method)'}

# Maps every "Dataset" label in the confusion CSV → (base_dataset, variant_name)
DATASET_VARIANT_MAP = {
    # ── Original 5 benchmark datasets ────────────────────────────────────────
    'MRPC (AA-KEA)':                         ('MRPC', 'AA-KEA'),
    'MRPC (WL)':                             ('MRPC', 'WL'),
    'MRPC (WL Accurate)':                    ('MRPC', 'WL Accurate'),
    'MRPC (KEA Enhanced)':                   ('MRPC', 'KEA Enhanced'),
    'MRPC (KEA BERT)':                       ('MRPC', 'KEA BERT'),
    'MRPC (GNN)':                            ('MRPC', 'GNN'),
    'MRPC (SNEA-BERT)':                      ('MRPC', 'SNEA-BERT'),
    'PAWS-Wiki':                             ('PAWS-Wiki', 'AA-KEA'),
    'PAWS-Wiki (SNEA-BERT)':                 ('PAWS-Wiki', 'SNEA-BERT'),
    'Semantic-KG Codex':                     ('Semantic-KG Codex', 'AA-KEA'),
    'Semantic-KG Combined':                  ('Semantic-KG Combined', 'AA-KEA'),
    'Sem-KG Comb (WL)':                      ('Semantic-KG Combined', 'WL'),
    'Semantic-KG (SNEA-BERT)':               ('Semantic-KG Combined', 'SNEA-BERT'),
    'Wiki Swap (AA-KEA)':                    ('Wiki Swap', 'AA-KEA'),
    'Wiki Swap (WL)':                        ('Wiki Swap', 'WL'),
    'Wiki Swap WL Clean':                    ('Wiki Swap', 'WL Clean'),
    'Wiki Swap (SNEA-BERT)':                 ('Wiki Swap', 'SNEA-BERT'),
    # ── New KG datasets (SNEA-BERT) ───────────────────────────────────────────
    'Semantic-KG Codex 400 (SNEA-BERT)':     ('Semantic-KG Codex 400', 'SNEA-BERT'),
    'Semantic-KG FindKG (SNEA-BERT)':        ('Semantic-KG FindKG', 'SNEA-BERT'),
    'Semantic-KG GloBI (SNEA-BERT)':         ('Semantic-KG GloBI', 'SNEA-BERT'),
    'Semantic-KG Oregano (SNEA-BERT)':       ('Semantic-KG Oregano', 'SNEA-BERT'),
    'STS12 (SNEA-BERT)':                     ('STS12', 'SNEA-BERT'),
    # ── SNEA-BERT α=0.3 (30% KG + 70% sentence-transformer) ─────────────────
    'MRPC (SNEA-BERT α=0.3)':               ('MRPC', 'SNEA-BERT α=0.3'),
    'PAWS-Wiki (SNEA-BERT α=0.3)':          ('PAWS-Wiki', 'SNEA-BERT α=0.3'),
    'Semantic-KG (SNEA-BERT α=0.3)':        ('Semantic-KG Combined', 'SNEA-BERT α=0.3'),
    'Wiki Swap (SNEA-BERT α=0.3)':          ('Wiki Swap', 'SNEA-BERT α=0.3'),
    'Semantic-KG Codex 400 (SNEA-α=0.3)':  ('Semantic-KG Codex 400', 'SNEA-BERT α=0.3'),
    'Semantic-KG FindKG (SNEA-α=0.3)':     ('Semantic-KG FindKG', 'SNEA-BERT α=0.3'),
    'Semantic-KG GloBI (SNEA-α=0.3)':      ('Semantic-KG GloBI', 'SNEA-BERT α=0.3'),
    'Semantic-KG Oregano (SNEA-α=0.3)':    ('Semantic-KG Oregano', 'SNEA-BERT α=0.3'),
    'STS12 (SNEA-α=0.3)':                  ('STS12', 'SNEA-BERT α=0.3'),
    # ── SNEA-BERT alpha sweep (α = 0.0 → 1.0) ─────────────────────────────────
    'MRPC (SNEA-α=0.0)':  ('MRPC', 'SNEA-BERT α=0.0'),
    'MRPC (SNEA-α=0.1)':  ('MRPC', 'SNEA-BERT α=0.1'),
    'MRPC (SNEA-α=0.2)':  ('MRPC', 'SNEA-BERT α=0.2'),
    'MRPC (SNEA-α=0.3)':  ('MRPC', 'SNEA-BERT α=0.3'),
    'MRPC (SNEA-α=0.4)':  ('MRPC', 'SNEA-BERT α=0.4'),
    'MRPC (SNEA-α=0.5)':  ('MRPC', 'SNEA-BERT α=0.5'),
    'MRPC (SNEA-α=0.6)':  ('MRPC', 'SNEA-BERT α=0.6'),
    'MRPC (SNEA-α=0.7)':  ('MRPC', 'SNEA-BERT α=0.7'),
    'MRPC (SNEA-α=0.8)':  ('MRPC', 'SNEA-BERT α=0.8'),
    'MRPC (SNEA-α=0.9)':  ('MRPC', 'SNEA-BERT α=0.9'),
    'MRPC (SNEA-α=1.0)':  ('MRPC', 'SNEA-BERT α=1.0'),
    'PAWS-Wiki (SNEA-α=0.0)':  ('PAWS-Wiki', 'SNEA-BERT α=0.0'),
    'PAWS-Wiki (SNEA-α=0.1)':  ('PAWS-Wiki', 'SNEA-BERT α=0.1'),
    'PAWS-Wiki (SNEA-α=0.2)':  ('PAWS-Wiki', 'SNEA-BERT α=0.2'),
    'PAWS-Wiki (SNEA-α=0.3)':  ('PAWS-Wiki', 'SNEA-BERT α=0.3'),
    'PAWS-Wiki (SNEA-α=0.4)':  ('PAWS-Wiki', 'SNEA-BERT α=0.4'),
    'PAWS-Wiki (SNEA-α=0.5)':  ('PAWS-Wiki', 'SNEA-BERT α=0.5'),
    'PAWS-Wiki (SNEA-α=0.6)':  ('PAWS-Wiki', 'SNEA-BERT α=0.6'),
    'PAWS-Wiki (SNEA-α=0.7)':  ('PAWS-Wiki', 'SNEA-BERT α=0.7'),
    'PAWS-Wiki (SNEA-α=0.8)':  ('PAWS-Wiki', 'SNEA-BERT α=0.8'),
    'PAWS-Wiki (SNEA-α=0.9)':  ('PAWS-Wiki', 'SNEA-BERT α=0.9'),
    'PAWS-Wiki (SNEA-α=1.0)':  ('PAWS-Wiki', 'SNEA-BERT α=1.0'),
    'STS12 (SNEA-α=0.0)':  ('STS12', 'SNEA-BERT α=0.0'),
    'STS12 (SNEA-α=0.1)':  ('STS12', 'SNEA-BERT α=0.1'),
    'STS12 (SNEA-α=0.2)':  ('STS12', 'SNEA-BERT α=0.2'),
    'STS12 (SNEA-α=0.3)':  ('STS12', 'SNEA-BERT α=0.3'),
    'STS12 (SNEA-α=0.4)':  ('STS12', 'SNEA-BERT α=0.4'),
    'STS12 (SNEA-α=0.5)':  ('STS12', 'SNEA-BERT α=0.5'),
    'STS12 (SNEA-α=0.6)':  ('STS12', 'SNEA-BERT α=0.6'),
    'STS12 (SNEA-α=0.7)':  ('STS12', 'SNEA-BERT α=0.7'),
    'STS12 (SNEA-α=0.8)':  ('STS12', 'SNEA-BERT α=0.8'),
    'STS12 (SNEA-α=0.9)':  ('STS12', 'SNEA-BERT α=0.9'),
    'STS12 (SNEA-α=1.0)':  ('STS12', 'SNEA-BERT α=1.0'),
    'Semantic-KG Combined (SNEA-α=0.0)':  ('Semantic-KG Combined', 'SNEA-BERT α=0.0'),
    'Semantic-KG Combined (SNEA-α=0.1)':  ('Semantic-KG Combined', 'SNEA-BERT α=0.1'),
    'Semantic-KG Combined (SNEA-α=0.2)':  ('Semantic-KG Combined', 'SNEA-BERT α=0.2'),
    'Semantic-KG Combined (SNEA-α=0.3)':  ('Semantic-KG Combined', 'SNEA-BERT α=0.3'),
    'Semantic-KG Combined (SNEA-α=0.4)':  ('Semantic-KG Combined', 'SNEA-BERT α=0.4'),
    'Semantic-KG Combined (SNEA-α=0.5)':  ('Semantic-KG Combined', 'SNEA-BERT α=0.5'),
    'Semantic-KG Combined (SNEA-α=0.6)':  ('Semantic-KG Combined', 'SNEA-BERT α=0.6'),
    'Semantic-KG Combined (SNEA-α=0.7)':  ('Semantic-KG Combined', 'SNEA-BERT α=0.7'),
    'Semantic-KG Combined (SNEA-α=0.8)':  ('Semantic-KG Combined', 'SNEA-BERT α=0.8'),
    'Semantic-KG Combined (SNEA-α=0.9)':  ('Semantic-KG Combined', 'SNEA-BERT α=0.9'),
    'Semantic-KG Combined (SNEA-α=1.0)':  ('Semantic-KG Combined', 'SNEA-BERT α=1.0'),
    'Semantic-KG Codex 400 (SNEA-α=0.0)':  ('Semantic-KG Codex 400', 'SNEA-BERT α=0.0'),
    'Semantic-KG Codex 400 (SNEA-α=0.1)':  ('Semantic-KG Codex 400', 'SNEA-BERT α=0.1'),
    'Semantic-KG Codex 400 (SNEA-α=0.2)':  ('Semantic-KG Codex 400', 'SNEA-BERT α=0.2'),
    'Semantic-KG Codex 400 (SNEA-α=0.3)':  ('Semantic-KG Codex 400', 'SNEA-BERT α=0.3'),
    'Semantic-KG Codex 400 (SNEA-α=0.4)':  ('Semantic-KG Codex 400', 'SNEA-BERT α=0.4'),
    'Semantic-KG Codex 400 (SNEA-α=0.5)':  ('Semantic-KG Codex 400', 'SNEA-BERT α=0.5'),
    'Semantic-KG Codex 400 (SNEA-α=0.6)':  ('Semantic-KG Codex 400', 'SNEA-BERT α=0.6'),
    'Semantic-KG Codex 400 (SNEA-α=0.7)':  ('Semantic-KG Codex 400', 'SNEA-BERT α=0.7'),
    'Semantic-KG Codex 400 (SNEA-α=0.8)':  ('Semantic-KG Codex 400', 'SNEA-BERT α=0.8'),
    'Semantic-KG Codex 400 (SNEA-α=0.9)':  ('Semantic-KG Codex 400', 'SNEA-BERT α=0.9'),
    'Semantic-KG Codex 400 (SNEA-α=1.0)':  ('Semantic-KG Codex 400', 'SNEA-BERT α=1.0'),
    'Semantic-KG FindKG (SNEA-α=0.0)':  ('Semantic-KG FindKG', 'SNEA-BERT α=0.0'),
    'Semantic-KG FindKG (SNEA-α=0.1)':  ('Semantic-KG FindKG', 'SNEA-BERT α=0.1'),
    'Semantic-KG FindKG (SNEA-α=0.2)':  ('Semantic-KG FindKG', 'SNEA-BERT α=0.2'),
    'Semantic-KG FindKG (SNEA-α=0.3)':  ('Semantic-KG FindKG', 'SNEA-BERT α=0.3'),
    'Semantic-KG FindKG (SNEA-α=0.4)':  ('Semantic-KG FindKG', 'SNEA-BERT α=0.4'),
    'Semantic-KG FindKG (SNEA-α=0.5)':  ('Semantic-KG FindKG', 'SNEA-BERT α=0.5'),
    'Semantic-KG FindKG (SNEA-α=0.6)':  ('Semantic-KG FindKG', 'SNEA-BERT α=0.6'),
    'Semantic-KG FindKG (SNEA-α=0.7)':  ('Semantic-KG FindKG', 'SNEA-BERT α=0.7'),
    'Semantic-KG FindKG (SNEA-α=0.8)':  ('Semantic-KG FindKG', 'SNEA-BERT α=0.8'),
    'Semantic-KG FindKG (SNEA-α=0.9)':  ('Semantic-KG FindKG', 'SNEA-BERT α=0.9'),
    'Semantic-KG FindKG (SNEA-α=1.0)':  ('Semantic-KG FindKG', 'SNEA-BERT α=1.0'),
    'Semantic-KG GloBI (SNEA-α=0.0)':  ('Semantic-KG GloBI', 'SNEA-BERT α=0.0'),
    'Semantic-KG GloBI (SNEA-α=0.1)':  ('Semantic-KG GloBI', 'SNEA-BERT α=0.1'),
    'Semantic-KG GloBI (SNEA-α=0.2)':  ('Semantic-KG GloBI', 'SNEA-BERT α=0.2'),
    'Semantic-KG GloBI (SNEA-α=0.3)':  ('Semantic-KG GloBI', 'SNEA-BERT α=0.3'),
    'Semantic-KG GloBI (SNEA-α=0.4)':  ('Semantic-KG GloBI', 'SNEA-BERT α=0.4'),
    'Semantic-KG GloBI (SNEA-α=0.5)':  ('Semantic-KG GloBI', 'SNEA-BERT α=0.5'),
    'Semantic-KG GloBI (SNEA-α=0.6)':  ('Semantic-KG GloBI', 'SNEA-BERT α=0.6'),
    'Semantic-KG GloBI (SNEA-α=0.7)':  ('Semantic-KG GloBI', 'SNEA-BERT α=0.7'),
    'Semantic-KG GloBI (SNEA-α=0.8)':  ('Semantic-KG GloBI', 'SNEA-BERT α=0.8'),
    'Semantic-KG GloBI (SNEA-α=0.9)':  ('Semantic-KG GloBI', 'SNEA-BERT α=0.9'),
    'Semantic-KG GloBI (SNEA-α=1.0)':  ('Semantic-KG GloBI', 'SNEA-BERT α=1.0'),
    'Semantic-KG Oregano (SNEA-α=0.0)':  ('Semantic-KG Oregano', 'SNEA-BERT α=0.0'),
    'Semantic-KG Oregano (SNEA-α=0.1)':  ('Semantic-KG Oregano', 'SNEA-BERT α=0.1'),
    'Semantic-KG Oregano (SNEA-α=0.2)':  ('Semantic-KG Oregano', 'SNEA-BERT α=0.2'),
    'Semantic-KG Oregano (SNEA-α=0.3)':  ('Semantic-KG Oregano', 'SNEA-BERT α=0.3'),
    'Semantic-KG Oregano (SNEA-α=0.4)':  ('Semantic-KG Oregano', 'SNEA-BERT α=0.4'),
    'Semantic-KG Oregano (SNEA-α=0.5)':  ('Semantic-KG Oregano', 'SNEA-BERT α=0.5'),
    'Semantic-KG Oregano (SNEA-α=0.6)':  ('Semantic-KG Oregano', 'SNEA-BERT α=0.6'),
    'Semantic-KG Oregano (SNEA-α=0.7)':  ('Semantic-KG Oregano', 'SNEA-BERT α=0.7'),
    'Semantic-KG Oregano (SNEA-α=0.8)':  ('Semantic-KG Oregano', 'SNEA-BERT α=0.8'),
    'Semantic-KG Oregano (SNEA-α=0.9)':  ('Semantic-KG Oregano', 'SNEA-BERT α=0.9'),
    'Semantic-KG Oregano (SNEA-α=1.0)':  ('Semantic-KG Oregano', 'SNEA-BERT α=1.0'),
    'Wiki Swap (SNEA-α=0.0)':  ('Wiki Swap', 'SNEA-BERT α=0.0'),
    'Wiki Swap (SNEA-α=0.1)':  ('Wiki Swap', 'SNEA-BERT α=0.1'),
    'Wiki Swap (SNEA-α=0.2)':  ('Wiki Swap', 'SNEA-BERT α=0.2'),
    'Wiki Swap (SNEA-α=0.3)':  ('Wiki Swap', 'SNEA-BERT α=0.3'),
    'Wiki Swap (SNEA-α=0.4)':  ('Wiki Swap', 'SNEA-BERT α=0.4'),
    'Wiki Swap (SNEA-α=0.5)':  ('Wiki Swap', 'SNEA-BERT α=0.5'),
    'Wiki Swap (SNEA-α=0.6)':  ('Wiki Swap', 'SNEA-BERT α=0.6'),
    'Wiki Swap (SNEA-α=0.7)':  ('Wiki Swap', 'SNEA-BERT α=0.7'),
    'Wiki Swap (SNEA-α=0.8)':  ('Wiki Swap', 'SNEA-BERT α=0.8'),
    'Wiki Swap (SNEA-α=0.9)':  ('Wiki Swap', 'SNEA-BERT α=0.9'),
    'Wiki Swap (SNEA-α=1.0)':  ('Wiki Swap', 'SNEA-BERT α=1.0'),
}

BASE_DATASETS = [
    'MRPC', 'PAWS-Wiki', 'Semantic-KG Codex', 'Semantic-KG Combined', 'Wiki Swap',
    # New datasets
    'Semantic-KG Codex 400', 'Semantic-KG FindKG', 'Semantic-KG GloBI',
    'Semantic-KG Oregano', 'STS12',
]

# Our variants in preferred display order
VARIANT_ORDER = list(dict.fromkeys([
    'AA-KEA', 'SNEA-BERT',
    'SNEA-BERT α=0.0', 'SNEA-BERT α=0.1', 'SNEA-BERT α=0.2', 'SNEA-BERT α=0.3',
    'SNEA-BERT α=0.4', 'SNEA-BERT α=0.5', 'SNEA-BERT α=0.6', 'SNEA-BERT α=0.7',
    'SNEA-BERT α=0.8', 'SNEA-BERT α=0.9', 'SNEA-BERT α=1.0',
    'WL', 'WL Accurate', 'WL Clean', 'KEA Enhanced', 'KEA BERT', 'GNN',
]))

# Shown in the main heatmap / rank plot (AA-KEA excluded; only alpha sweep + plain SNEA-BERT).
MAIN_OUR_VARIANTS = list(dict.fromkeys([
    'SNEA-BERT',
    'SNEA-BERT α=0.0', 'SNEA-BERT α=0.1', 'SNEA-BERT α=0.2', 'SNEA-BERT α=0.3',
    'SNEA-BERT α=0.4', 'SNEA-BERT α=0.5', 'SNEA-BERT α=0.6', 'SNEA-BERT α=0.7',
    'SNEA-BERT α=0.8', 'SNEA-BERT α=0.9', 'SNEA-BERT α=1.0',
]))

# Alpha-only variants used for "Our Best Variant" in the performance summary table.
ALPHA_VARIANTS = [v for v in MAIN_OUR_VARIANTS if 'α=' in v]

# Datasets shown in heatmap/rank plots (Semantic-KG Codex excluded — no alpha data,
# produces all-grey columns for our variants).
HEATMAP_DATASETS = [d for d in BASE_DATASETS if d != 'Semantic-KG Codex']

# Short display names for x-tick labels
METHOD_SHORT = {
    'AA-KEA':           'AA-KEA\n(Ours)',
    'SNEA-BERT':        'SNEA-BERT\n(Ours)',
    'SNEA-BERT α=0.3':  'SNEA-BERT\nα=0.3\n(Ours)',
    'WL':               'WL\n(Ours)',
    'WL Accurate':      'WL-Acc\n(Ours)',
    'WL Clean':         'WL-Clean\n(Ours)',
    'KEA Enhanced':     'KEA-Enh\n(Ours)',
    'KEA BERT':         'KEA-BERT\n(Ours)',
    'GNN':              'GNN\n(Ours)',
    'ROUGE-1':          'ROUGE-1',
    'ROUGE-2':          'ROUGE-2',
    'ROUGE-L':          'ROUGE-L',
    'BLEU':             'BLEU',
    'BERTScore':        'BERTScore',
    'all-MiniLM-L6-v2': 'MiniLM',
    'sentence-t5-base': 'T5-base',
    'SNEA-BERT α=0.0': 'SNEA-α=0.0\n(Ours)',
    'SNEA-BERT α=0.1': 'SNEA-α=0.1\n(Ours)',
    'SNEA-BERT α=0.2': 'SNEA-α=0.2\n(Ours)',
    'SNEA-BERT α=0.3': 'SNEA-α=0.3\n(Ours)',
    'SNEA-BERT α=0.4': 'SNEA-α=0.4\n(Ours)',
    'SNEA-BERT α=0.5': 'SNEA-α=0.5\n(Ours)',
    'SNEA-BERT α=0.6': 'SNEA-α=0.6\n(Ours)',
    'SNEA-BERT α=0.7': 'SNEA-α=0.7\n(Ours)',
    'SNEA-BERT α=0.8': 'SNEA-α=0.8\n(Ours)',
    'SNEA-BERT α=0.9': 'SNEA-α=0.9\n(Ours)',
    'SNEA-BERT α=1.0': 'SNEA-α=1.0\n(Ours)',
}

# Set to [] when nothing new to append; populate before running main().
NEW_DATASETS: list[dict] = []

# New KG datasets evaluated with SNEA-BERT — results read from overall_results.csv
# after running:  python semantic_kg_evaluation.py --mode multi
NEW_KG_DATASETS = [
    {
        'label':      'Semantic-KG Codex 400 (SNEA-BERT)',
        'output_dir': _HERE / 'output/semantic_kg_codex_400_snea_bert',
        'our_method': 'SNEA-BERT (Our Method)',
        'N': 400, 'N_pos': 200,
    },
    {
        'label':      'Semantic-KG FindKG (SNEA-BERT)',
        'output_dir': _HERE / 'output/semantic_kg_findkg_snea_bert',
        'our_method': 'SNEA-BERT (Our Method)',
        'N': 400, 'N_pos': 200,
    },
    {
        'label':      'Semantic-KG GloBI (SNEA-BERT)',
        'output_dir': _HERE / 'output/semantic_kg_globi_snea_bert',
        'our_method': 'SNEA-BERT (Our Method)',
        'N': 400, 'N_pos': 200,
    },
    {
        'label':      'Semantic-KG Oregano (SNEA-BERT)',
        'output_dir': _HERE / 'output/semantic_kg_oregano_snea_bert',
        'our_method': 'SNEA-BERT (Our Method)',
        'N': 400, 'N_pos': 200,
    },
    {
        'label':      'STS12 (SNEA-BERT)',
        'output_dir': _HERE / 'output/sts12_snea_bert',
        'our_method': 'SNEA-BERT (Our Method)',
        'N': 400, 'N_pos': 200,
    },
    # ── SNEA-BERT α=0.3 results ───────────────────────────────────────────────
    {
        'label':      'MRPC (SNEA-BERT α=0.3)',
        'output_dir': _HERE / 'output/mrpc_snea_bert_0_3',
        'our_method': 'SNEA-BERT α=0.3 (Our Method)',
        'N': 400, 'N_pos': 200,
    },
    {
        'label':      'PAWS-Wiki (SNEA-BERT α=0.3)',
        'output_dir': _HERE / 'output/paws_wiki_snea_bert_0_3',
        'our_method': 'SNEA-BERT α=0.3 (Our Method)',
        'N': 400, 'N_pos': 200,
    },
    {
        'label':      'Semantic-KG (SNEA-BERT α=0.3)',
        'output_dir': _HERE / 'output/semantic_kg_combined_snea_bert_0_3',
        'our_method': 'SNEA-BERT α=0.3 (Our Method)',
        'N': 400, 'N_pos': 200,
    },
    {
        'label':      'Wiki Swap (SNEA-BERT α=0.3)',
        'output_dir': _HERE / 'output/wikipedia_snea_bert_0_3',
        'our_method': 'SNEA-BERT α=0.3 (Our Method)',
        'N': 400, 'N_pos': 200,
    },
    {
        'label':      'Semantic-KG Codex 400 (SNEA-α=0.3)',
        'output_dir': _HERE / 'output/semantic_kg_codex_400_snea_bert_0_3',
        'our_method': 'SNEA-BERT α=0.3 (Our Method)',
        'N': 400, 'N_pos': 200,
    },
    {
        'label':      'Semantic-KG FindKG (SNEA-α=0.3)',
        'output_dir': _HERE / 'output/semantic_kg_findkg_snea_bert_0_3',
        'our_method': 'SNEA-BERT α=0.3 (Our Method)',
        'N': 400, 'N_pos': 200,
    },
    {
        'label':      'Semantic-KG GloBI (SNEA-α=0.3)',
        'output_dir': _HERE / 'output/semantic_kg_globi_snea_bert_0_3',
        'our_method': 'SNEA-BERT α=0.3 (Our Method)',
        'N': 400, 'N_pos': 200,
    },
    {
        'label':      'Semantic-KG Oregano (SNEA-α=0.3)',
        'output_dir': _HERE / 'output/semantic_kg_oregano_snea_bert_0_3',
        'our_method': 'SNEA-BERT α=0.3 (Our Method)',
        'N': 400, 'N_pos': 200,
    },
    {
        'label':      'STS12 (SNEA-α=0.3)',
        'output_dir': _HERE / 'output/sts12_snea_bert_0_3',
        'our_method': 'SNEA-BERT α=0.3 (Our Method)',
        'N': 400, 'N_pos': 200,
    },

    # ── SNEA-BERT alpha sweep ─────────────────────────────────────────────────
    {'label': 'MRPC (SNEA-α=0.0)', 'output_dir': _HERE / 'output/mrpc_snea_alpha_0p0', 'our_method': 'SNEA-BERT α=0.0 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'MRPC (SNEA-α=0.1)', 'output_dir': _HERE / 'output/mrpc_snea_alpha_0p1', 'our_method': 'SNEA-BERT α=0.1 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'MRPC (SNEA-α=0.2)', 'output_dir': _HERE / 'output/mrpc_snea_alpha_0p2', 'our_method': 'SNEA-BERT α=0.2 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'MRPC (SNEA-α=0.3)', 'output_dir': _HERE / 'output/mrpc_snea_alpha_0p3', 'our_method': 'SNEA-BERT α=0.3 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'MRPC (SNEA-α=0.4)', 'output_dir': _HERE / 'output/mrpc_snea_alpha_0p4', 'our_method': 'SNEA-BERT α=0.4 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'MRPC (SNEA-α=0.5)', 'output_dir': _HERE / 'output/mrpc_snea_alpha_0p5', 'our_method': 'SNEA-BERT α=0.5 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'MRPC (SNEA-α=0.6)', 'output_dir': _HERE / 'output/mrpc_snea_alpha_0p6', 'our_method': 'SNEA-BERT α=0.6 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'MRPC (SNEA-α=0.7)', 'output_dir': _HERE / 'output/mrpc_snea_alpha_0p7', 'our_method': 'SNEA-BERT α=0.7 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'MRPC (SNEA-α=0.8)', 'output_dir': _HERE / 'output/mrpc_snea_alpha_0p8', 'our_method': 'SNEA-BERT α=0.8 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'MRPC (SNEA-α=0.9)', 'output_dir': _HERE / 'output/mrpc_snea_alpha_0p9', 'our_method': 'SNEA-BERT α=0.9 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'MRPC (SNEA-α=1.0)', 'output_dir': _HERE / 'output/mrpc_snea_alpha_1p0', 'our_method': 'SNEA-BERT α=1.0 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'PAWS-Wiki (SNEA-α=0.0)', 'output_dir': _HERE / 'output/paws_wiki_snea_alpha_0p0', 'our_method': 'SNEA-BERT α=0.0 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'PAWS-Wiki (SNEA-α=0.1)', 'output_dir': _HERE / 'output/paws_wiki_snea_alpha_0p1', 'our_method': 'SNEA-BERT α=0.1 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'PAWS-Wiki (SNEA-α=0.2)', 'output_dir': _HERE / 'output/paws_wiki_snea_alpha_0p2', 'our_method': 'SNEA-BERT α=0.2 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'PAWS-Wiki (SNEA-α=0.3)', 'output_dir': _HERE / 'output/paws_wiki_snea_alpha_0p3', 'our_method': 'SNEA-BERT α=0.3 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'PAWS-Wiki (SNEA-α=0.4)', 'output_dir': _HERE / 'output/paws_wiki_snea_alpha_0p4', 'our_method': 'SNEA-BERT α=0.4 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'PAWS-Wiki (SNEA-α=0.5)', 'output_dir': _HERE / 'output/paws_wiki_snea_alpha_0p5', 'our_method': 'SNEA-BERT α=0.5 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'PAWS-Wiki (SNEA-α=0.6)', 'output_dir': _HERE / 'output/paws_wiki_snea_alpha_0p6', 'our_method': 'SNEA-BERT α=0.6 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'PAWS-Wiki (SNEA-α=0.7)', 'output_dir': _HERE / 'output/paws_wiki_snea_alpha_0p7', 'our_method': 'SNEA-BERT α=0.7 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'PAWS-Wiki (SNEA-α=0.8)', 'output_dir': _HERE / 'output/paws_wiki_snea_alpha_0p8', 'our_method': 'SNEA-BERT α=0.8 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'PAWS-Wiki (SNEA-α=0.9)', 'output_dir': _HERE / 'output/paws_wiki_snea_alpha_0p9', 'our_method': 'SNEA-BERT α=0.9 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'PAWS-Wiki (SNEA-α=1.0)', 'output_dir': _HERE / 'output/paws_wiki_snea_alpha_1p0', 'our_method': 'SNEA-BERT α=1.0 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'STS12 (SNEA-α=0.0)', 'output_dir': _HERE / 'output/sts12_snea_alpha_0p0', 'our_method': 'SNEA-BERT α=0.0 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'STS12 (SNEA-α=0.1)', 'output_dir': _HERE / 'output/sts12_snea_alpha_0p1', 'our_method': 'SNEA-BERT α=0.1 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'STS12 (SNEA-α=0.2)', 'output_dir': _HERE / 'output/sts12_snea_alpha_0p2', 'our_method': 'SNEA-BERT α=0.2 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'STS12 (SNEA-α=0.3)', 'output_dir': _HERE / 'output/sts12_snea_alpha_0p3', 'our_method': 'SNEA-BERT α=0.3 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'STS12 (SNEA-α=0.4)', 'output_dir': _HERE / 'output/sts12_snea_alpha_0p4', 'our_method': 'SNEA-BERT α=0.4 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'STS12 (SNEA-α=0.5)', 'output_dir': _HERE / 'output/sts12_snea_alpha_0p5', 'our_method': 'SNEA-BERT α=0.5 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'STS12 (SNEA-α=0.6)', 'output_dir': _HERE / 'output/sts12_snea_alpha_0p6', 'our_method': 'SNEA-BERT α=0.6 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'STS12 (SNEA-α=0.7)', 'output_dir': _HERE / 'output/sts12_snea_alpha_0p7', 'our_method': 'SNEA-BERT α=0.7 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'STS12 (SNEA-α=0.8)', 'output_dir': _HERE / 'output/sts12_snea_alpha_0p8', 'our_method': 'SNEA-BERT α=0.8 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'STS12 (SNEA-α=0.9)', 'output_dir': _HERE / 'output/sts12_snea_alpha_0p9', 'our_method': 'SNEA-BERT α=0.9 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'STS12 (SNEA-α=1.0)', 'output_dir': _HERE / 'output/sts12_snea_alpha_1p0', 'our_method': 'SNEA-BERT α=1.0 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Combined (SNEA-α=0.0)', 'output_dir': _HERE / 'output/semantic_kg_combined_snea_alpha_0p0', 'our_method': 'SNEA-BERT α=0.0 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Combined (SNEA-α=0.1)', 'output_dir': _HERE / 'output/semantic_kg_combined_snea_alpha_0p1', 'our_method': 'SNEA-BERT α=0.1 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Combined (SNEA-α=0.2)', 'output_dir': _HERE / 'output/semantic_kg_combined_snea_alpha_0p2', 'our_method': 'SNEA-BERT α=0.2 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Combined (SNEA-α=0.3)', 'output_dir': _HERE / 'output/semantic_kg_combined_snea_alpha_0p3', 'our_method': 'SNEA-BERT α=0.3 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Combined (SNEA-α=0.4)', 'output_dir': _HERE / 'output/semantic_kg_combined_snea_alpha_0p4', 'our_method': 'SNEA-BERT α=0.4 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Combined (SNEA-α=0.5)', 'output_dir': _HERE / 'output/semantic_kg_combined_snea_alpha_0p5', 'our_method': 'SNEA-BERT α=0.5 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Combined (SNEA-α=0.6)', 'output_dir': _HERE / 'output/semantic_kg_combined_snea_alpha_0p6', 'our_method': 'SNEA-BERT α=0.6 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Combined (SNEA-α=0.7)', 'output_dir': _HERE / 'output/semantic_kg_combined_snea_alpha_0p7', 'our_method': 'SNEA-BERT α=0.7 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Combined (SNEA-α=0.8)', 'output_dir': _HERE / 'output/semantic_kg_combined_snea_alpha_0p8', 'our_method': 'SNEA-BERT α=0.8 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Combined (SNEA-α=0.9)', 'output_dir': _HERE / 'output/semantic_kg_combined_snea_alpha_0p9', 'our_method': 'SNEA-BERT α=0.9 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Combined (SNEA-α=1.0)', 'output_dir': _HERE / 'output/semantic_kg_combined_snea_alpha_1p0', 'our_method': 'SNEA-BERT α=1.0 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Codex 400 (SNEA-α=0.0)', 'output_dir': _HERE / 'output/semantic_kg_codex_400_snea_alpha_0p0', 'our_method': 'SNEA-BERT α=0.0 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Codex 400 (SNEA-α=0.1)', 'output_dir': _HERE / 'output/semantic_kg_codex_400_snea_alpha_0p1', 'our_method': 'SNEA-BERT α=0.1 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Codex 400 (SNEA-α=0.2)', 'output_dir': _HERE / 'output/semantic_kg_codex_400_snea_alpha_0p2', 'our_method': 'SNEA-BERT α=0.2 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Codex 400 (SNEA-α=0.3)', 'output_dir': _HERE / 'output/semantic_kg_codex_400_snea_alpha_0p3', 'our_method': 'SNEA-BERT α=0.3 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Codex 400 (SNEA-α=0.4)', 'output_dir': _HERE / 'output/semantic_kg_codex_400_snea_alpha_0p4', 'our_method': 'SNEA-BERT α=0.4 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Codex 400 (SNEA-α=0.5)', 'output_dir': _HERE / 'output/semantic_kg_codex_400_snea_alpha_0p5', 'our_method': 'SNEA-BERT α=0.5 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Codex 400 (SNEA-α=0.6)', 'output_dir': _HERE / 'output/semantic_kg_codex_400_snea_alpha_0p6', 'our_method': 'SNEA-BERT α=0.6 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Codex 400 (SNEA-α=0.7)', 'output_dir': _HERE / 'output/semantic_kg_codex_400_snea_alpha_0p7', 'our_method': 'SNEA-BERT α=0.7 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Codex 400 (SNEA-α=0.8)', 'output_dir': _HERE / 'output/semantic_kg_codex_400_snea_alpha_0p8', 'our_method': 'SNEA-BERT α=0.8 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Codex 400 (SNEA-α=0.9)', 'output_dir': _HERE / 'output/semantic_kg_codex_400_snea_alpha_0p9', 'our_method': 'SNEA-BERT α=0.9 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Codex 400 (SNEA-α=1.0)', 'output_dir': _HERE / 'output/semantic_kg_codex_400_snea_alpha_1p0', 'our_method': 'SNEA-BERT α=1.0 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG FindKG (SNEA-α=0.0)', 'output_dir': _HERE / 'output/semantic_kg_findkg_snea_alpha_0p0', 'our_method': 'SNEA-BERT α=0.0 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG FindKG (SNEA-α=0.1)', 'output_dir': _HERE / 'output/semantic_kg_findkg_snea_alpha_0p1', 'our_method': 'SNEA-BERT α=0.1 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG FindKG (SNEA-α=0.2)', 'output_dir': _HERE / 'output/semantic_kg_findkg_snea_alpha_0p2', 'our_method': 'SNEA-BERT α=0.2 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG FindKG (SNEA-α=0.3)', 'output_dir': _HERE / 'output/semantic_kg_findkg_snea_alpha_0p3', 'our_method': 'SNEA-BERT α=0.3 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG FindKG (SNEA-α=0.4)', 'output_dir': _HERE / 'output/semantic_kg_findkg_snea_alpha_0p4', 'our_method': 'SNEA-BERT α=0.4 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG FindKG (SNEA-α=0.5)', 'output_dir': _HERE / 'output/semantic_kg_findkg_snea_alpha_0p5', 'our_method': 'SNEA-BERT α=0.5 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG FindKG (SNEA-α=0.6)', 'output_dir': _HERE / 'output/semantic_kg_findkg_snea_alpha_0p6', 'our_method': 'SNEA-BERT α=0.6 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG FindKG (SNEA-α=0.7)', 'output_dir': _HERE / 'output/semantic_kg_findkg_snea_alpha_0p7', 'our_method': 'SNEA-BERT α=0.7 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG FindKG (SNEA-α=0.8)', 'output_dir': _HERE / 'output/semantic_kg_findkg_snea_alpha_0p8', 'our_method': 'SNEA-BERT α=0.8 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG FindKG (SNEA-α=0.9)', 'output_dir': _HERE / 'output/semantic_kg_findkg_snea_alpha_0p9', 'our_method': 'SNEA-BERT α=0.9 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG FindKG (SNEA-α=1.0)', 'output_dir': _HERE / 'output/semantic_kg_findkg_snea_alpha_1p0', 'our_method': 'SNEA-BERT α=1.0 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG GloBI (SNEA-α=0.0)', 'output_dir': _HERE / 'output/semantic_kg_globi_snea_alpha_0p0', 'our_method': 'SNEA-BERT α=0.0 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG GloBI (SNEA-α=0.1)', 'output_dir': _HERE / 'output/semantic_kg_globi_snea_alpha_0p1', 'our_method': 'SNEA-BERT α=0.1 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG GloBI (SNEA-α=0.2)', 'output_dir': _HERE / 'output/semantic_kg_globi_snea_alpha_0p2', 'our_method': 'SNEA-BERT α=0.2 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG GloBI (SNEA-α=0.3)', 'output_dir': _HERE / 'output/semantic_kg_globi_snea_alpha_0p3', 'our_method': 'SNEA-BERT α=0.3 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG GloBI (SNEA-α=0.4)', 'output_dir': _HERE / 'output/semantic_kg_globi_snea_alpha_0p4', 'our_method': 'SNEA-BERT α=0.4 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG GloBI (SNEA-α=0.5)', 'output_dir': _HERE / 'output/semantic_kg_globi_snea_alpha_0p5', 'our_method': 'SNEA-BERT α=0.5 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG GloBI (SNEA-α=0.6)', 'output_dir': _HERE / 'output/semantic_kg_globi_snea_alpha_0p6', 'our_method': 'SNEA-BERT α=0.6 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG GloBI (SNEA-α=0.7)', 'output_dir': _HERE / 'output/semantic_kg_globi_snea_alpha_0p7', 'our_method': 'SNEA-BERT α=0.7 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG GloBI (SNEA-α=0.8)', 'output_dir': _HERE / 'output/semantic_kg_globi_snea_alpha_0p8', 'our_method': 'SNEA-BERT α=0.8 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG GloBI (SNEA-α=0.9)', 'output_dir': _HERE / 'output/semantic_kg_globi_snea_alpha_0p9', 'our_method': 'SNEA-BERT α=0.9 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG GloBI (SNEA-α=1.0)', 'output_dir': _HERE / 'output/semantic_kg_globi_snea_alpha_1p0', 'our_method': 'SNEA-BERT α=1.0 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Oregano (SNEA-α=0.0)', 'output_dir': _HERE / 'output/semantic_kg_oregano_snea_alpha_0p0', 'our_method': 'SNEA-BERT α=0.0 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Oregano (SNEA-α=0.1)', 'output_dir': _HERE / 'output/semantic_kg_oregano_snea_alpha_0p1', 'our_method': 'SNEA-BERT α=0.1 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Oregano (SNEA-α=0.2)', 'output_dir': _HERE / 'output/semantic_kg_oregano_snea_alpha_0p2', 'our_method': 'SNEA-BERT α=0.2 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Oregano (SNEA-α=0.3)', 'output_dir': _HERE / 'output/semantic_kg_oregano_snea_alpha_0p3', 'our_method': 'SNEA-BERT α=0.3 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Oregano (SNEA-α=0.4)', 'output_dir': _HERE / 'output/semantic_kg_oregano_snea_alpha_0p4', 'our_method': 'SNEA-BERT α=0.4 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Oregano (SNEA-α=0.5)', 'output_dir': _HERE / 'output/semantic_kg_oregano_snea_alpha_0p5', 'our_method': 'SNEA-BERT α=0.5 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Oregano (SNEA-α=0.6)', 'output_dir': _HERE / 'output/semantic_kg_oregano_snea_alpha_0p6', 'our_method': 'SNEA-BERT α=0.6 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Oregano (SNEA-α=0.7)', 'output_dir': _HERE / 'output/semantic_kg_oregano_snea_alpha_0p7', 'our_method': 'SNEA-BERT α=0.7 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Oregano (SNEA-α=0.8)', 'output_dir': _HERE / 'output/semantic_kg_oregano_snea_alpha_0p8', 'our_method': 'SNEA-BERT α=0.8 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Oregano (SNEA-α=0.9)', 'output_dir': _HERE / 'output/semantic_kg_oregano_snea_alpha_0p9', 'our_method': 'SNEA-BERT α=0.9 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Semantic-KG Oregano (SNEA-α=1.0)', 'output_dir': _HERE / 'output/semantic_kg_oregano_snea_alpha_1p0', 'our_method': 'SNEA-BERT α=1.0 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Wiki Swap (SNEA-α=0.0)', 'output_dir': _HERE / 'output/wikipedia_snea_alpha_0p0', 'our_method': 'SNEA-BERT α=0.0 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Wiki Swap (SNEA-α=0.1)', 'output_dir': _HERE / 'output/wikipedia_snea_alpha_0p1', 'our_method': 'SNEA-BERT α=0.1 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Wiki Swap (SNEA-α=0.2)', 'output_dir': _HERE / 'output/wikipedia_snea_alpha_0p2', 'our_method': 'SNEA-BERT α=0.2 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Wiki Swap (SNEA-α=0.3)', 'output_dir': _HERE / 'output/wikipedia_snea_alpha_0p3', 'our_method': 'SNEA-BERT α=0.3 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Wiki Swap (SNEA-α=0.4)', 'output_dir': _HERE / 'output/wikipedia_snea_alpha_0p4', 'our_method': 'SNEA-BERT α=0.4 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Wiki Swap (SNEA-α=0.5)', 'output_dir': _HERE / 'output/wikipedia_snea_alpha_0p5', 'our_method': 'SNEA-BERT α=0.5 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Wiki Swap (SNEA-α=0.6)', 'output_dir': _HERE / 'output/wikipedia_snea_alpha_0p6', 'our_method': 'SNEA-BERT α=0.6 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Wiki Swap (SNEA-α=0.7)', 'output_dir': _HERE / 'output/wikipedia_snea_alpha_0p7', 'our_method': 'SNEA-BERT α=0.7 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Wiki Swap (SNEA-α=0.8)', 'output_dir': _HERE / 'output/wikipedia_snea_alpha_0p8', 'our_method': 'SNEA-BERT α=0.8 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Wiki Swap (SNEA-α=0.9)', 'output_dir': _HERE / 'output/wikipedia_snea_alpha_0p9', 'our_method': 'SNEA-BERT α=0.9 (Our Method)', 'N': 400, 'N_pos': 200},
    {'label': 'Wiki Swap (SNEA-α=1.0)', 'output_dir': _HERE / 'output/wikipedia_snea_alpha_1p0', 'our_method': 'SNEA-BERT α=1.0 (Our Method)', 'N': 400, 'N_pos': 200},
]

# ---------------------------------------------------------------------------
# Helper: confusion matrix from precision / recall / N / N_pos
# ---------------------------------------------------------------------------

def compute_confusion(precision, recall, N, N_pos):
    """Return (TP, FP, TN, FN) as integers."""
    TP    = round(recall * N_pos)
    FN    = N_pos - TP
    TP_FP = round(TP / precision) if precision > 0 else 0
    FP    = TP_FP - TP
    TN    = N - TP - FP - FN
    return int(TP), int(FP), int(TN), int(FN)


# ---------------------------------------------------------------------------
# Append new dataset rows to the confusion CSV
# ---------------------------------------------------------------------------

def build_new_rows(existing_df: pd.DataFrame, datasets_override=None) -> list[dict]:
    rows = []
    for ds in (datasets_override or []):
        label     = ds['label']
        our_label = ds['our_method']
        N, N_pos  = ds['N'], ds['N_pos']
        res_dir   = Path(ds['results_dir'])
        base_lbl  = ds['baseline_label']

        results = pd.read_csv(res_dir / 'overall_results.csv')
        our_row = results[results['method'] == 'AA-KEA (Our Method)'].iloc[0]
        prec, rec, f1, auc = (
            our_row['precision'], our_row['recall'],
            our_row['f1'],        our_row['roc_auc'],
        )
        TP, FP, TN, FN = compute_confusion(prec, rec, N, N_pos)
        rows.append({
            'Dataset': label, 'Method': our_label, 'N': N,
            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
            'Precision': round(prec, 3), 'Recall': round(rec, 3),
            'F1': round(f1, 3), 'AUC': round(auc, 3),
        })

        base_rows = existing_df[
            (existing_df['Dataset'] == base_lbl) &
            (existing_df['Method'].isin(BASELINE_METHODS))
        ]
        for _, br in base_rows.iterrows():
            rows.append({
                'Dataset': label, 'Method': br['Method'], 'N': N,
                'TP': int(br['TP']), 'FP': int(br['FP']),
                'TN': int(br['TN']), 'FN': int(br['FN']),
                'Precision': float(br['Precision']), 'Recall': float(br['Recall']),
                'F1': float(br['F1']), 'AUC': float(br['AUC']),
            })
    return rows


# ---------------------------------------------------------------------------
# Build clean flat DataFrame: base_dataset × all methods (variants + baselines)
# ---------------------------------------------------------------------------

def build_main_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten confusion CSV into:
        Dataset (base) | Method (variant or baseline) | F1 | AUC | Precision | Recall

    Each our-method variant becomes its own row using the variant name as Method.
    Baseline values are deduplicated per base dataset.
    """
    rows: list[dict] = []
    seen_baselines: set = set()

    for ds_label, (base_ds, variant) in DATASET_VARIANT_MAP.items():
        sub = df[df['Dataset'] == ds_label]
        if sub.empty:
            continue

        # Our method variant
        our = sub[sub['Method'].isin(OUR_METHOD_LABELS)]
        if not our.empty:
            r = our.iloc[0]
            rows.append({
                'Dataset': base_ds, 'Method': variant,
                'MethodType': 'Our Variant',
                'F1': r['F1'], 'AUC': r['AUC'],
                'Precision': r['Precision'], 'Recall': r['Recall'],
            })

        # Baselines — one entry per (base_ds, method)
        for _, br in sub[sub['Method'].isin(BASELINE_METHODS)].iterrows():
            key = (base_ds, br['Method'])
            if key not in seen_baselines:
                seen_baselines.add(key)
                rows.append({
                    'Dataset': base_ds, 'Method': br['Method'],
                    'MethodType': 'Baseline',
                    'F1': br['F1'], 'AUC': br['AUC'],
                    'Precision': br['Precision'], 'Recall': br['Recall'],
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helper: draw a dual F1 / AUC heatmap pair
# ---------------------------------------------------------------------------

def _draw_dual_heatmap(pivot_f1: pd.DataFrame, pivot_auc: pd.DataFrame,
                        col_labels: list[str], suptitle: str,
                        out_path: Path, n_ours: int = 0):
    ncols = len(pivot_f1.columns)
    nrows = len(pivot_f1.index)
    fig, axes = plt.subplots(1, 2, figsize=(max(ncols * 2.0 + 4, 22),
                                             max(nrows * 1.3 + 3, 7)))

    for ax, pivot, metric in zip(axes, [pivot_f1, pivot_auc], ['F1 Score', 'ROC-AUC']):
        mask = pivot.isna()
        sns.heatmap(
            pivot, ax=ax,
            annot=True, fmt='.3f',
            annot_kws={'size': 11, 'weight': 'bold'},
            cmap='RdBu',
            vmin=0.5, vmax=1.0,
            mask=mask,
            linewidths=0.4, linecolor='#cccccc',
            cbar_kws={'label': metric, 'shrink': 0.75},
            xticklabels=col_labels,
        )
        ax.set_title(metric, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Dataset', fontsize=12)
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=10, rotation=0)

        # Navy border around "our variants" columns (if any)
        if n_ours > 0:
            ax.add_patch(plt.Rectangle((0, 0), n_ours, nrows,
                                        fill=False, edgecolor='navy',
                                        linewidth=2.5, clip_on=False))

    ours_p = mpatches.Patch(color='#4878d0', label='Our Method Variants')
    base_p = mpatches.Patch(color='#aec7e8', label='Baselines')
    fig.legend(handles=[ours_p, base_p], loc='lower center',
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.03))

    plt.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=350, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Saved: {out_path}')


# ---------------------------------------------------------------------------
# Plot 1 — Main heatmap: all variants + baselines × all datasets
# ---------------------------------------------------------------------------

def plot_heatmap(df: pd.DataFrame):
    main_df   = build_main_df(df)
    col_order = [m for m in MAIN_OUR_VARIANTS + BASELINE_METHODS
                 if m in main_df['Method'].unique()]
    datasets  = [d for d in HEATMAP_DATASETS if d in main_df['Dataset'].unique()]

    pivot_f1  = (main_df.pivot_table(index='Dataset', columns='Method',
                                      values='F1',  aggfunc='first')
                        .reindex(index=datasets, columns=col_order))
    pivot_auc = (main_df.pivot_table(index='Dataset', columns='Method',
                                      values='AUC', aggfunc='first')
                        .reindex(index=datasets, columns=col_order))

    n_ours = sum(1 for m in col_order if m in MAIN_OUR_VARIANTS)

    col_labels = [METHOD_SHORT.get(m, m) for m in col_order]
    _draw_dual_heatmap(
        pivot_f1, pivot_auc, col_labels,
        suptitle='All Methods × All Datasets  (Our Variants | Baselines)',
        out_path=CROSS_DIR / 'all_methods_heatmap.png',
        n_ours=n_ours,
    )


# ---------------------------------------------------------------------------
# Plot 2 — Variants-only heatmap: our variants × all datasets
# ---------------------------------------------------------------------------

def plot_variants_heatmap(df: pd.DataFrame):
    main_df   = build_main_df(df)
    var_df    = main_df[main_df['Method'].isin(VARIANT_ORDER)]
    col_order = [m for m in VARIANT_ORDER if m in var_df['Method'].unique()]
    datasets  = [d for d in BASE_DATASETS if d in var_df['Dataset'].unique()]

    pivot_f1  = (var_df.pivot_table(index='Dataset', columns='Method',
                                     values='F1',  aggfunc='first')
                       .reindex(index=datasets, columns=col_order))
    pivot_auc = (var_df.pivot_table(index='Dataset', columns='Method',
                                     values='AUC', aggfunc='first')
                       .reindex(index=datasets, columns=col_order))

    col_labels = [METHOD_SHORT.get(m, m) for m in col_order]
    _draw_dual_heatmap(
        pivot_f1, pivot_auc, col_labels,
        suptitle='Our Methodology Variants — Internal Comparison',
        out_path=CROSS_DIR / 'variants_comparison_heatmap.png',
        n_ours=len(col_order),   # all columns are ours
    )


# ---------------------------------------------------------------------------
# Plot 3 — Rank heatmap: per-dataset rank for each method (F1 and AUC)
# Methods not tested on a dataset are shown as grey (NaN).
# ---------------------------------------------------------------------------

def plot_ranks(df: pd.DataFrame):
    main_df   = build_main_df(df)
    datasets  = [d for d in HEATMAP_DATASETS if d in main_df['Dataset'].unique()]

    col_order = [m for m in MAIN_OUR_VARIANTS + BASELINE_METHODS
                 if m in main_df['Method'].unique()]
    n_ours    = sum(1 for m in col_order if m in MAIN_OUR_VARIANTS)

    # Build per-dataset rank pivot for F1 and AUC
    def make_rank_pivot(metric: str) -> pd.DataFrame:
        records = []
        for dataset in datasets:
            sub = (main_df[main_df['Dataset'] == dataset]
                   .pipe(lambda x: x[x['Method'].isin(col_order)])
                   .dropna(subset=[metric])
                   .copy())
            sub['rank'] = sub[metric].rank(ascending=False, method='min').astype(int)
            for _, row in sub.iterrows():
                records.append({'Dataset': row['Dataset'],
                                'Method':  row['Method'],
                                'rank':    row['rank']})
        pivot = (pd.DataFrame(records)
                   .pivot_table(index='Dataset', columns='Method',
                                values='rank', aggfunc='first')
                   .reindex(index=datasets, columns=col_order))
        return pivot

    pivot_f1  = make_rank_pivot('F1')
    pivot_auc = make_rank_pivot('AUC')

    col_labels = [METHOD_SHORT.get(m, m) for m in col_order]
    ncols = len(col_order)
    nrows = len(datasets)

    fig, axes = plt.subplots(1, 2, figsize=(max(ncols * 2.0 + 4, 22),
                                             max(nrows * 1.3 + 3, 7)))

    max_rank = int(max(pivot_f1.max().max(), pivot_auc.max().max()))

    for ax, pivot, metric in zip(axes, [pivot_f1, pivot_auc], ['F1', 'AUC']):
        mask = pivot.isna()
        # Rank 1 = best → invert colour scale (low rank number = dark blue = best)
        sns.heatmap(
            pivot, ax=ax,
            annot=True, fmt='.0f',
            annot_kws={'size': 11, 'weight': 'bold'},
            cmap='RdBu',              # rank 1 = blue (best), high rank = red (worst)
            vmin=1, vmax=max_rank,
            mask=mask,
            linewidths=0.4, linecolor='#cccccc',
            cbar_kws={'label': f'Rank (1 = best)  [{metric}]', 'shrink': 0.75},
            xticklabels=col_labels,
        )
        ax.set_title(f'{metric} Rank per Dataset  (1 = best)',
                     fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Dataset', fontsize=12)
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=10, rotation=0)

        # Grey fill for methods not tested on that dataset
        for i in range(nrows):
            for j in range(ncols):
                if mask.iloc[i, j]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1,
                                               fill=True, color='#d9d9d9',
                                               lw=0, zorder=0))

        # Navy border around "our variants" columns
        if n_ours > 0:
            ax.add_patch(plt.Rectangle((0, 0), n_ours, nrows,
                                        fill=False, edgecolor='navy',
                                        linewidth=2.5, clip_on=False))

    ours_p = mpatches.Patch(color='#4878d0', label='Our Method Variants')
    base_p = mpatches.Patch(color='#aec7e8', label='Baselines')
    grey_p = mpatches.Patch(color='#d9d9d9', label='Not tested on this dataset')
    fig.legend(handles=[ours_p, base_p, grey_p], loc='lower center',
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.03))

    plt.suptitle('Per-Dataset Method Rankings  (F1 | AUC)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = CROSS_DIR / 'all_methods_ranks.png'
    plt.savefig(out, dpi=350, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Saved: {out}')


# ---------------------------------------------------------------------------
# CSV summary export
# ---------------------------------------------------------------------------

def save_summary_csv(df: pd.DataFrame):
    main_df   = build_main_df(df)
    col_order = [m for m in VARIANT_ORDER + BASELINE_METHODS
                 if m in main_df['Method'].unique()]
    datasets  = [d for d in BASE_DATASETS if d in main_df['Dataset'].unique()]

    main_df['Method'] = pd.Categorical(main_df['Method'],
                                        categories=col_order, ordered=True)
    ordered = (main_df[main_df['Dataset'].isin(datasets)]
               .sort_values(['Dataset', 'Method'])
               .reset_index(drop=True))
    ordered['Dataset'] = pd.Categorical(ordered['Dataset'],
                                         categories=datasets, ordered=True)
    ordered = ordered.sort_values(['Dataset', 'Method']).reset_index(drop=True)

    out = CROSS_DIR / 'cross_dataset_summary.csv'
    ordered.to_csv(out, index=False)
    print(f'  ✓ Saved: {out}')


# ---------------------------------------------------------------------------
# Build rows from overall_results.csv produced by semantic_kg_evaluation.py
# ---------------------------------------------------------------------------

def rows_from_output_dirs(existing_df: pd.DataFrame) -> list[dict]:
    """
    For each entry in NEW_KG_DATASETS, read its overall_results.csv
    (produced by semantic_kg_evaluation.py --mode multi) and build confusion
    CSV rows for every method (our method + all baselines).
    Skips datasets already present in the confusion CSV.
    """
    existing_labels = set(existing_df['Dataset'].unique())
    rows = []

    for spec in NEW_KG_DATASETS:
        label = spec['label']
        if label in existing_labels:
            print(f'  Skipping {label} (already in CSV)')
            continue

        results_csv = spec['output_dir'] / 'overall_results.csv'
        if not results_csv.exists():
            print(f'  ⚠ {label}: overall_results.csv not found at {results_csv}')
            print(f'     → Run: python semantic_kg_evaluation.py --mode multi  first.')
            continue

        results = pd.read_csv(results_csv)
        N, N_pos = spec['N'], spec['N_pos']

        for _, r in results.iterrows():
            method = r['method']
            # Rename generic 'AA-KEA (Our Method)' to the correct variant label
            if method == 'AA-KEA (Our Method)':
                method = spec['our_method']

            prec = float(r['precision'])
            rec  = float(r['recall'])
            f1   = float(r['f1'])
            auc  = float(r['roc_auc'])
            TP, FP, TN, FN = compute_confusion(prec, rec, N, N_pos)

            rows.append({
                'Dataset': label, 'Method': method, 'N': N,
                'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
                'Precision': round(prec, 3), 'Recall': round(rec, 3),
                'F1': round(f1, 3), 'AUC': round(auc, 3),
            })

        print(f'  ✓ {label}: {len(results)} method rows loaded')

    return rows


# ---------------------------------------------------------------------------
# Plot 4 — Per-dataset performance summary table
# Shows for each dataset: our best variant vs best baseline, with good/bad verdict
# ---------------------------------------------------------------------------

def plot_performance_table(df: pd.DataFrame):
    main_df  = build_main_df(df)
    datasets = [d for d in BASE_DATASETS if d in main_df['Dataset'].unique()]

    rows = []
    for ds in datasets:
        sub      = main_df[main_df['Dataset'] == ds]
        # Only consider SNEA-BERT alpha variants as "our variants"
        our_sub  = sub[sub['Method'].isin(ALPHA_VARIANTS)]

        if our_sub.empty:
            continue

        best_our = our_sub.loc[our_sub['F1'].idxmax()]

        # Best competitor = best baseline method only (ROUGE, BLEU, BERTScore, ST)
        others = sub[sub['Method'].isin(BASELINE_METHODS)]
        if others.empty:
            continue
        best_other = others.loc[others['F1'].idxmax()]

        delta_f1  = best_our['F1']  - best_other['F1']
        delta_auc = best_our['AUC'] - best_other['AUC']

        if delta_f1 >= 0.03:
            verdict = '✔ Strong'
        elif delta_f1 >= 0:
            verdict = '~ Comparable'
        elif delta_f1 >= -0.05:
            verdict = '▼ Slight gap'
        else:
            verdict = '✘ Underperforms'

        # Tag competitor as Ours or Baseline
        other_type = best_other['MethodType']
        other_label = (f"{best_other['Method']} (Ours)"
                       if other_type == 'Our Variant' else best_other['Method'])

        rows.append({
            'Dataset':            ds,
            'Our Best Variant':   best_our['Method'],
            'Our F1':             round(best_our['F1'],  3),
            'Our AUC':            round(best_our['AUC'], 3),
            'Best Other Method':  other_label,
            'Other F1':           round(best_other['F1'],  3),
            'Other AUC':          round(best_other['AUC'], 3),
            'ΔF1':                round(delta_f1,  3),
            'ΔAUC':               round(delta_auc, 3),
            'Verdict':            verdict,
        })

    tbl_df = pd.DataFrame(rows)

    # ── Figure ────────────────────────────────────────────────────────────
    n = len(tbl_df)
    fig, ax = plt.subplots(figsize=(20, n * 0.65 + 2.5))
    ax.axis('off')

    col_keys   = list(tbl_df.columns)
    col_widths = [0.13, 0.12, 0.07, 0.07, 0.16, 0.08, 0.07, 0.06, 0.06, 0.11]

    table = ax.table(
        cellText=tbl_df.values.tolist(),
        colLabels=col_keys,
        cellLoc='center', loc='center',
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 2.0)

    # Header style
    for j in range(len(col_keys)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Row style by verdict
    VERDICT_COLORS = {
        '✔ Strong':       '#d4edda',
        '~ Comparable':   '#fff3cd',
        '▼ Slight gap':   '#fde8d8',
        '✘ Underperforms':'#f8d7da',
    }
    # Verdicts that represent our method winning / holding its own
    HIGHLIGHT_VERDICTS = {'✔ Strong', '~ Comparable'}

    for i, row in enumerate(tbl_df.itertuples(index=False)):
        bg      = VERDICT_COLORS.get(row.Verdict, '#f9f9f9')
        highlight = row.Verdict in HIGHLIGHT_VERDICTS
        for j in range(len(col_keys)):
            cell = table[i + 1, j]
            cell.set_facecolor(bg)
            if highlight:
                # Thick dark-green border on every cell in highlighted rows
                cell.set_edgecolor('#1a6e2e')
                cell.set_linewidth(2.0)
            else:
                cell.set_edgecolor('#cccccc')
                cell.set_linewidth(0.5)
        # Bold + underline the verdict cell for highlighted rows
        verdict_cell = table[i + 1, col_keys.index('Verdict')]
        if highlight:
            verdict_cell.set_text_props(fontweight='bold',
                                        color='#1a6e2e',
                                        fontstyle='italic')
        # Bold delta columns
        delta_f1_val  = tbl_df.iloc[i]['ΔF1']
        delta_auc_val = tbl_df.iloc[i]['ΔAUC']
        table[i + 1, col_keys.index('ΔF1')].set_text_props(
            color='#1a6e2e' if delta_f1_val >= 0 else '#a94442', fontweight='bold')
        table[i + 1, col_keys.index('ΔAUC')].set_text_props(
            color='#1a6e2e' if delta_auc_val >= 0 else '#a94442', fontweight='bold')

    # Legend patches
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=v) for v, c in VERDICT_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.02), frameon=True)

    plt.suptitle('Per-Dataset Performance Summary — Our Best Variant vs Best Other Method (All Methods)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = CROSS_DIR / 'performance_summary_table.png'
    plt.savefig(out, dpi=350, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ Saved: {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('=' * 70)
    print('Updating cross-dataset evaluation')
    print('=' * 70)

    existing = pd.read_csv(CONFUSION_CSV)
    existing_labels = existing['Dataset'].unique().tolist()
    print(f'\nExisting datasets ({len(existing_labels)}): {existing_labels}')

    if NEW_DATASETS:
        to_add  = [ds for ds in NEW_DATASETS if ds['label'] not in existing_labels]
        already = [ds['label'] for ds in NEW_DATASETS if ds['label'] in existing_labels]
        if already:
            print(f'Already present (skipping): {already}')
        if to_add:
            print(f'Adding {len(to_add)} new dataset(s): {[d["label"] for d in to_add]}')
            new_rows = build_new_rows(existing, datasets_override=to_add)
            existing = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            existing.to_csv(CONFUSION_CSV, index=False)
            print(f'✓ Updated: {CONFUSION_CSV}')
        else:
            print('All new datasets already present — skipping CSV update.')
    else:
        print('NEW_DATASETS is empty — regenerating plots from existing CSV only.')

    # Add new KG datasets from their overall_results.csv
    print('\nChecking new KG datasets (SNEA-BERT)...')
    kg_rows = rows_from_output_dirs(existing)
    if kg_rows:
        existing = pd.concat([existing, pd.DataFrame(kg_rows)], ignore_index=True)
        existing.to_csv(CONFUSION_CSV, index=False)
        print(f'✓ Appended {len(kg_rows)} rows from new KG datasets')

    print(f'\nTotal datasets in CSV : {existing["Dataset"].nunique()}')
    print(f'Total rows            : {len(existing)}')

    print('\nRegenerating plots & summary CSV...')
    plot_heatmap(existing)
    plot_variants_heatmap(existing)
    plot_ranks(existing)
    plot_performance_table(existing)
    save_summary_csv(existing)

    print('\n✓ Done. Outputs in:', CROSS_DIR)


if __name__ == '__main__':
    main()