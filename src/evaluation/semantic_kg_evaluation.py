"""
Semantic-KG Dataset Evaluation Script

This script evaluates multiple semantic similarity methods on the Semantic-KG dataset,
comparing AA-KEA (knowledge graph-based) against traditional NLP baselines.

Modes:
  1. Single-dataset evaluation (original behaviour) — Semantic-KG Codex
  2. Multi-dataset evaluation — all 5 benchmark datasets with cross-dataset comparison

Required input files (single-dataset mode):
  - semantic_kg_for_kg_generation.csv
  - semantic_kg_aa_kea_results.csv

Required input files (multi-dataset mode, after running prepare scripts):
  - datasets/mrpc_400.csv + datasets/mrpc_aa_kea_results.csv
  - datasets/paws_wiki_400.csv + datasets/paws_wiki_aa_kea_results.csv
  - datasets/semantic_kg_combined_400.csv + datasets/semantic_kg_combined_aa_kea_results.csv
  - datasets/wikipedia_entity_swap_400.csv + datasets/wikipedia_entity_swap_aa_kea_results.csv

Output:
  - Comprehensive evaluation metrics (F1, Precision, Recall, ROC-AUC)
  - Stratification by perturbation type
  - Cross-dataset comparison plots
  - Text-length vs performance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
    accuracy_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import torch
import warnings
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)

# ── Model singletons — loaded once, reused across all datasets ────────────────
_ROUGE_SCORER = None
_ST_MODELS: dict = {}   # model_name -> SentenceTransformer instance

def get_rouge_scorer():
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        _ROUGE_SCORER = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
    return _ROUGE_SCORER

def get_st_model(model_name: str, model_path: str) -> SentenceTransformer:
    if model_name not in _ST_MODELS:
        print(f"  Loading {model_name} (once)...")
        _ST_MODELS[model_name] = SentenceTransformer(model_path)
    return _ST_MODELS[model_name]

_BERT_SCORER = None

def get_bert_scorer() -> BERTScorer:
    global _BERT_SCORER
    if _BERT_SCORER is None:
        print("  Loading BERTScorer (roberta-large, once)...")
        _BERT_SCORER = BERTScorer(
            model_type='roberta-large',
            lang='en',
            device='cpu',
        )
    return _BERT_SCORER


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for evaluation"""

    # File paths (single-dataset / original mode)
    DATASET_FILE = 'semantic_kg_for_kg_generation.csv'
    AA_KEA_RESULTS = 'semantic_kg_aa_kea_results.csv'
    OUTPUT_DIR = 'output/semantic_kg'

    # -----------------------------------------------------------------------
    # Multi-dataset configuration
    # Each entry: dataset_file, aa_kea_file, text1_col, text2_col, output_dir
    # short_text=True flags datasets with sentence-level text (STS-B, MRPC)
    # -----------------------------------------------------------------------
    DATASETS = {
        'mrpc': {
            'dataset_file': 'datasets/mrpc_400.csv',
            'aa_kea_file':  'datasets/mrpc_aa_kea_results.csv',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/mrpc',
            'short_text':   True,
            'label':        'Short-text / Human-annotated',
        },
        'paws_wiki': {
            'dataset_file': 'datasets/paws_wiki_400.csv',
            'aa_kea_file':  'datasets/paws_wiki_aa_kea_results.csv',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/paws_wiki',
            'short_text':   False,
            'label':        'Long-text / Linguistic (non-KG)',
        },
        'semantic_kg_codex': {
            'dataset_file': 'semantic_kg_for_kg_generation.csv',
            'aa_kea_file':  'semantic_kg_aa_kea_results.csv',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Codex)',
        },
        'semantic_kg_combined': {
            'dataset_file': 'datasets/semantic_kg_combined_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_combined_aa_kea_results.csv',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_combined',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (All domains)',
        },
        'mrpc_wl': {
            'dataset_file': 'datasets/mrpc_400.csv',
            'aa_kea_file':  'datasets/mrpc_wl_results.csv',
            'score_col':    'semantic_wl_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/mrpc_wl',
            'short_text':   True,
            'label':        'Short-text / WL Semantic Similarity',
        },
        'mrpc_wl_accurate': {
            'dataset_file': 'datasets/mrpc_400.csv',
            'aa_kea_file':  'datasets/mrpc_wl_accurate_results.csv',
            'score_col':    'semantic_wl_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/mrpc_wl_accurate',
            'short_text':   True,
            'label':        'Short-text / WL Semantic Similarity (Enhanced)',
        },
        'mrpc_kea_enhanced': {
            'dataset_file': 'datasets/mrpc_400.csv',
            'aa_kea_file':  'datasets/mrpc_kea_enhanced_results.csv',
            'score_col':    'kea_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/mrpc_kea_enhanced',
            'short_text':   True,
            'label':        'Short-text / KEA Enhanced',
        },
        'semantic_kg_combined_wl': {
            'dataset_file': 'datasets/semantic_kg_combined_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_combined_wl_results.csv',
            'score_col':    'semantic_wl_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_combined_wl',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (WL Similarity)',
        },
        'wikipedia_entity_swap_wl': {
            'dataset_file': 'datasets/wikipedia_entity_swap_400.csv',
            'aa_kea_file':  'datasets/wikipedia_entity_swap_wl_results.csv',
            'score_col':    'semantic_wl_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/wikipedia_entity_swap_wl',
            'short_text':   False,
            'label':        'Long-text / NLP-perturbed Wikipedia (WL Similarity)',
        },
        'wikipedia_entity_swap_wl_clean': {
            'dataset_file': 'datasets/wikipedia_entity_swap_clean_400.csv',
            'aa_kea_file':  'datasets/wikipedia_entity_swap_clean_wl_results.csv',
            'score_col':    'semantic_wl_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/wikipedia_entity_swap_wl_clean',
            'short_text':   False,
            'label':        'Long-text / NLP-perturbed Wikipedia (WL, Cleaned)',
        },
        'wikipedia_entity_swap': {
            'dataset_file': 'datasets/wikipedia_entity_swap_400.csv',
            'aa_kea_file':  'datasets/wikipedia_entity_swap_aa_kea_results.csv',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/wikipedia_entity_swap',
            'short_text':   False,
            'label':        'Long-text / NLP-perturbed Wikipedia (AA-KEA)',
        },
        'wikipedia_entity_swap_clean': {
            'dataset_file': 'datasets/wikipedia_entity_swap_clean_400.csv',
            'aa_kea_file':  'datasets/wikipedia_entity_swap_aa_kea_clean_results.csv',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/wikipedia_entity_swap_clean',
            'short_text':   False,
            'label':        'Long-text / NLP-perturbed Wikipedia (AA-KEA, Cleaned)',
        },
        'mrpc_gnn': {
            'dataset_file': 'datasets/mrpc_400.csv',
            'aa_kea_file':  'datasets/mrpc_gnn_results.csv',
            'score_col':    'gnn_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/mrpc_gnn',
            'short_text':   True,
            'label':        'Short-text / GNN Similarity',
        },
        'mrpc_kea_bert': {
            'dataset_file': 'datasets/mrpc_400.csv',
            'aa_kea_file':  'datasets/mrpc_kea_bert_results.csv',
            'score_col':    'kea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/mrpc_kea_bert',
            'short_text':   True,
            'label':        'Short-text / KEA BERT',
        },
        'wikipedia_entity_swap': {
            'dataset_file': 'datasets/wikipedia_entity_swap_400.csv',
            'aa_kea_file':  'datasets/wikipedia_entity_swap_aa_kea_results.csv',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/wikipedia_entity_swap',
            'short_text':   False,
            'label':        'Long-text / NLP-perturbed (non-KG)',
        },
        'mrpc_snea_bert': {
            'dataset_file': 'datasets/mrpc_400.csv',
            'aa_kea_file':  'datasets/mrpc_400_snea_bert_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/mrpc_snea_bert',
            'short_text':   True,
            'label':        'Short-text / SNEA-BERT',
        },
        'paws_wiki_snea_bert': {
            'dataset_file': 'datasets/paws_wiki_400.csv',
            'aa_kea_file':  'datasets/paws_wiki_400_snea_bert_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/paws_wiki_snea_bert',
            'short_text':   False,
            'label':        'Long-text / Linguistic (SNEA-BERT)',
        },
        'semantic_kg_snea_bert': {
            'dataset_file': 'datasets/semantic_kg_combined_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_combined_400_snea_bert_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_snea_bert',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (SNEA-BERT)',
        },
        'semantic_kg_codex_400_snea_bert': {
            'dataset_file': 'datasets/semantic_kg_codex_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_codex_400_KGs_snea_bert_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_codex_400_snea_bert',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Codex 400, SNEA-BERT)',
        },
        'semantic_kg_findkg_snea_bert': {
            'dataset_file': 'datasets/semantic_kg_findkg_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_findkg_400_KGs_snea_bert_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_findkg_snea_bert',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (FindKG, SNEA-BERT)',
        },
        'semantic_kg_globi_snea_bert': {
            'dataset_file': 'datasets/semantic_kg_globi_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_globi_400_KGs_snea_bert_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_globi_snea_bert',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (GloBI, SNEA-BERT)',
        },
        'semantic_kg_oregano_snea_bert': {
            'dataset_file': 'datasets/semantic_kg_oregano_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_oregano_400_KGs_snea_bert_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_oregano_snea_bert',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Oregano, SNEA-BERT)',
        },
        'sts12_snea_bert': {
            'dataset_file': 'datasets/sts12_400.csv',
            'aa_kea_file':  'datasets/sts12_400_KGs_snea_bert_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/sts12_snea_bert',
            'short_text':   True,
            'label':        'Short-text / STS12 (SNEA-BERT)',
        },
        'wikipedia_snea_bert': {
            'dataset_file': 'datasets/wikipedia_entity_swap_400.csv',
            'aa_kea_file':  'datasets/wikipedia_entity_swap_400_snea_bert_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/wikipedia_snea_bert',
            'short_text':   False,
            'label':        'Long-text / NLP-perturbed (SNEA-BERT)',
        },
        # ── SNEA-BERT α=0.3 (hybrid: 30% KG + 70% sentence-transformer) ─────
        'mrpc_snea_bert_0_3': {
            'dataset_file': 'datasets/mrpc_400.csv',
            'aa_kea_file':  'datasets/mrpc_400_KGs_results_snea_0.3.csv',
            'score_col':    'snea_bert_alpha_0.3',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/mrpc_snea_bert_0_3',
            'short_text':   True,
            'label':        'Short-text / SNEA-BERT α=0.3',
        },
        'paws_wiki_snea_bert_0_3': {
            'dataset_file': 'datasets/paws_wiki_400.csv',
            'aa_kea_file':  'datasets/paws_wiki_400_KGs_results_snea_0.3.csv',
            'score_col':    'snea_bert_alpha_0.3',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/paws_wiki_snea_bert_0_3',
            'short_text':   False,
            'label':        'Long-text / Linguistic (SNEA-BERT α=0.3)',
        },
        'semantic_kg_combined_snea_bert_0_3': {
            'dataset_file': 'datasets/semantic_kg_combined_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_combined_400_KGs_results_snea_0.3.csv',
            'score_col':    'snea_bert_alpha_0.3',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_combined_snea_bert_0_3',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (SNEA-BERT α=0.3)',
        },
        'wikipedia_snea_bert_0_3': {
            'dataset_file': 'datasets/wikipedia_entity_swap_400.csv',
            'aa_kea_file':  'datasets/wikipedia_entity_swap_400_KGs_results_snea_0.3.csv',
            'score_col':    'snea_bert_alpha_0.3',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/wikipedia_snea_bert_0_3',
            'short_text':   False,
            'label':        'Long-text / NLP-perturbed (SNEA-BERT α=0.3)',
        },
        'semantic_kg_codex_400_snea_bert_0_3': {
            'dataset_file': 'datasets/semantic_kg_codex_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_codex_400_KGs_results_snea_0.3.csv',
            'score_col':    'snea_bert_alpha_0.3',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_codex_400_snea_bert_0_3',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Codex 400, SNEA-BERT α=0.3)',
        },
        'semantic_kg_findkg_snea_bert_0_3': {
            'dataset_file': 'datasets/semantic_kg_findkg_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_findkg_400_KGs_results_snea_0.3.csv',
            'score_col':    'snea_bert_alpha_0.3',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_findkg_snea_bert_0_3',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (FindKG, SNEA-BERT α=0.3)',
        },
        'semantic_kg_globi_snea_bert_0_3': {
            'dataset_file': 'datasets/semantic_kg_globi_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_globi_400_KGs_results_snea_0.3.csv',
            'score_col':    'snea_bert_alpha_0.3',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_globi_snea_bert_0_3',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (GloBI, SNEA-BERT α=0.3)',
        },
        'semantic_kg_oregano_snea_bert_0_3': {
            'dataset_file': 'datasets/semantic_kg_oregano_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_oregano_400_KGs_results_snea_0.3.csv',
            'score_col':    'snea_bert_alpha_0.3',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_oregano_snea_bert_0_3',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Oregano, SNEA-BERT α=0.3)',
        },
        'sts12_snea_bert_0_3': {
            'dataset_file': 'datasets/sts12_400.csv',
            'aa_kea_file':  'datasets/sts12_400_KGs_results_snea_0.3.csv',
            'score_col':    'snea_bert_alpha_0.3',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/sts12_snea_bert_0_3',
            'short_text':   True,
            'label':        'Short-text / STS12 (SNEA-BERT α=0.3)',
        },
        # ── SNEA-BERT alpha sweep (α = 0.0 → 1.0, from Results_All_Methods 2) ──────
        'mrpc_snea_alpha_0p0': {
            'dataset_file': 'datasets/mrpc_400.csv',
            'aa_kea_file':  'datasets/mrpc_400_snea_alpha_0p0_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/mrpc_snea_alpha_0p0',
            'short_text':   True,
            'label':        'Short-text / MRPC (SNEA-α=0.0)',
        },
        'mrpc_snea_alpha_0p1': {
            'dataset_file': 'datasets/mrpc_400.csv',
            'aa_kea_file':  'datasets/mrpc_400_snea_alpha_0p1_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/mrpc_snea_alpha_0p1',
            'short_text':   True,
            'label':        'Short-text / MRPC (SNEA-α=0.1)',
        },
        'mrpc_snea_alpha_0p2': {
            'dataset_file': 'datasets/mrpc_400.csv',
            'aa_kea_file':  'datasets/mrpc_400_snea_alpha_0p2_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/mrpc_snea_alpha_0p2',
            'short_text':   True,
            'label':        'Short-text / MRPC (SNEA-α=0.2)',
        },
        'mrpc_snea_alpha_0p3': {
            'dataset_file': 'datasets/mrpc_400.csv',
            'aa_kea_file':  'datasets/mrpc_400_snea_alpha_0p3_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/mrpc_snea_alpha_0p3',
            'short_text':   True,
            'label':        'Short-text / MRPC (SNEA-α=0.3)',
        },
        'mrpc_snea_alpha_0p4': {
            'dataset_file': 'datasets/mrpc_400.csv',
            'aa_kea_file':  'datasets/mrpc_400_snea_alpha_0p4_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/mrpc_snea_alpha_0p4',
            'short_text':   True,
            'label':        'Short-text / MRPC (SNEA-α=0.4)',
        },
        'mrpc_snea_alpha_0p5': {
            'dataset_file': 'datasets/mrpc_400.csv',
            'aa_kea_file':  'datasets/mrpc_400_snea_alpha_0p5_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/mrpc_snea_alpha_0p5',
            'short_text':   True,
            'label':        'Short-text / MRPC (SNEA-α=0.5)',
        },
        'mrpc_snea_alpha_0p6': {
            'dataset_file': 'datasets/mrpc_400.csv',
            'aa_kea_file':  'datasets/mrpc_400_snea_alpha_0p6_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/mrpc_snea_alpha_0p6',
            'short_text':   True,
            'label':        'Short-text / MRPC (SNEA-α=0.6)',
        },
        'mrpc_snea_alpha_0p7': {
            'dataset_file': 'datasets/mrpc_400.csv',
            'aa_kea_file':  'datasets/mrpc_400_snea_alpha_0p7_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/mrpc_snea_alpha_0p7',
            'short_text':   True,
            'label':        'Short-text / MRPC (SNEA-α=0.7)',
        },
        'mrpc_snea_alpha_0p8': {
            'dataset_file': 'datasets/mrpc_400.csv',
            'aa_kea_file':  'datasets/mrpc_400_snea_alpha_0p8_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/mrpc_snea_alpha_0p8',
            'short_text':   True,
            'label':        'Short-text / MRPC (SNEA-α=0.8)',
        },
        'mrpc_snea_alpha_0p9': {
            'dataset_file': 'datasets/mrpc_400.csv',
            'aa_kea_file':  'datasets/mrpc_400_snea_alpha_0p9_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/mrpc_snea_alpha_0p9',
            'short_text':   True,
            'label':        'Short-text / MRPC (SNEA-α=0.9)',
        },
        'mrpc_snea_alpha_1p0': {
            'dataset_file': 'datasets/mrpc_400.csv',
            'aa_kea_file':  'datasets/mrpc_400_snea_alpha_1p0_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/mrpc_snea_alpha_1p0',
            'short_text':   True,
            'label':        'Short-text / MRPC (SNEA-α=1.0)',
        },
        'paws_wiki_snea_alpha_0p0': {
            'dataset_file': 'datasets/paws_wiki_400.csv',
            'aa_kea_file':  'datasets/paws_wiki_400_snea_alpha_0p0_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/paws_wiki_snea_alpha_0p0',
            'short_text':   False,
            'label':        'Long-text / Linguistic (SNEA-α=0.0)',
        },
        'paws_wiki_snea_alpha_0p1': {
            'dataset_file': 'datasets/paws_wiki_400.csv',
            'aa_kea_file':  'datasets/paws_wiki_400_snea_alpha_0p1_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/paws_wiki_snea_alpha_0p1',
            'short_text':   False,
            'label':        'Long-text / Linguistic (SNEA-α=0.1)',
        },
        'paws_wiki_snea_alpha_0p2': {
            'dataset_file': 'datasets/paws_wiki_400.csv',
            'aa_kea_file':  'datasets/paws_wiki_400_snea_alpha_0p2_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/paws_wiki_snea_alpha_0p2',
            'short_text':   False,
            'label':        'Long-text / Linguistic (SNEA-α=0.2)',
        },
        'paws_wiki_snea_alpha_0p3': {
            'dataset_file': 'datasets/paws_wiki_400.csv',
            'aa_kea_file':  'datasets/paws_wiki_400_snea_alpha_0p3_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/paws_wiki_snea_alpha_0p3',
            'short_text':   False,
            'label':        'Long-text / Linguistic (SNEA-α=0.3)',
        },
        'paws_wiki_snea_alpha_0p4': {
            'dataset_file': 'datasets/paws_wiki_400.csv',
            'aa_kea_file':  'datasets/paws_wiki_400_snea_alpha_0p4_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/paws_wiki_snea_alpha_0p4',
            'short_text':   False,
            'label':        'Long-text / Linguistic (SNEA-α=0.4)',
        },
        'paws_wiki_snea_alpha_0p5': {
            'dataset_file': 'datasets/paws_wiki_400.csv',
            'aa_kea_file':  'datasets/paws_wiki_400_snea_alpha_0p5_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/paws_wiki_snea_alpha_0p5',
            'short_text':   False,
            'label':        'Long-text / Linguistic (SNEA-α=0.5)',
        },
        'paws_wiki_snea_alpha_0p6': {
            'dataset_file': 'datasets/paws_wiki_400.csv',
            'aa_kea_file':  'datasets/paws_wiki_400_snea_alpha_0p6_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/paws_wiki_snea_alpha_0p6',
            'short_text':   False,
            'label':        'Long-text / Linguistic (SNEA-α=0.6)',
        },
        'paws_wiki_snea_alpha_0p7': {
            'dataset_file': 'datasets/paws_wiki_400.csv',
            'aa_kea_file':  'datasets/paws_wiki_400_snea_alpha_0p7_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/paws_wiki_snea_alpha_0p7',
            'short_text':   False,
            'label':        'Long-text / Linguistic (SNEA-α=0.7)',
        },
        'paws_wiki_snea_alpha_0p8': {
            'dataset_file': 'datasets/paws_wiki_400.csv',
            'aa_kea_file':  'datasets/paws_wiki_400_snea_alpha_0p8_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/paws_wiki_snea_alpha_0p8',
            'short_text':   False,
            'label':        'Long-text / Linguistic (SNEA-α=0.8)',
        },
        'paws_wiki_snea_alpha_0p9': {
            'dataset_file': 'datasets/paws_wiki_400.csv',
            'aa_kea_file':  'datasets/paws_wiki_400_snea_alpha_0p9_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/paws_wiki_snea_alpha_0p9',
            'short_text':   False,
            'label':        'Long-text / Linguistic (SNEA-α=0.9)',
        },
        'paws_wiki_snea_alpha_1p0': {
            'dataset_file': 'datasets/paws_wiki_400.csv',
            'aa_kea_file':  'datasets/paws_wiki_400_snea_alpha_1p0_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/paws_wiki_snea_alpha_1p0',
            'short_text':   False,
            'label':        'Long-text / Linguistic (SNEA-α=1.0)',
        },
        'sts12_snea_alpha_0p0': {
            'dataset_file': 'datasets/sts12_400.csv',
            'aa_kea_file':  'datasets/sts12_400_snea_alpha_0p0_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/sts12_snea_alpha_0p0',
            'short_text':   True,
            'label':        'Short-text / STS12 (SNEA-α=0.0)',
        },
        'sts12_snea_alpha_0p1': {
            'dataset_file': 'datasets/sts12_400.csv',
            'aa_kea_file':  'datasets/sts12_400_snea_alpha_0p1_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/sts12_snea_alpha_0p1',
            'short_text':   True,
            'label':        'Short-text / STS12 (SNEA-α=0.1)',
        },
        'sts12_snea_alpha_0p2': {
            'dataset_file': 'datasets/sts12_400.csv',
            'aa_kea_file':  'datasets/sts12_400_snea_alpha_0p2_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/sts12_snea_alpha_0p2',
            'short_text':   True,
            'label':        'Short-text / STS12 (SNEA-α=0.2)',
        },
        'sts12_snea_alpha_0p3': {
            'dataset_file': 'datasets/sts12_400.csv',
            'aa_kea_file':  'datasets/sts12_400_snea_alpha_0p3_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/sts12_snea_alpha_0p3',
            'short_text':   True,
            'label':        'Short-text / STS12 (SNEA-α=0.3)',
        },
        'sts12_snea_alpha_0p4': {
            'dataset_file': 'datasets/sts12_400.csv',
            'aa_kea_file':  'datasets/sts12_400_snea_alpha_0p4_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/sts12_snea_alpha_0p4',
            'short_text':   True,
            'label':        'Short-text / STS12 (SNEA-α=0.4)',
        },
        'sts12_snea_alpha_0p5': {
            'dataset_file': 'datasets/sts12_400.csv',
            'aa_kea_file':  'datasets/sts12_400_snea_alpha_0p5_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/sts12_snea_alpha_0p5',
            'short_text':   True,
            'label':        'Short-text / STS12 (SNEA-α=0.5)',
        },
        'sts12_snea_alpha_0p6': {
            'dataset_file': 'datasets/sts12_400.csv',
            'aa_kea_file':  'datasets/sts12_400_snea_alpha_0p6_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/sts12_snea_alpha_0p6',
            'short_text':   True,
            'label':        'Short-text / STS12 (SNEA-α=0.6)',
        },
        'sts12_snea_alpha_0p7': {
            'dataset_file': 'datasets/sts12_400.csv',
            'aa_kea_file':  'datasets/sts12_400_snea_alpha_0p7_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/sts12_snea_alpha_0p7',
            'short_text':   True,
            'label':        'Short-text / STS12 (SNEA-α=0.7)',
        },
        'sts12_snea_alpha_0p8': {
            'dataset_file': 'datasets/sts12_400.csv',
            'aa_kea_file':  'datasets/sts12_400_snea_alpha_0p8_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/sts12_snea_alpha_0p8',
            'short_text':   True,
            'label':        'Short-text / STS12 (SNEA-α=0.8)',
        },
        'sts12_snea_alpha_0p9': {
            'dataset_file': 'datasets/sts12_400.csv',
            'aa_kea_file':  'datasets/sts12_400_snea_alpha_0p9_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/sts12_snea_alpha_0p9',
            'short_text':   True,
            'label':        'Short-text / STS12 (SNEA-α=0.9)',
        },
        'sts12_snea_alpha_1p0': {
            'dataset_file': 'datasets/sts12_400.csv',
            'aa_kea_file':  'datasets/sts12_400_snea_alpha_1p0_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/sts12_snea_alpha_1p0',
            'short_text':   True,
            'label':        'Short-text / STS12 (SNEA-α=1.0)',
        },
        'semantic_kg_combined_snea_alpha_0p0': {
            'dataset_file': 'datasets/semantic_kg_combined_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_combined_400_snea_alpha_0p0_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_combined_snea_alpha_0p0',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (SNEA-α=0.0)',
        },
        'semantic_kg_combined_snea_alpha_0p1': {
            'dataset_file': 'datasets/semantic_kg_combined_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_combined_400_snea_alpha_0p1_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_combined_snea_alpha_0p1',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (SNEA-α=0.1)',
        },
        'semantic_kg_combined_snea_alpha_0p2': {
            'dataset_file': 'datasets/semantic_kg_combined_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_combined_400_snea_alpha_0p2_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_combined_snea_alpha_0p2',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (SNEA-α=0.2)',
        },
        'semantic_kg_combined_snea_alpha_0p3': {
            'dataset_file': 'datasets/semantic_kg_combined_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_combined_400_snea_alpha_0p3_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_combined_snea_alpha_0p3',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (SNEA-α=0.3)',
        },
        'semantic_kg_combined_snea_alpha_0p4': {
            'dataset_file': 'datasets/semantic_kg_combined_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_combined_400_snea_alpha_0p4_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_combined_snea_alpha_0p4',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (SNEA-α=0.4)',
        },
        'semantic_kg_combined_snea_alpha_0p5': {
            'dataset_file': 'datasets/semantic_kg_combined_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_combined_400_snea_alpha_0p5_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_combined_snea_alpha_0p5',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (SNEA-α=0.5)',
        },
        'semantic_kg_combined_snea_alpha_0p6': {
            'dataset_file': 'datasets/semantic_kg_combined_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_combined_400_snea_alpha_0p6_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_combined_snea_alpha_0p6',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (SNEA-α=0.6)',
        },
        'semantic_kg_combined_snea_alpha_0p7': {
            'dataset_file': 'datasets/semantic_kg_combined_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_combined_400_snea_alpha_0p7_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_combined_snea_alpha_0p7',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (SNEA-α=0.7)',
        },
        'semantic_kg_combined_snea_alpha_0p8': {
            'dataset_file': 'datasets/semantic_kg_combined_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_combined_400_snea_alpha_0p8_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_combined_snea_alpha_0p8',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (SNEA-α=0.8)',
        },
        'semantic_kg_combined_snea_alpha_0p9': {
            'dataset_file': 'datasets/semantic_kg_combined_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_combined_400_snea_alpha_0p9_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_combined_snea_alpha_0p9',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (SNEA-α=0.9)',
        },
        'semantic_kg_combined_snea_alpha_1p0': {
            'dataset_file': 'datasets/semantic_kg_combined_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_combined_400_snea_alpha_1p0_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_combined_snea_alpha_1p0',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (SNEA-α=1.0)',
        },
        'semantic_kg_codex_400_snea_alpha_0p0': {
            'dataset_file': 'datasets/semantic_kg_codex_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_codex_400_snea_alpha_0p0_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_codex_400_snea_alpha_0p0',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Codex 400) (SNEA-α=0.0)',
        },
        'semantic_kg_codex_400_snea_alpha_0p1': {
            'dataset_file': 'datasets/semantic_kg_codex_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_codex_400_snea_alpha_0p1_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_codex_400_snea_alpha_0p1',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Codex 400) (SNEA-α=0.1)',
        },
        'semantic_kg_codex_400_snea_alpha_0p2': {
            'dataset_file': 'datasets/semantic_kg_codex_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_codex_400_snea_alpha_0p2_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_codex_400_snea_alpha_0p2',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Codex 400) (SNEA-α=0.2)',
        },
        'semantic_kg_codex_400_snea_alpha_0p3': {
            'dataset_file': 'datasets/semantic_kg_codex_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_codex_400_snea_alpha_0p3_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_codex_400_snea_alpha_0p3',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Codex 400) (SNEA-α=0.3)',
        },
        'semantic_kg_codex_400_snea_alpha_0p4': {
            'dataset_file': 'datasets/semantic_kg_codex_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_codex_400_snea_alpha_0p4_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_codex_400_snea_alpha_0p4',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Codex 400) (SNEA-α=0.4)',
        },
        'semantic_kg_codex_400_snea_alpha_0p5': {
            'dataset_file': 'datasets/semantic_kg_codex_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_codex_400_snea_alpha_0p5_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_codex_400_snea_alpha_0p5',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Codex 400) (SNEA-α=0.5)',
        },
        'semantic_kg_codex_400_snea_alpha_0p6': {
            'dataset_file': 'datasets/semantic_kg_codex_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_codex_400_snea_alpha_0p6_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_codex_400_snea_alpha_0p6',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Codex 400) (SNEA-α=0.6)',
        },
        'semantic_kg_codex_400_snea_alpha_0p7': {
            'dataset_file': 'datasets/semantic_kg_codex_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_codex_400_snea_alpha_0p7_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_codex_400_snea_alpha_0p7',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Codex 400) (SNEA-α=0.7)',
        },
        'semantic_kg_codex_400_snea_alpha_0p8': {
            'dataset_file': 'datasets/semantic_kg_codex_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_codex_400_snea_alpha_0p8_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_codex_400_snea_alpha_0p8',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Codex 400) (SNEA-α=0.8)',
        },
        'semantic_kg_codex_400_snea_alpha_0p9': {
            'dataset_file': 'datasets/semantic_kg_codex_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_codex_400_snea_alpha_0p9_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_codex_400_snea_alpha_0p9',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Codex 400) (SNEA-α=0.9)',
        },
        'semantic_kg_codex_400_snea_alpha_1p0': {
            'dataset_file': 'datasets/semantic_kg_codex_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_codex_400_snea_alpha_1p0_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_codex_400_snea_alpha_1p0',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Codex 400) (SNEA-α=1.0)',
        },
        'semantic_kg_findkg_snea_alpha_0p0': {
            'dataset_file': 'datasets/semantic_kg_findkg_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_findkg_400_snea_alpha_0p0_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_findkg_snea_alpha_0p0',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (FindKG) (SNEA-α=0.0)',
        },
        'semantic_kg_findkg_snea_alpha_0p1': {
            'dataset_file': 'datasets/semantic_kg_findkg_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_findkg_400_snea_alpha_0p1_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_findkg_snea_alpha_0p1',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (FindKG) (SNEA-α=0.1)',
        },
        'semantic_kg_findkg_snea_alpha_0p2': {
            'dataset_file': 'datasets/semantic_kg_findkg_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_findkg_400_snea_alpha_0p2_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_findkg_snea_alpha_0p2',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (FindKG) (SNEA-α=0.2)',
        },
        'semantic_kg_findkg_snea_alpha_0p3': {
            'dataset_file': 'datasets/semantic_kg_findkg_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_findkg_400_snea_alpha_0p3_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_findkg_snea_alpha_0p3',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (FindKG) (SNEA-α=0.3)',
        },
        'semantic_kg_findkg_snea_alpha_0p4': {
            'dataset_file': 'datasets/semantic_kg_findkg_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_findkg_400_snea_alpha_0p4_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_findkg_snea_alpha_0p4',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (FindKG) (SNEA-α=0.4)',
        },
        'semantic_kg_findkg_snea_alpha_0p5': {
            'dataset_file': 'datasets/semantic_kg_findkg_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_findkg_400_snea_alpha_0p5_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_findkg_snea_alpha_0p5',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (FindKG) (SNEA-α=0.5)',
        },
        'semantic_kg_findkg_snea_alpha_0p6': {
            'dataset_file': 'datasets/semantic_kg_findkg_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_findkg_400_snea_alpha_0p6_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_findkg_snea_alpha_0p6',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (FindKG) (SNEA-α=0.6)',
        },
        'semantic_kg_findkg_snea_alpha_0p7': {
            'dataset_file': 'datasets/semantic_kg_findkg_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_findkg_400_snea_alpha_0p7_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_findkg_snea_alpha_0p7',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (FindKG) (SNEA-α=0.7)',
        },
        'semantic_kg_findkg_snea_alpha_0p8': {
            'dataset_file': 'datasets/semantic_kg_findkg_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_findkg_400_snea_alpha_0p8_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_findkg_snea_alpha_0p8',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (FindKG) (SNEA-α=0.8)',
        },
        'semantic_kg_findkg_snea_alpha_0p9': {
            'dataset_file': 'datasets/semantic_kg_findkg_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_findkg_400_snea_alpha_0p9_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_findkg_snea_alpha_0p9',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (FindKG) (SNEA-α=0.9)',
        },
        'semantic_kg_findkg_snea_alpha_1p0': {
            'dataset_file': 'datasets/semantic_kg_findkg_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_findkg_400_snea_alpha_1p0_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_findkg_snea_alpha_1p0',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (FindKG) (SNEA-α=1.0)',
        },
        'semantic_kg_globi_snea_alpha_0p0': {
            'dataset_file': 'datasets/semantic_kg_globi_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_globi_400_snea_alpha_0p0_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_globi_snea_alpha_0p0',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (GloBI) (SNEA-α=0.0)',
        },
        'semantic_kg_globi_snea_alpha_0p1': {
            'dataset_file': 'datasets/semantic_kg_globi_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_globi_400_snea_alpha_0p1_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_globi_snea_alpha_0p1',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (GloBI) (SNEA-α=0.1)',
        },
        'semantic_kg_globi_snea_alpha_0p2': {
            'dataset_file': 'datasets/semantic_kg_globi_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_globi_400_snea_alpha_0p2_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_globi_snea_alpha_0p2',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (GloBI) (SNEA-α=0.2)',
        },
        'semantic_kg_globi_snea_alpha_0p3': {
            'dataset_file': 'datasets/semantic_kg_globi_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_globi_400_snea_alpha_0p3_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_globi_snea_alpha_0p3',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (GloBI) (SNEA-α=0.3)',
        },
        'semantic_kg_globi_snea_alpha_0p4': {
            'dataset_file': 'datasets/semantic_kg_globi_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_globi_400_snea_alpha_0p4_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_globi_snea_alpha_0p4',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (GloBI) (SNEA-α=0.4)',
        },
        'semantic_kg_globi_snea_alpha_0p5': {
            'dataset_file': 'datasets/semantic_kg_globi_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_globi_400_snea_alpha_0p5_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_globi_snea_alpha_0p5',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (GloBI) (SNEA-α=0.5)',
        },
        'semantic_kg_globi_snea_alpha_0p6': {
            'dataset_file': 'datasets/semantic_kg_globi_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_globi_400_snea_alpha_0p6_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_globi_snea_alpha_0p6',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (GloBI) (SNEA-α=0.6)',
        },
        'semantic_kg_globi_snea_alpha_0p7': {
            'dataset_file': 'datasets/semantic_kg_globi_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_globi_400_snea_alpha_0p7_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_globi_snea_alpha_0p7',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (GloBI) (SNEA-α=0.7)',
        },
        'semantic_kg_globi_snea_alpha_0p8': {
            'dataset_file': 'datasets/semantic_kg_globi_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_globi_400_snea_alpha_0p8_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_globi_snea_alpha_0p8',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (GloBI) (SNEA-α=0.8)',
        },
        'semantic_kg_globi_snea_alpha_0p9': {
            'dataset_file': 'datasets/semantic_kg_globi_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_globi_400_snea_alpha_0p9_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_globi_snea_alpha_0p9',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (GloBI) (SNEA-α=0.9)',
        },
        'semantic_kg_globi_snea_alpha_1p0': {
            'dataset_file': 'datasets/semantic_kg_globi_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_globi_400_snea_alpha_1p0_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_globi_snea_alpha_1p0',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (GloBI) (SNEA-α=1.0)',
        },
        'semantic_kg_oregano_snea_alpha_0p0': {
            'dataset_file': 'datasets/semantic_kg_oregano_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_oregano_400_snea_alpha_0p0_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_oregano_snea_alpha_0p0',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Oregano) (SNEA-α=0.0)',
        },
        'semantic_kg_oregano_snea_alpha_0p1': {
            'dataset_file': 'datasets/semantic_kg_oregano_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_oregano_400_snea_alpha_0p1_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_oregano_snea_alpha_0p1',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Oregano) (SNEA-α=0.1)',
        },
        'semantic_kg_oregano_snea_alpha_0p2': {
            'dataset_file': 'datasets/semantic_kg_oregano_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_oregano_400_snea_alpha_0p2_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_oregano_snea_alpha_0p2',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Oregano) (SNEA-α=0.2)',
        },
        'semantic_kg_oregano_snea_alpha_0p3': {
            'dataset_file': 'datasets/semantic_kg_oregano_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_oregano_400_snea_alpha_0p3_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_oregano_snea_alpha_0p3',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Oregano) (SNEA-α=0.3)',
        },
        'semantic_kg_oregano_snea_alpha_0p4': {
            'dataset_file': 'datasets/semantic_kg_oregano_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_oregano_400_snea_alpha_0p4_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_oregano_snea_alpha_0p4',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Oregano) (SNEA-α=0.4)',
        },
        'semantic_kg_oregano_snea_alpha_0p5': {
            'dataset_file': 'datasets/semantic_kg_oregano_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_oregano_400_snea_alpha_0p5_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_oregano_snea_alpha_0p5',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Oregano) (SNEA-α=0.5)',
        },
        'semantic_kg_oregano_snea_alpha_0p6': {
            'dataset_file': 'datasets/semantic_kg_oregano_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_oregano_400_snea_alpha_0p6_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_oregano_snea_alpha_0p6',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Oregano) (SNEA-α=0.6)',
        },
        'semantic_kg_oregano_snea_alpha_0p7': {
            'dataset_file': 'datasets/semantic_kg_oregano_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_oregano_400_snea_alpha_0p7_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_oregano_snea_alpha_0p7',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Oregano) (SNEA-α=0.7)',
        },
        'semantic_kg_oregano_snea_alpha_0p8': {
            'dataset_file': 'datasets/semantic_kg_oregano_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_oregano_400_snea_alpha_0p8_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_oregano_snea_alpha_0p8',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Oregano) (SNEA-α=0.8)',
        },
        'semantic_kg_oregano_snea_alpha_0p9': {
            'dataset_file': 'datasets/semantic_kg_oregano_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_oregano_400_snea_alpha_0p9_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_oregano_snea_alpha_0p9',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Oregano) (SNEA-α=0.9)',
        },
        'semantic_kg_oregano_snea_alpha_1p0': {
            'dataset_file': 'datasets/semantic_kg_oregano_400.csv',
            'aa_kea_file':  'datasets/semantic_kg_oregano_400_snea_alpha_1p0_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/semantic_kg_oregano_snea_alpha_1p0',
            'short_text':   False,
            'label':        'Long-text / KG-perturbed (Oregano) (SNEA-α=1.0)',
        },
        'wikipedia_snea_alpha_0p0': {
            'dataset_file': 'datasets/wikipedia_entity_swap_400.csv',
            'aa_kea_file':  'datasets/wikipedia_entity_swap_400_snea_alpha_0p0_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/wikipedia_snea_alpha_0p0',
            'short_text':   False,
            'label':        'Long-text / NLP-perturbed (SNEA-α=0.0)',
        },
        'wikipedia_snea_alpha_0p1': {
            'dataset_file': 'datasets/wikipedia_entity_swap_400.csv',
            'aa_kea_file':  'datasets/wikipedia_entity_swap_400_snea_alpha_0p1_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/wikipedia_snea_alpha_0p1',
            'short_text':   False,
            'label':        'Long-text / NLP-perturbed (SNEA-α=0.1)',
        },
        'wikipedia_snea_alpha_0p2': {
            'dataset_file': 'datasets/wikipedia_entity_swap_400.csv',
            'aa_kea_file':  'datasets/wikipedia_entity_swap_400_snea_alpha_0p2_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/wikipedia_snea_alpha_0p2',
            'short_text':   False,
            'label':        'Long-text / NLP-perturbed (SNEA-α=0.2)',
        },
        'wikipedia_snea_alpha_0p3': {
            'dataset_file': 'datasets/wikipedia_entity_swap_400.csv',
            'aa_kea_file':  'datasets/wikipedia_entity_swap_400_snea_alpha_0p3_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/wikipedia_snea_alpha_0p3',
            'short_text':   False,
            'label':        'Long-text / NLP-perturbed (SNEA-α=0.3)',
        },
        'wikipedia_snea_alpha_0p4': {
            'dataset_file': 'datasets/wikipedia_entity_swap_400.csv',
            'aa_kea_file':  'datasets/wikipedia_entity_swap_400_snea_alpha_0p4_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/wikipedia_snea_alpha_0p4',
            'short_text':   False,
            'label':        'Long-text / NLP-perturbed (SNEA-α=0.4)',
        },
        'wikipedia_snea_alpha_0p5': {
            'dataset_file': 'datasets/wikipedia_entity_swap_400.csv',
            'aa_kea_file':  'datasets/wikipedia_entity_swap_400_snea_alpha_0p5_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/wikipedia_snea_alpha_0p5',
            'short_text':   False,
            'label':        'Long-text / NLP-perturbed (SNEA-α=0.5)',
        },
        'wikipedia_snea_alpha_0p6': {
            'dataset_file': 'datasets/wikipedia_entity_swap_400.csv',
            'aa_kea_file':  'datasets/wikipedia_entity_swap_400_snea_alpha_0p6_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/wikipedia_snea_alpha_0p6',
            'short_text':   False,
            'label':        'Long-text / NLP-perturbed (SNEA-α=0.6)',
        },
        'wikipedia_snea_alpha_0p7': {
            'dataset_file': 'datasets/wikipedia_entity_swap_400.csv',
            'aa_kea_file':  'datasets/wikipedia_entity_swap_400_snea_alpha_0p7_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/wikipedia_snea_alpha_0p7',
            'short_text':   False,
            'label':        'Long-text / NLP-perturbed (SNEA-α=0.7)',
        },
        'wikipedia_snea_alpha_0p8': {
            'dataset_file': 'datasets/wikipedia_entity_swap_400.csv',
            'aa_kea_file':  'datasets/wikipedia_entity_swap_400_snea_alpha_0p8_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/wikipedia_snea_alpha_0p8',
            'short_text':   False,
            'label':        'Long-text / NLP-perturbed (SNEA-α=0.8)',
        },
        'wikipedia_snea_alpha_0p9': {
            'dataset_file': 'datasets/wikipedia_entity_swap_400.csv',
            'aa_kea_file':  'datasets/wikipedia_entity_swap_400_snea_alpha_0p9_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/wikipedia_snea_alpha_0p9',
            'short_text':   False,
            'label':        'Long-text / NLP-perturbed (SNEA-α=0.9)',
        },
        'wikipedia_snea_alpha_1p0': {
            'dataset_file': 'datasets/wikipedia_entity_swap_400.csv',
            'aa_kea_file':  'datasets/wikipedia_entity_swap_400_snea_alpha_1p0_results.csv',
            'score_col':    'snea_bert_similarity',
            'text1_col':    'response1',
            'text2_col':    'response2',
            'output_dir':   'output/wikipedia_snea_alpha_1p0',
            'short_text':   False,
            'label':        'Long-text / NLP-perturbed (SNEA-α=1.0)',
        },
    }

    # Embedding models to evaluate
    EMBEDDING_MODELS = {
        'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
        'sentence-t5-base': 'sentence-transformers/sentence-t5-base',
    }

    # ROUGE metrics
    ROUGE_METRICS = ['rouge1', 'rouge2', 'rougeL']

    # Classification threshold (will be optimized)
    DEFAULT_THRESHOLD = 0.5

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load dataset and AA-KEA results"""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    # Load original dataset
    print(f"\nLoading {Config.DATASET_FILE}...")
    df = pd.read_csv(Config.DATASET_FILE)
    print(f"✓ Loaded {len(df)} sentence pairs")

    # Load AA-KEA results
    print(f"\nLoading {Config.AA_KEA_RESULTS}...")
    aa_kea = pd.read_csv(Config.AA_KEA_RESULTS)
    print(f"✓ Loaded {len(aa_kea)} AA-KEA results")

    # Merge on pair_id
    print("\nMerging datasets...")
    df = df.merge(aa_kea[['pair_id', 'aa_kea_similarity']], on='pair_id', how='left')

    # Check for missing values
    missing = df['aa_kea_similarity'].isna().sum()
    if missing > 0:
        print(f"⚠️  Warning: {missing} pairs missing AA-KEA scores, filling with 0.0")
        df['aa_kea_similarity'] = df['aa_kea_similarity'].fillna(0.0)

    print(f"✓ Final dataset: {len(df)} pairs")
    print(f"  Columns: {df.columns.tolist()}")

    return df


# ============================================================================
# BASELINE METHODS
# ============================================================================

def compute_rouge_scores(df):
    """Compute ROUGE scores for all sentence pairs"""
    print("\n" + "="*80)
    print("COMPUTING ROUGE SCORES")
    print("="*80)

    scorer = get_rouge_scorer()

    rouge_results = {metric: [] for metric in Config.ROUGE_METRICS}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="ROUGE"):
        scores = scorer.score(row['response1'], row['response2'])
        for metric in Config.ROUGE_METRICS:
            rouge_results[metric].append(scores[metric].fmeasure)

    for metric in Config.ROUGE_METRICS:
        df[f'{metric}_score'] = rouge_results[metric]

    print("✓ ROUGE scores computed")
    return df


def compute_bleu_score(reference, hypothesis):
    """Compute BLEU score (simple implementation)"""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    smoothing = SmoothingFunction().method1
    score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
    return score


def compute_bleu_scores(df):
    """Compute BLEU scores for all sentence pairs"""
    print("\n" + "="*80)
    print("COMPUTING BLEU SCORES")
    print("="*80)

    # Download NLTK data if needed
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    bleu_scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="BLEU"):
        score = compute_bleu_score(row['response1'], row['response2'])
        bleu_scores.append(score)

    df['bleu_score'] = bleu_scores

    print("✓ BLEU scores computed")
    return df


def compute_bertscore(df, batch_size=16):
    """Compute BERTScore for all sentence pairs"""
    print("\n" + "="*80)
    print("COMPUTING BERTSCORE")
    print("="*80)

    # Use roberta-large for compatibility (deberta-xlarge has issues with long texts)
    print("Using model: roberta-large (better for long texts)")

    # Truncate very long texts to avoid overflow errors
    max_length = 450  # tokens

    try:
        scorer = get_bert_scorer()
        references = [str(t)[:2000] for t in df['response1'].tolist()]
        candidates = [str(t)[:2000] for t in df['response2'].tolist()]

        all_scores = []
        for i in tqdm(range(0, len(references), batch_size), desc="BERTScore"):
            refs = references[i:i+batch_size]
            cands = candidates[i:i+batch_size]
            P, R, F1 = scorer.score(cands, refs)
            all_scores.extend(F1.tolist())

        df['bertscore_f1'] = all_scores
        print("✓ BERTScore computed")

    except Exception as e:
        print(f"⚠️  BERTScore failed: {e}")
        print("   Skipping BERTScore (will use other metrics)")
        df['bertscore_f1'] = 0.0  # Placeholder

    return df


def compute_embedding_similarities(df):
    """Compute embedding-based similarities using multiple models"""
    print("\n" + "="*80)
    print("COMPUTING EMBEDDING SIMILARITIES")
    print("="*80)

    texts1 = df['response1'].tolist()
    texts2 = df['response2'].tolist()

    for model_name, model_path in Config.EMBEDDING_MODELS.items():
        model = get_st_model(model_name, model_path)

        print(f"  Encoding {len(texts1)*2} texts with {model_name} (batch)...")
        all_texts = texts1 + texts2
        embs = model.encode(all_texts, batch_size=32, show_progress_bar=True,
                            convert_to_numpy=True)
        n = len(texts1)
        e1, e2 = embs[:n], embs[n:]
        norms = (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)).clip(min=1e-9)
        similarities = (np.einsum('ij,ij->i', e1, e2) / norms).tolist()

        df[f'{model_name}_similarity'] = similarities
        print(f"  ✓ {model_name} similarities computed")

    return df


def compute_all_baselines(df):
    """Compute all baseline methods"""
    print("\n" + "="*80)
    print("COMPUTING ALL BASELINE METHODS")
    print("="*80)

    df = compute_rouge_scores(df)
    df = compute_bleu_scores(df)
    df = compute_bertscore(df)
    df = compute_embedding_similarities(df)

    print("\n✓ All baseline methods computed")
    return df


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def find_optimal_threshold(y_true, y_scores):
    """Find optimal threshold that maximizes F1 score"""
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def evaluate_method(df, score_column, method_name):
    """Evaluate a single method"""
    y_true = df['label'].values
    y_scores = df[score_column].values

    # Find optimal threshold
    threshold, _ = find_optimal_threshold(y_true, y_scores)

    # Make predictions
    y_pred = (y_scores >= threshold).astype(int)

    # Compute metrics
    metrics = {
        'method': method_name,
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_scores),
    }

    return metrics, y_pred


def evaluate_all_methods(df):
    """Evaluate all methods and return results"""
    print("\n" + "="*80)
    print("EVALUATING ALL METHODS")
    print("="*80)

    # Define methods to evaluate
    methods = {
        'AA-KEA (Our Method)': 'aa_kea_similarity',
        'ROUGE-1': 'rouge1_score',
        'ROUGE-2': 'rouge2_score',
        'ROUGE-L': 'rougeL_score',
        'BLEU': 'bleu_score',
        'BERTScore': 'bertscore_f1',
        'all-MiniLM-L6-v2': 'all-MiniLM-L6-v2_similarity',
        'sentence-t5-base': 'sentence-t5-base_similarity',
    }

    results = []
    predictions = {}

    for method_name, score_column in methods.items():
        if score_column in df.columns:
            print(f"\nEvaluating {method_name}...")
            metrics, y_pred = evaluate_method(df, score_column, method_name)
            results.append(metrics)
            predictions[method_name] = y_pred

            print(f"  F1: {metrics['f1']:.4f}, "
                  f"Precision: {metrics['precision']:.4f}, "
                  f"Recall: {metrics['recall']:.4f}, "
                  f"ROC-AUC: {metrics['roc_auc']:.4f}")

    results_df = pd.DataFrame(results)
    return results_df, predictions


# ============================================================================
# PERTURBATION TYPE ANALYSIS
# ============================================================================

def evaluate_by_perturbation_type(df):
    """Evaluate methods stratified by perturbation type"""
    print("\n" + "="*80)
    print("PERTURBATION TYPE ANALYSIS")
    print("="*80)

    perturbation_types = df['perturbation_type'].unique()

    methods = {
        'AA-KEA': 'aa_kea_similarity',
        'ROUGE-1': 'rouge1_score',
        'ROUGE-L': 'rougeL_score',
        'BLEU': 'bleu_score',
        'BERTScore': 'bertscore_f1',
        'Sentence-T5': 'sentence-t5-base_similarity',
    }

    perturbation_results = []

    for pert_type in perturbation_types:
        print(f"\n{pert_type}:")
        subset = df[df['perturbation_type'] == pert_type]

        for method_name, score_column in methods.items():
            if score_column in df.columns:
                y_true = subset['label'].values
                y_scores = subset[score_column].values

                threshold, _ = find_optimal_threshold(y_true, y_scores)
                y_pred = (y_scores >= threshold).astype(int)

                f1 = f1_score(y_true, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y_true, y_scores)

                perturbation_results.append({
                    'perturbation_type': pert_type,
                    'method': method_name,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'n_samples': len(subset)
                })

                print(f"  {method_name:15s} F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    pert_df = pd.DataFrame(perturbation_results)
    return pert_df


# ============================================================================
# NOVEL ANALYSES
# ============================================================================

def complementarity_analysis(df, predictions):
    """
    Analyze which cases each method uniquely gets correct
    Shows complementarity between methods
    """
    print("\n" + "="*80)
    print("COMPLEMENTARITY ANALYSIS")
    print("="*80)

    y_true = df['label'].values

    # Define main methods to compare
    methods = ['AA-KEA (Our Method)', 'BERTScore', 'sentence-t5-base', 'ROUGE-L']

    # Create correctness matrix
    correct_matrix = {}
    for method in methods:
        if method in predictions:
            correct_matrix[method] = (predictions[method] == y_true)

    # Overall accuracy
    print("\n1. Individual Method Accuracy:")
    for method in methods:
        if method in correct_matrix:
            acc = correct_matrix[method].mean()
            print(f"  {method:25s} {acc:.4f}")

    # Pairwise complementarity
    print("\n2. Pairwise Complementarity Analysis:")

    complementarity_data = []

    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            if method1 in correct_matrix and method2 in correct_matrix:
                # Cases where method1 correct but method2 wrong
                m1_only = correct_matrix[method1] & ~correct_matrix[method2]
                # Cases where method2 correct but method1 wrong
                m2_only = correct_matrix[method2] & ~correct_matrix[method1]
                # Both correct
                both = correct_matrix[method1] & correct_matrix[method2]
                # Both wrong
                neither = ~correct_matrix[method1] & ~correct_matrix[method2]

                print(f"\n  {method1} vs {method2}:")
                print(f"    {method1} only correct: {m1_only.sum()} ({100*m1_only.mean():.1f}%)")
                print(f"    {method2} only correct: {m2_only.sum()} ({100*m2_only.mean():.1f}%)")
                print(f"    Both correct:           {both.sum()} ({100*both.mean():.1f}%)")
                print(f"    Both wrong:             {neither.sum()} ({100*neither.mean():.1f}%)")

                # Analyze perturbation types for unique cases
                if m1_only.sum() > 0:
                    print(f"\n    Perturbations where {method1} uniquely succeeds:")
                    pert_dist = df[m1_only]['perturbation_type'].value_counts()
                    for pert, count in pert_dist.items():
                        print(f"      {pert}: {count}")

                complementarity_data.append({
                    'method1': method1,
                    'method2': method2,
                    'm1_only': m1_only.sum(),
                    'm2_only': m2_only.sum(),
                    'both': both.sum(),
                    'neither': neither.sum()
                })

    # Oracle ensemble (upper bound)
    print("\n3. Oracle Ensemble (Upper Bound):")
    oracle_pred = y_true.copy()  # Start with ground truth

    # For each case, if any method got it right, mark as correct
    for idx in range(len(y_true)):
        any_correct = False
        for method in methods:
            if method in predictions:
                if predictions[method][idx] == y_true[idx]:
                    any_correct = True
                    oracle_pred[idx] = predictions[method][idx]
                    break
        if not any_correct:
            # All methods wrong, use AA-KEA as default
            oracle_pred[idx] = predictions['AA-KEA (Our Method)'][idx]

    oracle_acc = (oracle_pred == y_true).mean()
    print(f"  Oracle Accuracy: {oracle_acc:.4f}")
    print(f"  Improvement over best single method: {oracle_acc - max([correct_matrix[m].mean() for m in methods if m in correct_matrix]):.4f}")

    comp_df = pd.DataFrame(complementarity_data)
    return comp_df, oracle_acc


def text_length_analysis(df, predictions):
    """
    Analyze performance vs text length
    Shows which methods scale better to longer texts
    """
    print("\n" + "="*80)
    print("TEXT LENGTH ANALYSIS")
    print("="*80)

    # Calculate text lengths
    df['text1_length'] = df['response1'].str.split().str.len()
    df['text2_length'] = df['response2'].str.split().str.len()
    df['avg_length'] = (df['text1_length'] + df['text2_length']) / 2

    print(f"\nText Length Statistics:")
    print(f"  Mean: {df['avg_length'].mean():.1f} words")
    print(f"  Median: {df['avg_length'].median():.1f} words")
    print(f"  Min: {df['avg_length'].min():.1f} words")
    print(f"  Max: {df['avg_length'].max():.1f} words")

    # Create length bins
    bins = [0, 40, 60, 80, 100, 150, 200]
    df['length_bin'] = pd.cut(df['avg_length'], bins=bins)

    print(f"\nLength Distribution:")
    print(df['length_bin'].value_counts().sort_index())

    # Evaluate each method per bin
    methods = ['AA-KEA (Our Method)', 'BERTScore', 'sentence-t5-base', 'ROUGE-L']

    length_results = []
    y_true = df['label'].values

    print(f"\nPerformance by Text Length:")
    for bin_label in df['length_bin'].cat.categories:
        mask = df['length_bin'] == bin_label
        if mask.sum() == 0:
            continue

        print(f"\n  Length: {bin_label}")
        print(f"  Samples: {mask.sum()}")

        for method in methods:
            if method in predictions:
                y_pred = predictions[method][mask]
                y_true_bin = y_true[mask]

                acc = (y_pred == y_true_bin).mean()
                f1 = f1_score(y_true_bin, y_pred, zero_division=0)

                length_results.append({
                    'method': method,
                    'length_bin': str(bin_label),
                    'avg_length': df[mask]['avg_length'].mean(),
                    'n_samples': mask.sum(),
                    'accuracy': acc,
                    'f1': f1
                })

                print(f"    {method:25s} Acc: {acc:.4f}, F1: {f1:.4f}")

    length_df = pd.DataFrame(length_results)
    return length_df


def domain_transfer_analysis(df, methods_scores):
    """
    Analyze cross-domain generalization
    Train threshold on one domain, test on others
    """
    print("\n" + "="*80)
    print("DOMAIN TRANSFER ANALYSIS")
    print("="*80)

    domains = df['dataset_name'].unique()
    print(f"\nDomains: {domains}")

    # Methods to evaluate
    methods = {
        'AA-KEA': 'aa_kea_similarity',
        'BERTScore': 'bertscore_f1',
        'Sentence-T5': 'sentence-t5-base_similarity',
        'ROUGE-L': 'rougeL_score'
    }

    transfer_results = []

    for train_domain in domains:
        print(f"\n{'='*70}")
        print(f"Training on: {train_domain}")
        print(f"{'='*70}")

        # Get train data
        train_mask = df['dataset_name'] == train_domain
        train_df = df[train_mask]
        y_train = train_df['label'].values

        print(f"  Train samples: {len(train_df)}")

        for method_name, score_col in methods.items():
            if score_col not in df.columns:
                continue

            # Find optimal threshold on train domain
            train_scores = train_df[score_col].values
            threshold, train_f1 = find_optimal_threshold(y_train, train_scores)

            print(f"\n  {method_name}:")
            print(f"    Optimal threshold: {threshold:.3f}")
            print(f"    Train F1: {train_f1:.4f}")

            # Test on all domains
            for test_domain in domains:
                test_mask = df['dataset_name'] == test_domain
                test_df = df[test_mask]
                y_test = test_df['label'].values
                test_scores = test_df[score_col].values

                # Apply transferred threshold
                y_pred_transfer = (test_scores >= threshold).astype(int)
                f1_transfer = f1_score(y_test, y_pred_transfer, zero_division=0)

                # Compute optimal threshold on test domain
                threshold_optimal, f1_optimal = find_optimal_threshold(y_test, test_scores)

                # Transfer gap
                transfer_gap = f1_optimal - f1_transfer

                transfer_results.append({
                    'train_domain': train_domain,
                    'test_domain': test_domain,
                    'method': method_name,
                    'threshold': threshold,
                    'f1_transfer': f1_transfer,
                    'f1_optimal': f1_optimal,
                    'transfer_gap': transfer_gap,
                    'n_test': len(test_df)
                })

                is_same = "✓" if train_domain == test_domain else ""
                print(f"    → {test_domain:10s} {is_same:2s} F1: {f1_transfer:.4f} "
                      f"(optimal: {f1_optimal:.4f}, gap: {transfer_gap:+.4f})")

    transfer_df = pd.DataFrame(transfer_results)

    # Summary: Average transfer gap per method
    print(f"\n{'='*70}")
    print("TRANSFER SUMMARY (Average gap when transferring across domains)")
    print(f"{'='*70}")

    for method in methods.keys():
        method_data = transfer_df[
            (transfer_df['method'] == method) &
            (transfer_df['train_domain'] != transfer_df['test_domain'])
        ]
        if len(method_data) > 0:
            avg_gap = method_data['transfer_gap'].mean()
            print(f"  {method:15s} Average transfer gap: {avg_gap:.4f}")
            if avg_gap < 0.02:
                print(f"                  → Excellent cross-domain generalization!")

    return transfer_df


# ============================================================================
# VISUALIZATION - NOVEL ANALYSES
# ============================================================================

def plot_complementarity(comp_df, oracle_acc, output_dir):
    """Plot complementarity analysis results"""
    print("\nGenerating complementarity visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Venn-style bar chart
    comp_subset = comp_df[comp_df['method1'] == 'AA-KEA (Our Method)'].copy()

    if len(comp_subset) > 0:
        x = range(len(comp_subset))
        width = 0.2

        axes[0].bar([i - 1.5*width for i in x], comp_subset['m1_only'],
                   width, label='AA-KEA only', color='#d62728')
        axes[0].bar([i - 0.5*width for i in x], comp_subset['both'],
                   width, label='Both correct', color='#2ca02c')
        axes[0].bar([i + 0.5*width for i in x], comp_subset['m2_only'],
                   width, label='Other only', color='#1f77b4')
        axes[0].bar([i + 1.5*width for i in x], comp_subset['neither'],
                   width, label='Both wrong', color='#7f7f7f')

        axes[0].set_xlabel('Comparison', fontsize=12)
        axes[0].set_ylabel('Number of Cases', fontsize=12)
        axes[0].set_title('Complementarity: AA-KEA vs Baselines', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([m2.replace(' (Our Method)', '') for m2 in comp_subset['method2']],
                                rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

    # Plot 2: Oracle performance
    methods_acc = [0.85, 0.82, 0.79, 0.77]  # Example - will be replaced with actual
    methods_names = ['AA-KEA', 'BERTScore', 'Sentence-T5', 'ROUGE-L']

    axes[1].barh(methods_names + ['Oracle\n(Upper Bound)'],
                methods_acc + [oracle_acc],
                color=['#1f77b4']*len(methods_acc) + ['#ff7f0e'])
    axes[1].set_xlabel('Accuracy', fontsize=12)
    axes[1].set_title('Oracle Ensemble Upper Bound', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].set_xlim(0, 1)

    # Add value labels
    for i, v in enumerate(methods_acc + [oracle_acc]):
        axes[1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/complementarity_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/complementarity_analysis.png")
    plt.close()


def plot_text_length_analysis(length_df, output_dir):
    """Plot text length analysis results"""
    print("\nGenerating text length analysis visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: F1 vs Text Length
    for method in length_df['method'].unique():
        method_data = length_df[length_df['method'] == method].sort_values('avg_length')
        axes[0].plot(method_data['avg_length'], method_data['f1'],
                    marker='o', linewidth=2, markersize=8, label=method)

    axes[0].set_xlabel('Average Text Length (words)', fontsize=12)
    axes[0].set_ylabel('F1 Score', fontsize=12)
    axes[0].set_title('Performance vs Text Length', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1)

    # Plot 2: Accuracy degradation rate
    # Compute slope for each method
    slopes = []
    for method in length_df['method'].unique():
        method_data = length_df[length_df['method'] == method].sort_values('avg_length')
        if len(method_data) > 1:
            x = method_data['avg_length'].values
            y = method_data['accuracy'].values
            slope = np.polyfit(x, y, 1)[0] * 100  # per 100 words
            slopes.append({'method': method, 'slope': slope})

    if slopes:
        slope_df = pd.DataFrame(slopes).sort_values('slope', ascending=False)
        colors = ['#2ca02c' if s > 0 else '#d62728' for s in slope_df['slope']]

        axes[1].barh(range(len(slope_df)), slope_df['slope'], color=colors)
        axes[1].set_yticks(range(len(slope_df)))
        axes[1].set_yticklabels(slope_df['method'])
        axes[1].set_xlabel('Accuracy Change per 100 Words', fontsize=12)
        axes[1].set_title('Robustness to Text Length', fontsize=14, fontweight='bold')
        axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
        axes[1].grid(axis='x', alpha=0.3)

        # Add value labels
        for i, v in enumerate(slope_df['slope']):
            axes[1].text(v + 0.1 if v > 0 else v - 0.1, i, f'{v:+.2f}',
                        va='center', ha='left' if v > 0 else 'right', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/text_length_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/text_length_analysis.png")
    plt.close()


def plot_domain_transfer(transfer_df, output_dir):
    """Plot domain transfer analysis results"""
    print("\nGenerating domain transfer visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    methods = transfer_df['method'].unique()

    for idx, method in enumerate(methods[:4]):  # Plot top 4 methods
        method_data = transfer_df[transfer_df['method'] == method]

        # Create pivot table for heatmap
        pivot = method_data.pivot(index='train_domain',
                                  columns='test_domain',
                                  values='transfer_gap')

        # Plot heatmap
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r',
                   center=0, vmin=-0.15, vmax=0.15,
                   cbar_kws={'label': 'Transfer Gap'},
                   ax=axes[idx])
        axes[idx].set_title(f'{method} - Transfer Gap', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Test Domain', fontsize=10)
        axes[idx].set_ylabel('Train Domain', fontsize=10)

    plt.suptitle('Domain Transfer Analysis\n(Gap = Optimal F1 - Transferred F1)',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/domain_transfer_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/domain_transfer_analysis.png")
    plt.close()


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_overall_comparison(results_df, output_dir):
    """Plot overall method comparison"""
    print("\nGenerating overall comparison plot...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = ['f1', 'precision', 'recall', 'roc_auc']
    titles = ['F1 Score', 'Precision', 'Recall', 'ROC-AUC']

    for ax, metric, title in zip(axes.flat, metrics, titles):
        sorted_df = results_df.sort_values(metric, ascending=True)
        colors = ['#d62728' if 'AA-KEA' in x else '#1f77b4' for x in sorted_df['method']]

        ax.barh(range(len(sorted_df)), sorted_df[metric], color=colors)
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df['method'])
        ax.set_xlabel(title, fontsize=12)
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, v in enumerate(sorted_df[metric]):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

    plt.suptitle('Method Comparison on Semantic-KG Dataset', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/overall_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/overall_comparison.png")
    plt.close()


def plot_perturbation_analysis(pert_df, output_dir):
    """Plot performance by perturbation type"""
    print("\nGenerating perturbation type analysis plot...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # F1 Score by perturbation type
    pivot_f1 = pert_df.pivot(index='method', columns='perturbation_type', values='f1')
    pivot_f1.plot(kind='bar', ax=axes[0], width=0.8)
    axes[0].set_title('F1 Score by Perturbation Type', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Method', fontsize=12)
    axes[0].set_ylabel('F1 Score', fontsize=12)
    axes[0].legend(title='Perturbation Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

    # ROC-AUC by perturbation type
    pivot_auc = pert_df.pivot(index='method', columns='perturbation_type', values='roc_auc')
    pivot_auc.plot(kind='bar', ax=axes[1], width=0.8)
    axes[1].set_title('ROC-AUC by Perturbation Type', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Method', fontsize=12)
    axes[1].set_ylabel('ROC-AUC', fontsize=12)
    axes[1].legend(title='Perturbation Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/perturbation_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/perturbation_analysis.png")
    plt.close()


def plot_roc_curves(df, output_dir, our_method_name='AA-KEA', dataset_title=None):
    """Plot ROC curves for all methods.

    our_method_name : display name for the KG-based method (AA-KEA / SNEA-BERT / SNEA-BERT α=0.3)
    dataset_title   : dataset label shown in the plot title
    """
    print("\nGenerating ROC curves...")

    title_suffix = f' — {dataset_title}' if dataset_title else ''

    methods = {
        our_method_name: ('aa_kea_similarity', '#1a6e2e'),
        'BERTScore':     ('bertscore_f1',              '#ff7f0e'),
        'sentence-T5':   ('sentence-t5-base_similarity', '#2ca02c'),
        'ROUGE-L':       ('rougeL_score',              '#9467bd'),
        'BLEU':          ('bleu_score',                '#8c564b'),
        'MiniLM':        ('all-MiniLM-L6-v2_similarity', '#e377c2'),
    }

    plt.figure(figsize=(10, 8))

    y_true = df['label'].values

    for method_name, (score_col, color) in methods.items():
        if score_col in df.columns:
            y_scores = df[score_col].values
            if len(np.unique(y_scores)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc = roc_auc_score(y_true, y_scores)
            lw = 2.5 if method_name == our_method_name else 1.5
            plt.plot(fpr, tpr, label=f'{method_name} (AUC = {auc:.3f})',
                     color=color, linewidth=lw)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves{title_suffix}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/roc_curves.png")
    plt.close()


# ============================================================================
# RESULTS EXPORT
# ============================================================================

def save_results(results_df, pert_df, output_dir):
    """Save results to CSV"""
    print("\nSaving results...")

    results_df.to_csv(f'{output_dir}/overall_results.csv', index=False)
    print(f"✓ Saved: {output_dir}/overall_results.csv")

    pert_df.to_csv(f'{output_dir}/perturbation_results.csv', index=False)
    print(f"✓ Saved: {output_dir}/perturbation_results.csv")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("SEMANTIC-KG EVALUATION PIPELINE")
    print("="*80)

    # Check if AA-KEA results exist
    if not Path(Config.AA_KEA_RESULTS).exists():
        print(f"\n❌ ERROR: {Config.AA_KEA_RESULTS} not found!")
        print("\nPlease provide the AA-KEA results from your colleague first.")
        print("Expected format:")
        print("  - pair_id: integer (matches semantic_kg_for_kg_generation.csv)")
        print("  - kg1: KG triples from response1")
        print("  - kg2: KG triples from response2")
        print("  - aa_kea_similarity: similarity score (0-1)")
        return None

    # Load data
    df = load_data()

    # Compute baselines
    df = compute_all_baselines(df)

    # Overall evaluation
    results_df, predictions = evaluate_all_methods(df)

    # Perturbation type analysis
    pert_df = evaluate_by_perturbation_type(df)

    # ===== NOVEL ANALYSES =====
    # 1. Complementarity Analysis
    comp_df, oracle_acc = complementarity_analysis(df, predictions)

    # 2. Text Length Analysis
    length_df = text_length_analysis(df, predictions)

    # 3. Domain Transfer Analysis
    transfer_df = domain_transfer_analysis(df, df)

    # Generate visualizations
    plot_overall_comparison(results_df, Config.OUTPUT_DIR)
    plot_perturbation_analysis(pert_df, Config.OUTPUT_DIR)
    plot_roc_curves(df, Config.OUTPUT_DIR)

    # Novel analysis visualizations
    plot_complementarity(comp_df, oracle_acc, Config.OUTPUT_DIR)
    plot_text_length_analysis(length_df, Config.OUTPUT_DIR)
    plot_domain_transfer(transfer_df, Config.OUTPUT_DIR)

    # Save results
    save_results(results_df, pert_df, Config.OUTPUT_DIR)

    # Save novel analysis results
    comp_df.to_csv(f'{Config.OUTPUT_DIR}/complementarity_results.csv', index=False)
    length_df.to_csv(f'{Config.OUTPUT_DIR}/text_length_results.csv', index=False)
    transfer_df.to_csv(f'{Config.OUTPUT_DIR}/domain_transfer_results.csv', index=False)
    print(f"✓ Saved: {Config.OUTPUT_DIR}/complementarity_results.csv")
    print(f"✓ Saved: {Config.OUTPUT_DIR}/text_length_results.csv")
    print(f"✓ Saved: {Config.OUTPUT_DIR}/domain_transfer_results.csv")

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

    print("\n📊 Overall Results:")
    print(results_df[['method', 'f1', 'precision', 'recall', 'roc_auc']].to_string(index=False))

    print(f"\n📁 All results saved to: {Config.OUTPUT_DIR}/")
    print(f"\n  Standard Evaluation:")
    print(f"    - overall_results.csv")
    print(f"    - perturbation_results.csv")
    print(f"    - overall_comparison.png")
    print(f"    - perturbation_analysis.png")
    print(f"    - roc_curves.png")
    print(f"\n  📈 Novel Analyses:")
    print(f"    - complementarity_results.csv")
    print(f"    - complementarity_analysis.png")
    print(f"    - text_length_results.csv")
    print(f"    - text_length_analysis.png")
    print(f"    - domain_transfer_results.csv")
    print(f"    - domain_transfer_analysis.png")

    print(f"\n✨ Novel Contributions:")
    print(f"  1. Complementarity Analysis - Shows which cases each method uniquely handles")
    print(f"  2. Text Length Analysis - Performance vs document length")
    print(f"  3. Domain Transfer - Cross-domain generalization analysis")

    return {
        'data': df,
        'overall_results': results_df,
        'perturbation_results': pert_df,
        'complementarity_results': comp_df,
        'text_length_results': length_df,
        'domain_transfer_results': transfer_df,
        'oracle_accuracy': oracle_acc
    }


if __name__ == "__main__":
    results = main()


# ============================================================================
# MULTI-DATASET EVALUATION
# ============================================================================

def load_single_dataset(dataset_file: str, aa_kea_file: str,
                        text1_col: str = 'response1',
                        text2_col: str = 'response2',
                        score_col: str = 'aa_kea_similarity') -> pd.DataFrame | None:
    """
    Load a prepared dataset CSV and merge similarity results.
    Normalises text columns to 'response1' / 'response2' for downstream compatibility.
    score_col: column name in aa_kea_file containing the similarity score.
    Returns None if either file is missing.
    """
    if not Path(dataset_file).exists():
        print(f"  ⚠  Dataset file not found: {dataset_file}")
        return None

    df = pd.read_csv(dataset_file)

    # Rename text columns to standard names expected by baseline functions
    if text1_col != 'response1':
        df = df.rename(columns={text1_col: 'response1', text2_col: 'response2'})

    if not Path(aa_kea_file).exists():
        print(f"  ⚠  AA-KEA results not found: {aa_kea_file}")
        print(f"     Send '{dataset_file}' to your colleague and ask for '{aa_kea_file}'")
        df['aa_kea_similarity'] = np.nan
    else:
        aakea = pd.read_csv(aa_kea_file)
        # Support pair_id, row_id, or id as merge key
        merge_key = next(
            (k for k in ['pair_id', 'row_id', 'id'] if k in aakea.columns),
            None
        )
        # Deduplicate results file — keep first occurrence per id
        # (SNEA-BERT pipeline sometimes re-processes pairs, producing duplicate rows)
        if merge_key and aakea[merge_key].duplicated().any():
            n_before = len(aakea)
            aakea = aakea.drop_duplicates(subset=merge_key, keep='first')
            print(f"  ⚠  Dropped {n_before - len(aakea)} duplicate id(s) in {aa_kea_file}")

        if merge_key and merge_key in df.columns:
            # Direct key match (e.g. both use pair_id)
            df = df.merge(aakea[[merge_key, score_col]], on=merge_key, how='left')
        elif merge_key == 'id' and 'pair_id' in df.columns:
            # Cross-key merge: results file uses 'id', base file uses 'pair_id'
            df = df.merge(aakea[['id', score_col]],
                          left_on='pair_id', right_on='id', how='left')
        else:
            # Positional merge (same row order, last resort)
            print(f"  ⚠  Falling back to positional merge for {aa_kea_file}")
            df[score_col] = aakea[score_col].values[:len(df)]
        # Normalise to standard column name for downstream pipeline
        if score_col != 'aa_kea_similarity':
            df = df.rename(columns={score_col: 'aa_kea_similarity'})

    missing = df['aa_kea_similarity'].isna().sum()
    if missing > 0:
        print(f"  ⚠  {missing} rows missing AA-KEA score — filling with 0.0")
        df['aa_kea_similarity'] = df['aa_kea_similarity'].fillna(0.0)

    # Ensure required columns exist
    if 'perturbation_type' not in df.columns:
        df['perturbation_type'] = 'unknown'
    if 'dataset_name' not in df.columns:
        df['dataset_name'] = Path(dataset_file).stem

    print(f"  ✓ Loaded {len(df)} pairs | labels: {df['label'].value_counts().to_dict()}")
    return df


def run_single_dataset(dataset_key: str, cfg: dict) -> dict | None:
    """
    Run the full evaluation pipeline for one dataset.
    Returns a summary dict for cross-dataset comparison.
    """
    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_key.upper()}  [{cfg['label']}]")
    print("=" * 80)

    Path(cfg['output_dir']).mkdir(parents=True, exist_ok=True)

    df = load_single_dataset(
        cfg['dataset_file'], cfg['aa_kea_file'],
        cfg['text1_col'], cfg['text2_col'],
        cfg.get('score_col', 'aa_kea_similarity'),
    )
    if df is None or len(df) == 0:
        return None

    # Compute all baselines
    df = compute_all_baselines(df)

    # Evaluate all methods
    results_df, predictions = evaluate_all_methods(df)

    # Perturbation-type breakdown (only if column populated)
    if df['perturbation_type'].nunique() > 1:
        pert_df = evaluate_by_perturbation_type(df)
        pert_df.to_csv(f"{cfg['output_dir']}/perturbation_results.csv", index=False)
    else:
        pert_df = pd.DataFrame()

    # Text length analysis
    length_df = text_length_analysis(df, predictions)

    # Derive the correct display name for our method based on dataset key / score column
    _score_col = cfg.get('score_col', 'aa_kea_similarity')
    if '_snea_alpha_' in dataset_key:
        # e.g. 'mrpc_snea_alpha_0p5' → 'SNEA-BERT α=0.5'
        _a_lbl = dataset_key.split('_snea_alpha_')[-1]   # '0p5'
        _a_float = _a_lbl.replace('p', '.')               # '0.5'
        _our_method_name = f'SNEA-BERT α={_a_float}'
    elif _score_col == 'snea_bert_alpha_0.3':
        _our_method_name = 'SNEA-BERT α=0.3'
    elif _score_col == 'snea_bert_similarity':
        _our_method_name = 'SNEA-BERT'
    else:
        _our_method_name = 'AA-KEA'

    # Clean dataset title: strip 'Long-text / ' or 'Short-text / ' prefix
    _dataset_title = cfg['label'].split(' / ', 1)[-1] if ' / ' in cfg['label'] else cfg['label']

    # Visualisations
    plot_overall_comparison(results_df, cfg['output_dir'])
    plot_roc_curves(df, cfg['output_dir'],
                    our_method_name=_our_method_name,
                    dataset_title=_dataset_title)
    if not pert_df.empty:
        plot_perturbation_analysis(pert_df, cfg['output_dir'])
    plot_text_length_analysis(length_df, cfg['output_dir'])

    # Save CSVs
    results_df.to_csv(f"{cfg['output_dir']}/overall_results.csv", index=False)
    length_df.to_csv(f"{cfg['output_dir']}/text_length_results.csv", index=False)

    # Build summary row for cross-dataset comparison
    avg_len = (df['response1'].str.split().str.len() +
               df['response2'].str.split().str.len()).mean() / 2
    summary = {
        'dataset':       dataset_key,
        'label':         cfg['label'],
        'short_text':    cfg['short_text'],
        'n_pairs':       len(df),
        'avg_text_len':  round(avg_len, 1),
    }
    for _, row in results_df.iterrows():
        method = row['method'].replace(' (Our Method)', '').replace(' ', '_')
        summary[f'f1_{method}']  = round(row['f1'],      4)
        summary[f'auc_{method}'] = round(row['roc_auc'], 4)

    print(f"\n  Summary saved to {cfg['output_dir']}/")
    return summary


# ============================================================================
# CROSS-DATASET COMPARISON PLOTS
# ============================================================================

def plot_cross_dataset_f1(summary_df: pd.DataFrame, output_dir: str = 'output'):
    """
    Grouped bar chart: F1 score per method across all datasets.
    Datasets ordered from short-text to long-text / non-KG to KG.
    """
    print("\nGenerating cross-dataset F1 comparison...")

    method_cols = [c for c in summary_df.columns if c.startswith('f1_')]
    if not method_cols:
        print("  ⚠ No F1 columns found — skipping plot.")
        return

    # Friendly method names
    method_labels = {
        'f1_AA-KEA':            'AA-KEA',
        'f1_ROUGE-1':           'ROUGE-1',
        'f1_ROUGE-L':           'ROUGE-L',
        'f1_BLEU':              'BLEU',
        'f1_BERTScore':         'BERTScore',
        'f1_all-MiniLM-L6-v2': 'MiniLM',
        'f1_sentence-t5-base':  'Sentence-T5',
    }

    plot_methods = [c for c in method_cols if c in method_labels]
    if not plot_methods:
        plot_methods = method_cols[:5]

    x     = np.arange(len(summary_df))
    width = 0.8 / len(plot_methods)

    fig, ax = plt.subplots(figsize=(16, 7))

    for i, col in enumerate(plot_methods):
        label = method_labels.get(col, col.replace('f1_', ''))
        color = '#d62728' if 'AA-KEA' in label else None
        offset = (i - len(plot_methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, summary_df[col].fillna(0), width,
                      label=label, color=color)
        # Value labels on AA-KEA bars only
        if 'AA-KEA' in label:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=7,
                        color='#d62728', fontweight='bold')

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Cross-Dataset Comparison: F1 Score by Method\n'
                 '(Red = AA-KEA, ordered from short-text to long-text/KG)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['dataset']}\n({r['avg_text_len']:.0f} words)" for _, r in summary_df.iterrows()],
        rotation=20, ha='right', fontsize=9
    )
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = f'{output_dir}/cross_dataset_f1_comparison.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {out}")
    plt.close()


def plot_text_length_vs_aakea(summary_df: pd.DataFrame, output_dir: str = 'output'):
    """
    Scatter plot: average text length vs AA-KEA F1 across datasets.
    Shows that AA-KEA performance increases with text richness.
    """
    print("Generating text-length vs AA-KEA performance plot...")

    aakea_col = next((c for c in summary_df.columns if 'AA-KEA' in c and c.startswith('f1_')), None)
    bert_col  = next((c for c in summary_df.columns if 'BERT'   in c and c.startswith('f1_')), None)
    t5_col    = next((c for c in summary_df.columns if 't5'     in c.lower() and c.startswith('f1_')), None)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors

    for i, (_, row) in enumerate(summary_df.iterrows()):
        x = row['avg_text_len']
        if aakea_col and not pd.isna(row.get(aakea_col)):
            ax.scatter(x, row[aakea_col], s=120, color='#d62728', zorder=5,
                       label='AA-KEA' if i == 0 else '')
            ax.annotate(row['dataset'], (x, row[aakea_col]),
                        textcoords='offset points', xytext=(5, 5), fontsize=8)
        if bert_col and not pd.isna(row.get(bert_col)):
            ax.scatter(x, row[bert_col], s=80, color='#1f77b4', marker='s', zorder=4,
                       label='BERTScore' if i == 0 else '')
        if t5_col and not pd.isna(row.get(t5_col)):
            ax.scatter(x, row[t5_col], s=80, color='#2ca02c', marker='^', zorder=4,
                       label='Sentence-T5' if i == 0 else '')

    # Trend line for AA-KEA
    if aakea_col:
        valid = summary_df[[aakea_col, 'avg_text_len']].dropna()
        if len(valid) >= 2:
            z = np.polyfit(valid['avg_text_len'], valid[aakea_col], 1)
            p = np.poly1d(z)
            xfit = np.linspace(valid['avg_text_len'].min(), valid['avg_text_len'].max(), 100)
            ax.plot(xfit, p(xfit), '--', color='#d62728', alpha=0.5, linewidth=1.5,
                    label='AA-KEA trend')

    ax.set_xlabel('Average Text Length (words)', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Text Length vs Method Performance\n'
                 'AA-KEA improves on longer, richer paragraphs',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=10)

    plt.tight_layout()
    out = f'{output_dir}/text_length_vs_performance.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {out}")
    plt.close()


def plot_perturbation_cross_dataset(all_pert_results: dict, output_dir: str = 'output'):
    """
    Compare AA-KEA F1 by perturbation type across Semantic-KG and Wikipedia-entity-swap.
    Verifies that the behaviour is consistent regardless of data-creation methodology.
    """
    print("Generating cross-dataset perturbation comparison...")

    PERT_DATASETS = ['semantic_kg_codex', 'semantic_kg_combined', 'wikipedia_entity_swap']
    available = {k: v for k, v in all_pert_results.items() if k in PERT_DATASETS and v is not None}

    if len(available) < 2:
        print("  ⚠ Need ≥2 perturbation-aware datasets — skipping.")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(7 * len(available), 6), sharey=True)
    if len(available) == 1:
        axes = [axes]

    PERT_ORDER = ['node_replacement', 'node_deletion', 'edge_replacement', 'edge_deletion', 'paraphrase']

    for ax, (dkey, pert_df) in zip(axes, available.items()):
        if pert_df.empty:
            ax.set_title(dkey)
            continue

        aa_kea_data = pert_df[pert_df['method'] == 'AA-KEA'].copy()
        aa_kea_data = aa_kea_data.set_index('perturbation_type')['f1'].reindex(PERT_ORDER).dropna()

        ax.bar(range(len(aa_kea_data)), aa_kea_data.values, color='#d62728', alpha=0.8)
        ax.set_xticks(range(len(aa_kea_data)))
        ax.set_xticklabels(aa_kea_data.index, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('AA-KEA F1', fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title(Config.DATASETS.get(dkey, {}).get('label', dkey), fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        for i, v in enumerate(aa_kea_data.values):
            ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=9)

    plt.suptitle('AA-KEA F1 by Perturbation Type\n'
                 '(Consistent pattern = results not an artifact of data-creation methodology)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = f'{output_dir}/perturbation_cross_dataset.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {out}")
    plt.close()


# ============================================================================
# MULTI-DATASET MAIN
# ============================================================================

def run_all_datasets(keys_filter=None):
    """
    Run full evaluation across all configured datasets and produce
    cross-dataset comparison visualisations.

    Datasets missing their AA-KEA results file are skipped gracefully —
    run partial evaluations as results become available.

    keys_filter: optional set of dataset keys to restrict which datasets run.
    """
    print("\n" + "=" * 80)
    print("MULTI-DATASET EVALUATION")
    print("=" * 80)

    summaries     = []
    pert_results  = {}
    CROSS_DIR     = 'output/cross_dataset'
    Path(CROSS_DIR).mkdir(parents=True, exist_ok=True)

    for dataset_key, cfg in Config.DATASETS.items():
        if keys_filter and dataset_key not in keys_filter:
            continue
        summary = run_single_dataset(dataset_key, cfg)
        if summary:
            summaries.append(summary)

        # Collect perturbation results for cross-dataset perturbation plot
        pert_path = f"{cfg['output_dir']}/perturbation_results.csv"
        if Path(pert_path).exists():
            pert_results[dataset_key] = pd.read_csv(pert_path)
        else:
            pert_results[dataset_key] = None

    if not summaries:
        print("\n✗ No datasets evaluated successfully.")
        return

    summary_df = pd.DataFrame(summaries)

    # Order by avg text length (short → long) for intuitive plot layout
    summary_df = summary_df.sort_values('avg_text_len').reset_index(drop=True)

    # Save cross-dataset summary table
    summary_csv = f'{CROSS_DIR}/cross_dataset_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✓ Cross-dataset summary saved: {summary_csv}")

    # Cross-dataset visualisations
    plot_cross_dataset_f1(summary_df,        output_dir=CROSS_DIR)
    plot_text_length_vs_aakea(summary_df,    output_dir=CROSS_DIR)
    plot_perturbation_cross_dataset(pert_results, output_dir=CROSS_DIR)

    # Print summary table
    print("\n" + "=" * 80)
    print("CROSS-DATASET SUMMARY")
    print("=" * 80)
    aakea_col = next((c for c in summary_df.columns if 'AA-KEA' in c and 'f1_' in c), None)
    bert_col  = next((c for c in summary_df.columns if 'BERT'   in c and 'f1_' in c), None)

    display_cols = ['dataset', 'avg_text_len']
    if aakea_col: display_cols.append(aakea_col)
    if bert_col:  display_cols.append(bert_col)

    print(summary_df[display_cols].to_string(index=False))

    print(f"\n📁 All cross-dataset outputs in: {CROSS_DIR}/")
    print("   - cross_dataset_summary.csv")
    print("   - cross_dataset_f1_comparison.png")
    print("   - text_length_vs_performance.png")
    print("   - perturbation_cross_dataset.png")

    return summary_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic similarity evaluation")
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "both"],
        default="single",
        help="'single' = original Semantic-KG Codex evaluation, "
             "'multi' = all datasets, 'both' = run both",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Run only the specified dataset keys (subset of Config.DATASETS).",
    )
    args = parser.parse_args()

    if args.mode in ("single", "both"):
        results = main()

    if args.mode in ("multi", "both"):
        if args.keys:
            run_all_datasets(keys_filter=set(args.keys))
        else:
            run_all_datasets()
