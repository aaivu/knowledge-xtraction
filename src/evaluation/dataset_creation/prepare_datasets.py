"""
prepare_datasets.py

Downloads and prepares benchmark datasets for AA-KEA evaluation.

Datasets prepared:
  1. MRPC (GLUE)        - 400 pairs (200 similar + 200 dissimilar)
  2. PAWS-Wiki          - 400 pairs (200 similar + 200 dissimilar)
  3. Semantic-KG        - 100 random pairs per domain (codex/findkg/globi/oregano) = 400 total

Output format (matches semantic_kg_for_kg_generation.csv):
  pair_id, response1, response2, label, perturbation_type, dataset_name

Outputs saved to: datasets/
  - datasets/mrpc_400.csv
  - datasets/paws_wiki_400.csv
  - datasets/semantic_kg_combined_400.csv
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = Path("datasets")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# HELPERS
# ============================================================================

def save_dataset(df, filename, name):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"  ✓ Saved {name}: {path}")
    print(f"    {len(df)} pairs | label dist: {df['label'].value_counts().to_dict()}")
    return path


def sample_balanced(df, label_col, n_each, seed=SEED):
    """Sample n_each rows per label value."""
    parts = []
    for val in df[label_col].unique():
        subset = df[df[label_col] == val]
        n = min(n_each, len(subset))
        parts.append(subset.sample(n=n, random_state=seed))
    result = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return result


# ============================================================================
# MRPC
# ============================================================================

def prepare_mrpc(n_each=200):
    """Download GLUE MRPC and prepare 400 balanced pairs."""
    print("\n" + "="*60)
    print("Preparing MRPC...")

    # MRPC train + validation combined (~4k pairs)
    train = load_dataset("glue", "mrpc", split="train").to_pandas()
    val   = load_dataset("glue", "mrpc", split="validation").to_pandas()
    raw   = pd.concat([train, val], ignore_index=True)

    print(f"  Raw size: {len(raw)} | labels: {raw['label'].value_counts().to_dict()}")

    sampled = sample_balanced(raw, 'label', n_each)

    result = pd.DataFrame({
        'pair_id':          range(len(sampled)),
        'response1':        sampled['sentence1'].values,
        'response2':        sampled['sentence2'].values,
        'label':            sampled['label'].values,
        'perturbation_type': sampled['label'].map({1: 'paraphrase', 0: 'non_paraphrase'}).values,
        'dataset_name':     'mrpc',
    })

    save_dataset(result, 'mrpc_400.csv', 'MRPC')
    return result


# ============================================================================
# PAWS-Wiki
# ============================================================================

def prepare_paws_wiki(n_each=200):
    """Download PAWS labeled_final and prepare 400 balanced pairs."""
    print("\n" + "="*60)
    print("Preparing PAWS-Wiki...")

    raw = load_dataset("paws", "labeled_final", split="train").to_pandas()
    print(f"  Raw size: {len(raw)} | labels: {raw['label'].value_counts().to_dict()}")

    sampled = sample_balanced(raw, 'label', n_each)

    result = pd.DataFrame({
        'pair_id':          range(len(sampled)),
        'response1':        sampled['sentence1'].values,
        'response2':        sampled['sentence2'].values,
        'label':            sampled['label'].values,
        'perturbation_type': sampled['label'].map({1: 'paraphrase', 0: 'non_paraphrase'}).values,
        'dataset_name':     'paws_wiki',
    })

    save_dataset(result, 'paws_wiki_400.csv', 'PAWS-Wiki')
    return result


# ============================================================================
# SEMANTIC-KG (HuggingFace: QiyaoWei/Semantic-KG)
# ============================================================================

# Column name candidates for each field
_TEXT1_CANDIDATES  = ['statement_1', 'sentence_1', 'text_1', 'response1', 'paragraph_1', 'text1', 'sent1']
_TEXT2_CANDIDATES  = ['statement_2', 'sentence_2', 'text_2', 'response2', 'paragraph_2', 'text2', 'sent2']
_LABEL_CANDIDATES  = ['label', 'labels', 'similarity_label', 'is_similar']
_PERT_CANDIDATES   = ['perturbation_type', 'perturbation', 'pert_type', 'type']


def _pick_col(columns, candidates, fallback_idx=0):
    for c in candidates:
        if c in columns:
            return c
    return columns[fallback_idx]


def prepare_semantic_kg(n_per_domain=100):
    """
    Download QiyaoWei/Semantic-KG for all 4 domains and sample 100 pairs each.
    Tries domain-specific configs first, falls back to single dataset with domain filter.
    """
    print("\n" + "="*60)
    print("Preparing Semantic-KG (4 domains, 100 each)...")

    domains = ['codex', 'findkg', 'globi', 'oregano']
    all_dfs = []

    for domain in domains:
        print(f"\n  [{domain}]")
        df_domain = _load_semantic_kg_domain(domain, n_per_domain)
        if df_domain is not None:
            all_dfs.append(df_domain)
        else:
            print(f"  ✗ Skipping {domain}")

    if not all_dfs:
        print("  ✗ Could not load any Semantic-KG domain — check dataset availability.")
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined['pair_id'] = range(len(combined))

    save_dataset(combined, 'semantic_kg_combined_400.csv', 'Semantic-KG (all domains)')
    return combined


def _load_semantic_kg_domain(domain, n):
    """Try multiple loading strategies for a Semantic-KG domain."""

    # Strategy 1: named config  e.g. load_dataset("QiyaoWei/Semantic-KG", "codex")
    try:
        raw = load_dataset("QiyaoWei/Semantic-KG", domain, split="train").to_pandas()
        print(f"    Loaded via config '{domain}': {len(raw)} rows | cols: {raw.columns.tolist()}")
        return _format_semantic_kg_rows(raw, domain, n)
    except Exception as e:
        print(f"    Config load failed: {e}")

    # Strategy 2: default split, filter by domain column
    try:
        raw = load_dataset("QiyaoWei/Semantic-KG", split="train").to_pandas()
        domain_col = next((c for c in raw.columns if 'domain' in c.lower() or 'dataset' in c.lower()), None)
        if domain_col:
            raw = raw[raw[domain_col].str.lower() == domain]
        print(f"    Loaded via default split filtered on '{domain_col}': {len(raw)} rows")
        return _format_semantic_kg_rows(raw, domain, n)
    except Exception as e:
        print(f"    Default split failed: {e}")

    return None


def _format_semantic_kg_rows(raw, domain, n):
    """Map raw Semantic-KG columns to standard format and sample n rows."""
    cols = raw.columns.tolist()

    t1   = _pick_col(cols, _TEXT1_CANDIDATES, 0)
    t2   = _pick_col(cols, _TEXT2_CANDIDATES, 1)
    lbl  = _pick_col(cols, _LABEL_CANDIDATES, -1)
    pert = _pick_col(cols, _PERT_CANDIDATES, None) if any(c in cols for c in _PERT_CANDIDATES) else None

    print(f"    Mapped cols → text1='{t1}', text2='{t2}', label='{lbl}', pert='{pert}'")

    # Drop rows with missing text
    raw = raw.dropna(subset=[t1, t2])

    sample = raw.sample(n=min(n, len(raw)), random_state=SEED).reset_index(drop=True)

    result = pd.DataFrame({
        'pair_id':           range(len(sample)),
        'response1':         sample[t1].astype(str).values,
        'response2':         sample[t2].astype(str).values,
        'label':             sample[lbl].astype(int).values,
        'perturbation_type': sample[pert].astype(str).values if pert else 'unknown',
        'dataset_name':      domain,
    })

    print(f"    ✓ {len(result)} pairs | labels: {result['label'].value_counts().to_dict()}")
    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("BENCHMARK DATASET PREPARATION")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    results = {}

    try:
        results['mrpc'] = prepare_mrpc()
    except Exception as e:
        print(f"\n✗ MRPC failed: {e}")

    try:
        results['paws_wiki'] = prepare_paws_wiki()
    except Exception as e:
        print(f"\n✗ PAWS-Wiki failed: {e}")

    try:
        results['semantic_kg'] = prepare_semantic_kg()
    except Exception as e:
        print(f"\n✗ Semantic-KG failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, df in results.items():
        if df is not None:
            print(f"  {name:30s} {len(df):>4d} pairs")
    print(f"\nAll files saved to: {OUTPUT_DIR.resolve()}/")

    print("\nNext step: send CSV files to colleague to run AA-KEA.")
    print("Expected AA-KEA result filenames (place in datasets/):")
    for name in results:
        print(f"  datasets/{name}_aa_kea_results.csv")


if __name__ == "__main__":
    main()
