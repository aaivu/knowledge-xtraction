"""
prepare_new_datasets.py

Prepares 5 new benchmark datasets for AA-KEA evaluation:

  1. Semantic-KG / Codex    — 400 pairs (200 similar + 200 dissimilar)
  2. Semantic-KG / FinDKG   — 400 pairs (200 similar + 200 dissimilar)
  3. Semantic-KG / GloBI    — 400 pairs (200 similar + 200 dissimilar)
  4. Semantic-KG / Oregano  — 400 pairs (200 similar + 200 dissimilar)
  5. STS-12                 — 400 pairs (200 similar + 200 dissimilar)
                              binary threshold: score >= 3.0 → label=1

Source:
  Semantic-KG : HuggingFace QiyaoWei/Semantic-KG (per-domain configs)
  STS-12      : HuggingFace mteb/sts12-sts (SemEval 2012 combined test)

Output format (matches existing pipeline):
  pair_id, response1, response2, label, perturbation_type, dataset_name

Outputs:
  datasets/semantic_kg_codex_400.csv
  datasets/semantic_kg_findkg_400.csv
  datasets/semantic_kg_globi_400.csv
  datasets/semantic_kg_oregano_400.csv
  datasets/sts12_400.csv
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = Path("datasets")
OUTPUT_DIR.mkdir(exist_ok=True)

N_EACH = 200      # per label → 400 total per dataset
STS_THRESHOLD = 3.0   # score >= threshold → similar (label=1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_dataset(df: pd.DataFrame, filename: str, name: str) -> None:
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    dist = df['label'].value_counts().to_dict()
    print(f"  ✓ Saved {name}: {path}")
    print(f"    {len(df)} pairs | label dist: {dist}")


def sample_balanced(df: pd.DataFrame, label_col: str, n_each: int) -> pd.DataFrame:
    """Sample n_each rows per label; takes as many as available if fewer exist."""
    parts = []
    for val in sorted(df[label_col].unique()):
        subset = df[df[label_col] == val]
        n = min(n_each, len(subset))
        parts.append(subset.sample(n=n, random_state=SEED))
        print(f"    label={val}: {len(subset)} available → sampled {n}")
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


_TEXT1   = ['statement_1', 'sentence_1', 'text_1', 'response1', 'paragraph_1', 'text1', 'sent1']
_TEXT2   = ['statement_2', 'sentence_2', 'text_2', 'response2', 'paragraph_2', 'text2', 'sent2']
_LABEL   = ['label', 'labels', 'similarity_label', 'is_similar']
_PERT    = ['perturbation_type', 'perturbation', 'pert_type', 'type']


def _pick(columns: list, candidates: list, default: str | None = None) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return default


# ─────────────────────────────────────────────────────────────────────────────
# Semantic-KG domains  (QiyaoWei/Semantic-KG)
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_MAP = {
    'codex':   ('semantic_kg_codex_400.csv',   'semantic_kg_codex'),
    'findkg':  ('semantic_kg_findkg_400.csv',  'semantic_kg_findkg'),
    'globi':   ('semantic_kg_globi_400.csv',   'semantic_kg_globi'),
    'oregano': ('semantic_kg_oregano_400.csv', 'semantic_kg_oregano'),
}


def _load_domain_raw(domain: str) -> pd.DataFrame | None:
    """Try loading a single Semantic-KG domain config from HuggingFace."""

    # Strategy 1 — named config + train split
    for split in ('train', 'test', 'validation'):
        try:
            raw = load_dataset(
                "QiyaoWei/Semantic-KG", domain, split=split
            ).to_pandas()
            print(f"    Loaded config='{domain}' split='{split}': "
                  f"{len(raw)} rows | cols: {raw.columns.tolist()}")
            return raw
        except Exception as e:
            print(f"    config='{domain}' split='{split}' failed: {e}")

    # Strategy 2 — combined dataset filtered by domain column
    for split in ('train', 'test'):
        try:
            raw = load_dataset("QiyaoWei/Semantic-KG", split=split).to_pandas()
            dom_col = next(
                (c for c in raw.columns
                 if any(k in c.lower() for k in ('domain', 'dataset', 'source'))),
                None,
            )
            if dom_col:
                raw = raw[raw[dom_col].str.lower().str.contains(domain, na=False)]
            print(f"    Combined split='{split}' filtered on '{dom_col}': "
                  f"{len(raw)} rows")
            if len(raw) > 0:
                return raw
        except Exception as e:
            print(f"    Combined split='{split}' failed: {e}")

    return None


def prepare_semantic_kg_domain(domain: str, n_each: int = N_EACH) -> pd.DataFrame | None:
    filename, dataset_name = _DOMAIN_MAP[domain]
    print(f"\n{'='*60}")
    print(f"Preparing Semantic-KG / {domain.upper()} …")

    raw = _load_domain_raw(domain)
    if raw is None or len(raw) == 0:
        print(f"  ✗ Could not load {domain}")
        return None

    cols  = raw.columns.tolist()
    t1    = _pick(cols, _TEXT1)
    t2    = _pick(cols, _TEXT2)
    lbl   = _pick(cols, _LABEL)
    pert  = _pick(cols, _PERT)

    if not t1 or not t2 or not lbl:
        print(f"  ✗ Could not identify required columns. Available: {cols}")
        return None

    print(f"  Mapped → text1='{t1}'  text2='{t2}'  "
          f"label='{lbl}'  pert='{pert}'")

    raw = raw.dropna(subset=[t1, t2, lbl])
    raw[lbl] = raw[lbl].astype(int)

    print(f"  Label distribution before sampling:")
    sampled = sample_balanced(raw, lbl, n_each)

    result = pd.DataFrame({
        'pair_id':           range(len(sampled)),
        'response1':         sampled[t1].astype(str).values,
        'response2':         sampled[t2].astype(str).values,
        'label':             sampled[lbl].values,
        'perturbation_type': sampled[pert].astype(str).values if pert else 'unknown',
        'dataset_name':      dataset_name,
    })

    save_dataset(result, filename, f"Semantic-KG / {domain}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STS-12  (mteb/sts12-sts)
# Binary threshold: score >= STS_THRESHOLD → label=1 (similar)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_sts12(n_each: int = N_EACH) -> pd.DataFrame | None:
    print(f"\n{'='*60}")
    print("Preparing STS-12 …")

    raw = None

    # Try several known HuggingFace sources for STS-12
    sources = [
        ("mteb/sts12-sts",           "test"),
        ("mteb/sts12-sts",           "train"),
        ("stsb_multi_mt",            "en", "test"),
    ]

    for args in sources:
        try:
            if len(args) == 2:
                ds_name, split = args
                raw = load_dataset(ds_name, split=split).to_pandas()
            else:
                ds_name, config, split = args
                raw = load_dataset(ds_name, config, split=split).to_pandas()
            print(f"  Loaded '{ds_name}': {len(raw)} rows | cols: {raw.columns.tolist()}")
            if len(raw) > 0:
                break
        except Exception as e:
            print(f"  '{args[0]}' failed: {e}")

    if raw is None or len(raw) == 0:
        print("  ✗ Could not load STS-12")
        return None

    # Identify columns
    cols   = raw.columns.tolist()
    s1_col = _pick(cols, ['sentence1', 'sentence_1', 'sent1', 'text1', 's1'])
    s2_col = _pick(cols, ['sentence2', 'sentence_2', 'sent2', 'text2', 's2'])
    sc_col = _pick(cols, ['score', 'similarity_score', 'label', 'gold_score'])

    if not s1_col or not s2_col or not sc_col:
        print(f"  ✗ Cannot identify columns. Available: {cols}")
        return None

    print(f"  Mapped → s1='{s1_col}'  s2='{s2_col}'  score='{sc_col}'")
    print(f"  Score range: {raw[sc_col].min():.2f} – {raw[sc_col].max():.2f}")
    print(f"  Binary threshold: score >= {STS_THRESHOLD} → label=1")

    raw = raw.dropna(subset=[s1_col, s2_col, sc_col])

    # Normalise score to 0-5 if it looks like 0-1 range
    max_score = raw[sc_col].max()
    if max_score <= 1.0:
        raw[sc_col] = raw[sc_col] * 5.0
        print(f"  Score normalised ×5 (was 0–1 range)")

    raw['label'] = (raw[sc_col] >= STS_THRESHOLD).astype(int)

    print(f"  Label distribution before sampling:")
    sampled = sample_balanced(raw, 'label', n_each)

    result = pd.DataFrame({
        'pair_id':           range(len(sampled)),
        'response1':         sampled[s1_col].astype(str).values,
        'response2':         sampled[s2_col].astype(str).values,
        'label':             sampled['label'].values,
        'perturbation_type': sampled['label'].map(
            {1: 'similar', 0: 'dissimilar'}
        ).values,
        'dataset_name':      'sts12',
    })

    save_dataset(result, 'sts12_400.csv', 'STS-12')
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("NEW BENCHMARK DATASET PREPARATION")
    print("=" * 60)
    print(f"Target : {N_EACH * 2} pairs per dataset ({N_EACH} per label)")
    print(f"Output : {OUTPUT_DIR.resolve()}/")

    results = {}

    # Semantic-KG — 4 domains independently
    for domain in ('codex', 'findkg', 'globi', 'oregano'):
        try:
            results[domain] = prepare_semantic_kg_domain(domain)
        except Exception as e:
            print(f"\n✗ semantic_kg/{domain} failed: {e}")
            results[domain] = None

    # STS-12
    try:
        results['sts12'] = prepare_sts12()
    except Exception as e:
        print(f"\n✗ STS-12 failed: {e}")
        results['sts12'] = None

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, df in results.items():
        status = f"{len(df)} pairs" if df is not None else "FAILED"
        print(f"  {name:<20} {status}")

    print(f"\nAll files saved to: {OUTPUT_DIR.resolve()}/")
    print("\nExpected AA-KEA result filenames (place in datasets/):")
    filenames = {
        'codex':   'semantic_kg_codex_aa_kea_results.csv',
        'findkg':  'semantic_kg_findkg_aa_kea_results.csv',
        'globi':   'semantic_kg_globi_aa_kea_results.csv',
        'oregano': 'semantic_kg_oregano_aa_kea_results.csv',
        'sts12':   'sts12_aa_kea_results.csv',
    }
    for name, fname in filenames.items():
        print(f"  datasets/{fname}")


if __name__ == "__main__":
    main()