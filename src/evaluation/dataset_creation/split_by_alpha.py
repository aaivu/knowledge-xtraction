"""
split_by_alpha.py
=================
Reads each *_KGs_results.csv file from 'Results_All_Methods 2/' and splits
it into separate per-alpha result CSVs, one file per alpha value (0.0 → 1.0).

Only SNEA-BERT alpha columns are extracted. Each output file has columns:
    pair_id, snea_bert_similarity

Output files are written to datasets/:
    <dataset>_snea_alpha_0p0_results.csv
    <dataset>_snea_alpha_0p1_results.csv
    ...
    <dataset>_snea_alpha_1p0_results.csv

Run from src/evaluation/:
    python dataset_creation/split_by_alpha.py
"""

from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent.parent   # src/evaluation/
DATASETS_DIR = HERE / 'datasets'
SOURCE_DIR = HERE.parent / 'Results_All_Methods 2'

# Maps output alpha label → source column name in the combined file
ALPHA_COLS = {
    '0p0': 'snea_bert_alpha_0.0',
    '0p1': 'snea_bert_alpha_0.1',
    '0p2': 'snea_bert_alpha_0.2',
    '0p3': 'snea_bert_alpha_0.3',
    '0p4': 'snea_bert_alpha_0.4',
    '0p5': 'snea_bert_alpha_0.5',
    '0p6': 'snea_bert_alpha_0.6',
    '0p7': 'snea_bert_alpha_0.7',
    '0p8': 'snea_bert_alpha_0.8',
    '0p9': 'snea_bert_alpha_0.9',
    '1p0': 'snea_bert_alpha_1.0_SNEA_alone',
}

# Source combined files to process (from Results_All_Methods 2/)
SOURCE_FILES = [
    'mrpc_400_KGs_results.csv',
    'paws_wiki_400_KGs_results.csv',
    'pubmedqa_ranked_faithfulness_400_KGs_results.csv',
    'semantic_kg_codex_400_KGs_results.csv',
    'semantic_kg_combined_400_KGs_results.csv',
    'semantic_kg_findkg_400_KGs_results.csv',
    'semantic_kg_globi_400_KGs_results.csv',
    'semantic_kg_oregano_400_KGs_results.csv',
    'sts12_400_KGs_results.csv',
    'wikipedia_entity_swap_400_KGs_results.csv',
]


def derive_base_name(filename: str) -> str:
    """Strip the '_KGs_results.csv' suffix to get the dataset base name."""
    return filename.replace('_KGs_results.csv', '')


def split_file(src_path: Path) -> None:
    base = derive_base_name(src_path.name)
    df = pd.read_csv(src_path)
    print(f'  Loaded {len(df)} rows.')

    # Normalise id column
    if 'id' in df.columns and 'pair_id' not in df.columns:
        df = df.rename(columns={'id': 'pair_id'})

    if 'pair_id' not in df.columns:
        print(f'  ⚠ No pair_id column in {src_path.name} — skipping.')
        return

    for alpha_label, src_col in ALPHA_COLS.items():
        if src_col not in df.columns:
            print(f'  ⚠ Column {src_col} not found in {src_path.name} — skipping α={alpha_label}.')
            continue

        out_df = df[['pair_id', src_col]].copy()
        out_df = out_df.rename(columns={src_col: 'snea_bert_similarity'})
        out_df = out_df.dropna(subset=['snea_bert_similarity'])

        out_name = f'{base}_snea_alpha_{alpha_label}_results.csv'
        out_path = DATASETS_DIR / out_name
        out_df.to_csv(out_path, index=False)
        print(f'  ✓ {out_name}  ({len(out_df)} rows)')


def main():
    print(f'Source directory: {SOURCE_DIR}')
    print(f'Output directory: {DATASETS_DIR}\n')
    for fname in SOURCE_FILES:
        src = SOURCE_DIR / fname
        if not src.exists():
            print(f'⚠ Not found: {fname} — skipping.')
            continue
        print(f'Processing: {fname}')
        split_file(src)
        print()
    print('Done.')


if __name__ == '__main__':
    main()