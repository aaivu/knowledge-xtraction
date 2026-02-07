# Knowledge Graph Evaluation

This directory contains the evaluation scripts for comparing KG-based semantic similarity with baseline methods on STS benchmarks.

## Directory Structure

```
src/evaluation/
├── kg_evaluation.py          # Main evaluation script
├── stsb_merged.csv            # Merged STS-B + STS12 dataset
├── aa_kea_results.csv         # Pre-computed KG similarity scores
├── output/                    # Generated results
│   └── stsb_merged_roc_comparison.png
└── README.md                  # This file
```

## Required Files

- **stsb_merged.csv**: Contains 200 sentence pairs from STS-B and STS12 datasets with human similarity scores (0-1 range)
- **aa_kea_results.csv**: Pre-computed knowledge graph similarity scores using the AA-KEA algorithm

## Usage

### Run the evaluation:

```bash
cd src/evaluation
python kg_evaluation.py
```

### Or from project root:

```bash
cd /Users/admin/Desktop/FYP_Benchmarking
python src/evaluation/kg_evaluation.py
```

## Output

The script will:
1. Load the merged dataset and pre-computed KG scores
2. Compute baseline similarity methods:
   - Text Similarity (all-MiniLM-L6-v2 embeddings)
   - Word Overlap (Jaccard similarity)
   - Character N-gram (3-gram similarity)
3. Evaluate all methods against human judgments:
   - Pearson & Spearman correlations
   - MSE & MAE
   - ROC curves & AUC scores
4. Generate comparison plot: `output/stsb_merged_roc_comparison.png`

## Methods Compared

| Method | Description | Type |
|--------|-------------|------|
| **KG Similarity (AA-KEA)** | Pre-computed using AA-KEA algorithm on knowledge graphs | Your method |
| **Text Similarity** | Semantic similarity using all-MiniLM-L6-v2 embeddings | Strong baseline |
| **Word Overlap** | Jaccard similarity on tokenized words | Weak baseline |
| **Character N-gram** | 3-gram character-level similarity | Weak baseline |

## Evaluation Metrics

- **Correlation**: Pearson (linear) and Spearman (rank-based)
- **Error**: MSE and MAE
- **Classification**: ROC curves and AUC (threshold=0.7, balanced classes)

## Notes

- ROC threshold of 0.7 gives balanced classes (54% positive, 46% negative)
- All baseline methods use fair comparisons (no training on STS-B data)
- Dataset composition: 100 pairs from STS-B + 100 pairs from STS12