# Benchmarking Evaluation — How to Reproduce Results

This README explains every script in the evaluation pipeline, what it produces, and the order to run them.

---

## Project Overview

We benchmark **KG-based semantic similarity methods** (AA-KEA, SNEA-BERT, WL variants, KEA-BERT, GNN) against standard NLP baselines (ROUGE-L, BLEU, BERTScore, Sentence-BERT, MiniLM, T5-base) across 10 datasets.

All scripts live under `src/evaluation/`. Run them **from that directory** unless noted otherwise.

```
cd src/evaluation
```

---

## Step 0 — Install Dependencies

```bash
# Activate your virtualenv first
source ../../venv/bin/activate

pip install pandas numpy matplotlib seaborn scipy scikit-learn
pip install rouge-score bert-score sentence-transformers
pip install nltk datasets tqdm
```

---

## Step 1 — Create Datasets

Run these once to generate the input CSV files in `datasets/`. Skip if the files already exist.

### 1a. Core benchmark datasets (MRPC, PAWS-Wiki, Semantic-KG Combined)

```bash
python dataset_creation/prepare_datasets.py
```

**Produces:**
- `datasets/mrpc_400.csv`
- `datasets/paws_wiki_400.csv`
- `datasets/semantic_kg_combined_400.csv`

---

### 1b. New Semantic-KG domain datasets + STS12

```bash
python dataset_creation/prepare_new_datasets.py
```

**Produces:**
- `datasets/semantic_kg_codex_400.csv`
- `datasets/semantic_kg_findkg_400.csv`
- `datasets/semantic_kg_globi_400.csv`
- `datasets/semantic_kg_oregano_400.csv`
- `datasets/sts12_400.csv`

---

### 1c. Wikipedia Entity Swap dataset

```bash
python dataset_creation/create_wikipedia_entity_swap_dataset.py
```

**Produces:**
- `datasets/wikipedia_entity_swap_400.csv`

---

### 1d. PubMedQA Ranked Faithfulness dataset

```bash
python dataset_creation/create_pubmedqa_ranked_faithfulness_dataset.py
```

**Produces:**
- `datasets/pubmedqa_ranked_faithfulness_400.csv`

> **Note:** This dataset has 100 questions × 4 faithfulness levels (L4 > L3 > L2 > L1). Level 4 = faithful paraphrase, Level 1 = context-ignoring hallucination. Used to test whether a metric can rank answers by faithfulness.

---

### 1e. PubMedQA Contextual dataset (optional)

Requires Python ≤ 3.13 and spaCy.

```bash
python dataset_creation/create_pubmedqa_contextual_dataset.py
```

**Produces:**
- `datasets/pubmedqa_contextual_400.csv`

---

## Step 2 — Run Our Method (SNEA-BERT / AA-KEA) on Each Dataset

The KG similarity scores from our method must be pre-computed and saved as result CSVs before running evaluations. These are already present in `datasets/` for all datasets. If you need to regenerate them, run the KG extraction pipeline separately.

**Required pre-computed result files** (already in `datasets/`):

| Dataset | Our Method Result File |
|---|---|
| MRPC | `datasets/mrpc_400_snea_bert_results.csv` |
| PAWS-Wiki | `datasets/paws_wiki_400_snea_bert_results.csv` |
| Semantic-KG Combined | `datasets/semantic_kg_combined_400_snea_bert_results.csv` |
| Wikipedia Entity Swap | `datasets/wikipedia_entity_swap_400_snea_bert_results.csv` |
| Semantic-KG Codex 400 | `datasets/semantic_kg_codex_400_KGs_snea_bert_results.csv` |
| Semantic-KG FindKG | `datasets/semantic_kg_findkg_400_KGs_snea_bert_results.csv` |
| Semantic-KG GloBI | `datasets/semantic_kg_globi_400_KGs_snea_bert_results.csv` |
| Semantic-KG Oregano | `datasets/semantic_kg_oregano_400_KGs_snea_bert_results.csv` |
| STS12 | `datasets/sts12_400_KGs_snea_bert_results.csv` |
| PubMedQA Ranked Faithfulness | `datasets/pubmedqa_ranked_faithfulness_400_snea_bert_results.csv` |

---

## Step 3 — Run Per-Dataset Evaluation (F1, AUC, Precision, Recall)

This evaluates our method against all baselines on each individual dataset and writes results to `output/<dataset_name>/`.

```bash
python semantic_kg_evaluation.py --mode multi
```

**What it does:**
- Loads each dataset + our SNEA-BERT result file
- Computes ROUGE-L, BLEU, BERTScore, MiniLM, T5-base baseline scores
- Sweeps thresholds → reports max F1 (Precision, Recall) and ROC-AUC
- Stratifies results by perturbation type and text length

**Produces** (for each dataset under `output/<dataset>/`):
- `overall_results.csv` — F1, AUC, Precision, Recall per method
- `overall_comparison.png` — bar chart
- `roc_curves.png` — ROC curves for all methods
- `perturbation_results.csv` / `perturbation_analysis.png`
- `text_length_results.csv` / `text_length_analysis.png`

---

## Step 4 — Cross-Dataset Comparison (Heatmaps, Ranks, Summary Table)

After Step 3 completes, regenerate all cross-dataset plots.

```bash
python update_cross_dataset.py
```

**What it does:**
- Reads `output/cross_dataset/all_methods_confusion.csv` (master results file)
- Adds any new KG dataset results from their `overall_results.csv`
- Generates heatmaps, rank plots, and a per-dataset performance summary table

**Produces** (under `output/cross_dataset/`):

| File | Description |
|---|---|
| `all_methods_heatmap.png` | F1 and AUC heatmap — AA-KEA + SNEA-BERT vs all baselines × all datasets. Grey cells = method not tested on that dataset. RdBu colormap (blue = high). |
| `all_methods_ranks.png` | Per-dataset rank (1 = best) for each method by F1 and AUC. |
| `variants_comparison_heatmap.png` | Internal comparison of all our method variants only. |
| `performance_summary_table.png` | Per-dataset verdict table — our best variant vs best competing method (all methods considered). Green border = Strong or Comparable. |
| `cross_dataset_summary.csv` | Full flat table of all results. |

> **Note on grey cells in the heatmap:** Some variants (WL, GNN, KEA-BERT etc.) were only evaluated on a subset of datasets. They appear in `variants_comparison_heatmap.png` but are excluded from `all_methods_heatmap.png` to keep the main comparison fair. AA-KEA and SNEA-BERT are shown with grey where not tested.

---

## Step 5 — Ranked Faithfulness Evaluation (PubMedQA)

Tests whether each method can correctly rank 4 answers by faithfulness to a source medical document.

```bash
python ranked_faithfulness_eval.py
```

**What it does:**
- Loads `datasets/pubmedqa_ranked_faithfulness_400.csv` + SNEA-BERT scores
- Computes ROUGE-L, BLEU, BERTScore, Sentence-BERT baselines
- Reports Kendall's τ, Spearman ρ, Perfect Rank %, Boundary Accuracy per method

**Produces** (under `output/pubmedqa_ranked_faithfulness/`):

| File | Description |
|---|---|
| `ranking_results.csv` | Summary: mean Kendall τ, Spearman ρ, Perfect rank %, boundary accuracies |
| `boundary_accuracy.csv` | Per-boundary accuracy: L4>L3, L3>L2, L2>L1 |
| `per_question_taus.csv` | Kendall τ per question per method |
| `kendall_tau_comparison.png` | Bar chart of mean τ per method |
| `score_by_level.png` | Score distributions per faithfulness level |
| `boundary_heatmap.png` | Boundary accuracy heatmap |
| `tau_distribution.png` | Tau distribution per method |
| `l3_l2_failures.csv` | Questions where L3>L2 boundary fails |

---

## Step 6 — Dataset Statistical Validation (PubMedQA)

Run this **before** interpreting perfect rank % results. Validates that the dataset's level separation is statistically sound, and explains why text-overlap methods get 0% perfect rank.

```bash
python pubmedqa_dataset_stats.py
```

**What it does:**
- Per-level descriptive stats (mean, std, min, max) per method
- One-way ANOVA across 4 faithfulness levels
- Pairwise Welch t-tests between adjacent levels (L4↔L3, L3↔L2, L2↔L1)
- Cohen's d effect size
- **Inversion rate**: % of questions where a lower-level answer scores *higher* than the upper-level answer — explains why ROUGE/BLEU/BERTScore get 0% perfect rank (100% inversion at L4>L3 boundary due to lexical bias toward near-copies over paraphrases)

**Produces** (under `output/pubmedqa_ranked_faithfulness/dataset_stats/`):

| File | Description |
|---|---|
| `stats_table.png` | Full stats: mean scores per level, ANOVA p, inversion rates, Cohen's d |
| `score_distributions.png` | Violin + box plots per method per faithfulness level |
| `cohens_d_heatmap.png` | Effect size at each boundary per method |
| `inversion_rate_bar.png` | Key chart: % of questions with inverted ordering per boundary |
| `stats_summary.csv` | All stats in CSV form |

---

## Output Directory Map

```
output/
├── cross_dataset/                  ← Step 4 outputs
│   ├── all_methods_confusion.csv   ← Master results CSV (DO NOT delete)
│   ├── all_methods_heatmap.png
│   ├── all_methods_ranks.png
│   ├── variants_comparison_heatmap.png
│   ├── performance_summary_table.png
│   └── cross_dataset_summary.csv
│
├── mrpc/                           ← Step 3 outputs (one dir per dataset)
├── mrpc_snea_bert/
├── paws_wiki/
├── paws_wiki_snea_bert/
├── semantic_kg/
├── semantic_kg_codex_400_snea_bert/
├── semantic_kg_findkg_snea_bert/
├── semantic_kg_globi_snea_bert/
├── semantic_kg_oregano_snea_bert/
├── semantic_kg_combined_snea_bert/
├── wikipedia_entity_swap/
├── wikipedia_snea_bert/
├── sts12_snea_bert/
│
└── pubmedqa_ranked_faithfulness/   ← Steps 5 & 6 outputs
    ├── ranking_results.csv
    ├── boundary_accuracy.csv
    ├── ...
    └── dataset_stats/
        ├── stats_table.png
        ├── score_distributions.png
        ├── cohens_d_heatmap.png
        └── inversion_rate_bar.png
```

---

## Quick Reference — Script → Output

| Goal | Script | Key Output |
|---|---|---|
| Create all datasets | `dataset_creation/prepare_datasets.py` + `prepare_new_datasets.py` | `datasets/*.csv` |
| Evaluate on all datasets | `semantic_kg_evaluation.py --mode multi` | `output/*/overall_results.csv` |
| Cross-dataset heatmaps & table | `update_cross_dataset.py` | `output/cross_dataset/*.png` |
| Ranked faithfulness ranking | `ranked_faithfulness_eval.py` | `output/pubmedqa_ranked_faithfulness/ranking_results.csv` |
| Validate faithfulness dataset | `pubmedqa_dataset_stats.py` | `output/pubmedqa_ranked_faithfulness/dataset_stats/` |

---

## Important Files — Do Not Delete

| File | Why |
|---|---|
| `output/cross_dataset/all_methods_confusion.csv` | Master confusion matrix CSV — all variant + baseline results across all datasets. Regenerating this from scratch requires re-running all evaluations. |
| `datasets/*_snea_bert_results.csv` | Pre-computed KG similarity scores from our method. Re-generating requires the full KG extraction pipeline. |
| `datasets/pubmedqa_ranked_faithfulness_400.csv` | Custom-built dataset — not from HuggingFace, built by `create_pubmedqa_ranked_faithfulness_dataset.py`. |