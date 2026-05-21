# KGX Graph Similarity

Scores and analyzes knowledge graph similarity for benchmark KG pairs and LLM-generated QA knowledge graphs.

## Setup

Install dependencies from the repository root:

```bash
pip install -r ../requirements.txt
```

## Benchmark Scoring

Input: `S3KG_Benchmarking/Data/*.csv`

Output: `S3KG_Benchmarking/Results_All_Methods_KGSim/*_results.csv`

Run all benchmark files:

```bash
python s3kg_benchmarking_scores.py
```

Run a quick check on 5 rows in a temporary folder:

```bash
python s3kg_benchmarking_scores.py \
  --limit 5 \
  --methods s3kg_alpha_0.5 \
  --input S3KG_Benchmarking/Data/mrpc_400_KGs.csv \
  --output _tmp_score_checks/s3kg_benchmarking/mrpc_400_KGs_results.csv
```

## LLM KG Scoring

Input: `LLM_Evaluation/KGs_With_Temps/<temperature>/*.csv`

Output: `LLM_Evaluation/Results_KGs_With_Temps/<temperature>/*.csv`

Scores added:

- `s3kg_gold_llm`: LLM answer KG vs gold answer KG
- `s3kg_ctx_llm`: LLM answer KG vs supporting context KG

Run all temperature files:

```bash
python llm_evalution_score_s3kg_temps.py
```

Run a 5-row temporary check:

```bash
python llm_evalution_score_s3kg_temps.py \
  --limit 5 \
  --input LLM_Evaluation/KGs_With_Temps/0/mesaqa_google_gemma-7b-it_answers_KGs.csv \
  --output _tmp_score_checks/llm_eval/mesaqa_google_gemma-7b-it_answers_KGs.csv
```

## Mean Score Analysis

Input: `LLM_Evaluation/Results_KGs_With_Temps/`

Output: `LLM_Evaluation/Results_KGs_With_Temps/Analysis_plot_temps/`

Creates mean score CSVs, plots, and paper table files for GoldSim, CtxSim, and CUS.

```bash
python llm_mean_score_analysis.py
```

## Bottom 5% Filtering

Input: `Filtered_5%_rows_for_TAU/*_scored.csv`

Output:

- `Filtered_5%_rows_for_TAU/low_gold/`
- `Filtered_5%_rows_for_TAU/low_context/`
- `Filtered_5%_rows_for_TAU/filtered/`

```bash
cd Filtered_5%_rows_for_TAU
python filter_low_scores.py
```

## Main Scripts

- `s3kg_benchmarking_scores.py`
- `llm_evalution_score_s3kg_temps.py`
- `llm_mean_score_analysis.py`
- `Filtered_5%_rows_for_TAU/filter_low_scores.py`
- `Methods/s3kg.py`
