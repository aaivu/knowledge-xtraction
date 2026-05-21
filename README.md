# Knowledge Extraction Pipeline

A comprehensive system for extracting and analyzing knowledge graphs from text documents.

## Project Structure

```
knowledge-xtraction/
├── graph-construction/      # Step 1: Knowledge graph extraction from text
├── KGX-Graph-Similarity/    # Graph similarity evaluation and comparison
├── evaluation/              # Evaluation metrics and datasets
└── Triplet analyzing unit/  # Triplet-level analysis tools
```

## Getting Started

### Step 1: Run Graph Construction

Extract knowledge graphs from your text data according to the [Graph Construction README](./graph-construction/README.md).

```bash
cd graph-construction

# Basic usage with default model (Groq Llama 3.3 70B)
python main.py --input_csv input/data.csv

# With custom output path and model
python main.py --input_csv input/data.csv --output_csv output/results.csv --model gpt-4
```

**What you need:**
1. An input CSV file with format: `id, paragraph_1, paragraph_2, ...`
2. Set up the appropriate API key in `.env` (see [Graph Construction README](./graph-construction/README.md) for details)
3. Run the extraction

See [Graph Construction README](./graph-construction/README.md) for:
- Detailed argument documentation
- Supported LLM models and providers
- Input/output format specifications
- Environment setup instructions

---

### 2. Graph Similarity Benchmarking

**Folder:** `KGX-Graph-Similarity/`

**Input:** Benchmark CSVs in `KGX-Graph-Similarity/S3KG_Benchmarking/Data/` with `kg_1` and `kg_2` columns.

**What it does:** Scores each KG pair with graph similarity methods, including S3KG across alpha settings, WL kernel, KEA variants, TransE, RotatE, and semantic WL.

**Output:** Result CSVs in `KGX-Graph-Similarity/S3KG_Benchmarking/Results_All_Methods_KGSim/` with one score column per method.

```bash
python KGX-Graph-Similarity/s3kg_benchmarking_scores.py
```

### 3. LLM KG Similarity Scoring

**Folder:** `KGX-Graph-Similarity/`

**Input:** LLM QA KG CSVs in `KGX-Graph-Similarity/LLM_Evaluation/KGs_With_Temps/`. Files are grouped by temperature and contain `kg_gold`, `kg_llm`, and `kg_context`.

**What it does:** Compares the LLM answer KG against the gold answer KG and the supporting context KG using S3KG with alpha `0.5`.

**Output:** Scored CSVs in `KGX-Graph-Similarity/LLM_Evaluation/Results_KGs_With_Temps/` with `s3kg_gold_llm` and `s3kg_ctx_llm`.

```bash
python KGX-Graph-Similarity/llm_evalution_score_s3kg_temps.py
```

### 4. LLM Score Analysis

**Folder:** `KGX-Graph-Similarity/`

**Input:** Scored LLM KG CSVs from `KGX-Graph-Similarity/LLM_Evaluation/Results_KGs_With_Temps/`.

**What it does:** Computes mean GoldSim, CtxSim, and CUS scores by dataset, model, and temperature. It also creates paper-ready table CSV/LaTeX files and plots.

**Output:** Analysis files in `KGX-Graph-Similarity/LLM_Evaluation/Results_KGs_With_Temps/Analysis_plot_temps/`.

```bash
python KGX-Graph-Similarity/llm_mean_score_analysis.py
```

### 5. Bottom 5% Filtering for Triplet Analysis

**Folder:** `KGX-Graph-Similarity/Filtered_5%_rows_for_TAU/`

**Input:** `*_scored.csv` files with `gold_similarity` and `context_similarity` columns.

**What it does:** Finds the bottom 5% cutoff separately for gold similarity and context similarity. It exports low-gold rows, low-context rows, and a filtered set where both scores are above the 5th percentile.

**Output:** `low_gold/`, `low_context/`, and `filtered/`.

```bash
cd KGX-Graph-Similarity/Filtered_5%_rows_for_TAU
python filter_low_scores.py
```

### 6. Triplet Analysis

**Folder:** `Triplet analyzing unit/`

**Input:** Annotated triplet files or filtered low-score cases.

**What it does:** Performs triplet-level comparison, including aligned triplets, entity differences, relation differences, and annotator agreement.

**Output:** Excel/CSV files for detailed triplet-level review.

```bash
cd "Triplet analyzing unit"
python run_triplet_analysis.py
python run_triplet_analysis_kea.py
```

### 7. Evaluation

**Folder:** `evaluation/`

**Input:** Benchmark datasets and precomputed similarity result CSVs.

**What it does:** Runs dataset-level evaluation, ranked faithfulness evaluation, and cross-dataset comparisons.

**Output:** Evaluation CSVs and plots under the evaluation output folders.

## Installation

```bash
# Install dependencies for the entire project
pip install -r requirements.txt
```

Each submodule may have additional dependencies in their respective `requirements.txt` files.

## Dependencies

- **Core ML/DL**: torch, torch-geometric, scikit-learn, scipy
- **Graph Processing**: networkx, grakel
- **NLP**: sentence-transformers, transformers
- **LLM APIs**: openai, groq, google-genai
- **Data & Utilities**: pandas, matplotlib, tqdm, pydantic, python-dotenv