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

### Step 2: Evaluate Graph Similarity

Analyze and compare knowledge graphs using various similarity metrics.

See [KGX-Graph-Similarity README](./KGX-Graph-Similarity/README.md) for evaluation methods.

---

### Step 3: Analyze Triplets

Perform triplet-level analysis and comparison of extracted knowledge graphs.

See [Triplet Analyzing Unit README](./Triplet%20analyzing%20unit/README.md) for:
- Triplet classification (aligned, entity_different, relation_different)
- Comparison methods (standard and KEA-style scoring)
- Agreement analysis across annotators
- Excel-based output for detailed examination

```bash
cd "triplet-analyzing-unit"

# Standard triplet analysis
python run_triplet_analysis.py

# KEA-style (head/relation/tail) scoring
python run_triplet_analysis_kea.py
```

---

### Step 4: Run Evaluations

Execute evaluation pipelines on datasets.

See [Evaluation README](./evaluation/README.md) for benchmark and evaluation details.

---

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