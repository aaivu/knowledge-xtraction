# Triplet Analysis

This repository provides tools and scripts for analyzing knowledge graph triplets, comparing annotation agreements, and evaluating alignment between predicted and reference triplets. It is designed for research and evaluation in knowledge extraction and graph-based NLP tasks.

## Project Structure

- `agreement_3_annotators.py` — Analyze agreement among three annotators on triplet annotations.
- `eval_aligned_pr.py` — Evaluate precision and recall for aligned triplets.
- `run_triplet_analysis_kea.py` — Run triplet analysis using the KEA method.
- `run_triplet_analysis.py` — Main script to run triplet analysis.
- `requirements.txt` — Python dependencies for the project.
- `kg_compare/` — Module for comparing and processing knowledge graph triplets:
  - `compare_kea.py`, `compare.py` — Comparison utilities for triplet data.
  - `embedder.py` — Embedding utilities for triplet analysis.
  - `io_utils.py`, `parse_utils.py` — Input/output and parsing utilities.
  - `Results/` — Contains result files:
    - `agreement_items.csv` — Annotator agreement data.
    - `Annotations - triplet_analysis.csv` — Triplet annotation data.

## Getting Started

### Prerequisites

- Python 3.7+
- Recommended: Create a virtual environment

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aaivu/knowledge-xtraction.git
   cd knowledge-xtraction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

- To analyze annotator agreement:
  ```bash
  python agreement_3_annotators.py
  ```
- To evaluate aligned precision/recall:
  ```bash
  python eval_aligned_pr.py
  ```
- To run triplet analysis:
  ```bash
  python run_triplet_analysis.py
  ```
  or
  ```bash
  python run_triplet_analysis_kea.py
  ```

## Data

- manually annotated ground truth datasets are in the `Annotated datasets` directory.
