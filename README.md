# Knowledge Xtraction

A web UI for extracting and comparing Knowledge Graphs from text using LLMs.

## Setup

**Requirements:** Python 3.10+, a [Groq API key](https://console.groq.com/)

```bash
git clone https://github.com/aaivu/knowledge-xtraction.git
cd knowledge-xtraction
pip install flask sentence-transformers networkx numpy torch
```

## Run

```bash
cd website
C:\Python313\python.exe app.py
```

Open **http://127.0.0.1:5000** in your browser.

> If `python app.py` gives `ModuleNotFoundError: No module named 'flask'`, use the full path `C:\Python313\python.exe app.py` instead (the venv doesn't have the packages).

## Usage

1. Click **⚙ Settings** → enter your Groq API key → save
2. Paste a *Gold Answer* and two *Supporting Context* paragraphs
3. Click **Generate Knowledge Graphs** → view the extracted KGs
4. Click **Calculate S3KG Similarity** → see similarity scores
5. Click **Analyse with KGAnalytica** → see triplet-level analysis

## Project Structure

```
website/         # Flask app (app.py) + frontend (templates/, static/)
src/
  graph-construction/      # KG extractor + LLM factory
  KGX-Graph-Similarity/    # SNEA-SBERT similarity
  library-kganalytica/     # KRPO Multi extractor + KGAnalytica
  evaluation/              # Offline evaluation scripts
```

**`SyntaxError: invalid syntax` in `snea_sbert_similarity.py`**  
The file has unresolved git merge conflict markers. Restore the clean version:
```bash
git checkout stash@{0} -- src/KGX-Graph-Similarity/snea_sbert_similarity.py
```

**Server crashes immediately after a file change**  
Flask's debug reloader detects file changes and restarts. With `use_reloader=False` this should not happen. If it does, simply re-run `python app.py` from the `website/` directory.
