# Graph Construction

Extracts knowledge graphs (KGs) from any number of text paragraphs using a single-prompt few-shot LLM call. For each row in the input CSV the pipeline sends all N paragraphs together in one prompt and returns N aligned knowledge graphs, one per paragraph.

## How it works

1. **Few-shot prompting** — three worked examples (with known-good KGs) are embedded directly in the prompt so the model understands the expected output format without fine-tuning.
2. **Single LLM call per row** — all N paragraphs are sent together; the model returns `knowledge_graph1` … `knowledge_graphN` in one JSON response.
3. **Validation** — any triplet that is not a 3-string list, or that still contains placeholder tokens (`entity1`, `relation`, etc.), is dropped.
4. **Normalisation** — every subject, relation, and object is lowercased, underscores are replaced with spaces, and extra whitespace is collapsed.
5. **Checkpointing** — rows already present in the output CSV are skipped, so a run can be safely resumed after an interruption.

## Quick start

```bash
python main.py --input_csv input/data.csv
```

With explicit output path and model:

```bash
python main.py --input_csv input/data.csv --output_csv output/kg_results.csv --model llama-3.3-70b-versatile
```

## Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--input_csv` | *(required)* | Path to input CSV |
| `--output_csv` | `output/kg_results.csv` | Path to output CSV |
| `--model` | `llama-3.3-70b-versatile` | LLM model name (see Supported models below) |

## Input format

```csv
id,paragraph_1,paragraph_2,paragraph_3,...
1,"Text 1...","Text 2...","Text 3..."
2,"Text 1...","Text 2..."
```

- `id` — unique row identifier
- `paragraph_1`, `paragraph_2`, `paragraph_3`, … — any number of paragraph columns; each row can have a different count
- The pipeline detects N dynamically and instructs the model to return exactly N graphs

## Output format

```csv
id,paragraph_1,kg_1,paragraph_2,kg_2,paragraph_3,kg_3,...
1,"Text 1...","[[\"subject\",\"relation\",\"object\"],...]","Text 2...","[...]","Text 3...","[...]"
```

Columns are interleaved: `paragraph_1, kg_1, paragraph_2, kg_2, …` for as many paragraphs as the row has. Each `kg_N` column contains a JSON array of `[subject, relation, object]` triplets.

## Environment variables

Create a `.env` file in this directory. Only one key is needed depending on the provider you use:

| Variable | Required for |
|----------|-------------|
| `GROQ_API_KEY` | Groq models — `llama-*`, `mixtral-*` (default) |
| `GEMINI_API_KEY` | Google Gemini models — `gemini-*` |
| `OPENAI_API_KEY` | OpenAI models — `gpt-*` |

HuggingFace models are loaded locally and need no API key.

Example `.env` for Groq:

```
GROQ_API_KEY=gsk_...
```

## Supported models

The provider is inferred from the model name:

| Model name contains | Provider |
|---------------------|----------|
| `llama`, `mixtral`, `groq` | Groq |
| `gemini` | Google Gemini |
| `gpt`, `openai` | OpenAI |
| anything else | HuggingFace (local) |
