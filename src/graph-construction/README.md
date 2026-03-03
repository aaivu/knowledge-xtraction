# Knowledge Graph Construction Pipeline

A scalable pipeline for extracting and verifying knowledge graphs from text paragraphs using LLMs. The pipeline supports dynamic paragraph counts, iterative refinement, and configurable LLM parameters.

## Overview

The pipeline performs the following steps:

1. **Extraction**: Extracts triplets from paragraphs using a configured LLM
2. **Verification**: Optionally verifies extracted triplets against source text
3. **Refinement**: Re-generates graphs if verification fails (up to `max_time` iterations)
4. **Output**: Saves verified knowledge graphs to CSV files

## Quick Start

### Basic Usage

```bash
python main.py input.csv
```

### Fast Execution

For quick results with minimal verification:

```bash
python main.py input.csv \
  --extract_llm llama-3.3-70b-versatile \
  --skip_verification \
  --max_time 1
```

### Custom Triplet Count (Optional)

If you want the LLM to extract approximately a specific number of triplets:

```bash
python main.py input.csv --num_triplets 15
```

**Note**: If `--num_triplets` is not provided, the LLM decides the triplet count freely.

## Command-Line Arguments

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input_csv` | string | Path to input CSV file. If not provided, processes all CSVs in `input/` folder |

### Optional Arguments

#### LLM Selection

| Flag | Default | Description |
|------|---------|-------------|
| `--extract_llm` | `mistralai/Mistral-7B-Instruct-v0.2` | LLM model for knowledge graph extraction |
| `--verification_llm1` | `mistralai/Mistral-7B-Instruct-v0.2` | Primary LLM for triplet verification |
| `--verification_llm2` | `mistralai/Mistral-7B-Instruct-v0.2` | Secondary LLM for verification (backup) |

#### Temperature Control

| Flag | Default | Range | Description |
|------|---------|-------|-------------|
| `--extract_temperature` | `0.0` | 0.0-1.0 | Temperature for extraction LLM (0=deterministic, 1=creative) |
| `--verify_temperature` | `0.0` | 0.0-1.0 | Temperature for verification LLM (0=deterministic, 1=creative) |

**Note**: Temperature values default to 0 if not provided, ensuring deterministic outputs.

#### Pipeline Control

| Flag | Default | Type | Description |
|------|---------|------|-------------|
| `--max_time` | `3` | int | Maximum iterations for extraction/verification loop |
| `--num_triplets` | `None` | int | Target number of triplets per graph (if not provided, LLM decides freely) |
| `--skip_verification` | `False` | flag | Skip verification step (extraction only) |
| `--output_dir` | `output` | string | Directory for output CSV files |
| `--verbose` | `False` | flag | Enable verbose logging for debugging |

## Input File Format

CSV file with the following structure:

```csv
id,paragraph_1,paragraph_2,paragraph_3,...
1,Text for first paragraph,Text for second paragraph,Text for third paragraph,...
2,Another first paragraph,Another second paragraph,Another third paragraph,...
```

**Requirements**:
- First column must be `id` (unique identifier)
- Paragraph columns must be named `paragraph_1`, `paragraph_2`, etc.
- Supports dynamic number of paragraphs per row

## Output Format

Output CSV with knowledge graphs:

```csv
id,paragraph_1,kg_1,paragraph_2,kg_2,paragraph_3,kg_3,...
1,Text for first paragraph,"[[""subject"",""relation"",""object""]]",Text for second paragraph,"[[""subject2"",""relation2"",""object2""]]",...
```

Each `kg_N` column contains JSON array of triplets: `[["subject", "relation", "object"], ...]`

## Examples

### Example 1: Fast Extraction (No Verification)

```bash
python main.py data.csv \
  --extract_llm llama-3.3-70b-versatile \
  --skip_verification \
  --max_time 1
```

**Use case**: Quick prototype, bulk processing, fast feedback
- Skips verification for speed
- Single extraction pass
- Deterministic output (default temperature 0)

### Example 2: Quality-First Extraction

```bash
python main.py data.csv \
  --extract_llm gpt-4 \
  --verification_llm1 gpt-4o-mini \
  --max_time 5
```

**Use case**: High-quality knowledge graphs
- Uses advanced models
- Multiple refinement iterations
- LLM decides triplet count freely