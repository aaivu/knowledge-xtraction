# Library Usage (`construct_kgs`)

This library entrypoint is:

```python
from main import construct_kgs
```

## Input CSV Type

`construct_kgs` expects an input CSV with this format:

- `id`
- `paragraph_1`
- `paragraph_2`
- `paragraph_3`
- ...

Example header:

```csv
id,paragraph_1,paragraph_2,paragraph_3
```

## Function Arguments

```python
construct_kgs(
	input_csv: str,
	output_csv: str,
	extract_llm: str = "llama-3.3-70b-versatile",
	verbose: bool = False,
) -> str
```

Argument notes:

- `input_csv`: path to input CSV (must follow the paragraph format above)
- `output_csv`: path to output CSV
- `extract_llm`: extraction model (default `llama-3.3-70b-versatile`)
- `verbose`: enable debug logs

## Required `.env` Keys

Set keys based on the model/provider you use:

- `GROQ_API_KEY` (required for `llama-3.3-70b-versatile` and other Groq models)
- `GROQ_API_KEY2` to `GROQ_API_KEY20` (optional, for key rotation)
- `OPENAI_API_KEY` (required for OpenAI models)
- `GEMINI_API_KEY` (required for Gemini models)

Example:

```env
GROQ_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
```

---

# Function Usage (`construct_graph`)

For constructing a knowledge graph from a single paragraph, use:

```python
from main import construct_graph
```

## Function Signature

```python
construct_graph(
    paragraph: str,
    llm: str = "llama-3.3-70b-versatile"
) -> Graph
```

## Function Arguments

- `paragraph` (str): The input text/paragraph to extract knowledge graph from
- `llm` (str): The LLM model to use for extraction (default: `llama-3.3-70b-versatile`)

## Return Type

**Returns**: `Graph` object containing:

- `triples`: List of triplets (subject, relation, object)
  - Each triplet is a `Triplet` object with:
    - `subject`: The subject entity
    - `relation`: The relationship type
    - `object`: The object entity

## Usage Examples

### Basic Usage (with default LLM)

```python
from main import construct_graph

paragraph = "Albert Einstein was born in Germany. He developed the theory of relativity."
graph = construct_graph(paragraph)

# Access the triplets
for triplet in graph.triples:
    print(f"{triplet.subject} -> {triplet.relation} -> {triplet.object}")
```

### With Custom LLM

```python
from main import construct_graph

paragraph = "Paris is the capital of France."
graph = construct_graph(
    paragraph=paragraph,
    llm="mistralai/Mistral-7B-Instruct-v0.2"
)

# Process the graph
triplets = graph.triples
```

### Output Format

The `Graph` object's `triples` attribute contains a list of extracted relationships. Example output:

```
Albert Einstein -> born_in -> Germany
Albert Einstein -> developed -> theory of relativity
Paris -> capital_of -> France
```

## Required `.env` Keys for `construct_graph`

Same as `construct_kgs` - set keys based on the LLM model you use:

- `GROQ_API_KEY` (for Groq models like `llama-3.3-70b-versatile`)
- `OPENAI_API_KEY` (for OpenAI models)
- `GEMINI_API_KEY` (for Gemini models)

---

# Function Usage (`construct_graphs`)

For constructing knowledge graphs from multiple paragraphs at once, use:

```python
from main import construct_graphs
```

## Function Signature

```python
construct_graphs(
    paragraphs: List[str],
    llm: str = "llama-3.3-70b-versatile"
) -> DynamicKnowledgeGraphs
```

## Function Arguments

- `paragraphs` (List[str]): List of paragraphs to extract knowledge graphs from
- `llm` (str): The LLM model to use for extraction (default: `llama-3.3-70b-versatile`)

## Return Type

**Returns**: `DynamicKnowledgeGraphs` object containing:

- `graphs`: Dictionary of indexed graphs (keyed by paragraph index starting at 1)
- `get_graph(i)`: Method to retrieve a specific graph by index

Each graph contains:
- `triples`: List of triplets (subject, relation, object)

## Usage Examples

### Basic Usage (with default LLM)

```python
from main import construct_graphs

paragraphs = [
    "Albert Einstein was born in Germany. He developed the theory of relativity.",
    "Paris is the capital of France.",
    "The Great Wall of China was built over many centuries."
]

kgs = construct_graphs(paragraphs)

# Access graphs by index (1-based indexing)
for i in range(1, len(paragraphs) + 1):
    graph = kgs.get_graph(i)
    print(f"\n--- Paragraph {i} ---")
    for triplet in graph.triples:
        print(f"{triplet.subject} -> {triplet.relation} -> {triplet.object}")
```

### With Custom LLM

```python
from main import construct_graphs

paragraphs = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers."
]

kgs = construct_graphs(
    paragraphs=paragraphs,
    llm="mistralai/Mistral-7B-Instruct-v0.2"
)

# Process all graphs
for i in range(1, len(paragraphs) + 1):
    graph = kgs.get_graph(i)
    print(f"Paragraph {i} has {len(graph.triples)} triplets")
```

### Accessing All Graphs

```python
from main import construct_graphs

paragraphs = ["Text 1", "Text 2", "Text 3"]
kgs = construct_graphs(paragraphs)

# Get all graphs as a dictionary
all_graphs = kgs.graphs

# Iterate through all graphs
for index, graph in all_graphs.items():
    print(f"Graph {index}: {len(graph.triples)} triplets")
```

### Output Format

Each paragraph gets its own knowledge graph. Example for 3 paragraphs:

```
--- Paragraph 1 ---
Albert Einstein -> born_in -> Germany
Albert Einstein -> developed -> theory of relativity

--- Paragraph 2 ---
Paris -> capital_of -> France

--- Paragraph 3 ---
Great Wall -> located_in -> China
Great Wall -> built_over -> many centuries
```

## Required `.env` Keys for `construct_graphs`

Same as `construct_graph` and `construct_kgs`:

- `GROQ_API_KEY` (for Groq models like `llama-3.3-70b-versatile`)
- `OPENAI_API_KEY` (for OpenAI models)
- `GEMINI_API_KEY` (for Gemini models)

## Difference Between Functions

| Function | Input | Output | Use Case |
|----------|-------|--------|----------|
| `construct_graph` | Single paragraph (str) | Single Graph | Extract KG from one text |
| `construct_graphs` | Multiple paragraphs (List[str]) | DynamicKnowledgeGraphs | Extract KGs from multiple texts at once |
| `construct_kgs` | CSV file with paragraphs | CSV file with KGs | Batch process from file with persistence |
