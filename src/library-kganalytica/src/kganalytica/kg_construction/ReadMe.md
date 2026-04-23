# Knowledge Graph Construction API

Three public functions for extracting knowledge graphs using KRPOMultiAlign with N-paragraph label alignment.

## API Functions

### 1. construct_graph

Extract a single knowledge graph from one paragraph.

```python
from kganalytica.kg_construction import construct_graph

graph = construct_graph(
    paragraph="Albert Einstein developed the theory of relativity.",
    model="llama-3.1-8b-instant",
    api_key=None,
    max_rounds=3
)
# Returns: [['Albert_Einstein', 'developed', 'relativity'], ...]
```

### 2. construct_graphs

Extract N aligned knowledge graphs from N paragraphs simultaneously.

```python
from kganalytica.kg_construction import construct_graphs

result = construct_graphs(
    paragraphs=["Paris is in France.", "France is in Europe."],
    model="llama-3.1-8b-instant",
    api_key=None,
    max_rounds=3
)
# Returns: {"graph_1": [...], "graph_2": [...]}
```

### 3. construct_kgs

Process rows with multiple paragraphs (CSV-like format).

```python
from kganalytica.kg_construction import construct_kgs

rows = [{
    "id": "doc_1",
    "paragraph_1": "Text about topic.",
    "paragraph_2": "More text about topic."
}]

result = construct_kgs(
    rows=rows,
    model="llama-3.1-8b-instant",
    api_key=None,
    max_rounds=3
)
# Returns: [{"id": "doc_1", "kg_1": [...], "kg_2": [...]}]
```

## Configuration

### Environment Variables

Set ONE provider API key (choose Groq, OpenAI, or Gemini):

```env
# Groq (Recommended - includes Llama, Mistral, Qwen)
GROQ_API_KEY=gsk_...

# OR OpenAI (for GPT models)
OPENAI_API_KEY=sk-...

# OR Gemini (for Google models)
GEMINI_API_KEY=...
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
