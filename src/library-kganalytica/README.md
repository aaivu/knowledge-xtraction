# kganalytica

**kganalytica** is a Python research library designed for
**constructing, analyzing, and comparing knowledge graphs generated from
natural language text**.

The library enables researchers and developers to transform unstructured
text such as paragraphs, documents, or model outputs into **structured
knowledge graphs (KGs)** represented as triplets:

subject → relation → object

By converting textual information into graph structures, the library
allows deeper analysis of factual information and relationships
contained within the text.

---

# Why This Library Is Important

Modern large language models generate answers in natural language, but
evaluating whether those answers **faithfully represent the underlying
facts** is challenging.

Traditional evaluation methods rely on:

- surface-level text similarity
- token overlap metrics
- embedding similarity

These methods often fail to capture **whether the factual relationships
are correct**.

Knowledge graphs provide a structured representation of information
where:

- entities represent real-world objects
- relations represent factual connections between entities

By constructing knowledge graphs from text, it becomes possible to
compare **facts rather than sentences**.

---

# What This Library Enables

Using **kganalytica**, a paragraph can be converted into a knowledge
graph containing multiple factual triplets.

Example:

Input paragraph:

Albert Einstein was born in Germany.\
He developed the theory of relativity.

Extracted knowledge graph:

Albert Einstein → born_in → Germany\
Albert Einstein → developed → theory_of_relativity

Once two texts are converted into knowledge graphs, the library can
compare them to measure **how similar their factual content is**.

This allows comparison between:

- context vs ground truth answers
- ground truth vs model-generated answers
- knowledge graphs extracted from different systems

---

# Core Capabilities

## Knowledge Graph Construction

Convert text paragraphs into structured knowledge graphs using LLM-based
extraction.

Supported inputs include:

- single paragraphs
- multiple paragraphs
- CSV datasets containing text fields

---

## Triplet-Level Graph Comparison

The library enables detailed comparison between two knowledge graphs by
identifying:

- aligned triplets
- missing triplets
- extra triplets
- entity mismatches
- relation mismatches

This allows precise identification of **where factual differences
occur**.

---

## Semantic and Structural Graph Similarity

kganalytica measures similarity between two knowledge graphs using a
hybrid approach that combines:

**Semantic similarity**\
Triplet meaning is compared using sentence embeddings.

**Structural similarity**\
Graph topology and relational structure are compared using graph
kernels.

This allows similarity measurement beyond simple text matching.

---

# Research Applications

The library is particularly useful for:

### LLM Evaluation

Evaluate whether a language model's output preserves the factual
relationships present in the context.

### Knowledge Graph Research

Analyze and compare graph structures generated from different extraction
models.

### Information Extraction Systems

Benchmark triplet extraction pipelines and analyze their errors.

---

# Installation

Install directly from PyPI:

```bash
pip install kganalytica
```

---

# Quick Example

```python
from kganalytica import construct_graph

paragraph = '''
Albert Einstein was born in Germany.
He developed the theory of relativity.
'''

graph = construct_graph(paragraph)

for t in graph.triples:
    print(t.subject, "->", t.relation, "->", t.object)
```

Output:

Albert Einstein -\> born_in -\> Germany\
Albert Einstein -\> developed -\> theory_of_relativity

---

# Graph Similarity Example

```python
from kganalytica import kg_similarity

kg1 = [
    ("Einstein", "born_in", "Germany"),
    ("Einstein", "developed", "Relativity")
]

kg2 = [
    ("Albert Einstein", "born in", "Germany"),
    ("Albert Einstein", "created", "Theory of Relativity")
]

score = kg_similarity(kg1, kg2)

print("Similarity:", score)
```

---

# Project Motivation

The library was developed to support research on **evaluating contextual
understanding in large language models through knowledge graph
comparison**.

By analyzing information at the **triplet level rather than the sentence
level**, researchers can better understand whether an AI system has
captured the correct relationships between entities.

---

# License

MIT License
