#!/usr/bin/env python
"""Comprehensive test of KRPOMultiAlign library with N-paragraph extraction."""

import sys
sys.path.insert(0, 'src')

from kganalytica import construct_graph, construct_graphs, construct_kgs
import json

print("\n" + "=" * 70)
print("KRPOMultiAlign Library - Comprehensive API Test")
print("=" * 70)

# ============================================================================
# TEST 1: Single Paragraph Extraction
# ============================================================================
print("\n[TEST 1] Single Paragraph → Single Knowledge Graph")
print("-" * 70)

text1 = "Marie Curie was a Polish physicist who discovered radium and polonium."
result1 = construct_graph(text1, model="llama-3.1-8b-instant", max_rounds=1)

print(f"Input: {text1}")
print(f"Output triplets ({len(result1)}):")
for i, triplet in enumerate(result1, 1):
    print(f"  {i}. {triplet}")

# ============================================================================
# TEST 2: Multiple Paragraphs → Multiple Aligned Knowledge Graphs
# ============================================================================
print("\n[TEST 2] N Paragraphs → N Aligned Knowledge Graphs (Label Alignment)")
print("-" * 70)

paragraphs = [
    "Python is a high-level programming language created by Guido van Rossum.",
    "Python is widely used for data science and machine learning applications.",
    "The Python Software Foundation maintains Python and hosts community resources."
]

result2 = construct_graphs(paragraphs, model="llama-3.1-8b-instant", max_rounds=1)

print(f"Input: {len(paragraphs)} paragraphs")
for i, para in enumerate(paragraphs, 1):
    print(f"  P{i}: {para}")

print(f"\nOutput: {len(result2)} aligned knowledge graphs")
for graph_key in sorted(result2.keys()):
    triplets = result2[graph_key]
    print(f"\n  {graph_key.upper()} ({len(triplets)} triplet{'s' if len(triplets) != 1 else ''}):")
    for j, triplet in enumerate(triplets, 1):
        print(f"    {j}. {triplet}")

# ============================================================================
# TEST 3: Row-Based Processing (CSV-like Input)
# ============================================================================
print("\n[TEST 3] Row-Based Processing (CSV Format) → Structured Output")
print("-" * 70)

rows = [
    {
        "id": "paper_001",
        "paragraph_1": "Graph neural networks combine graph theory with deep learning.",
        "paragraph_2": "GNNs are used for node classification and link prediction tasks.",
        "paragraph_3": "Popular GNN architectures include GCN and GraphSAGE."
    }
]

result3 = construct_kgs(rows, model="llama-3.1-8b-instant", max_rounds=1)

print(f"Input: {len(rows)} row(s) with multiple paragraphs each")
print(f"\nRow structure:")
for key, value in rows[0].items():
    if key != "id":
        print(f"  {key}: {value}")

print(f"\nOutput: {len(result3)} processed row(s)")
for i, output_row in enumerate(result3, 1):
    print(f"\n  Row #{i} (ID: {output_row['id']}):")
    for key in sorted(output_row.keys()):
        if key != 'id':
            triplets = output_row[key]
            print(f"    {key.upper()}: {len(triplets)} triplet{'s' if len(triplets) != 1 else ''}")
            for j, triplet in enumerate(triplets, 1):
                print(f"      {j}. {triplet}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("✓ All tests completed successfully!")
print("=" * 70)
print("\nKey Features Demonstrated:")
print("  1. Single paragraph extraction with clean triplet output")
print("  2. N-paragraph simultaneous extraction with label alignment")
print("     (graph_1, graph_2, graph_3, ...)")
print("  3. Row-based processing with structured input/output")
print("     (paragraph_N → kg_N transformation)")
print("\nAPI Pattern:")
print("  - All functions support: model selection, api_key fallback, max_rounds")
print("  - GROQ_API_KEY environment variable used for authentication")
print("  - Clear input/output structure for integration")
print("\n")
