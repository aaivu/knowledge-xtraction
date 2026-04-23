#!/usr/bin/env python
"""Quick test of the KRPOMultiAlign API."""

import sys
sys.path.insert(0, 'src')

from kganalytica import construct_graph, construct_graphs, construct_kgs

# Test 1: Single paragraph
print("=" * 60)
print("TEST 1: construct_graph (single paragraph)")
print("=" * 60)
try:
    result = construct_graph(
        "Albert Einstein developed the theory of relativity.",
        model="llama-3.1-8b-instant",
        api_key=None,
        max_rounds=1
    )
    print(f"✓ Result type: {type(result)}")
    print(f"✓ Number of triplets: {len(result)}")
    if result:
        print(f"✓ Sample: {result[0]}")
except Exception as e:
    print(f"✗ Error: {e}")
print()

# Test 2: Multiple paragraphs
print("=" * 60)
print("TEST 2: construct_graphs (multiple paragraphs)")
print("=" * 60)
try:
    results = construct_graphs(
        ["Paris is the capital of France.", "The Eiffel Tower is in Paris."],
        model="llama-3.1-8b-instant",
        api_key=None,
        max_rounds=1
    )
    print(f"✓ Result type: {type(results)}")
    print(f"✓ Keys: {list(results.keys())}")
    if results:
        for key, triplets in results.items():
            print(f"  {key}: {len(triplets)} triplets")
except Exception as e:
    print(f"✗ Error: {e}")
print()

# Test 3: Row-based input
print("=" * 60)
print("TEST 3: construct_kgs (row-based input)")
print("=" * 60)
try:
    rows = [
        {
            "id": "doc_1",
            "paragraph_1": "Python is a programming language.",
            "paragraph_2": "It was created by Guido van Rossum."
        }
    ]
    results = construct_kgs(rows, model="llama-3.1-8b-instant", api_key=None, max_rounds=1)
    print(f"✓ Result type: {type(results)}")
    print(f"✓ Number of results: {len(results)}")
    if results:
        print(f"✓ First result keys: {list(results[0].keys())}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60)
print("API tests completed")
print("=" * 60)
