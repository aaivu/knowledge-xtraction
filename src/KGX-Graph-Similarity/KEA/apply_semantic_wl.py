"""
Apply Semantic WL Kernel similarity to all KG pairs in the CSV file.

This script reads the CSV with kg1 and kg2 columns, computes similarity scores
using the Semantic WL Kernel, and adds a new column with the results.
"""

import pandas as pd
import ast
from tqdm import tqdm
from semantic_wl_kernel import calculate_semantic_wl_similarity

# Path to CSV file
CSV_PATH = r"D:\FYP\aaivu\knowledge-xtraction\src\KGX-Graph-Similarity\KEA\mrpc_400_KGs.csv"


def parse_kg(kg_str):
    """
    Parse KG string from CSV to list of triplets.
    
    Args:
        kg_str: String representation of KG (e.g., "[['a', 'b', 'c'], ...]")
        
    Returns:
        List of triplets or empty list if parsing fails
    """
    if pd.isna(kg_str) or kg_str == "" or kg_str == "[]":
        return []
    
    try:
        kg = ast.literal_eval(kg_str)
        if isinstance(kg, list):
            return kg
        return []
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing KG: {e}")
        return []


def main():
    print(f"Loading CSV from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize similarity column
    similarity_scores = []
    
    print("\nComputing Semantic WL Kernel similarity for each KG pair...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        kg1 = parse_kg(row['kg_1'])
        kg2 = parse_kg(row['kg_2'])
        
        # Handle empty KGs
        if len(kg1) == 0 or len(kg2) == 0:
            similarity = 0.0
        else:
            try:
                similarity = calculate_semantic_wl_similarity(kg1, kg2, verbose=False)
            except Exception as e:
                print(f"\nError at row {idx}: {e}")
                similarity = 0.0
        
        similarity_scores.append(similarity)
    
    # Add new column
    df['semantic_wl_similarity'] = similarity_scores
    
    # Save updated CSV
    df.to_csv(CSV_PATH, index=False)
    print(f"\nCSV updated with 'semantic_wl_similarity' column!")
    print(f"Saved to: {CSV_PATH}")
    
    # Show summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Mean similarity: {df['semantic_wl_similarity'].mean():.4f}")
    print(f"Median similarity: {df['semantic_wl_similarity'].median():.4f}")
    print(f"Min similarity: {df['semantic_wl_similarity'].min():.4f}")
    print(f"Max similarity: {df['semantic_wl_similarity'].max():.4f}")
    print(f"Std similarity: {df['semantic_wl_similarity'].std():.4f}")
    
    # Show first few results
    print("\n--- First 10 Results ---")
    print(df[['id', 'semantic_wl_similarity']].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
