"""
Apply Semantic WL Kernel similarity to all KG pairs in the CSV file.

This script reads the CSV with kg1 and kg2 columns, computes similarity scores
using the Semantic WL Kernel, and adds a new column with the results.

Usage:
    python apply_semantic_wl.py                    # Run with enhanced method (default)
    python apply_semantic_wl.py --legacy           # Run with original method
    python apply_semantic_wl.py --preset balanced  # Use a specific preset
    python apply_semantic_wl.py --csv path/to/file.csv  # Specify CSV path
"""

import pandas as pd
import ast
import argparse
from tqdm import tqdm
from semantic_wl_kernel import (
    calculate_semantic_wl_similarity,
    semantic_wl_kernel_similarity,
    semantic_wl_kernel_nodewise_similarity,
    get_preset_config,
    set_sbert_model,
    clear_embedding_cache
)

# Default CSV path
DEFAULT_CSV_PATH = r"D:\FYP\aaivu\knowledge-xtraction\src\KGX-Graph-Similarity\KEA\mrpc_400_KGs.csv"


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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Apply Semantic WL Kernel similarity')
    parser.add_argument('--csv', type=str, default=DEFAULT_CSV_PATH,
                        help='Path to the CSV file')
    parser.add_argument('--legacy', action='store_true',
                        help='Use legacy (original) method instead of enhanced')
    parser.add_argument('--preset', type=str, default=None,
                        choices=['fast', 'balanced', 'accurate', 'legacy'],
                        help='Use a specific preset configuration')
    parser.add_argument('--model', type=str, default=None,
                        choices=['default', 'accurate', 'biomedical', 'multilingual'],
                        help='SBERT model to use')
    parser.add_argument('--output-col', type=str, default='semantic_wl_similarity',
                        help='Name of the output column')
    parser.add_argument('--groupwise', action='store_true', help='Use groupwise mean similarity as main score')
    args = parser.parse_args()
    
    CSV_PATH = args.csv
    
    # Set SBERT model if specified
    if args.model:
        print(f"Setting SBERT model to: {args.model}")
        set_sbert_model(args.model)
    
    # Determine which method to use
    if args.preset:
        config = get_preset_config(args.preset)
        method_name = f"Preset: {args.preset}"
    elif args.legacy:
        config = None
        method_name = "Legacy (original)"
    else:
        config = get_preset_config('balanced')
        method_name = "Enhanced (balanced preset)"
    
    print(f"\n{'='*60}")
    print(f"SEMANTIC WL KERNEL - KG Similarity Computation")
    print(f"{'='*60}")
    print(f"Method: {method_name}")
    print(f"CSV: {CSV_PATH}")
    print(f"Output column: {args.output_col}")
    
    print(f"\nLoading CSV...")
    df = pd.read_csv(CSV_PATH)
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Clear cache for fresh start
    clear_embedding_cache()
    
    # Initialize similarity column
    similarity_scores = []

    print(f"\nComputing similarity for each KG pair...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        kg1 = parse_kg(row['kg_1'])
        kg2 = parse_kg(row['kg_2'])
        
        # Handle empty KGs
        if len(kg1) == 0 or len(kg2) == 0:
            similarity = 0.0
            groupwise_scores = []
            groupwise_mean = 0.0
        else:
            try:
                if config is not None:
                    # Use preset configuration
                    result = semantic_wl_kernel_similarity(kg1, kg2, **config, verbose=False)
                    similarity = result['similarity']
                    groupwise_scores = None
                    groupwise_mean = None
                else:
                    # Use legacy method
                    similarity = calculate_semantic_wl_similarity(kg1, kg2, enhanced=False, verbose=False)
                    groupwise_scores = None
                    groupwise_mean = None
                # Run combined node group similarity
                nodewise_result = semantic_wl_kernel_nodewise_similarity(kg1, kg2)
                groupwise_scores = nodewise_result['groupwise_similarities']
                groupwise_mean = nodewise_result['similarity']
                if args.groupwise:
                    similarity = groupwise_mean
            except Exception as e:
                print(f"\nError at row {idx}: {e}")
                similarity = 0.0
                groupwise_scores = []
                groupwise_mean = 0.0
        similarity_scores.append(similarity)

    # Add new column: only save the mean groupwise similarity (not the vector)
    df[args.output_col] = similarity_scores

    # Save updated CSV (only mean similarity, not vector)
    output_path = CSV_PATH.replace('.csv', f'_{args.preset or ("legacy" if args.legacy else "enhanced")}.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Show summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Mean similarity: {df[args.output_col].mean():.4f}")
    print(f"Median similarity: {df[args.output_col].median():.4f}")
    print(f"Min similarity: {df[args.output_col].min():.4f}")
    print(f"Max similarity: {df[args.output_col].max():.4f}")
    print(f"Std similarity: {df[args.output_col].std():.4f}")
    print(f"\n--- First 10 Results ---")
    if 'id' in df.columns:
        print(df[['id', args.output_col]].head(10).to_string(index=False))
    else:
        print(df[[args.output_col]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
