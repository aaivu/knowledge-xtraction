"""
Knowledge Graph Evaluation for STS Benchmarks

This script evaluates knowledge graph-based semantic similarity against baseline methods
on merged STS-B and STS12 datasets.

Required files (in same directory):
- stsb_merged.csv: Merged dataset with sentence pairs and human similarity scores
- aa_kea_results.csv: Pre-computed KG similarity scores using AA-KEA algorithm

Output:
- output/stsb_merged_roc_comparison.png: ROC curves comparing all methods

Usage:
    cd src/evaluation
    python kg_evaluation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    adjusted_rand_score, normalized_mutual_info_score,
    mean_squared_error, mean_absolute_error,
    f1_score, accuracy_score
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.stats import spearmanr, pearsonr
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
import ast
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def parse_kg(kg_string):
    """Parse knowledge graph string representation"""
    try:
        if pd.isna(kg_string):
            return []
        if isinstance(kg_string, str):
            kg_list = ast.literal_eval(kg_string)
        else:
            kg_list = kg_string
        return kg_list if isinstance(kg_list, list) else []
    except:
        return []


# ============================================================================
# MODELS LOADING
# ============================================================================

def load_models():
    """Load embedding model"""
    print("Loading models...")

    # Embedding model for KG and text similarity
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Embedding model loaded")

    return embedding_model

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def kg_to_text(kg_triples):
    """Convert KG triples to text"""
    if not kg_triples:
        return ""
    texts = []
    for triple in kg_triples:
        if len(triple) == 3:
            entity1, relation, entity2 = triple
            text = f"{entity1} {relation} {entity2}"
            texts.append(text)
    return " . ".join(texts)

def compute_kg_similarity(gold_kg, llm_kg, model):
    """Compute KG similarity using embeddings"""
    gold_text = kg_to_text(gold_kg)
    llm_text = kg_to_text(llm_kg)

    if not gold_text or not llm_text:
        return 0.0

    gold_embedding = model.encode([gold_text])
    llm_embedding = model.encode([llm_text])
    similarity = cosine_similarity(gold_embedding, llm_embedding)[0][0]
    return float(similarity)

def compute_text_similarity(text1, text2, model):
    """Compute text similarity using embeddings (semantic similarity)"""
    if pd.isna(text1) or pd.isna(text2):
        return 0.0
    embeddings = model.encode([str(text1), str(text2)])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(similarity)

def compute_word_overlap_similarity(text1, text2):
    """Compute word overlap similarity using Jaccard index"""
    if pd.isna(text1) or pd.isna(text2):
        return 0.0

    # Tokenize and convert to lowercase
    words1 = set(str(text1).lower().split())
    words2 = set(str(text2).lower().split())

    # Jaccard similarity: intersection / union
    if len(words1.union(words2)) == 0:
        return 0.0

    jaccard = len(words1.intersection(words2)) / len(words1.union(words2))
    return float(jaccard)

def compute_char_ngram_similarity(text1, text2, n=3):
    """Compute character n-gram similarity"""
    if pd.isna(text1) or pd.isna(text2):
        return 0.0

    text1 = str(text1).lower()
    text2 = str(text2).lower()

    # Generate character n-grams
    def get_ngrams(text, n):
        return set([text[i:i+n] for i in range(len(text) - n + 1)])

    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)

    # Jaccard similarity on n-grams
    if len(ngrams1.union(ngrams2)) == 0:
        return 0.0

    similarity = len(ngrams1.intersection(ngrams2)) / len(ngrams1.union(ngrams2))
    return float(similarity)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_merged_dataset(dataset_csv='stsb_merged.csv', kg_similarity_csv='aa_kea_results.csv'):
    """
    Load merged STS dataset with pre-computed KG similarity scores

    Args:
        dataset_csv: Path to merged dataset CSV file (default: stsb_merged.csv)
        kg_similarity_csv: Path to KG similarity results CSV file (default: aa_kea_results.csv)
    Returns:
        DataFrame with sentences, scores, and KG similarity scores
    """
    print("="*70)
    print("LOADING MERGED DATASET")
    print("="*70)

    # Check if files exist
    if not Path(dataset_csv).exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_csv}")
    if not Path(kg_similarity_csv).exists():
        raise FileNotFoundError(f"KG similarity file not found: {kg_similarity_csv}")

    # Load dataset CSV
    print(f"\nLoading {dataset_csv}...")
    df = pd.read_csv(dataset_csv)
    print(f"✓ Loaded {len(df)} rows")
    print(f"  Columns: {df.columns.tolist()}")

    # Load KG similarity results CSV
    print(f"\nLoading {kg_similarity_csv}...")
    kg_sim_df = pd.read_csv(kg_similarity_csv)
    print(f"✓ Loaded {len(kg_sim_df)} rows with pre-computed KG similarity scores")
    print(f"  Columns: {kg_sim_df.columns.tolist()}")

    # Check for missing/empty aa_kea_similarity values
    missing_count = kg_sim_df['aa_kea_similarity'].isna().sum()
    if missing_count > 0:
        print(f"\n⚠️  WARNING: Found {missing_count} rows with missing aa_kea_similarity")
        print("   These rows likely have empty KGs for both sentences")
        print("   Filling missing values with 0.0 (no similarity)...")
        kg_sim_df['aa_kea_similarity'] = kg_sim_df['aa_kea_similarity'].fillna(0.0)

    # Merge datasets on row_id
    print("\nMerging dataset with KG similarity scores...")
    merged_df = df.merge(
        kg_sim_df[['row_id', 'aa_kea_similarity']],
        on='row_id',
        how='left'
    )

    # Rename aa_kea_similarity to kg_sim_pred for consistency
    merged_df['kg_sim_pred'] = merged_df['aa_kea_similarity']
    merged_df = merged_df.drop(columns=['aa_kea_similarity'])

    # Final check for any NaN values after merge
    final_nan_count = merged_df['kg_sim_pred'].isna().sum()
    if final_nan_count > 0:
        print(f"\n⚠️  WARNING: Found {final_nan_count} rows with NaN after merge")
        print("   Filling with 0.0...")
        merged_df['kg_sim_pred'] = merged_df['kg_sim_pred'].fillna(0.0)

    print(f"✓ Merged dataset created: {len(merged_df)} rows")
    print(f"  Final columns: {merged_df.columns.tolist()}")

    # Display sample data
    print("\nSample data:")
    sample_cols = ['row_id', 'sentence1', 'sentence2', 'score', 'kg_sim_pred']
    print(merged_df[sample_cols].head(3))

    print("\nKG Similarity Score Statistics:")
    print(f"  Range: {merged_df['kg_sim_pred'].min():.4f} - {merged_df['kg_sim_pred'].max():.4f}")
    print(f"  Mean: {merged_df['kg_sim_pred'].mean():.4f}")

    return merged_df


def compute_predictions(df, embedding_model):
    """
    Compute similarity predictions using multiple methods
    Note: KG similarity scores are pre-computed and already loaded in kg_sim_pred column

    Args:
        df: DataFrame with sentence1, sentence2, and kg_sim_pred columns
        embedding_model: SentenceTransformer model
    Returns:
        DataFrame with prediction columns added
    """
    print("\n" + "="*70)
    print("COMPUTING PREDICTIONS")
    print("="*70)

    # Check if kg_sim_pred already exists (pre-computed)
    if 'kg_sim_pred' in df.columns:
        print("\n✓ KG similarity scores already loaded from aa_kea_results.csv")
        print(f"  KG Similarity range: {df['kg_sim_pred'].min():.4f} - {df['kg_sim_pred'].max():.4f}")
    else:
        print("\n⚠ Warning: kg_sim_pred column not found, will be computed using embeddings")
        print("Computing KG similarity predictions...")
        df['kg_sim_pred'] = df.apply(
            lambda row: compute_kg_similarity(row['kg1'], row['kg2'], embedding_model),
            axis=1
        )

    print("\nComputing text similarity (embedding-based)...")
    df['text_sim_pred'] = df.apply(
        lambda row: compute_text_similarity(row['sentence1'], row['sentence2'], embedding_model),
        axis=1
    )

    print("Computing word overlap similarity (Jaccard)...")
    df['word_overlap_pred'] = df.apply(
        lambda row: compute_word_overlap_similarity(row['sentence1'], row['sentence2']),
        axis=1
    )

    print("Computing character n-gram similarity...")
    df['char_ngram_pred'] = df.apply(
        lambda row: compute_char_ngram_similarity(row['sentence1'], row['sentence2']),
        axis=1
    )

    # Check if scores need normalization (if max > 1, assume 0-5 scale)
    max_score = df['score'].max()
    if max_score > 1.0:
        print(f"\nNormalizing scores from 0-5 to 0-1 range...")
        df['score_normalized'] = df['score'] / 5.0
    else:
        print(f"\nScores already in 0-1 range, using as-is...")
        df['score_normalized'] = df['score']

    print("\n✓ All predictions computed")
    print(f"\nPrediction ranges:")
    print(f"  KG Similarity:      {df['kg_sim_pred'].min():.4f} - {df['kg_sim_pred'].max():.4f}")
    print(f"  Text Similarity:    {df['text_sim_pred'].min():.4f} - {df['text_sim_pred'].max():.4f}")
    print(f"  Word Overlap:       {df['word_overlap_pred'].min():.4f} - {df['word_overlap_pred'].max():.4f}")
    print(f"  Char N-gram:        {df['char_ngram_pred'].min():.4f} - {df['char_ngram_pred'].max():.4f}")
    print(f"  Human Score:        {df['score_normalized'].min():.4f} - {df['score_normalized'].max():.4f}")

    return df


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_correlation(df):
    """
    Evaluate correlation of predictions with human judgments

    Args:
        df: DataFrame with score_normalized and prediction columns
    Returns:
        Dictionary with correlation metrics
    """
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)

    human_scores = df['score_normalized'].values

    # Define all methods
    methods = {
        'kg': 'kg_sim_pred',
        'text': 'text_sim_pred',
        'word_overlap': 'word_overlap_pred',
        'char_ngram': 'char_ngram_pred'
    }

    results = {'pearson': {}, 'spearman': {}, 'mse': {}, 'mae': {}}

    # Compute metrics for each method
    for method_name, column_name in methods.items():
        if column_name in df.columns:
            # Pearson correlation (linear relationship)
            pearson, _ = pearsonr(human_scores, df[column_name].values)
            results['pearson'][method_name] = pearson

            # Spearman correlation (rank-based)
            spearman, _ = spearmanr(human_scores, df[column_name].values)
            results['spearman'][method_name] = spearman

            # MSE and MAE
            mse = mean_squared_error(human_scores, df[column_name].values)
            results['mse'][method_name] = mse

            mae = mean_absolute_error(human_scores, df[column_name].values)
            results['mae'][method_name] = mae

    # Print results
    print("\nKG Similarity (AA-KEA):")
    print(f"  Pearson Correlation:  {results['pearson']['kg']:.4f}")
    print(f"  Spearman Correlation: {results['spearman']['kg']:.4f}")
    print(f"  MSE:                  {results['mse']['kg']:.4f}")
    print(f"  MAE:                  {results['mae']['kg']:.4f}")

    print("\nText Similarity (Embeddings):")
    print(f"  Pearson Correlation:  {results['pearson']['text']:.4f}")
    print(f"  Spearman Correlation: {results['spearman']['text']:.4f}")
    print(f"  MSE:                  {results['mse']['text']:.4f}")
    print(f"  MAE:                  {results['mae']['text']:.4f}")

    print("\nWord Overlap (Jaccard):")
    print(f"  Pearson Correlation:  {results['pearson']['word_overlap']:.4f}")
    print(f"  Spearman Correlation: {results['spearman']['word_overlap']:.4f}")
    print(f"  MSE:                  {results['mse']['word_overlap']:.4f}")
    print(f"  MAE:                  {results['mae']['word_overlap']:.4f}")

    print("\nCharacter N-gram:")
    print(f"  Pearson Correlation:  {results['pearson']['char_ngram']:.4f}")
    print(f"  Spearman Correlation: {results['spearman']['char_ngram']:.4f}")
    print(f"  MSE:                  {results['mse']['char_ngram']:.4f}")
    print(f"  MAE:                  {results['mae']['char_ngram']:.4f}")

    print("\n" + "="*70)
    print("Interpretation:")
    print("  Correlation >0.8: Excellent alignment with humans")
    print("  Correlation 0.6-0.8: Good alignment")
    print("  Correlation <0.6: Moderate/weak alignment")
    print("="*70)

    return results


def evaluate_roc_analysis(df, metrics, output_filename='stsb_roc_comparison.png', threshold=0.75):
    """
    Perform ROC analysis and generate comparison plot

    Args:
        df: DataFrame with score_normalized and prediction columns
        metrics: Dictionary with correlation metrics from evaluate_correlation
        output_filename: Path to save the ROC plot
        threshold: Threshold for binary classification (default 0.7 for balanced classes)
                   0.7 = score ≥ 3.5/5 considered "highly similar"
    Returns:
        Dictionary with AUC scores
    """
    print("\n" + "="*70)
    print("ROC CURVE ANALYSIS")
    print("="*70)

    # Create output directory if it doesn't exist
    output_dir = Path(output_filename).parent
    if output_dir and str(output_dir) != '.':
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output directory ensured: {output_dir}")

    human_scores = df['score_normalized'].values

    # Convert continuous human scores to binary labels
    binary_labels = (human_scores >= threshold).astype(int)

    n_positive = binary_labels.sum()
    n_negative = len(binary_labels) - n_positive
    pct_positive = 100 * n_positive / len(binary_labels)

    print(f"\nBinary Classification Threshold: Human score ≥ {threshold} (≥ {threshold*5:.1f}/5)")
    print(f"Positive samples (highly similar): {n_positive} ({pct_positive:.1f}%)")
    print(f"Negative samples (less similar):   {n_negative} ({100-pct_positive:.1f}%)")

    # Warn if classes are very imbalanced
    if pct_positive < 30 or pct_positive > 70:
        print(f"⚠️  Note: Classes are imbalanced. Consider adjusting threshold for balanced evaluation.")

    # Define methods and colors
    methods_config = {
        'kg_sim_pred': {'label': 'KG Similarity (AA-KEA)', 'color': '#1f77b4'},
        'text_sim_pred': {'label': 'Text Similarity (Embeddings)', 'color': '#ff7f0e'},
        'word_overlap_pred': {'label': 'Word Overlap (Jaccard)', 'color': '#2ca02c'},
        'char_ngram_pred': {'label': 'Char N-gram', 'color': '#d62728'}
    }

    # Compute ROC curves and AUC scores for all methods
    roc_data = {}
    print(f"\nArea Under ROC Curve (AUC):")

    for col_name, config in methods_config.items():
        if col_name in df.columns:
            fpr, tpr, _ = roc_curve(binary_labels, df[col_name].values)
            auroc = roc_auc_score(binary_labels, df[col_name].values)
            roc_data[col_name] = {'fpr': fpr, 'tpr': tpr, 'auroc': auroc, **config}
            print(f"  {config['label']:40s} {auroc:.4f}")

    # Plot ROC curves
    plt.figure(figsize=(12, 9))

    for col_name, data in roc_data.items():
        plt.plot(data['fpr'], data['tpr'],
                label=f"{data['label']} (AUC = {data['auroc']:.3f})",
                linewidth=2, color=data['color'])

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: Method Comparison on STS Benchmarks', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.axis('square')

    # Build summary table dynamically
    summary_lines = ["Performance Summary:", "="*50]
    summary_lines.append(f"{'Method':<30} {'AUC':>8} {'Corr*':>8}")
    summary_lines.append("="*50)

    method_names = {
        'kg_sim_pred': 'KG Similarity',
        'text_sim_pred': 'Text Similarity',
        'word_overlap_pred': 'Word Overlap',
        'char_ngram_pred': 'Char N-gram'
    }

    for col_name, display_name in method_names.items():
        if col_name in roc_data:
            auroc = roc_data[col_name]['auroc']
            method_key = col_name.replace('_pred', '').replace('_sim', '').replace('kg', 'kg')

            # Map column names to metrics keys
            metrics_map = {
                'kg_sim_pred': 'kg',
                'text_sim_pred': 'text',
                'word_overlap_pred': 'word_overlap',
                'char_ngram_pred': 'char_ngram'
            }

            metric_key = metrics_map.get(col_name, '')
            corr = metrics['pearson'].get(metric_key, 0.0)
            summary_lines.append(f"{display_name:<30} {auroc:>8.3f} {corr:>8.3f}")

    summary_lines.append("="*50)
    summary_lines.append("*Pearson correlation with human scores")
    summary_text = "\n".join(summary_lines)

    plt.figtext(0.02, 0.02, summary_text, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
                family='monospace')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()

    # Determine which method aligns best
    method_results = {}
    for col_name in roc_data.keys():
        metrics_map = {
            'kg_sim_pred': ('KG Similarity', 'kg'),
            'text_sim_pred': ('Text Similarity', 'text'),
            'word_overlap_pred': ('Word Overlap', 'word_overlap'),
            'char_ngram_pred': ('Char N-gram', 'char_ngram')
        }

        if col_name in metrics_map:
            display_name, metric_key = metrics_map[col_name]
            method_results[display_name] = {
                'auc': roc_data[col_name]['auroc'],
                'corr': metrics['pearson'][metric_key]
            }

    best_auc = max(method_results.items(), key=lambda x: x[1]['auc'])
    best_corr = max(method_results.items(), key=lambda x: x[1]['corr'])

    print(f"\nBest performing method:")
    print(f"  By AUC (classification): {best_auc[0]} (AUC = {best_auc[1]['auc']:.3f})")
    print(f"  By Correlation: {best_corr[0]} (Pearson = {best_corr[1]['corr']:.3f})")

    if best_auc[0] == best_corr[0]:
        print(f"\n{best_auc[0]} is consistently the best method!")
    else:
        print(f"\nDifferent methods perform best for different metrics")
        print(f"Consider ensemble approach for optimal performance")

    print("\n" + "="*70)
    print(f"✓ ROC analysis complete and plot saved as '{output_filename}'")

    # Return all AUC scores
    auroc_results = {metrics_map[col][1]: roc_data[col]['auroc']
                     for col in roc_data.keys() if col in metrics_map}

    return {
        'auroc': auroc_results,
        'best_auc': best_auc,
        'best_corr': best_corr,
        'roc_data': roc_data
    }


# ============================================================================
# DEEP ANALYSIS FUNCTIONS
# ============================================================================

def complementarity_analysis(df, threshold=0.75):
    """
    Analyze which cases each method uniquely gets correct
    """
    print("\n" + "="*70)
    print("COMPLEMENTARITY ANALYSIS")
    print("="*70)

    # Create binary labels (similar/dissimilar)
    y_true = (df['score_normalized'] >= threshold).astype(int)

    # Define methods and their predictions
    methods = {
        'AA-KEA': 'kg_sim_pred',
        'Text Embedding': 'text_sim_pred',
        'Word Overlap': 'word_overlap_pred',
        'Char N-gram': 'char_ngram_pred'
    }

    # Find optimal threshold for each method and make predictions
    predictions = {}
    for method_name, score_col in methods.items():
        scores = df[score_col].values

        # Find optimal threshold
        best_f1 = 0
        best_threshold = 0.5
        for t in np.arange(0, 1.01, 0.01):
            y_pred = (scores >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        predictions[method_name] = (scores >= best_threshold).astype(int)
        print(f"\n{method_name}:")
        print(f"  Optimal threshold: {best_threshold:.3f}")
        print(f"  F1 Score: {best_f1:.4f}")
        print(f"  Accuracy: {(predictions[method_name] == y_true).mean():.4f}")

    # Pairwise complementarity
    print("\n" + "="*70)
    print("PAIRWISE COMPLEMENTARITY")
    print("="*70)

    comp_results = []

    for method1, method2 in [('AA-KEA', 'Text Embedding'),
                             ('AA-KEA', 'Word Overlap'),
                             ('AA-KEA', 'Char N-gram')]:
        pred1 = predictions[method1]
        pred2 = predictions[method2]

        m1_only = (pred1 == y_true) & (pred2 != y_true)
        m2_only = (pred2 == y_true) & (pred1 != y_true)
        both_correct = (pred1 == y_true) & (pred2 == y_true)
        both_wrong = (pred1 != y_true) & (pred2 != y_true)

        print(f"\n{method1} vs {method2}:")
        print(f"  {method1} only correct: {m1_only.sum()} ({100*m1_only.mean():.1f}%)")
        print(f"  {method2} only correct: {m2_only.sum()} ({100*m2_only.mean():.1f}%)")
        print(f"  Both correct: {both_correct.sum()} ({100*both_correct.mean():.1f}%)")
        print(f"  Both wrong: {both_wrong.sum()} ({100*both_wrong.mean():.1f}%)")

        comp_results.append({
            'method1': method1,
            'method2': method2,
            'm1_only': m1_only.sum(),
            'm2_only': m2_only.sum(),
            'both_correct': both_correct.sum(),
            'both_wrong': both_wrong.sum(),
            'm1_only_indices': df[m1_only].index.tolist(),
            'm2_only_indices': df[m2_only].index.tolist()
        })

    # Oracle ensemble
    print("\n" + "="*70)
    print("ORACLE ENSEMBLE")
    print("="*70)

    oracle_pred = y_true.copy()
    for idx in range(len(y_true)):
        # If any method got it right, use that
        for method in predictions.values():
            if method[idx] == y_true[idx]:
                oracle_pred[idx] = method[idx]
                break

    oracle_acc = (oracle_pred == y_true).mean()
    best_single = max([accuracy_score(y_true, p) for p in predictions.values()])

    print(f"  Best single method: {best_single:.4f}")
    print(f"  Oracle ensemble: {oracle_acc:.4f}")
    print(f"  Improvement: +{(oracle_acc - best_single):.4f} ({100*(oracle_acc - best_single)/best_single:.1f}%)")

    return comp_results, predictions, y_true


def case_by_case_analysis(df, comp_results, predictions, y_true, n_cases=10):
    """
    Analyze specific cases where methods differ
    """
    print("\n" + "="*70)
    print("CASE-BY-CASE ANALYSIS")
    print("="*70)

    # Focus on AA-KEA vs Text Embedding
    comp = comp_results[0]  # AA-KEA vs Text Embedding

    # Cases where AA-KEA succeeds but Text Embedding fails
    print(f"\n{'='*70}")
    print(f"CASES WHERE AA-KEA SUCCEEDS BUT TEXT EMBEDDING FAILS")
    print(f"{'='*70}")

    aa_kea_only_indices = comp['m1_only_indices'][:n_cases]

    for idx in aa_kea_only_indices:
        row = df.loc[idx]
        print(f"\nCase {idx}:")
        print(f"  Sentence 1: {row['sentence1']}")
        print(f"  Sentence 2: {row['sentence2']}")
        print(f"  Human Score: {row['score']:.2f}/5 ({'Similar' if row['score_normalized'] >= 0.75 else 'Dissimilar'})")
        print(f"  AA-KEA: {row['kg_sim_pred']:.3f} → {'Similar' if predictions['AA-KEA'][idx] == 1 else 'Dissimilar'} ✓")
        print(f"  Text Emb: {row['text_sim_pred']:.3f} → {'Similar' if predictions['Text Embedding'][idx] == 1 else 'Dissimilar'} ✗")
        print(f"  KG1: {row.get('kg1', 'N/A')}")
        print(f"  KG2: {row.get('kg2', 'N/A')}")

    # Cases where Text Embedding succeeds but AA-KEA fails
    print(f"\n{'='*70}")
    print(f"CASES WHERE TEXT EMBEDDING SUCCEEDS BUT AA-KEA FAILS")
    print(f"{'='*70}")

    text_emb_only_indices = comp['m2_only_indices'][:n_cases]

    for idx in text_emb_only_indices:
        row = df.loc[idx]
        print(f"\nCase {idx}:")
        print(f"  Sentence 1: {row['sentence1']}")
        print(f"  Sentence 2: {row['sentence2']}")
        print(f"  Human Score: {row['score']:.2f}/5 ({'Similar' if row['score_normalized'] >= 0.75 else 'Dissimilar'})")
        print(f"  AA-KEA: {row['kg_sim_pred']:.3f} → {'Similar' if predictions['AA-KEA'][idx] == 1 else 'Dissimilar'} ✗")
        print(f"  Text Emb: {row['text_sim_pred']:.3f} → {'Similar' if predictions['Text Embedding'][idx] == 1 else 'Dissimilar'} ✓")
        print(f"  KG1: {row.get('kg1', 'N/A')}")
        print(f"  KG2: {row.get('kg2', 'N/A')}")


def correlation_deep_dive(df):
    """
    Analyze why AA-KEA has lower correlation
    """
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS - WHY IS AA-KEA LOWER?")
    print("="*70)

    human_scores = df['score_normalized'].values

    methods = {
        'AA-KEA': 'kg_sim_pred',
        'Text Embedding': 'text_sim_pred',
        'Word Overlap': 'word_overlap_pred',
        'Char N-gram': 'char_ngram_pred'
    }

    # 1. Score distribution analysis
    print("\n1. SCORE DISTRIBUTION ANALYSIS")
    print("-" * 70)

    for method_name, score_col in methods.items():
        scores = df[score_col].values
        print(f"\n{method_name}:")
        print(f"  Mean: {scores.mean():.4f}")
        print(f"  Std:  {scores.std():.4f}")
        print(f"  Min:  {scores.min():.4f}")
        print(f"  Max:  {scores.max():.4f}")
        print(f"  Range: {scores.max() - scores.min():.4f}")

    print(f"\nHuman Scores:")
    print(f"  Mean: {human_scores.mean():.4f}")
    print(f"  Std:  {human_scores.std():.4f}")
    print(f"  Range: {human_scores.max() - human_scores.min():.4f}")

    # 2. Outlier analysis
    print("\n2. OUTLIER ANALYSIS")
    print("-" * 70)

    for method_name, score_col in methods.items():
        scores = df[score_col].values

        # Find cases with large prediction errors
        errors = np.abs(scores - human_scores)
        large_errors = errors > np.percentile(errors, 90)  # Top 10% errors

        print(f"\n{method_name}:")
        print(f"  Mean Absolute Error: {errors.mean():.4f}")
        print(f"  Large errors (>90th percentile): {large_errors.sum()}")

        if large_errors.sum() > 0:
            print(f"  Examples of large errors:")
            large_error_indices = np.where(large_errors)[0][:3]
            for idx in large_error_indices:
                row = df.iloc[idx]
                print(f"    Case {idx}: Human={row['score']:.1f}, "
                      f"Predicted={scores[idx]:.3f}, Error={errors[idx]:.3f}")

    # 3. Score compression analysis
    print("\n3. SCORE COMPRESSION ANALYSIS")
    print("-" * 70)

    for method_name, score_col in methods.items():
        scores = df[score_col].values

        # Check if scores are compressed (not using full range)
        human_range = human_scores.max() - human_scores.min()
        method_range = scores.max() - scores.min()
        compression = method_range / human_range

        print(f"\n{method_name}:")
        print(f"  Range usage: {method_range:.3f} / {human_range:.3f} = {compression:.2f}")
        if compression < 0.8:
            print(f"  ⚠️  Scores are compressed! Not using full range.")

    # 4. Ceiling/Floor effects
    print("\n4. CEILING/FLOOR EFFECTS")
    print("-" * 70)

    for method_name, score_col in methods.items():
        scores = df[score_col].values

        ceiling = (scores > 0.95).sum()
        floor = (scores < 0.05).sum()

        print(f"\n{method_name}:")
        print(f"  At ceiling (>0.95): {ceiling} ({100*ceiling/len(scores):.1f}%)")
        print(f"  At floor (<0.05): {floor} ({100*floor/len(scores):.1f}%)")
        if ceiling > 0.2 * len(scores) or floor > 0.2 * len(scores):
            print(f"  ⚠️  Strong ceiling/floor effects!")

    # 5. Rank correlation vs linear correlation
    print("\n5. RANK vs LINEAR CORRELATION")
    print("-" * 70)

    from scipy.stats import spearmanr, pearsonr

    for method_name, score_col in methods.items():
        scores = df[score_col].values

        pearson, _ = pearsonr(human_scores, scores)
        spearman, _ = spearmanr(human_scores, scores)

        print(f"\n{method_name}:")
        print(f"  Pearson (linear):  {pearson:.4f}")
        print(f"  Spearman (rank):   {spearman:.4f}")
        print(f"  Difference: {abs(pearson - spearman):.4f}")

        if abs(pearson - spearman) > 0.05:
            print(f"  ⚠️  Large difference suggests non-linear relationship")


def visualize_score_distributions(df, output_dir='output'):
    """
    Visualize score distributions to understand differences
    """
    print("\n" + "="*70)
    print("GENERATING DISTRIBUTION VISUALIZATIONS")
    print("="*70)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = {
        'AA-KEA': 'kg_sim_pred',
        'Text Embedding': 'text_sim_pred',
        'Word Overlap': 'word_overlap_pred',
        'Char N-gram': 'char_ngram_pred'
    }

    # 1. Scatter plots against human scores
    for idx, (method_name, score_col) in enumerate(methods.items()):
        ax = axes[idx // 2, idx % 2]

        ax.scatter(df['score_normalized'], df[score_col], alpha=0.6, s=30)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect correlation')

        # Add best fit line
        z = np.polyfit(df['score_normalized'], df[score_col], 1)
        p = np.poly1d(z)
        ax.plot([0, 1], p([0, 1]), 'b-', linewidth=2, alpha=0.7, label='Best fit')

        # Pearson correlation
        corr = df[[score_col, 'score_normalized']].corr().iloc[0, 1]

        ax.set_xlabel('Human Score (Normalized)', fontsize=11)
        ax.set_ylabel(f'{method_name} Score', fontsize=11)
        ax.set_title(f'{method_name}\nPearson r = {corr:.4f}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    plt.suptitle('Method Scores vs Human Judgments', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/score_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/score_distributions.png")
    plt.close()

    # 2. Distribution histograms
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Human scores
    axes[0, 0].hist(df['score_normalized'], bins=20, alpha=0.7, color='gray', edgecolor='black')
    axes[0, 0].set_xlabel('Score', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Human Scores', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)

    # Method scores
    for idx, (method_name, score_col) in enumerate(methods.items(), 1):
        row = (idx) // 3
        col = (idx) % 3

        axes[row, col].hist(df[score_col], bins=20, alpha=0.7, edgecolor='black')
        axes[row, col].set_xlabel('Score', fontsize=11)
        axes[row, col].set_ylabel('Frequency', fontsize=11)
        axes[row, col].set_title(method_name, fontsize=12, fontweight='bold')
        axes[row, col].grid(alpha=0.3)

        # Add statistics
        mean = df[score_col].mean()
        std = df[score_col].std()
        axes[row, col].axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean={mean:.2f}')
        axes[row, col].legend(fontsize=9)

    plt.suptitle('Score Distribution Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/score_histograms.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/score_histograms.png")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("KNOWLEDGE GRAPH EVALUATION FOR STS BENCHMARKS")
    print("="*70)

    # Verify required files exist
    required_files = ['stsb_merged.csv', 'aa_kea_results.csv']
    missing_files = [f for f in required_files if not Path(f).exists()]

    if missing_files:
        print("\nERROR: Required files not found in current directory:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure you are running this script from: src/evaluation/")
        print("Usage: cd src/evaluation && python kg_evaluation.py")
        return None

    # Load models
    embedding_model = load_models()

    # Load merged dataset with pre-computed KG similarity scores
    dataset = load_merged_dataset(
        dataset_csv='stsb_merged.csv',
        kg_similarity_csv='aa_kea_results.csv'
    )

    # Compute predictions (text similarity baselines)
    # KG similarity is already loaded from aa_kea_results.csv
    dataset = compute_predictions(dataset, embedding_model)

    # Evaluate correlation with human judgments
    correlation_metrics = evaluate_correlation(dataset)

    # Perform ROC analysis
    roc_results = evaluate_roc_analysis(
        dataset,
        correlation_metrics,
        output_filename='output/stsb_merged_roc_comparison.png',
        threshold=0.75  # 0.7 = score ≥ 3.5/5 (balanced: 54% positive, 46% negative)
    )

    # ===== DEEP ANALYSIS =====
    print("\n" + "="*70)
    print("DEEP ANALYSIS - Understanding AA-KEA's Behavior")
    print("="*70)

    # 1. Complementarity analysis
    comp_results, predictions, y_true = complementarity_analysis(dataset, threshold=0.75)

    # 2. Case-by-case analysis
    case_by_case_analysis(dataset, comp_results, predictions, y_true, n_cases=5)

    # 3. Correlation deep dive
    correlation_deep_dive(dataset)

    # 4. Visualizations
    visualize_score_distributions(dataset, output_dir='output')

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print("\nResults saved:")
    print("  - ROC plot: output/stsb_merged_roc_comparison.png")
    print(f"\nEvaluated {len(dataset)} sentence pairs from merged STS-B + STS12 dataset")
    print("\nDataset composition:")
    print("  - Rows 0-99: STS-B data")
    print("  - Rows 100-199: STS12 data")
    print("\nMethod comparison:")
    print("  1. KG Similarity (AA-KEA): Pre-computed from aa_kea_results.csv")
    print("  2. Text Similarity (Embeddings): all-MiniLM-L6-v2 (semantic baseline)")
    print("  3. Word Overlap (Jaccard): Simple token-based overlap")
    print("  4. Character N-gram: Character-level 3-gram similarity")
    print("\nNote: All baselines use fair comparisons (no training on STS-B data)")

    return {
        'data': dataset,
        'correlation': correlation_metrics,
        'roc': roc_results
    }


if __name__ == "__main__":
    results = main()

