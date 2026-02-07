import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    adjusted_rand_score, normalized_mutual_info_score,
    mean_squared_error, mean_absolute_error
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


def evaluate_roc_analysis(df, metrics, output_filename='stsb_roc_comparison.png', threshold=0.7):
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
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("KNOWLEDGE GRAPH EVALUATION FOR STS BENCHMARKS")
    print("="*70)

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
        output_filename='stsb_merged_roc_comparison.png',
        threshold=0.7  # 0.7 = score ≥ 3.5/5 (balanced: 54% positive, 46% negative)
    )

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print("\nResults saved:")
    print("  - ROC plot: stsb_merged_roc_comparison.png")
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

