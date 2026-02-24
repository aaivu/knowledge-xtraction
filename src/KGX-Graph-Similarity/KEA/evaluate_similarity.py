"""
Evaluate Semantic WL Kernel Similarity Scores

This script compares the calculated similarity scores from the Semantic WL Kernel
against the ground truth scores from the STS-B dataset.

It computes and reports:
- Correlation: Pearson and Spearman coefficients
- Error Metrics: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
- A scatter plot to visualize the relationship between predicted and actual scores.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# --- File Paths ---
GROUND_TRUTH_PATH = r"D:\FYP\aaivu\knowledge-xtraction\src\evaluation\stsb_merged.csv"
CALCULATED_PATH = r"D:\FYP\aaivu\knowledge-xtraction\src\KGX-Graph-Similarity\KEA\mistralai_Mistral-7B-Instruct-v0.2_stsb2_kgv.csv"
OUTPUT_PLOT_PATH = r"D:\FYP\aaivu\knowledge-xtraction\src\KGX-Graph-Similarity\KEA\similarity_evaluation_scatter.png"


def main():
    """
    Main function to load data, compute metrics, and generate plot.
    """
    print("--- Similarity Score Evaluation ---")

    # --- 1. Load Data ---
    try:
        print(f"Loading ground truth data from: {GROUND_TRUTH_PATH}")
        df_truth = pd.read_csv(GROUND_TRUTH_PATH)
        
        print(f"Loading calculated scores from: {CALCULATED_PATH}")
        df_calc = pd.read_csv(CALCULATED_PATH, engine='python', quotechar='"', escapechar='\\')
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure file paths are correct.")
        return

    # --- 2. Filter and Merge Data ---
    # Filter out rows where kg1 or kg2 is empty
    df_calc = df_calc[df_calc['kg1'].notna() & (df_calc['kg1'] != '[]')]
    df_calc = df_calc[df_calc['kg2'].notna() & (df_calc['kg2'] != '[]')]
    
    print(f"Filtered calculated scores. Rows remaining: {len(df_calc)}")

    # Assuming 'row_id' is the common key. If not, we can use index.
    if 'row_id' in df_truth.columns and 'row_id' in df_calc.columns:
        df_merged = pd.merge(df_truth, df_calc, on='row_id', suffixes=('_truth', '_calc'))
    else:
        # Fallback to merging on index if row_id is not present
        df_merged = df_truth.join(df_calc, lsuffix='_truth', rsuffix='_calc')

    # Extract relevant columns
    ground_truth_scores = df_merged['score']
    predicted_scores_0_1 = df_merged['semantic_wl_similarity']
    
    # Scale predicted scores from [0, 1] to [0, 5] to match ground truth
    predicted_scores_0_5 = predicted_scores_0_1 #* 5.0
    
    print(f"\nData loaded and merged. Total pairs: {len(df_merged)}")

    # --- 3. Calculate Metrics ---
    print("\n--- Performance Metrics ---")
    
    # Correlation
    pearson_corr, p_pearson = pearsonr(ground_truth_scores, predicted_scores_0_5)
    spearman_corr, p_spearman = spearmanr(ground_truth_scores, predicted_scores_0_5)
    
    print(f"Pearson Correlation: {pearson_corr:.4f} (p-value: {p_pearson:.4f})")
    print(f"Spearman Correlation: {spearman_corr:.4f} (p-value: {p_spearman:.4f})")
    
    # Error Metrics
    mae = mean_absolute_error(ground_truth_scores, predicted_scores_0_5)
    mse = mean_squared_error(ground_truth_scores, predicted_scores_0_5)
    rmse = np.sqrt(mse)
    
    print(f"\nMean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # --- 4. Generate Visualization ---
    print("\n--- Generating Visualization ---")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot with regression line
    sns.regplot(
        x=ground_truth_scores, 
        y=predicted_scores_0_5, 
        ax=ax,
        scatter_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'w', 'linewidths': 0.5},
        line_kws={'color': 'red', 'linestyle': '--', 'linewidth': 2}
    )
    
    ax.set_title('Ground Truth vs. Predicted Similarity (Semantic WL Kernel)', fontsize=16, pad=20)
    ax.set_xlabel('Ground Truth Score (STS-B)', fontsize=12)
    ax.set_ylabel('Predicted Score (Scaled 0-5)', fontsize=12)
    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, 5.5)
    ax.set_aspect('equal', adjustable='box')
    
    # Add correlation text
    stats_text = (
        f"Pearson: {pearson_corr:.3f}\n"
        f"Spearman: {spearman_corr:.3f}\n"
        f"RMSE: {rmse:.3f}"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    try:
        plt.savefig(OUTPUT_PLOT_PATH, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to: {OUTPUT_PLOT_PATH}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.show()


if __name__ == "__main__":
    main()
