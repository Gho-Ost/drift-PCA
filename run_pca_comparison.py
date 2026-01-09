import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import patchworklib as pw
import logging
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from analysis_methods.drift_component_analysis import DriftComponentAnalysis
from visualization_utils import create_scatter_plot, add_biplot_arrows

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_and_scale_data(dataset_name="sea"):
    """Load pre and post drift data for a given dataset."""
    pre_path = f"stream_datasets/{dataset_name}_pre.csv"
    post_path = f"stream_datasets/{dataset_name}_post.csv"

    if not os.path.exists(pre_path) or not os.path.exists(post_path):
        raise FileNotFoundError(f"Data files for {dataset_name} not found in stream_datasets/")

    df_pre = pd.read_csv(pre_path)
    df_post = pd.read_csv(post_path)

    X_pre = df_pre.iloc[:, :-1]
    y_pre = df_pre.iloc[:, -1].values
    X_post = df_post.iloc[:, :-1]
    y_post = df_post.iloc[:, -1].values
    
    feature_names = list(X_pre.columns)

    # Scale data
    scaler = StandardScaler()
    X_pre_scaled = scaler.fit_transform(X_pre)
    X_post_scaled = scaler.transform(X_post)

    return X_pre_scaled, y_pre, X_post_scaled, y_post, feature_names


def run_comparison():
    logger.info("Starting PCA vs Drift-PCA comparison")
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Load data
    try:
        X_pre, y_pre, X_post, y_post, feature_names = load_and_scale_data("sea")
        logger.info(f"Loaded SEA dataset: {len(X_pre)} pre-drift samples, {len(X_post)} post-drift samples")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # --- Standard PCA ---
    logger.info("Running Standard PCA...")
    start_time = time.time()
    pca = PCA(n_components=2)
    pca.fit(X_pre)
    
    X_pca_pre = pca.transform(X_pre)
    pca_fit_time = time.time() - start_time
    
    start_time = time.time()
    X_pca_post = pca.transform(X_post)
    pca_transform_time = time.time() - start_time
    
    # --- Drift PCA ---
    logger.info("Running Drift PCA...")
    start_time = time.time()
    dca = DriftComponentAnalysis(n_components=2)
    dca.fit(X_pre, X_post)
    
    X_dca_pre = dca.transform(X_pre)
    dca_fit_time = time.time() - start_time
    
    start_time = time.time()
    X_dca_post = dca.transform(X_post)
    dca_transform_time = time.time() - start_time

    # --- Visualization ---
    logger.info("Generating visualizations...")
    
    # Row 1: PCA
    fig_pca_pre = create_scatter_plot(
        X_pca_pre, y_pre, "PC1", "PC2", "Standard PCA (Pre-Drift Data)", pca_fit_time, show_time=True
    )
    add_biplot_arrows(fig_pca_pre, pca, X_pca_pre, feature_names=feature_names)
    
    # Apply global axis limits for PCA row based on both pre and post
    x_min = min(X_pca_pre[:, 0].min(), X_pca_post[:, 0].min())
    x_max = max(X_pca_pre[:, 0].max(), X_pca_post[:, 0].max())
    y_min = min(X_pca_pre[:, 1].min(), X_pca_post[:, 1].min())
    y_max = max(X_pca_pre[:, 1].max(), X_pca_post[:, 1].max())
    
    fig_pca_pre.set_xlim(x_min, x_max)
    fig_pca_pre.set_ylim(y_min, y_max)
    
    fig_pca_post = create_scatter_plot(
        X_pca_post, y_post, "PC1", "PC2", "Standard PCA (Post-Drift Data)", pca_transform_time, show_time=True  # Using show_time to show transform time
    )
    fig_pca_post.set_xlim(x_min, x_max)
    fig_pca_post.set_ylim(y_min, y_max)

    # Row 2: Drift PCA
    fig_dca_pre = create_scatter_plot(
        X_dca_pre, y_pre, "D1", "D2", "Drift PCA (Pre-Drift Data)", dca_fit_time, show_time=True
    )
    add_biplot_arrows(fig_dca_pre, dca.pca, X_dca_pre, feature_names=feature_names) # Use internal pca object
    
    # Apply global axis limits for DCA row
    x_min_d = min(X_dca_pre[:, 0].min(), X_dca_post[:, 0].min())
    x_max_d = max(X_dca_pre[:, 0].max(), X_dca_post[:, 0].max())
    y_min_d = min(X_dca_pre[:, 1].min(), X_dca_post[:, 1].min())
    y_max_d = max(X_dca_pre[:, 1].max(), X_dca_post[:, 1].max())
    
    fig_dca_pre.set_xlim(x_min_d, x_max_d)
    fig_dca_pre.set_ylim(y_min_d, y_max_d)

    fig_dca_post = create_scatter_plot(
        X_dca_post, y_post, "D1", "D2", "Drift PCA (Post-Drift Data)", dca_transform_time, show_time=True
    )
    fig_dca_post.set_xlim(x_min_d, x_max_d)
    fig_dca_post.set_ylim(y_min_d, y_max_d)
    add_biplot_arrows(fig_dca_post, dca.pca, X_dca_post, feature_names=feature_names)

    # Combine
    row1 = fig_pca_pre | fig_pca_post
    row2 = fig_dca_pre | fig_dca_post
    
    final_layout = row1 / row2
    
    output_path = "results/comparison_result.png"
    final_layout.savefig(output_path, dpi=150)
    logger.info(f"Comparison saved to {output_path}")


if __name__ == "__main__":
    run_comparison()
