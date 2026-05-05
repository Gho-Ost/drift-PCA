import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Setup parent path so relative imports work if run as script
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis_methods.algorithm_comparison import (
    TruncatedSVDFitter,
    PCAFitter,
    UMAPFitter,
    SSNPFitter,
    plot_algorithm_scatter,
    plot_compass_rose,
    plot_algorithm_drift_compass
)
from scripts.run_dca import load_and_scale_binary_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def compute_diff_matrix(X_pre, X_post, y_pre, y_post):
    classes = np.intersect1d(np.unique(y_pre), np.unique(y_post))
    diff_vectors = []
    
    for c in classes:
        X_ref_c = X_pre[y_pre == c]
        X_cur_c = X_post[y_post == c]
        
        if len(X_ref_c) < 2 or len(X_cur_c) < 2:
            continue
            
        mean_ref = np.mean(X_ref_c, axis=0)
        std_ref = np.std(X_ref_c, axis=0)
        mean_cur = np.mean(X_cur_c, axis=0)
        std_cur = np.std(X_cur_c, axis=0)
        
        diff_vectors.append(mean_cur - mean_ref)
        diff_vectors.append(std_cur - std_ref)
        
    return np.array(diff_vectors)

def run_comparison():
    parser = argparse.ArgumentParser(description="Run Algorithm Comparison")
    parser.add_argument("--data_dir", type=str, default="data/thu_stream_datasets", help="Directory containing the dataset files")
    parser.add_argument("--results_dir", type=str, default="results_comparison", help="Directory to save the results")
    parser.add_argument("--dataset", type=str, default="thu_linear_sudden_f5", help="Name of the dataset (without _pre.csv)")
    
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    
    logger.info(f"Loading data for {args.dataset}...")
    try:
        X_pre, y_pre, X_post, y_post, feature_names, classes = load_and_scale_binary_data(
            dataset_name=args.dataset, data_dir=args.data_dir
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    c0, c1 = classes[0], classes[1]
    
    logger.info("Computing diff matrix...")
    diff_matrix = compute_diff_matrix(X_pre, X_post, y_pre, y_post)

    fitters = [
        TruncatedSVDFitter(),
        PCAFitter(),
        UMAPFitter(),
        SSNPFitter()
    ]

    # Setup the visual grid. N fitters, 4 columns (Pre, Post, Loadings, Drift Vectors)
    n_algorithms = len(fitters)
    
    fig = plt.figure(figsize=(24, 5 * n_algorithms))
    gs = gridspec.GridSpec(n_algorithms, 4, wspace=0.3, hspace=0.4)
    
    for row_idx, fitter in enumerate(fitters):
        logger.info(f"Running {fitter.name}...")
        
        # Fit and transform
        fitter.fit(X_pre, X_post, y_pre, y_post, diff_matrix)
        X_pre_trans, X_post_trans = fitter.transform(X_pre, X_post)
        
        # Calculate shared axis limits
        x_min = min(X_pre_trans[:, 0].min(), X_post_trans[:, 0].min())
        x_max = max(X_pre_trans[:, 0].max(), X_post_trans[:, 0].max())
        y_min = min(X_pre_trans[:, 1].min(), X_post_trans[:, 1].min())
        y_max = max(X_pre_trans[:, 1].max(), X_post_trans[:, 1].max())
        
        # Add a small margin
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        
        x_lims = (x_min - x_margin, x_max + x_margin)
        y_lims = (y_min - y_margin, y_max + y_margin)
        
        # Draw Pre Scatter
        ax_pre = fig.add_subplot(gs[row_idx, 0])
        plot_algorithm_scatter(X_pre_trans, y_pre, ax_pre, f"{fitter.name} - PRE Drift", is_pre=True, c0=c0, c1=c1)
        ax_pre.set_xlim(x_lims)
        ax_pre.set_ylim(y_lims)
        
        # Draw Post Scatter
        ax_post = fig.add_subplot(gs[row_idx, 1])
        plot_algorithm_scatter(X_post_trans, y_post, ax_post, f"{fitter.name} - POST Drift", is_pre=False, c0=c0, c1=c1)
        ax_post.set_xlim(x_lims)
        ax_post.set_ylim(y_lims)
        
        # Draw Loadings
        ax_loadings = fig.add_subplot(gs[row_idx, 2])
        plot_compass_rose(
            fitter.components_, ax_loadings, feature_names=feature_names, 
            loading_scale_factors=fitter.loading_scale_factors_
        )
        
        # Draw Drift Vectors if supported
        ax_drift = fig.add_subplot(gs[row_idx, 3])
        vectors_trans = fitter.transform_vectors(diff_matrix)
        if vectors_trans is not None:
            plot_algorithm_drift_compass(vectors_trans, ax_drift, classes=classes)
        else:
            ax_drift.set_title("No Drift Vectors")
            ax_drift.axis('off')
            ax_drift.text(0.5, 0.5, "Algorithm not fitted on\nmean/std difference vectors.", 
                          ha='center', va='center', fontsize=12, color='gray', transform=ax_drift.transAxes)

    plt.suptitle(f"Algorithm Comparison on {args.dataset}", fontsize=18, y=0.92)
    
    combined_path = os.path.join(args.results_dir, f"{args.dataset}_comparison.png")
    fig.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved comparison plot to {combined_path}")

if __name__ == "__main__":
    run_comparison()
