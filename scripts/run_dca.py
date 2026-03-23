import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Setup parent path so relative imports work if run as script
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis_methods.drift_component_analysis import DriftComponentAnalysis2
from analysis_methods.dca2_utils import (
    plot_dca_scatter,
    plot_loadings_compass,
    plot_drift_compass
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_and_scale_binary_data(dataset_name="sea", data_dir="stream_datasets"):
    """Load pre and post drift data, verifying it is binary."""
    pre_path = os.path.join(data_dir, f"{dataset_name}_pre.csv")
    post_path = os.path.join(data_dir, f"{dataset_name}_post.csv")

    if not os.path.exists(pre_path) or not os.path.exists(post_path):
        raise FileNotFoundError(f"Data files for {dataset_name} not found in {data_dir}/")

    df_pre = pd.read_csv(pre_path)
    df_post = pd.read_csv(post_path)

    X_pre = df_pre.iloc[:, :-1].values
    y_pre = df_pre.iloc[:, -1].values
    X_post = df_post.iloc[:, :-1].values
    y_post = df_post.iloc[:, -1].values
    
    feature_names = list(df_pre.columns[:-1])

    # Check for binary classification
    classes = np.unique(np.concatenate([y_pre, y_post]))
    if len(classes) != 2:
        raise ValueError(f"Dataset {dataset_name} has {len(classes)} classes, but DCA2 requires exactly 2 classes.")

    # Scale data
    scaler = StandardScaler()
    X_pre_scaled = scaler.fit_transform(X_pre)
    X_post_scaled = scaler.transform(X_post)

    return X_pre_scaled, y_pre, X_post_scaled, y_post, feature_names, classes

def run_dca():
    parser = argparse.ArgumentParser(description="Run Drift PCA Comparison (V2 Unscaled)")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the dataset files")
    parser.add_argument("--results_dir", type=str, default="results_dca2", help="Directory to save the results")
    parser.add_argument("--dataset", type=str, default="sea", help="Name of the dataset")
    parser.add_argument("--model", type=str, choices=["svc", "rf"], default="svc", help="Pre-drift model to use for boundary")
    parser.add_argument("--no_boundary", action="store_true", help="Do not draw decision boundary")
    parser.add_argument("--by_class", action="store_true", help="Calculate drift vectors separated by class")
    
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    
    try:
        X_pre, y_pre, X_post, y_post, feature_names, classes = load_and_scale_binary_data(
            dataset_name=args.dataset, data_dir=args.data_dir
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Train model on Pre-drift data ONLY
    pre_drift_model = None
    if not args.no_boundary:
        if args.model == "svc":
            pre_drift_model = SVC(kernel='rbf', probability=True, random_state=42)
        else:
            pre_drift_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            
        pre_drift_model.fit(X_pre, y_pre)

    # Fit DriftComponentAnalysis2 using SVD
    dca = DriftComponentAnalysis2(n_components=2, by_class=args.by_class)
    dca.fit(X_pre, X_post, y_pre, y_post)

    # Setup the unified GridSpec figure
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1])
    
    ax_scatter = fig.add_subplot(gs[0, :])     
    ax_loadings = fig.add_subplot(gs[1, 0])     
    ax_drift = fig.add_subplot(gs[1, 1])        

    # Plot 1: Main Scatter Plot with Optional Boundaries (Top spanning)
    contour = plot_dca_scatter(
        X_pre, y_pre, X_post, y_post, dca, ax=ax_scatter, 
        pre_drift_model=pre_drift_model
    )
    if contour is not None:
        # Add colorbar purely for the boundary probability on the side of ax_scatter
        cbar_ax = fig.add_axes([0.91, 0.55, 0.02, 0.3])
        fig.colorbar(contour, cax=cbar_ax, label="Probability of Class 1")
        plt.subplots_adjust(right=0.88, hspace=0.3, wspace=0.3)

    # Plot 2: Loadings Compass Rose (Bottom Left)
    plot_loadings_compass(dca, ax=ax_loadings, feature_names=feature_names)

    # Plot 3: Drift Compass Rose (Bottom Right)
    plot_drift_compass(dca, ax=ax_drift, classes=classes)

    if contour is None:
        plt.tight_layout()

    combined_path = os.path.join(args.results_dir, f"{args.dataset}_dca2_combined.png")
    fig.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    run_dca()
