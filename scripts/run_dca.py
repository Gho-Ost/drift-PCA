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
from sklearn.inspection import permutation_importance

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

def load_and_scale_data(dataset_name="sea", data_dir="stream_datasets", ignore_classes=False, has_target=True):
    """Load pre and post drift data."""
    pre_path = os.path.join(data_dir, f"{dataset_name}_pre.csv")
    post_path = os.path.join(data_dir, f"{dataset_name}_post.csv")

    if not os.path.exists(pre_path) or not os.path.exists(post_path):
        raise FileNotFoundError(f"Data files for {dataset_name} not found in {data_dir}/")

    df_pre = pd.read_csv(pre_path)
    df_post = pd.read_csv(post_path)

    if has_target:
        X_pre = df_pre.iloc[:, :-1].values
        y_pre = df_pre.iloc[:, -1].values
        X_post = df_post.iloc[:, :-1].values
        y_post = df_post.iloc[:, -1].values
        feature_names = list(df_pre.columns[:-1])
    else:
        X_pre = df_pre.values
        X_post = df_post.values
        y_pre = np.zeros(len(X_pre), dtype=int)
        y_post = np.zeros(len(X_post), dtype=int)
        feature_names = list(df_pre.columns)

    if ignore_classes or not has_target:
        y_pre = np.zeros(len(X_pre), dtype=int)
        y_post = np.zeros(len(X_post), dtype=int)
        classes = np.array([0])
    else:
        classes = np.unique(np.concatenate([y_pre, y_post]))
        if len(classes) > 6:
            raise ValueError(f"Dataset {dataset_name} has {len(classes)} classes. Maximum supported is 6.")

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
    parser.add_argument("--discrete_boundary", action="store_true", help="Display hard decision boundaries instead of class probabilities")
    parser.add_argument("--drift_mode", type=str, choices=["data", "global", "per-class"], default="per-class", 
                        help="Drift calculation mode: 'data' (ignore classes), 'global' (use classes but calc global drift), 'per-class' (calc drift per class)")
    parser.add_argument("--no_target", action="store_true", help="Dataset does not have a target/class column. All columns will be used as features.")
    parser.add_argument("--unscaled_loadings", action="store_true", help="Disable scaling of feature loadings by singular values in the Biplot")
    parser.add_argument("--color_scheme", type=str, choices=["class", "drift"], default="class", help="Color scheme for plots")
    parser.add_argument("--highlight_misclassifications", action="store_true", help="Highlight post-drift points misclassified by the pre-drift model")
    parser.add_argument("--hide_pre_drift_points", action="store_true", help="Do not draw pre-drift points on the scatter plot")
    parser.add_argument("--feature_importance", action="store_true", help="Use model explainability to color code feature importance on the Loadings compass rose")

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    
    try:
        X_pre, y_pre, X_post, y_post, feature_names, classes = load_and_scale_data(
            dataset_name=args.dataset, 
            data_dir=args.data_dir, 
            ignore_classes=(args.drift_mode == "data"),
            has_target=(not args.no_target)
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Determine default color scheme
    color_scheme = args.color_scheme
    if color_scheme is None:
        if args.drift_mode == "data" or len(classes) == 1:
            color_scheme = "drift"
        else:
            color_scheme = "class"

    # Validate highlight_misclassifications
    if args.highlight_misclassifications:
        if color_scheme == "drift":
            logger.warning("Misclassifications cannot be highlighted with the 'drift' color scheme. Disabling highlight.")
            args.highlight_misclassifications = False
        elif args.drift_mode == "data" or args.no_target or len(classes) < 2:
            logger.warning("Misclassifications cannot be highlighted without class targets. Disabling highlight.")
            args.highlight_misclassifications = False

    # Validate feature_importance
    if args.feature_importance:
        if args.drift_mode == "data" or args.no_target or len(classes) < 2:
            logger.warning("Feature importance requires class targets. Disabling feature importance.")
            args.feature_importance = False

    # Validate boundary
    if not args.no_boundary:
        if color_scheme == "drift":
            logger.warning("Decision boundary cannot be displayed with the 'drift' color scheme. Boundary disabled.")
            args.no_boundary = True
        elif args.drift_mode == "data" or args.no_target or len(classes) < 2:
            args.no_boundary = True

    needs_model = (not args.no_boundary) or args.highlight_misclassifications or args.feature_importance

    # Train model on Pre-drift data ONLY
    pre_drift_model = None
    if needs_model:
        if args.model == "svc":
            pre_drift_model = SVC(kernel='rbf', probability=True, random_state=42)
        else:
            pre_drift_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            
        logger.info(f"Training {args.model} model on Pre-drift data...")
        pre_drift_model.fit(X_pre, y_pre)

    feature_importances = None
    if args.feature_importance and pre_drift_model is not None:
        logger.info("Calculating feature importances using permutation importance...")
        result = permutation_importance(pre_drift_model, X_pre, y_pre, n_repeats=5, random_state=42, n_jobs=-1)
        feature_importances = np.maximum(result.importances_mean, 0)

    # Fit DriftComponentAnalysis2 using SVD
    by_class = (args.drift_mode == "per-class")
    dca = DriftComponentAnalysis2(n_components=2, by_class=by_class)
    dca.fit(X_pre, X_post, y_pre, y_post)

    # Setup the unified GridSpec figure
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1])
    
    ax_scatter = fig.add_subplot(gs[0, :])     
    ax_loadings = fig.add_subplot(gs[1, 0])     
    ax_drift = fig.add_subplot(gs[1, 1])        

    # Plot 1: Main Scatter Plot with Optional Boundaries (Top spanning)
    contour_info = plot_dca_scatter(
        X_pre, y_pre, X_post, y_post, dca, ax=ax_scatter, 
        pre_drift_model=pre_drift_model, color_scheme=color_scheme,
        discrete_boundary=args.discrete_boundary,
        draw_boundary=not args.no_boundary,
        highlight_misclassifications=args.highlight_misclassifications,
        hide_pre_drift_points=args.hide_pre_drift_points
    )
    
    if isinstance(contour_info, tuple):
        contour, is_discrete = contour_info
    else:
        contour, is_discrete = contour_info, False

    if contour is not None and not is_discrete and len(classes) == 2:
        # Add colorbar purely for the boundary probability on the side of ax_scatter
        cbar_ax = fig.add_axes([0.91, 0.55, 0.02, 0.3])
        fig.colorbar(contour, cax=cbar_ax, label=f"Probability of Class {classes[1]}")
        plt.subplots_adjust(right=0.88, hspace=0.3, wspace=0.3)

    # Plot 2: Loadings Compass Rose (Bottom Left)
    plot_loadings_compass(dca, ax=ax_loadings, feature_names=feature_names, scale_loadings=(not args.unscaled_loadings), feature_importances=feature_importances)

    # Plot 3: Drift Compass Rose (Bottom Right)
    plot_drift_compass(dca, ax=ax_drift, classes=classes if args.drift_mode != "data" else None, color_scheme=color_scheme)

    if contour is None:
        plt.tight_layout()

    combined_path = os.path.join(args.results_dir, f"{args.dataset}_dca2_combined.png")
    fig.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    run_dca()
