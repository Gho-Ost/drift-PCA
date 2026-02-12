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
from visualization_utils import create_scatter_plot, add_biplot_arrows, add_drift_info_to_plot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

def calculate_shapley_values(X_pre, X_post, y_pre, y_post, class_idx=1):
    """
    Calculate Shapley values for the dataset.
    Note: class_idx=1 assumes binary classification (targets 0 and 1), and we want positive class explanations.
    """
    logger.info("Training model for Shapley values...")
    y = np.concatenate((y_pre, y_post))
    
    # Train simple model (Random Forest)
    try:
        le_full = LabelEncoder()
        le_full.fit(y) 
        y_pre_num = le_full.transform(y_pre)
        # y_post_num = le_full.transform(y_post)
        
        # Train on Pre and explain Pre and Post
        clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        clf.fit(X_pre, y_pre_num)
        
        # Generate Explanations (SHAP)
        logger.info("Computing SHAP values...")
        # using TreeExplainer which is faster for trees
        explainer = shap.TreeExplainer(clf)
        
        # check_additivity=False to save time
        shap_values_pre_raw = explainer.shap_values(X_pre, check_additivity=False) 
        shap_values_post_raw = explainer.shap_values(X_post, check_additivity=False)

        # Handling different shap versions/outputs
        # For binary clf, shap_values is a list of [n_samples, n_features] for each class
        if isinstance(shap_values_pre_raw, list):
            # Select class index. If binary 0/1, usually index 1 is the positive class.
            c_idx = class_idx if class_idx < len(shap_values_pre_raw) else 0
            shap_values_pre = shap_values_pre_raw[c_idx]
            shap_values_post = shap_values_post_raw[c_idx]
        else:
            # If it returns a single array (e.g. regression or new shap versions), use as is
            # If 3D array (samples, features, outputs)
            if len(shap_values_pre_raw.shape) == 3:
                shap_values_pre = shap_values_pre_raw[:, :, class_idx]
                shap_values_post = shap_values_post_raw[:, :, class_idx]
            else:
                shap_values_pre = shap_values_pre_raw
                shap_values_post = shap_values_post_raw
            
    except Exception as e:
        logger.warning(f"Model training/explanation failed: {e}")
        return None, None

    return shap_values_pre, shap_values_post

def calculate_incorrect_classifications(X_pre, X_post, y_pre, y_post):
    """
    Filter data to only return incorrectly classified examples.
    Trains a model on Pre data (split train/test) and evaluates on Pre-test and Post.
    Returns the subsets of X corresponding to errors.
    """
    y_all = np.concatenate((y_pre, y_post))
    le_full = LabelEncoder()
    le_full.fit(y_all) 
    y_pre_num = le_full.transform(y_pre)
    y_post_num = le_full.transform(y_post)
    
    clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)

    # Split Pre data to have a holdout for "Pre-drift errors"
    X_train, X_test, y_train, y_test = train_test_split(X_pre, y_pre_num, test_size=0.3, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred_test = clf.predict(X_test)
    y_pred_post = clf.predict(X_post)

    mask_pre = y_pred_test != y_test
    mask_post = y_pred_post != y_post_num

    incorrect_clfs_pre = X_test[mask_pre]
    incorrect_pre_y = y_test[mask_pre]
    
    incorrect_clfs_post = X_post[mask_post]
    incorrect_post_y = y_post[mask_post]

    return incorrect_clfs_pre, incorrect_clfs_post, incorrect_pre_y, incorrect_post_y



def run_comparison(args):
    logger.info(f"Starting PCA vs Drift-PCA comparison for dataset: {args.dataset}")
    
    # Ensure results directory exists
    output_dir = f"results/{args.dataset}{'_Anchor' if args.add_anchor_point else ''}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data
    try:
        X_pre, y_pre, X_post, y_post, feature_names = load_and_scale_data(args.dataset)
        logger.info(f"Loaded {args.dataset} dataset: {len(X_pre)} pre, {len(X_post)} post")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Calculate Representations
    
    # Data (Raw)
    # Already loaded as X_pre, X_post
    
    # Shapley
    logger.info("Calculating Shapley values...")
    shap_pre, shap_post = calculate_shapley_values(X_pre, X_post, y_pre, y_post)
    if shap_pre is None:
        logger.error("Shapley calculation failed, skipping related comparisons.")
    
    # Incorrect Classifications (Errors)
    logger.info("Calculating Incorrect Classifications...")
    err_pre, err_post, err_y_pre, err_y_post = calculate_incorrect_classifications(X_pre, X_post, y_pre, y_post)
    logger.info(f"Errors found: Pre={len(err_pre)}, Post={len(err_post)}")

    # Define the dictionary of available datasets for loop
    # Structure: Key -> { "train": (X_train_pre, X_train_post), "vis": (X_vis_pre, y_vis_pre, X_vis_post, y_vis_post) }
    
    data_dict = {}
    
    # Raw Data
    data_dict["Data"] = {
        "train": (X_pre, X_post),
        "vis": (X_pre, y_pre, X_post, y_post),
        "features": feature_names
    }
    
    # Shapley
    if shap_pre is not None:
        data_dict["Shapley"] = {
            "train": (shap_pre, shap_post), 
            "vis": (shap_pre, y_pre, shap_post, y_post),
            "features": feature_names # SHAP has same feature dimensions
        }
        
    # Errors
    # Only add if we have enough samples to fit PCA (>1)
    if len(err_pre) > 2 and len(err_post) > 2:
        data_dict["Errors"] = {
            "train": (err_pre, err_post),
            "vis": (err_pre, err_y_pre, err_post, err_y_post),
            "features": feature_names
        }
    else:
        logger.warning("Not enough error samples for 'Errors' orientation/visualization")

    # Loop: Orientation x Visualization
    
    orient_keys = list(data_dict.keys())
    vis_keys = list(data_dict.keys())
    
    for orient_name in orient_keys:
        for vis_name in vis_keys:
            if orient_name != "Shapley" and vis_name == "Shapley":
                logger.info(f"--- Skipping: Orientation={orient_name}, Visualization={vis_name} ---")
                continue

            logger.info(f"--- Processing: Orientation={orient_name}, Visualization={vis_name} ---")
            
            orient_data = data_dict[orient_name]
            vis_data = data_dict[vis_name]
            
            X_train_pre, X_train_post = orient_data["train"]
            X_vis_pre, y_vis_pre, X_vis_post, y_vis_post = vis_data["vis"]
            vis_features = vis_data["features"]
            
            # Standard PCA for comparison
            pca = PCA(n_components=2)
            t0 = time.time()
            pca.fit(X_train_pre)
            pca_fit_time = time.time() - t0
            
            t0 = time.time()
            X_pca_pre_vis = pca.transform(X_vis_pre)
            pca_trans_time = time.time() - t0
            
            X_pca_post_vis = pca.transform(X_vis_post)
            
            # Drift PCA
            dca = DriftComponentAnalysis(n_components=2, add_anchor_point=args.add_anchor_point)
            t0 = time.time()
            dca.fit(X_train_pre, X_train_post)
            dca_fit_time = time.time() - t0
            
            t0 = time.time()
            X_dca_pre_vis = dca.transform(X_vis_pre)
            dca_trans_time = time.time() - t0
            
            X_dca_post_vis = dca.transform(X_vis_post)
            
            # Visualization
            
            # Determine global limits for the plot to maintain scale
            # PCA Cluster
            pca_all = np.vstack([X_pca_pre_vis, X_pca_post_vis])
            x_min_p, x_max_p = pca_all[:, 0].min(), pca_all[:, 0].max()
            y_min_p, y_max_p = pca_all[:, 1].min(), pca_all[:, 1].max()
            
            # DCA Cluster
            dca_all = np.vstack([X_dca_pre_vis, X_dca_post_vis])
            x_min_d, x_max_d = dca_all[:, 0].min(), dca_all[:, 0].max()
            y_min_d, y_max_d = dca_all[:, 1].min(), dca_all[:, 1].max()
            
            # Plots
            # 1. Standard PCA - Pre
            fig_p1 = create_scatter_plot(
                X_pca_pre_vis, y_vis_pre, "PC1", "PC2", 
                f"PCA ({orient_name}) on Pre {vis_name}", pca_fit_time, show_time=True
            )
            fig_p1.axvline(0, color='k', linestyle='--')
            fig_p1.axhline(0, color='k', linestyle='--')
            fig_p1.set_xlim(x_min_p - 0.1 * (x_max_p - x_min_p), x_max_p + 0.1 * (x_max_p - x_min_p)) # Make sure the 0.0 point is included in the figure
            fig_p1.set_ylim(y_min_p - 0.1 * (y_max_p - y_min_p), y_max_p + 0.1 * (y_max_p - y_min_p))
            add_biplot_arrows(fig_p1, pca, X_pca_pre_vis, feature_names=vis_features)
            
            # 2. Standard PCA - Post
            fig_p2 = create_scatter_plot(
                X_pca_post_vis, y_vis_post, "PC1", "PC2", 
                f"PCA ({orient_name}) on Post {vis_name}", pca_trans_time, show_time=True
            )
            fig_p2.axvline(0, color='k', linestyle='--')
            fig_p2.axhline(0, color='k', linestyle='--')
            fig_p2.set_xlim(x_min_p - 0.1 * (x_max_p - x_min_p), x_max_p + 0.1 * (x_max_p - x_min_p))
            fig_p2.set_ylim(y_min_p - 0.1 * (y_max_p - y_min_p), y_max_p + 0.1 * (y_max_p - y_min_p))
            add_biplot_arrows(fig_p2, pca, X_pca_post_vis, feature_names=vis_features)

            # 3. Drift PCA - Pre
            fig_d1 = create_scatter_plot(
                X_dca_pre_vis, y_vis_pre, "D1", "D2", 
                f"Drift PCA ({orient_name}) on Pre {vis_name}", dca_fit_time, show_time=True
            )
            fig_d1.axvline(0, color='k', linestyle='--')
            fig_d1.axhline(0, color='k', linestyle='--')
            fig_d1.set_xlim(x_min_d - 0.1 * (x_max_d - x_min_d), x_max_d + 0.1 * (x_max_d - x_min_d))
            fig_d1.set_ylim(y_min_d - 0.1 * (y_max_d - y_min_d), y_max_d + 0.1 * (y_max_d - y_min_d))
            add_biplot_arrows(fig_d1, dca.pca, X_dca_pre_vis, feature_names=vis_features)
            add_drift_info_to_plot(fig_d1, dca, args.add_anchor_point, args.add_drift_vectors)


            # 4. Drift PCA - Post
            fig_d2 = create_scatter_plot(
                X_dca_post_vis, y_vis_post, "D1", "D2", 
                f"Drift PCA ({orient_name}) on Post {vis_name}", dca_trans_time, show_time=True
            )
            fig_d2.axvline(0, color='k', linestyle='--')
            fig_d2.axhline(0, color='k', linestyle='--')
            fig_d2.set_xlim(x_min_d - 0.1 * (x_max_d - x_min_d), x_max_d + 0.1 * (x_max_d - x_min_d))
            fig_d2.set_ylim(y_min_d - 0.1 * (y_max_d - y_min_d), y_max_d + 0.1 * (y_max_d - y_min_d))
            add_biplot_arrows(fig_d2, dca.pca, X_dca_post_vis, feature_names=vis_features)
            add_drift_info_to_plot(fig_d2, dca, args.add_anchor_point, args.add_drift_vectors)

            
            # Combine
            final_utils = (fig_p1 | fig_p2) / (fig_d1 | fig_d2)
            
            out_name = os.path.join(output_dir, f"comparison_O-{orient_name}_V-{vis_name}.png")
            final_utils.savefig(out_name, dpi=120)
            logger.info(f"Saved {out_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Drift PCA Comparison")
    parser.add_argument("--dataset", type=str, default="sea", help="Name of the dataset (e.g., sea, elec)")
    parser.add_argument("--add_anchor_point", action='store_true', help="Add anchor point to Drift PCA")
    parser.add_argument("--add_drift_vectors", action='store_true', help="Add drift vectors to plots")

    args = parser.parse_args()
    
    run_comparison(args)
