import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import json
import logging

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
# Workaround for seaborn and pandas >= 2.2
pd.options.mode.chained_assignment = None
try:
    pd.set_option('mode.use_inf_as_na', True)
except pd.errors.OptionError:
    pass
import pandas._config.config as pd_config
try:
    pd_config.register_option('mode.use_inf_as_null', True, 'use_inf_as_null')
except pd_config.OptionError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def create_synthetic_drift_classes(
    name,
    means_pre_c0, stds_pre_c0,
    means_pre_c1, stds_pre_c1,
    means_post_c0, stds_post_c0,
    means_post_c1, stds_post_c1,
    n_samples=4000,
    data_dir=None,
    vis_dir=None,
    save_to_disk=True
):
    """
    Generate synthetic drift dataset with separate parameters for classes 0 and 1.
    """
    n_features = len(means_pre_c0)
    
    n_samples_pre = n_samples // 2
    n_samples_post = n_samples - n_samples_pre
    
    # We will generate balanced classes (50% class 0, 50% class 1) for both pre and post by default
    n_pre_c0 = n_samples_pre // 2
    n_pre_c1 = n_samples_pre - n_pre_c0
    
    n_post_c0 = n_samples_post // 2
    n_post_c1 = n_samples_post - n_post_c0

    # Generate Pre-Drift Data
    X_pre_c0 = np.random.normal(loc=means_pre_c0, scale=stds_pre_c0, size=(n_pre_c0, n_features))
    X_pre_c1 = np.random.normal(loc=means_pre_c1, scale=stds_pre_c1, size=(n_pre_c1, n_features))
    X_pre = np.vstack([X_pre_c0, X_pre_c1])
    y_pre = np.array([0]*n_pre_c0 + [1]*n_pre_c1)
    
    # Shuffle pre
    idx_pre = np.random.permutation(n_samples_pre)
    X_pre = X_pre[idx_pre]
    y_pre = y_pre[idx_pre]

    # Generate Post-Drift Data
    X_post_c0 = np.random.normal(loc=means_post_c0, scale=stds_post_c0, size=(n_post_c0, n_features))
    X_post_c1 = np.random.normal(loc=means_post_c1, scale=stds_post_c1, size=(n_post_c1, n_features))
    X_post = np.vstack([X_post_c0, X_post_c1])
    y_post = np.array([0]*n_post_c0 + [1]*n_post_c1)
    
    # Shuffle post
    idx_post = np.random.permutation(n_samples_post)
    X_post = X_post[idx_post]
    y_post = y_post[idx_post]
    
    # Create DataFrames
    cols = [str(i) for i in range(n_features)]
    
    df_pre = pd.DataFrame(X_pre, columns=cols)
    df_pre['target'] = y_pre
    
    df_post = pd.DataFrame(X_post, columns=cols)
    df_post['target'] = y_post
    
    df_pre_plot = df_pre.copy()
    df_pre_plot['Drift_Stage'] = 'Pre-Drift'
    
    df_post_plot = df_post.copy()
    df_post_plot['Drift_Stage'] = 'Post-Drift'
    
    df_combined = pd.concat([df_pre_plot, df_post_plot], ignore_index=True)

    if not save_to_disk:
        return X_pre, y_pre, X_post, y_post, df_combined
    
    # Create Directories
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_dir is None:
        data_dir = os.path.join(project_root, "data", "synth_stream_datasets")
    if vis_dir is None:
        vis_dir = os.path.join(project_root, "data", "synth_datasets_vis")
        
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Save Data
    pre_path = os.path.join(data_dir, f"{name}_pre.csv")
    post_path = os.path.join(data_dir, f"{name}_post.csv")
    
    df_pre.to_csv(pre_path, index=False)
    df_post.to_csv(post_path, index=False)
    logging.info(f"Saved datasets for {name} to:\n- {pre_path}\n- {post_path}")
    
    # Generating visualization
    df_combined['Stage_Target'] = df_combined['Drift_Stage'] + " - Class " + df_combined['target'].astype(str)
    
    g = sns.pairplot(
        df_combined,
        vars=cols,
        hue='Stage_Target',
        plot_kws={'alpha': 0.5, 's': 15},
        diag_kind='kde'
    )
    g.fig.suptitle(f'Synthetic Class Drift Dataset: {name}\n(Features: {n_features}, Samples: {n_samples})', y=1.02)
    
    vis_path = os.path.join(vis_dir, f"{name}_vis.png")
    g.savefig(vis_path, dpi=150)
    plt.close()
    logging.info(f"Saved visualization to: {vis_path}")


def generate_from_json(json_path, n_samples=4000, data_dir=None, vis_dir=None):
    with open(json_path, 'r') as f:
        datasets = json.load(f)
        
    success_count = 0
    for i, params in enumerate(datasets):
        logging.info(f"--- Generating scenario {i+1}/{len(datasets)}: {params['name']} ---")
        logging.info(f"Description: {params.get('desc', '')}")
        try:
            create_synthetic_drift_classes(
                name=params['name'],
                means_pre_c0=params['means_pre_c0'],
                stds_pre_c0=params['stds_pre_c0'],
                means_pre_c1=params['means_pre_c1'],
                stds_pre_c1=params['stds_pre_c1'],
                means_post_c0=params['means_post_c0'],
                stds_post_c0=params['stds_post_c0'],
                means_post_c1=params['means_post_c1'],
                stds_post_c1=params['stds_post_c1'],
                n_samples=params.get('n_samples', n_samples),
                data_dir=data_dir,
                vis_dir=vis_dir
            )
            success_count += 1
        except Exception as e:
            logging.error(f"Failed to generate {params['name']}: {e}")
            
    logging.info(f"Finished generating datasets. {success_count}/{len(datasets)} successful.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic class drift datasets from JSON")
    parser.add_argument("--json", type=str, default="synthetic_data_classes.json", help="Path to JSON file with dataset definitions")
    parser.add_argument("--n_samples", type=int, default=4000, help="Total number of samples (split equally)")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory to save CSV data")
    parser.add_argument("--vis_dir", type=str, default=None, help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # If path is relative, assume it's relative to project root if not found locally
    json_path = args.json
    if not os.path.exists(json_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_path = os.path.join(project_root, args.json)
        
    if os.path.exists(json_path):
        generate_from_json(
            json_path=json_path,
            n_samples=args.n_samples,
            data_dir=args.data_dir,
            vis_dir=args.vis_dir
        )
    else:
        logging.error(f"Error: JSON file not found at {args.json} or {json_path}")
