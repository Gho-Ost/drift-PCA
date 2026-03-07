import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

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


def create_synthetic_drift(
    name,
    means_pre,
    stds_pre,
    means_post,
    stds_post,
    n_samples=4000
):
    """
    Generate synthetic drift dataset.
    infers number of features from means_pre length.
    n_samples is divided equally between pre and post drift.
    """
    n_features = len(means_pre)
    
    # Assertions for input shapes
    assert len(stds_pre) == n_features, "stds_pre must match means_pre length"
    assert len(means_post) == n_features, "means_post must match means_pre length"
    assert len(stds_post) == n_features, "stds_post must match means_pre length"
    
    n_samples_pre = n_samples // 2
    n_samples_post = n_samples - n_samples_pre
    
    # Generate Pre-Drift Data
    # Shape: (n_samples_pre, n_features)
    X_pre = np.random.normal(loc=means_pre, scale=stds_pre, size=(n_samples_pre, n_features))
    
    # Assign two random classes (0 or 1)
    y_pre = np.random.randint(0, 2, size=n_samples_pre)
    
    # Generate Post-Drift Data
    X_post = np.random.normal(loc=means_post, scale=stds_post, size=(n_samples_post, n_features))
    y_post = np.random.randint(0, 2, size=n_samples_post)
    
    # Create DataFrames
    cols = [str(i) for i in range(n_features)]
    
    df_pre = pd.DataFrame(X_pre, columns=cols)
    df_pre['target'] = y_pre
    
    df_post = pd.DataFrame(X_post, columns=cols)
    df_post['target'] = y_post
    
    # Create Directories
    data_dir = "data/synth_stream_datasets"
    vis_dir = "data/synth_datasets_vis"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Save Data
    pre_path = os.path.join(data_dir, f"{name}_pre.csv")
    post_path = os.path.join(data_dir, f"{name}_post.csv")
    
    df_pre.to_csv(pre_path, index=False)
    df_post.to_csv(post_path, index=False)
    print(f"Saved datasets to:\n- {pre_path}\n- {post_path}")
    
    df_pre_plot = df_pre.copy()
    df_pre_plot['Drift_Stage'] = 'Pre-Drift'
    
    df_post_plot = df_post.copy()
    df_post_plot['Drift_Stage'] = 'Post-Drift'
    
    df_combined = pd.concat([df_pre_plot, df_post_plot], ignore_index=True)
    
    print("Generating visualization...")
    g = sns.pairplot(
        df_combined,
        vars=cols,
        hue='Drift_Stage',
        palette={'Pre-Drift': 'blue', 'Post-Drift': 'red'},
        plot_kws={'alpha': 0.5, 's': 15},
        diag_kind='hist'
    )
    g.fig.suptitle(f'Synthetic Drift Dataset: {name}\n(Features: {n_features}, Samples: {n_samples})', y=1.02)
    
    vis_path = os.path.join(vis_dir, f"{name}_vis.png")
    g.savefig(vis_path, dpi=150)
    plt.close()
    print(f"Saved visualization to: {vis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic drift datasets")
    parser.add_argument("--name", type=str, required=True, help="Custom name for the dataset")
    parser.add_argument("--means_pre", type=float, nargs='+', required=True, help="List of means for pre-drift features")
    parser.add_argument("--stds_pre", type=float, nargs='+', required=True, help="List of standard deviations for pre-drift features")
    parser.add_argument("--means_post", type=float, nargs='+', required=True, help="List of means for post-drift features")
    parser.add_argument("--stds_post", type=float, nargs='+', required=True, help="List of standard deviations for post-drift features")
    parser.add_argument("--n_samples", type=int, default=4000, help="Total number of samples (split equally)")
    
    args = parser.parse_args()
    
    create_synthetic_drift(
        name=args.name,
        means_pre=args.means_pre,
        stds_pre=args.stds_pre,
        means_post=args.means_post,
        stds_post=args.stds_post,
        n_samples=args.n_samples
    )
