import numpy as np
import pandas as pd
import seaborn as sns
import logging

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

logger = logging.getLogger(__name__)


def get_paired_class_palette():
    """
    Returns up to 12 colors from a paired palette (supporting up to 6 classes).
    Format: Class 0 Pre, Class 0 Post, Class 1 Pre, Class 1 Post, etc.
    """
    return [
        '#a6cee3', '#1f78b4', # Blue
        '#fb9a99', '#e31a1c', # Red
        '#b2df8a', '#33a02c', # Green
        '#fdbf6f', '#ff7f00', # Orange
        '#cab2d6', '#6a3d9a', # Purple
        '#ffff99', '#b15928'  # Brown
    ]

def plot_dca_scatter(X_pre, y_pre, X_post, y_post, dca, ax, pre_drift_model=None, grid_points=200, color_scheme='class', discrete_boundary=False, draw_boundary=True, highlight_misclassifications=False, hide_pre_drift_points=False):
    """
    Main scatter plot comprising unscaled data values.
    Optionally draws a decision boundary from the inverse-transformed PCA grid queried on the pre_drift_model.
    Uses predict_proba if available for smoother RdBu contour backgrounds.
    """
    # Transform Data into PCA space
    X_pre_proj = dca.transform(X_pre)
    X_post_proj = dca.transform(X_post)
    
    classes = np.unique(np.concatenate([y_pre, y_post]))
    if color_scheme == 'class' and len(classes) > 6:
        logger.warning(f"Too many classes ({len(classes)}) for paired class palette. Maximum is 6.")
        raise ValueError(f"Too many classes ({len(classes)}) for paired class palette. Maximum is 6.")
        
    # Set hue strings
    if color_scheme == 'drift':
        y_pre_hue = np.full(len(y_pre), "Pre-drift", dtype=object)
        y_post_hue = np.full(len(y_post), "Post-drift", dtype=object)
    else:
        y_pre_hue = np.array([f"Class {c} Pre" for c in y_pre], dtype=object)
        y_post_hue = np.array([f"Class {c} Post" for c in y_post], dtype=object)
    
    contour = None
    
    is_discrete = discrete_boundary or len(classes) > 2
    
    # Draw decision boundary based purely on PRE-DRIFT model
    if pre_drift_model is not None and draw_boundary:
        all_proj = np.vstack([X_pre_proj, X_post_proj])
        x_min, x_max = all_proj[:, 0].min() - 1, all_proj[:, 0].max() + 1
        y_min, y_max = all_proj[:, 1].min() - 1, all_proj[:, 1].max() + 1
        
        # Grid in 2D Space
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_points),
                             np.linspace(y_min, y_max, grid_points))
        grid_2d = np.c_[xx.ravel(), yy.ravel()]
        
        # Untransform to ND query space
        grid_nd = dca.inverse_transform(grid_2d)
        
        def darken_hex(hex_color, factor=0.4):
            rgb = mcolors.hex2color(hex_color)
            return mcolors.to_hex([c * factor for c in rgb])

        full_palette = get_paired_class_palette()
        light_colors = [full_palette[2*i] for i in range(len(classes))]
        dark_colors = [full_palette[2*i+1] for i in range(len(classes))]
        extreme_colors = [darken_hex(c) for c in dark_colors]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        
        # Query PRE-DRIFT model
        if hasattr(pre_drift_model, "predict_proba") and not is_discrete:
            Z = pre_drift_model.predict_proba(grid_nd)[:, 1]
            Z = Z.reshape(xx.shape)
            # Use extreme colors and dark colors with a white center to make boundaries highly defined
            cmap = LinearSegmentedColormap.from_list("custom_proba", [extreme_colors[0], dark_colors[0], "#ffffff", dark_colors[1], extreme_colors[1]])
            contour = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 21), alpha=0.5, cmap=cmap, vmin=0, vmax=1)
        else:
            Z_pred = pre_drift_model.predict(grid_nd)
            Z_idx = np.array([class_to_idx.get(v, 0) for v in Z_pred])
            Z = Z_idx.reshape(xx.shape)
            
            # Use dark_colors with low alpha to contrast against light pre-drift points
            cmap = ListedColormap(dark_colors)
            levels = np.arange(len(classes) + 1) - 0.5
            contour = ax.contourf(xx, yy, Z, levels=levels, alpha=0.3, cmap=cmap)
    
    # Plot Scatter points
    if color_scheme == 'drift':
        palette = ["#e31a1c", "#1f78b4"] # Red for pre, Blue for post
        hue_order = ["Pre-drift", "Post-drift"]
    else:
        full_palette = get_paired_class_palette()
        palette = full_palette[:len(classes)*2]
        hue_order = []
        for c in classes:
            hue_order.extend([f"Class {c} Pre", f"Class {c} Post"])
    
    if hide_pre_drift_points:
        plot_df = pd.DataFrame({
            "Component 1": X_post_proj[:, 0],
            "Component 2": X_post_proj[:, 1],
            "Label": y_post_hue
        })
        if highlight_misclassifications and pre_drift_model is not None:
            y_post_pred = pre_drift_model.predict(X_post)
            misclassified_flag = (y_post != y_post_pred)
        else:
            misclassified_flag = np.zeros(len(y_post), dtype=bool)
    else:
        plot_df = pd.DataFrame({
            "Component 1": np.concatenate([X_pre_proj[:, 0], X_post_proj[:, 0]]),
            "Component 2": np.concatenate([X_pre_proj[:, 1], X_post_proj[:, 1]]),
            "Label": np.concatenate([y_pre_hue, y_post_hue])
        })
        if highlight_misclassifications and pre_drift_model is not None:
            y_post_pred = pre_drift_model.predict(X_post)
            misclassified = (y_post != y_post_pred)
            misclassified_flag = np.concatenate([np.zeros(len(y_pre), dtype=bool), misclassified])
        else:
            misclassified_flag = np.zeros(len(y_pre) + len(y_post), dtype=bool)

    plot_df["Misclassified_Flag"] = misclassified_flag

    if highlight_misclassifications and pre_drift_model is not None:
        # Plot correctly classified points and pre-drift points normally
        sns.scatterplot(
            data=plot_df[~plot_df["Misclassified_Flag"]], x="Component 1", y="Component 2", hue="Label",
            palette=palette, hue_order=hue_order, alpha=0.8, s=60, ax=ax, legend=True
        )
        # Overlay misclassified points with a prominent black border
        sns.scatterplot(
            data=plot_df[plot_df["Misclassified_Flag"]], x="Component 1", y="Component 2", hue="Label",
            palette=palette, hue_order=hue_order, alpha=0.8, s=65, edgecolor="black", linewidth=1.5, ax=ax, legend=False
        )
        
        # Add custom legend entry for misclassifications
        handles, labels = ax.get_legend_handles_labels()
        misclassified_handle = mlines.Line2D([], [], color='none', marker='o', 
                                             markeredgecolor='black', markerfacecolor='none', 
                                             markersize=8, markeredgewidth=1.5, label='Misclassified')
        handles.append(misclassified_handle)
        labels.append('Misclassified')
        ax.legend(handles=handles, labels=labels)
    else:
        sns.scatterplot(
            data=plot_df, x="Component 1", y="Component 2", hue="Label",
            palette=palette, hue_order=hue_order, alpha=0.8, s=60, ax=ax
        )
    
    ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax.axvline(0, color='grey', linestyle='--', alpha=0.5)
    
    # Display Variance Info
    # evr = dca.explained_variance_ratio_
    # ax.set_xlabel(f"Component 1 (Explained Var: {evr[0]*100:.1f}%)")
    # ax.set_ylabel(f"Component 2 (Explained Var: {evr[1]*100:.1f}%)" if len(evr) > 1 else "Component 2")
    
    # Display Energy Info
    eer = dca.explained_energy_ratio_
    # eer = dca.loading_scale_factors_
    ax.set_xlabel(f"Component 1 (Explained Energy: {eer[0]*100:.1f}%)")
    ax.set_ylabel(f"Component 2 (Explained Energy: {eer[1]*100:.1f}%)" if len(eer) > 1 else "Component 2")

    ax.set_title("DCA Output: Pre vs Post Drift Data")
    
    return contour, is_discrete

def plot_loadings_compass(dca, ax, feature_names=None, scale_loadings=False, feature_importances=None):
    """
    Plots a compass rose containing loadings (PCA components representation).
    Optionally scales them by the raw singular values.
    If feature_importances are provided, colors the arrows by importance and adds a colorbar.
    """
    loadings = dca.pca.components_[:2].T
    if scale_loadings:
        loadings = loadings * dca.loading_scale_factors_[:2]
        title_suffix = "Scaled"
    else:
        title_suffix = "Unscaled"
    
    ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax.axvline(0, color='grey', linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    
    max_val = np.max(np.abs(loadings)) * 1.2
    if max_val == 0:
        max_val = 1.0
        
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    
    if feature_importances is not None:
        norm = mcolors.Normalize(vmin=np.min(feature_importances), vmax=np.max(feature_importances))
        cmap = plt.get_cmap('inferno_r') # Grays
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        colors = cmap(norm(feature_importances))
    else:
        colors = ["k"] * len(loadings)

    for i, arrow in enumerate(loadings):
        color = colors[i]
        ax.arrow(0, 0, arrow[0], arrow[1], color=color, alpha=0.8, 
                 width=0.005*max_val, length_includes_head=True, zorder=10)
        
        label = feature_names[i] if feature_names is not None else f"F{i+1}"
        ax.text(arrow[0] * 1.1, arrow[1] * 1.1, label, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"))
                
    if feature_importances is not None:
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.05)
        cbar.set_label('Feature Importance')

    # ax.set_title("Loadings Compass Rose (Unscaled)")
    # evr = dca.explained_variance_ratio_
    # ax.set_xlabel(f"Component 1 ({evr[0]*100:.1f}%)")
    # ax.set_ylabel(f"Component 2 ({evr[1]*100:.1f}%)" if len(evr) > 1 else "Component 2")

    ax.set_title(f"Loadings Compass Rose ({title_suffix})")
    eer = dca.explained_energy_ratio_
    # eer = dca.loading_scale_factors_
    ax.set_xlabel(f"Component 1 ({eer[0]*100:.1f}%)")
    ax.set_ylabel(f"Component 2 ({eer[1]*100:.1f}%)" if len(eer) > 1 else "Component 2")

def plot_drift_compass(dca, ax, classes=None, color_scheme='class'):
    """
    Plots a compass rose containing differences of Mean and Std deviations mapped to PCA.
    """
    ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax.axvline(0, color='grey', linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    
    if dca.diff_vectors is None or len(dca.diff_vectors) == 0:
        ax.set_title("Drift Compass: No drift vectors found")
        return
        
    vectors_trans = dca.transform(dca.diff_vectors)
    max_val = np.max(np.abs(vectors_trans)) * 1.2
    if max_val == 0:
        max_val = 1.0
        
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    
    labels = []
    colors = []
    
    is_global = len(vectors_trans) == 2 and (color_scheme == 'drift' or classes is None or len(classes) > 1)
    
    if is_global:
        labels = ["Global Mean Diff", "Global Std Diff"]
        colors = ["#800080", "#ffd700"] # purple (strong), yellow (light)
    else:
        full_palette = get_paired_class_palette()
        if classes is not None and len(vectors_trans) == len(classes) * 2:
            for i, c in enumerate(classes):
                labels.extend([f"Class {c} Mean Diff", f"Class {c} Std Diff"])
                colors.extend([full_palette[2*i+1], full_palette[2*i]])
        else:
            for i in range(len(vectors_trans)//2):
                labels.extend([f"C{i} Mean Diff", f"C{i} Std Diff"])
                colors.extend([full_palette[2*i+1], full_palette[2*i]])
                
    for i, vec in enumerate(vectors_trans):
        color = colors[i % len(colors)]
        label = labels[i] if i < len(labels) else f"Vector {i}"
        
        ax.arrow(0, 0, vec[0], vec[1], color=color, alpha=0.8,
                 width=0.005*max_val, length_includes_head=True, zorder=10, label=label)
                 
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                
    # ax.set_title("Drift Mean/Std Compass Rose")
    # eer = dca.explained_energy_ratio_
    # ax.set_xlabel(f"Component 1 ({eer[0]*100:.1f}%)")
    # ax.set_ylabel(f"Component 2 ({eer[1]*100:.1f}%)" if len(eer) > 1 else "Component 2")
 
    ax.set_title("Drift Mean/Std Compass Rose")
    eer = dca.explained_energy_ratio_
    # eer = dca.loading_scale_factors_
    ax.set_xlabel(f"Component 1 ({eer[0]*100:.1f}%)")
    ax.set_ylabel(f"Component 2 ({eer[1]*100:.1f}%)" if len(eer) > 1 else "Component 2")