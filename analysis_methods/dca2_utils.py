import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
import logging

logger = logging.getLogger(__name__)

# paired_class_palette = [
#     '#a6cee3',
#     '#1f78b4', # Blue
#     '#b2df8a',
#     '#33a02c', # Green
#     '#fb9a99',
#     '#e31a1c', # Red
#     '#fdbf6f',
#     '#ff7f00', # Orange
#     '#cab2d6',
#     '#6a3d9a', # Purple
#     '#ffff99',
#     '#b15928'  # Brown
# ]

def get_paired_class_palette():
    """
    Returns 4 colors from a paired palette.
    Format: Class 0 Pre, Class 0 Post, Class 1 Pre, Class 1 Post
    """
    return ["#a6cee3", "#1f78b4", "#fb9a99", "#e31a1c"]

def plot_dca_scatter(X_pre, y_pre, X_post, y_post, dca, ax, pre_drift_model=None, grid_points=200):
    """
    Main scatter plot comprising unscaled data values.
    Optionally draws a decision boundary from the inverse-transformed PCA grid queried on the pre_drift_model.
    Uses predict_proba if available for smoother RdBu contour backgrounds.
    """
    # Transform Data into PCA space
    X_pre_proj = dca.transform(X_pre)
    X_post_proj = dca.transform(X_post)
    
    classes = np.unique(np.concatenate([y_pre, y_post]))
    if len(classes) != 2:
        raise ValueError("dca scatter plot expects exactly 2 classes")
    
    c0, c1 = classes[0], classes[1]
    
    # Set hue strings
    y_pre_hue = np.where(y_pre == c0, f"Class {c0} Pre", f"Class {c1} Pre")
    y_post_hue = np.where(y_post == c0, f"Class {c0} Post", f"Class {c1} Post")
    
    contour = None
    
    # Draw decision boundary based purely on PRE-DRIFT model
    if pre_drift_model is not None:
        all_proj = np.vstack([X_pre_proj, X_post_proj])
        x_min, x_max = all_proj[:, 0].min() - 1, all_proj[:, 0].max() + 1
        y_min, y_max = all_proj[:, 1].min() - 1, all_proj[:, 1].max() + 1
        
        # Grid in 2D Space
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_points),
                             np.linspace(y_min, y_max, grid_points))
        grid_2d = np.c_[xx.ravel(), yy.ravel()]
        
        # Untransform to ND query space
        grid_nd = dca.inverse_transform(grid_2d)
        
        # Query PRE-DRIFT model
        if hasattr(pre_drift_model, "predict_proba"):
            Z = pre_drift_model.predict_proba(grid_nd)[:, 1]
            Z = Z.reshape(xx.shape)
            contour = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 21), alpha=0.4, cmap=plt.cm.RdBu_r, vmin=0, vmax=1)
        else:
            Z = pre_drift_model.predict(grid_nd)
            Z = Z.reshape(xx.shape)
            contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu_r)
    
    # Plot Scatter points
    palette = get_paired_class_palette()
    hue_order = [f"Class {c0} Pre", f"Class {c0} Post", f"Class {c1} Pre", f"Class {c1} Post"]
    
    plot_df = pd.DataFrame({
        "Component 1": np.concatenate([X_pre_proj[:, 0], X_post_proj[:, 0]]),
        "Component 2": np.concatenate([X_pre_proj[:, 1], X_post_proj[:, 1]]),
        "Label": np.concatenate([y_pre_hue, y_post_hue])
    })
    
    # Corrected hue_order referencing to class name array instead of colors
    sns.scatterplot(
        data=plot_df, x="Component 1", y="Component 2", hue="Label",
        palette=palette, hue_order=hue_order, alpha=0.8, s=60, ax=ax
    )
    
    ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax.axvline(0, color='grey', linestyle='--', alpha=0.5)
    
    # Display Variance Info
    evr = dca.explained_variance_ratio_
    ax.set_xlabel(f"Component 1 (Explained Var: {evr[0]*100:.1f}%)")
    ax.set_ylabel(f"Component 2 (Explained Var: {evr[1]*100:.1f}%)" if len(evr) > 1 else "Component 2")
    ax.set_title("DCA Output: Pre vs Post Drift Data")
    
    return contour

def plot_loadings_compass(dca, ax, feature_names=None):
    """
    Plots a compass rose containing unscaled loadings (PCA components representation).
    """
    loadings = dca.pca.components_[:2].T
    
    ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax.axvline(0, color='grey', linestyle='--', alpha=0.5)
    
    max_val = np.max(np.abs(loadings)) * 1.2
    if max_val == 0:
        max_val = 1.0
        
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    
    for i, arrow in enumerate(loadings):
        ax.arrow(0, 0, arrow[0], arrow[1], color="k", alpha=0.7, 
                 width=0.005*max_val, length_includes_head=True, zorder=10)
        
        label = feature_names[i] if feature_names is not None else f"F{i+1}"
        ax.text(arrow[0] * 1.1, arrow[1] * 1.1, label, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"))
                
    ax.set_title("Loadings Compass Rose (Unscaled)")
    evr = dca.explained_variance_ratio_
    ax.set_xlabel(f"Component 1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"Component 2 ({evr[1]*100:.1f}%)" if len(evr) > 1 else "Component 2")

def plot_drift_compass(dca, ax, classes=None):
    """
    Plots a compass rose containing unscaled differences of Mean and Std deviations mapped to PCA.
    """
    ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax.axvline(0, color='grey', linestyle='--', alpha=0.5)
    
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
    if classes is not None and len(classes) == 2 and len(vectors_trans) == 4:
        labels = [f"Class {classes[0]} Mean Diff", f"Class {classes[0]} Std Diff", 
                  f"Class {classes[1]} Mean Diff", f"Class {classes[1]} Std Diff"]
    elif len(vectors_trans) == 2:
        labels = ["Global Mean Diff", "Global Std Diff"]
    else:
        for i in range(len(vectors_trans)//2):
            labels.extend([f"C{i} Mean Diff", f"C{i} Std Diff"])
            
    colors = ['#e7298a', '#7570b3', '#d95f02', '#1b9e77']
    
    for i, vec in enumerate(vectors_trans):
        color = colors[i % len(colors)]
        label = labels[i] if i < len(labels) else f"Vector {i}"
        
        ax.arrow(0, 0, vec[0], vec[1], color=color, alpha=0.8,
                 width=0.005*max_val, length_includes_head=True, zorder=10)
                 
        ax.text(vec[0] * 1.1, vec[1] * 1.1, label, ha='center', va='center', color=color, 
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"))
                
    ax.set_title("Drift Mean/Std Compass Rose (Unscaled)")
    evr = dca.explained_variance_ratio_
    ax.set_xlabel(f"Component 1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"Component 2 ({evr[1]*100:.1f}%)" if len(evr) > 1 else "Component 2")