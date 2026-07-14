import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD, PCA
import logging
import umap

from ssnp.code.ssnp import SSNP

ARROW_HEAD_WIDTH_SCALE = 6
logger = logging.getLogger(__name__)

# Using the paired class palette https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12
def get_paired_class_palette():
    return [
        '#a6cee3', '#1f78b4', # Blue
        '#fb9a99', '#e31a1c', # Red
        '#b2df8a', '#33a02c', # Green
        '#fdbf6f', '#ff7f00', # Orange
        '#cab2d6', '#6a3d9a', # Purple
        '#ffff99', '#b15928'  # Brown
    ]

class BaseFitter:
    def __init__(self, name):
        self.name = name
        self.components_ = None
        self.explained_variance_ratio_ = None # Optional, only some support this
        self.loading_scale_factors_ = None    # Set only if we want to scale arrows
    
    def fit(self, X_pre, X_post, y_pre, y_post, diff_matrix):
        raise NotImplementedError
        
    def transform(self, X_pre, X_post):
        raise NotImplementedError
        
    def transform_vectors(self, diff_matrix):
        return None

class TruncatedSVDFitter(BaseFitter):
    def __init__(self, n_components=2):
        super().__init__("Truncated SVD")
        self.model = TruncatedSVD(n_components=n_components, random_state=42)
        
    def fit(self, X_pre, X_post, y_pre, y_post, diff_matrix):
        # TruncatedSVD uniquely fits specifically to the drift vectors themselves
        if len(diff_matrix) == 0:
            logger.warning("Empty diff matrix, fitting TruncatedSVD on X_pre")
            self.model.fit(X_pre)
        else:
            self.model.fit(diff_matrix)
        
        self.components_ = self.model.components_
        self.explained_variance_ratio_ = self.model.explained_variance_ratio_
        # Set explicitly for drawing scaled loadings analogously to dca2_utils 
        self.loading_scale_factors_ = self.model.singular_values_
        return self
        
    def transform(self, X_pre, X_post):
        return self.model.transform(X_pre), self.model.transform(X_post)

    def transform_vectors(self, diff_matrix):
        return self.model.transform(diff_matrix)

class PCAFitter(BaseFitter):
    def __init__(self, n_components=2):
        super().__init__("PCA")
        self.model = PCA(n_components=n_components, random_state=42)
        
    def fit(self, X_pre, X_post, y_pre, y_post, diff_matrix):
        self.model.fit(X_pre)
        self.components_ = self.model.components_
        # Scale loading arrows by component standard deviations (singular values / sqrt(N-1))
        self.loading_scale_factors_ = np.sqrt(self.model.explained_variance_)
        return self
        
    def transform(self, X_pre, X_post):
        return self.model.transform(X_pre), self.model.transform(X_post)

class UMAPFitter(BaseFitter):
    def __init__(self, n_components=2):
        super().__init__("UMAP")
        self.model = umap.UMAP(n_components=n_components, random_state=42)
        
    def fit(self, X_pre, X_post, y_pre, y_post, diff_matrix):
        self.components_ = None
        self.model.fit(X_pre)
        return self
        
    def transform(self, X_pre, X_post):
        pre_emb = self.model.embedding_
        post_emb = self.model.transform(X_post)
        return pre_emb, post_emb

class SSNPFitter(BaseFitter):
    def __init__(self, n_components=2):
        super().__init__("SSNP")
        # SSNP always outputs 2D in its bottleneck layer by default
        
    def fit(self, X_pre, X_post, y_pre, y_post, diff_matrix):
        self.model = SSNP(epochs=50, verbose=0)
        self.model.fit(X_pre, y_pre)
        
        self.y_post = y_post
        self.components_ = None
        return self
        
    def transform(self, X_pre, X_post):
        return self.model.transform(X_pre), self.model.transform(X_post)

# --- Plotting Utilities ---

def plot_algorithm_scatter(X_proj, y, ax, title, is_pre=True, classes=None):
    if classes is None:
        classes = np.unique(y)
        
    full_palette = get_paired_class_palette()
    
    # Generate hue_order and colors for the classes
    hue_order = []
    colors = []
    for idx, c in enumerate(classes):
        if is_pre:
            hue_order.append(f"Class {c} Pre")
            colors.append(full_palette[(2 * idx + 1) % len(full_palette)])
        else:
            hue_order.append(f"Class {c} Post")
            colors.append(full_palette[(2 * idx + 1) % len(full_palette)])
            
    suffix = "Pre" if is_pre else "Post"
    labels = np.array([f"Class {c} {suffix}" for c in y], dtype=object)
    
    df = pd.DataFrame({
        "Comp 1": X_proj[:, 0],
        "Comp 2": X_proj[:, 1],
        "Label": labels
    })
    
    # Filter hue_order and colors to keep only those actually present in y
    present_labels = np.unique(labels)
    hue_order_filtered = [l for l in hue_order if l in present_labels]
    colors_filtered = [colors[hue_order.index(l)] for l in hue_order_filtered]
    
    sns.scatterplot(
        data=df, x="Comp 1", y="Comp 2", hue="Label",
        palette=colors_filtered, hue_order=hue_order_filtered, 
        alpha=0.6, s=25, ax=ax, edgecolor='white', linewidth=0.3
    )
    ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax.axvline(0, color='grey', linestyle='--', alpha=0.5)
    
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(prop={'size': 8}, loc='best')

def plot_compass_rose(components, ax, feature_names=None, loading_scale_factors=None):
    if components is None or len(components) == 0:
        ax.set_title("No Linear Loadings")
        ax.axis('off')
        ax.text(0.5, 0.5, "Manifold Learning:\nNo Linear Loadings", 
                ha='center', va='center', fontsize=12, color='gray', transform=ax.transAxes)
        return
        
    loadings = components[:2].T
    title_suffix = "Unscaled"
    
    # Scale mappings if singular values supplied
    if loading_scale_factors is not None and len(loading_scale_factors) >= 2:
        loadings = loadings * loading_scale_factors[:2]
        title_suffix = "Scaled"
        
    ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax.axvline(0, color='grey', linestyle='--', alpha=0.5)
    
    max_val = np.max(np.abs(loadings)) * 1.2
    if max_val == 0:
        max_val = 1.0
        
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    
    for i, arrow in enumerate(loadings):
        arrow_x = arrow[0]
        arrow_y = arrow[1] if len(arrow) > 1 else 0.0
        
        ax.arrow(0, 0, arrow_x, arrow_y, color="k", alpha=0.7, head_width=ARROW_HEAD_WIDTH_SCALE*(0.005*max_val),
                 width=0.005*max_val, length_includes_head=True, zorder=10)
        
        label = feature_names[i] if feature_names is not None else f"F{i+1}"
        ax.text(arrow_x * 1.1, arrow_y * 1.1, label, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))
                
    ax.set_title(f"Loadings Compass ({title_suffix})", fontsize=12)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

def plot_algorithm_drift_compass(vectors_trans, ax, classes=None, classes_computed=None):
    if vectors_trans is None or len(vectors_trans) == 0:
        ax.set_title("No Drift Vectors")
        ax.axis('off')
        ax.text(0.5, 0.5, "Algorithm not fitted on\nmean/std difference vectors.", 
                ha='center', va='center', fontsize=12, color='gray', transform=ax.transAxes)
        return
        
    ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax.axvline(0, color='grey', linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    
    max_val = np.max(np.abs(vectors_trans)) * 1.2
    if max_val == 0:
        max_val = 1.0
        
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    
    labels = []
    colors = []
    
    full_palette = get_paired_class_palette()
    
    is_global = len(vectors_trans) == 2 and (classes_computed is None or len(classes_computed) == 0)
    
    if is_global:
        labels = ["Global Mean Diff", "Global Std Diff"]
        colors = ["#800080", "#ffd700"]
    else:
        if classes_computed is not None and len(vectors_trans) == len(classes_computed) * 2:
            for c in classes_computed:
                if classes is not None and c in classes:
                    i = list(classes).index(c)
                else:
                    i = len(classes) if classes is not None else 0
                labels.extend([f"Class {c} Mean Diff", f"Class {c} Std Diff"])
                colors.extend([full_palette[(2*i+1) % len(full_palette)], full_palette[(2*i) % len(full_palette)]])
        elif classes is not None and len(vectors_trans) == len(classes) * 2:
            for i, c in enumerate(classes):
                labels.extend([f"Class {c} Mean Diff", f"Class {c} Std Diff"])
                colors.extend([full_palette[(2*i+1) % len(full_palette)], full_palette[(2*i) % len(full_palette)]])
        else:
            for i in range(len(vectors_trans)//2):
                labels.extend([f"C{i} Mean Diff", f"C{i} Std Diff"])
                colors.extend([full_palette[(2*i+1) % len(full_palette)], full_palette[(2*i) % len(full_palette)]])
                
    import matplotlib.lines as mlines
    handles = []
    for i, vec in enumerate(vectors_trans):
        color = colors[i % len(colors)] if len(colors) > 0 else 'k'
        label = labels[i] if i < len(labels) else f"Vector {i}"
        
        ax.arrow(0, 0, vec[0], vec[1], color=color, alpha=0.8, head_width=ARROW_HEAD_WIDTH_SCALE*(0.005*max_val),
                 width=0.005*max_val, length_includes_head=True, zorder=10)
        
        # Create a legend handle for this vector
        handle = mlines.Line2D([], [], color=color, marker='>', markersize=5, label=label, alpha=0.8)
        handles.append(handle)
                 
    ax.legend(handles=handles, prop={'size': 7.5}, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Drift Vectors Compass", fontsize=12)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
