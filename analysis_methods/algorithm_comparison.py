import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD, SparsePCA, FastICA
from sklearn.manifold import TSNE
import logging
import umap

from ssnp.code.ssnp import SSNP

logger = logging.getLogger(__name__)

# Reusing the palette from dca2_utils for consistency
def get_paired_class_palette():
    return ["#a6cee3", "#1f78b4", "#fb9a99", "#e31a1c"]

class BaseFitter:
    def __init__(self, name):
        self.name = name
        self.components_ = None
        self.explained_variance_ratio_ = None # Optional, only some support this
        self.loading_scale_factors_ = None    # Set only if we want to scale arrows
    
    def fit(self, X_pre, X_post, y_pre, y_post, diff_matrix, strategy="pre_fit"):
        raise NotImplementedError
        
    def transform(self, X_pre, X_post, strategy="pre_fit"):
        raise NotImplementedError
        
    def transform_vectors(self, diff_matrix):
        return None

class TruncatedSVDFitter(BaseFitter):
    def __init__(self, n_components=2):
        super().__init__("Truncated SVD")
        self.model = TruncatedSVD(n_components=n_components, random_state=42)
        
    def fit(self, X_pre, X_post, y_pre, y_post, diff_matrix, strategy="pre_fit"):
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
        
    def transform(self, X_pre, X_post, strategy="pre_fit"):
        return self.model.transform(X_pre), self.model.transform(X_post)

    def transform_vectors(self, diff_matrix):
        return self.model.transform(diff_matrix)

class SparsePCAFitter(BaseFitter):
    def __init__(self, n_components=2):
        super().__init__("Sparse PCA")
        self.model = SparsePCA(n_components=n_components, random_state=42)
        
    def fit(self, X_pre, X_post, y_pre, y_post, diff_matrix, strategy="pre_fit"):
        if strategy == "joint_fit":
            self.model.fit(np.vstack([X_pre, X_post]))
        else:
            self.model.fit(X_pre)
            
        self.components_ = self.model.components_
        return self
        
    def transform(self, X_pre, X_post, strategy="pre_fit"):
        if strategy == "separate_fit":
            post_model = SparsePCA(n_components=self.model.n_components, random_state=42)
            post_model.fit(X_post)
            return self.model.transform(X_pre), post_model.transform(X_post)
        elif strategy == "joint_fit":
            X_stacked = np.vstack([X_pre, X_post])
            X_trans = self.model.transform(X_stacked)
            return X_trans[:len(X_pre)], X_trans[len(X_pre):]
        else: # pre_fit
            return self.model.transform(X_pre), self.model.transform(X_post)

class FastICAFitter(BaseFitter):
    def __init__(self, n_components=2):
        super().__init__("Fast ICA")
        self.model = FastICA(n_components=n_components, random_state=42, max_iter=1000)
        
    def fit(self, X_pre, X_post, y_pre, y_post, diff_matrix, strategy="pre_fit"):
        if strategy == "joint_fit":
            self.model.fit(np.vstack([X_pre, X_post]))
        else:
            self.model.fit(X_pre)
            
        self.components_ = self.model.mixing_.T if self.model.mixing_ is not None else self.model.components_
        return self
        
    def transform(self, X_pre, X_post, strategy="pre_fit"):
        if strategy == "separate_fit":
            post_model = FastICA(n_components=self.model.n_components, random_state=42, max_iter=1000)
            X_post_trans = post_model.fit_transform(X_post)
            return self.model.transform(X_pre), X_post_trans
        elif strategy == "joint_fit":
            X_stacked = np.vstack([X_pre, X_post])
            X_trans = self.model.transform(X_stacked)
            return X_trans[:len(X_pre)], X_trans[len(X_pre):]
        else:
            return self.model.transform(X_pre), self.model.transform(X_post)

class TSNEFitter(BaseFitter):
    def __init__(self, n_components=2):
        super().__init__("t-SNE")
        self.model = TSNE(n_components=n_components, random_state=42)
        
    def fit(self, X_pre, X_post, y_pre, y_post, diff_matrix, strategy="pre_fit"):
        self.components_ = None
        
        if strategy == "pre_fit":
            logger.warning("t-SNE natively prohibits .transform() functionality on unseen data. "
                           "Injecting fail-safe: automatically collapsing to `joint_fit` for t-SNE evaluation.")
            strategy = "joint_fit"
            
        if strategy == "joint_fit":
            X_stacked = np.vstack([X_pre, X_post])
            X_trans = self.model.fit_transform(X_stacked)
            self._pre_emb = X_trans[:len(X_pre)]
            self._post_emb = X_trans[len(X_pre):]
        elif strategy == "separate_fit":
            self._pre_emb = self.model.fit_transform(X_pre)
            post_model = TSNE(n_components=self.model.n_components, random_state=42)
            self._post_emb = post_model.fit_transform(X_post)
            
        return self
        
    def transform(self, X_pre, X_post, strategy="pre_fit"):
        return self._pre_emb, self._post_emb

class UMAPFitter(BaseFitter):
    def __init__(self, n_components=2):
        super().__init__("UMAP")
        self.model = umap.UMAP(n_components=n_components, random_state=42)
        
    def fit(self, X_pre, X_post, y_pre, y_post, diff_matrix, strategy="pre_fit"):
        self.components_ = None
        if strategy == "joint_fit":
            X_stacked = np.vstack([X_pre, X_post])
            X_trans = self.model.fit_transform(X_stacked)
            self._pre_emb = X_trans[:len(X_pre)]
            self._post_emb = X_trans[len(X_pre):]
        elif strategy == "pre_fit":
            self.model.fit(X_pre)
        return self
        
    def transform(self, X_pre, X_post, strategy="pre_fit"):
        if strategy == "joint_fit":
            return self._pre_emb, self._post_emb
        elif strategy == "separate_fit":
            pre_emb = self.model.fit_transform(X_pre)
            post_model = umap.UMAP(n_components=self.model.n_components, random_state=42)
            post_emb = post_model.fit_transform(X_post)
            return pre_emb, post_emb
        else: # pre_fit
            pre_emb = self.model.embedding_
            post_emb = self.model.transform(X_post)
            return pre_emb, post_emb

class SSNPFitter(BaseFitter):
    def __init__(self, n_components=2):
        super().__init__("SSNP")
        # SSNP always outputs 2D in its bottleneck layer by default
        
    def fit(self, X_pre, X_post, y_pre, y_post, diff_matrix, strategy="pre_fit"):
        self.model = SSNP(epochs=50, verbose=0)
        
        if strategy == "joint_fit":
            X_stacked = np.vstack([X_pre, X_post])
            y_stacked = np.hstack([y_pre, y_post])
            self.model.fit(X_stacked, y_stacked)
        else:
            self.model.fit(X_pre, y_pre)
        
        self.y_post = y_post
        self.components_ = None
        return self
        
    def transform(self, X_pre, X_post, strategy="pre_fit"):
        if strategy == "separate_fit":
            from ssnp.code.ssnp import SSNP
            post_model = SSNP(epochs=50, verbose=0)
            post_model.fit(X_post, self.y_post)
            return self.model.transform(X_pre), post_model.transform(X_post)
        else:
            return self.model.transform(X_pre), self.model.transform(X_post)

# --- Plotting Utilities ---

def plot_algorithm_scatter(X_proj, y, ax, title, is_pre=True, c0=0, c1=1):
    palette = get_paired_class_palette()
    
    if is_pre:
        hue_order = [f"Class {c0} Pre", f"Class {c1} Pre"]
        colors = [palette[0], palette[2]]
        labels = np.where(y == c0, hue_order[0], hue_order[1])
    else:
        hue_order = [f"Class {c0} Post", f"Class {c1} Post"]
        colors = [palette[1], palette[3]]
        labels = np.where(y == c0, hue_order[0], hue_order[1])
        
    df = pd.DataFrame({
        "Comp 1": X_proj[:, 0],
        "Comp 2": X_proj[:, 1],
        "Label": labels
    })
    
    sns.scatterplot(
        data=df, x="Comp 1", y="Comp 2", hue="Label",
        palette=colors, hue_order=hue_order, alpha=0.6, s=40, ax=ax, edgecolor=None
    )
    
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
        
        ax.arrow(0, 0, arrow_x, arrow_y, color="k", alpha=0.7, 
                 width=0.005*max_val, length_includes_head=True, zorder=10)
        
        label = feature_names[i] if feature_names is not None else f"F{i+1}"
        ax.text(arrow_x * 1.1, arrow_y * 1.1, label, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))
                
    ax.set_title(f"Loadings Compass ({title_suffix})", fontsize=12)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

def plot_algorithm_drift_compass(vectors_trans, ax, classes=None):
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
                
    ax.set_title("Drift Vectors Compass", fontsize=12)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
