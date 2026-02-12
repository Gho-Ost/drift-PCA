from sklearn.decomposition import PCA
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DriftComponentAnalysis:
    """
    Drift-Oriented PCA implementation.
    
    This class identifies directions of maximum change (drift) between two datasets
    (reference and current) and projects data onto these directions.
    It can compute drift globally or per-class.
    """
    
    def __init__(self, n_components=2, by_class=False, add_anchor_point=False):
        self.n_components = n_components
        self.by_class = by_class
        self.add_anchor_point = add_anchor_point
        self.pca = PCA(n_components=n_components)
        self.components_ = None
        self.diff_vectors = None
        
    def fit(self, X_ref, X_cur, y_ref=None, y_cur=None):
        """
        Fit the Drift PCA model.
        
        Args:
            X_ref: Reference data (pre-drift)
            X_cur: Current data (post-drift)
            y_ref: Reference labels (required if by_class=True)
            y_cur: Current labels (required if by_class=True)
            
        Returns:
            self
        """
        X_ref = np.array(X_ref)
        X_cur = np.array(X_cur)
        
        diff_vectors = []
        
        if self.by_class:
            if y_ref is None or y_cur is None:
                raise ValueError("Labels y_ref and y_cur are required when by_class=True")
                
            y_ref = np.array(y_ref)
            y_cur = np.array(y_cur)
            
            # Find common classes
            classes = np.intersect1d(np.unique(y_ref), np.unique(y_cur))
            
            for c in classes:
                X_ref_c = X_ref[y_ref == c]
                X_cur_c = X_cur[y_cur == c]
                
                # Check consistency - need at least 2 samples to compute std
                if len(X_ref_c) < 2 or len(X_cur_c) < 2:
                    continue
                    
                mean_ref = np.mean(X_ref_c, axis=0)
                std_ref = np.std(X_ref_c, axis=0)
                mean_cur = np.mean(X_cur_c, axis=0)
                std_cur = np.std(X_cur_c, axis=0)
                
                diff_vectors.append(mean_cur - mean_ref)
                diff_vectors.append(std_cur - std_ref)
                
        else:
            # Global drift
            mean_ref = np.mean(X_ref, axis=0)
            std_ref = np.std(X_ref, axis=0)
            mean_cur = np.mean(X_cur, axis=0)
            std_cur = np.std(X_cur, axis=0)

            # print("Means", mean_ref)
            # print("Std", std_ref)
            
            if self.add_anchor_point:
                diff_vectors.append(np.zeros(len(mean_ref))) # Origin anchor (no drift)

            diff_vectors.append(mean_cur - mean_ref)
            diff_vectors.append(std_cur - std_ref)
            
        diff_vectors = np.array(diff_vectors)
        # print(diff_vectors)
        
        # Handle edge case: no variance in drift vectors (e.g. no change)
        if np.all(diff_vectors == 0) or len(diff_vectors) == 0:
            # Fallback to standard PCA on X_ref if no drift detected
            logger.warning("No drift detected or empty diff vectors. Falling back to standard PCA on reference data.")
            self.pca.fit(X_ref)
        else:
             # If we have fewer diff vectors than components, we might run into issues with full PCA
             # But usually sklearn handles n_samples < n_components
            self.pca.fit(diff_vectors)
            
        # print("Explained variance ratio:", self.pca.explained_variance_ratio_)
        self.components_ = self.pca.components_
        self.diff_vectors = diff_vectors # for mean and std vectors
        return self
        
    def transform(self, X):
        return self.pca.transform(X)
