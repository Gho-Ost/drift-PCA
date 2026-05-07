from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DriftComponentAnalysis2:
    """
    Drift-Oriented PCA implementation utilizing TruncatedSVD.
    Computes diff between mean and std vectors for 2 specific classes
    and performs SVD on those difference vectors.
    """
    def __init__(self, n_components=2, by_class=False):
        self.n_components = n_components
        self.by_class = by_class
        self.pca = TruncatedSVD(n_components=n_components, random_state=42)
        self.diff_vectors = None
        
    def fit(self, X_ref, X_cur, y_ref, y_cur):
        X_ref = np.array(X_ref)
        X_cur = np.array(X_cur)
        y_ref = np.array(y_ref)
        y_cur = np.array(y_cur)
        
        diff_vectors = []
        if self.by_class:
            classes = np.intersect1d(np.unique(y_ref), np.unique(y_cur))
            if len(classes) != 2:
                logger.warning(f"Expected 2 classes, found {len(classes)}. This method is optimized for binary classification.")
                
            for c in classes:
                X_ref_c = X_ref[y_ref == c]
                X_cur_c = X_cur[y_cur == c]
                
                if len(X_ref_c) < 2 or len(X_cur_c) < 2:
                    continue
                    
                mean_ref = np.mean(X_ref_c, axis=0)
                std_ref = np.std(X_ref_c, axis=0)
                mean_cur = np.mean(X_cur_c, axis=0)
                std_cur = np.std(X_cur_c, axis=0)
                
                diff_vectors.append(mean_cur - mean_ref)
                diff_vectors.append(std_cur - std_ref)
        else:
            mean_ref = np.mean(X_ref, axis=0)
            std_ref = np.std(X_ref, axis=0)
            mean_cur = np.mean(X_cur, axis=0)
            std_cur = np.std(X_cur, axis=0)
            
            diff_vectors.append(mean_cur - mean_ref)
            diff_vectors.append(std_cur - std_ref)
            
        self.diff_vectors = np.array(diff_vectors)
        
        if len(self.diff_vectors) == 0 or np.all(self.diff_vectors == 0):
            logger.warning("No drift detected, fitting on X_ref")
            self.pca.fit(X_ref)
        else:
            self.pca.fit(self.diff_vectors)
            
        return self
        
    def transform(self, X):
        return self.pca.transform(X)
        
    def inverse_transform(self, X):
        return self.pca.inverse_transform(X)
        
    @property
    def explained_variance_ratio_(self):
        """
        Returns the traditional explained variance ratio.
        
        Purpose: None for the primary UI. Useful only for internal debugging 
        or if the user explicitly requests traditional variance metrics.
        
        Reasoning: Because the drift difference vectors are not centered around 
        zero, traditional variance (distance from the mean of the vectors) 
        decouples from SVD's optimization (distance from the origin). This 
        will output counterintuitive percentages for uncentered concept drift.
        """
        return self.pca.explained_variance_ratio_

    @property
    def explained_energy_ratio_(self):
        """
        Returns the explained energy ratio (squared singular values).
        
        Purpose: To label the X and Y axes on the scatter plots 
        (e.g., "Component 1 (Explained Energy: 88.7%)").
        
        Reasoning: TruncatedSVD solves for the axes that capture the maximum 
        "energy" (second moment) relative to the origin. Calculating the ratio 
        of the squared singular values accurately reflects how the algorithm 
        prioritized the axes and guarantees Component 1 is always the maximum.
        """
        squared_singular_values = self.pca.singular_values_ ** 2
        total_energy = np.sum(squared_singular_values)
        if total_energy == 0:
            return np.zeros_like(squared_singular_values)
        return squared_singular_values / total_energy

    @property
    def loading_scale_factors_(self):
        """
        Returns the raw singular values to be used as scaling multipliers.
        
        Purpose: To multiply against the raw unit-vector loadings to create 
        a true Biplot, determining the visual length of the loading arrows.
        
        Reasoning: Singular values represent the true 1D magnitude (length) 
        along each component. By scaling the loadings with the raw singular 
        values rather than the squared energy ratio, the visual lengths of 
        the arrows perfectly reflect the linear magnitude of drift caused 
        by each feature, preventing visual "punishment" of smaller drifts.
        """
        # print(self.pca.singular_values_)
        return self.pca.singular_values_


# class DriftComponentAnalysis:
#     """
#     Drift-Oriented PCA implementation.
    
#     This class identifies directions of maximum change (drift) between two datasets
#     (reference and current) and projects data onto these directions.
#     It can compute drift globally or per-class.
#     """
    
#     def __init__(self, n_components=2, by_class=False, add_anchor_point=False, use_svd=False):
#         self.n_components = n_components
#         self.by_class = by_class
#         self.use_svd = use_svd
#         self.add_anchor_point = add_anchor_point
#         if self.use_svd:
#             logger.info("Using TruncatedSVD; anchor point is disabled as data is inherently unshifted.")
#             self.add_anchor_point = False
#             self.pca = TruncatedSVD(n_components=n_components)
#         else:
#             self.pca = PCA(n_components=n_components)
#         self.components_ = None
#         self.diff_vectors = None
        
#     def fit(self, X_ref, X_cur, y_ref=None, y_cur=None):
#         """
#         Fit the Drift PCA model.
        
#         Args:
#             X_ref: Reference data (pre-drift)
#             X_cur: Current data (post-drift)
#             y_ref: Reference labels (required if by_class=True)
#             y_cur: Current labels (required if by_class=True)
            
#         Returns:
#             self
#         """
#         X_ref = np.array(X_ref)
#         X_cur = np.array(X_cur)
        
#         diff_vectors = []
        
#         if self.by_class:
#             if y_ref is None or y_cur is None:
#                 raise ValueError("Labels y_ref and y_cur are required when by_class=True")
                
#             y_ref = np.array(y_ref)
#             y_cur = np.array(y_cur)
            
#             # Find common classes
#             classes = np.intersect1d(np.unique(y_ref), np.unique(y_cur))
            
#             for c in classes:
#                 X_ref_c = X_ref[y_ref == c]
#                 X_cur_c = X_cur[y_cur == c]
                
#                 # Check consistency - need at least 2 samples to compute std
#                 if len(X_ref_c) < 2 or len(X_cur_c) < 2:
#                     continue
                    
#                 mean_ref = np.mean(X_ref_c, axis=0)
#                 std_ref = np.std(X_ref_c, axis=0)
#                 mean_cur = np.mean(X_cur_c, axis=0)
#                 std_cur = np.std(X_cur_c, axis=0)
                
#                 diff_vectors.append(mean_cur - mean_ref)
#                 diff_vectors.append(std_cur - std_ref)
                
#         else:
#             # Global drift
#             mean_ref = np.mean(X_ref, axis=0)
#             std_ref = np.std(X_ref, axis=0)
#             mean_cur = np.mean(X_cur, axis=0)
#             std_cur = np.std(X_cur, axis=0)

#             # print("Means", mean_ref)
#             # print("Std", std_ref)
            
#             if self.add_anchor_point:
#                 diff_vectors.append(np.zeros(len(mean_ref))) # Origin anchor (no drift)

#             diff_vectors.append(mean_cur - mean_ref)
#             diff_vectors.append(std_cur - std_ref)
            
#         diff_vectors = np.array(diff_vectors)
#         # print(diff_vectors)
        
#         # Handle edge case: no variance in drift vectors (e.g. no change)
#         if np.all(diff_vectors == 0) or len(diff_vectors) == 0:
#             # Fallback to standard PCA on X_ref if no drift detected
#             logger.warning("No drift detected or empty diff vectors. Falling back to standard PCA on reference data.")
#             self.pca.fit(X_ref)
#         else:
#              # If we have fewer diff vectors than components, we might run into issues with full PCA
#              # But usually sklearn handles n_samples < n_components
#             self.pca.fit(diff_vectors)
            
#         # print("Explained variance ratio:", self.pca.explained_variance_ratio_)
#         self.components_ = self.pca.components_
#         self.diff_vectors = diff_vectors # for mean and std vectors
#         return self
        
#     def transform(self, X):
#         return self.pca.transform(X)

#     def inverse_transform(self, X):
#         """Transform data back to its original space."""
#         return self.pca.inverse_transform(X)
