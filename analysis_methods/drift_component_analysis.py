from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DriftComponentAnalysis:
    """
    Drift-Oriented PCA implementation utilizing TruncatedSVD.
    Computes diff between mean and std vectors for all classes
    and performs SVD on those difference vectors.
    """
    def __init__(self, n_components=2, by_class=False):
        self.n_components = n_components
        self.by_class = by_class
        self.pca = TruncatedSVD(n_components=n_components, random_state=42)
        self.diff_vectors = None
        self.classes_ = None
        
    def fit(self, X_ref, X_cur, y_ref, y_cur):
        X_ref = np.array(X_ref)
        X_cur = np.array(X_cur)
        y_ref = np.array(y_ref)
        y_cur = np.array(y_cur)
        
        diff_vectors = []
        classes_computed = []
        if self.by_class:
            classes = np.intersect1d(np.unique(y_ref), np.unique(y_cur))
                
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
                classes_computed.append(c)
        else:
            mean_ref = np.mean(X_ref, axis=0)
            std_ref = np.std(X_ref, axis=0)
            mean_cur = np.mean(X_cur, axis=0)
            std_cur = np.std(X_cur, axis=0)
            
            diff_vectors.append(mean_cur - mean_ref)
            diff_vectors.append(std_cur - std_ref)
            
        self.diff_vectors = np.array(diff_vectors)
        self.classes_ = np.array(classes_computed) if self.by_class else None
        
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
        Returns the GLOBAL explained energy ratio relative to the total matrix energy.
        """
        if self.diff_vectors is None or np.all(self.diff_vectors == 0):
            return np.zeros(self.n_components)
            
        # Capture the energy of the 2 visualized components
        visualized_energy = self.pca.singular_values_ ** 2
        
        # Capture the absolute total energy of the entire drift matrix (Frobenius Norm squared)
        global_total_energy = np.sum(self.diff_vectors ** 2)
        
        if global_total_energy == 0:
            return np.zeros_like(visualized_energy)
            
        return visualized_energy / global_total_energy


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
        return self.pca.singular_values_
