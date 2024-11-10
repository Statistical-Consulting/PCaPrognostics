import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import os


class DimensionReduction:
    """Base class for dimension reduction methods"""

    def __init__(self):
        self.is_fitted = False
        self.scaler = StandardScaler()

    def fit(self, X):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)


class PCADimensionReduction(DimensionReduction):
    """PCA-based dimension reduction"""

    def __init__(self, n_components=None, variance_threshold=0.95):
        super().__init__()
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pca = None

    def fit(self, X):
        # Scale data
        X_scaled = self.scaler.fit_transform(X)

        # If n_components not specified, use variance threshold
        if self.n_components is None:
            self.pca = PCA()
            self.pca.fit(X_scaled)
            cumsum = np.cumsum(self.pca.explained_variance_ratio_)
            self.n_components = np.argmax(cumsum >= self.variance_threshold) + 1

        # Fit PCA with determined number of components
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_scaled)
        self.is_fitted = True

        # Store explained variance information
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.explained_variance_ = self.pca.explained_variance_

        return self

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before transform")
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)

    def get_feature_names(self):
        """Returns feature names for the transformed data"""
        return [f'PC{i + 1}' for i in range(self.n_components)]