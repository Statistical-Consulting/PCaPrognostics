import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class PCADimensionReduction:
    """PCA-based dimension reduction with variance threshold"""

    def __init__(self, variance_threshold=0.95, n_components=None, random_state=42):
        self.variance_threshold = variance_threshold
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.n_components_selected = None
        self.is_fitted = False
        self.feature_names_ = None

    def fit(self, X):
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)

        # First fit PCA with all components if using variance threshold
        if self.n_components is None:
            initial_pca = PCA(random_state=self.random_state)
            initial_pca.fit(X_scaled)

            # Find number of components needed for threshold
            cumsum = np.cumsum(initial_pca.explained_variance_ratio_)
            self.n_components_selected = np.argmax(cumsum >= self.variance_threshold) + 1

            logger.info(f"Selected {self.n_components_selected} components explaining "
                        f"{cumsum[self.n_components_selected - 1] * 100:.1f}% of variance")
        else:
            self.n_components_selected = self.n_components

        # Fit final PCA with selected number of components
        self.pca = PCA(n_components=self.n_components_selected,
                       random_state=self.random_state)
        self.pca.fit(X_scaled)

        # Generate feature names for transformed data
        self.feature_names_ = [f'PC{i + 1}' for i in range(self.n_components_selected)]

        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before transforming data")

        # Scale the data
        X_scaled = self.scaler.transform(X)

        # Apply PCA transformation
        X_transformed = self.pca.transform(X_scaled)

        # Convert to DataFrame with PC names
        return pd.DataFrame(
            X_transformed,
            index=X.index,
            columns=self.feature_names_
        )

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_feature_loadings(self):
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before getting feature loadings")

        return pd.DataFrame(
            self.pca.components_.T,
            columns=self.feature_names_,
            index=self.feature_names_original if hasattr(self, 'feature_names_original')
            else None
        )