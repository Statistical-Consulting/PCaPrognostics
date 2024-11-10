import os
import numpy as np
from preprocessing.data_loader import DataLoader
from preprocessing.dimension_reduction import PCADimensionReduction
import logging

logger = logging.getLogger(__name__)


class DataContainer:
    """Container for data loading and preprocessing"""

    def __init__(self, data_config=None, project_root=None):
        """
        Parameters
        ----------
        data_config : dict, optional
            Configuration with options:
            - use_pca : bool
            - pca_threshold : float
            - gene_type : str
            - use_imputed : bool
            - validation_split : float
            - use_cohorts : bool
        project_root : str, optional
            Path to project root directory
        """
        # Default configuration
        self.config = {
            'use_pca': False,
            'pca_threshold': 0.95,
            'gene_type': 'intersection',
            'use_imputed': True,
            'validation_split': 0.2,
            'use_cohorts': True
        }

        # Update with provided config
        if data_config is not None:
            self.config.update(data_config)

        self.project_root = project_root
        self.pca = None
        self.groups = None

    def load_data(self):
        """Load and preprocess data"""
        try:
            if self.project_root is None:
                raise ValueError("project_root must be provided")

            logger.info("Loading data...")

            # Load data
            loader = DataLoader(self.project_root)
            X, pdata = loader.get_merged_data(
                gene_type=self.config['gene_type'],
                use_imputed=self.config['use_imputed']
            )

            # Convert to survival format
            y = loader.prepare_survival_data(pdata)

            # Extract cohort information
            self.groups = np.array([idx.split('.')[0] for idx in X.index])

            # Apply PCA if configured
            if self.config['use_pca']:
                logger.info("Applying PCA...")
                self.pca = PCADimensionReduction(
                    variance_threshold=self.config['pca_threshold']
                )
                X = self.pca.fit_transform(X)

            logger.info(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def get_groups(self):
        """Return cohort labels"""
        return self.groups

    def get_train_val_split(self, X, y):
        """Create train/validation split"""
        try:
            if self.config['use_cohorts'] and self.groups is not None:
                # Cohort-based split
                unique_cohorts = np.unique(self.groups)
                val_cohort = np.random.choice(unique_cohorts)
                val_mask = self.groups == val_cohort
                train_mask = ~val_mask

                logger.info(f"Using cohort {val_cohort} for validation")
            else:
                # Random split
                indices = np.random.permutation(len(X))
                split = int(len(X) * (1 - self.config['validation_split']))
                train_mask = indices[:split]
                val_mask = indices[split:]

                logger.info(f"Using random {self.config['validation_split'] * 100}% validation split")

            X_train = X[train_mask]
            y_train = y[train_mask]
            X_val = X[val_mask]
            y_val = y[val_mask]

            logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
            return X_train, y_train, X_val, y_val

        except Exception as e:
            logger.error(f"Error creating train/val split: {str(e)}")
            raise