import os
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from utils.evaluation import cindex_score
from utils.resampling import NestedResamplingCV
from preprocessing.data_container import DataContainer
import logging

logger = logging.getLogger(__name__)


class BaseSurvivalModel(BaseEstimator):
    """Base class for all survival models.

    Provides common functionality for both pipeline-based models (like RSF)
    and directly trained models (like DeepSurv)
    """

    def __init__(self):
        self.is_fitted = False
        self.model = None
        self.cv_results_ = None
        self.data_container = None

    def fit(self, X, y, data_container=None, **kwargs):

        try:
            logger.info("Starting model training...")
            logger.info(f"Input data shape: X={X.shape}")

            # Store or create DataContainer
            self.data_container = data_container or DataContainer()

            # Get cohort information if available
            groups = self.data_container.get_groups() if data_container else None

            # Train model based on type
            if hasattr(self, 'use_pipeline') and self.use_pipeline:
                # Pipeline based models (RSF)
                self._fit_pipeline(X, y, groups=groups, **kwargs)
            else:
                # Direct models (DeepSurv)
                X_train, y_train, X_val, y_val = self.data_container.get_train_val_split(X, y)
                validation_data = (X_val, y_val) if X_val is not None else None
                self._fit_direct(X_train, y_train, validation_data=validation_data, **kwargs)

            self.is_fitted = True
            logger.info("Model training completed successfully")
            return self

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise

    def _fit_pipeline(self, X, y, groups=None, pipeline_steps=None, params_cv=None,
                      use_cohort_cv=True, n_splits_inner=5, refit=True):
        """Pipeline-based training implementation (RSF).


        """
        if pipeline_steps is None:
            raise ValueError("pipeline_steps required for pipeline-based models")

        # Create pipeline
        pipe = Pipeline(pipeline_steps)

        if params_cv:
            # Nested CV with parameter search
            cv = NestedResamplingCV(
                n_splits_inner=n_splits_inner,
                use_cohort_cv=use_cohort_cv,
                random_state=42
            )

            self.cv_results_ = cv.fit(
                estimator=pipe,
                X=X,
                y=y,
                groups=groups,
                param_grid=params_cv,
                scoring=cindex_score
            )

            if refit:
                # Refit on full data with best parameters
                best_params = self.cv_results_['best_params']
                pipe.set_params(**best_params)
                pipe.fit(X, y)
        else:
            # Simple fit without CV
            pipe.fit(X, y)

        self.model = pipe

    def _fit_direct(self, X, y, validation_data=None, **kwargs):
        """Direct training implementation (DeepSurv).

        """
        raise NotImplementedError

    def predict(self, X):
        """Make predictions for new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")
        return self.model.predict(X)

    def save(self, path, fname):
        """Save model and results.


        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        # Create directories
        model_dir = os.path.join(path, 'model')
        results_dir = os.path.join(path, 'results')
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Save model
        if hasattr(self, 'use_pipeline') and self.use_pipeline:
            # RSF: Save pipeline
            with open(os.path.join(model_dir, f"{fname}.pkl"), 'wb') as f:
                pickle.dump(self.model, f)
        else:
            # DeepSurv: Model specific save
            self._save_model(model_dir, fname)

        # Save CV results if available
        if hasattr(self, 'cv_results_'):
            results_file = os.path.join(results_dir, f"{fname}_cv_results.csv")
            pd.DataFrame(self.cv_results_).to_csv(results_file)
            logger.info(f"Saved CV results to {results_file}")

    def _save_model(self, path, fname):
        """Model-specific save implementation.

        """
        raise NotImplementedError

    def get_params(self, deep=True):
        """Get parameters for CV"""
        return {
            'use_nested_cv': self.use_nested_cv
        }

    def set_params(self, **parameters):
        """Set parameters for CV"""
        for parameter, value in parameters.items():
            if parameter == 'use_nested_cv':
                setattr(self, parameter, value)
            else:
                # Training parameters
                setattr(self, parameter, value)
        return self