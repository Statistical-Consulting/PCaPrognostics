
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sklearn.pipeline import Pipeline
import logging
from .base_model import BaseSurvivalModel

logger = logging.getLogger(__name__)


class RSFModel(BaseSurvivalModel):
    """Random Survival Forest Implementation"""

    def __init__(self):
        super().__init__()
        self.use_pipeline = True

    def _get_default_pipeline(self):
        """Create default pipeline with standard scaler and RSF"""
        return [
            ('scaler', StandardScaler()),
            ('rsf', RandomSurvivalForest(
                n_estimators=100,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=42
            ))
        ]

    def fit(self, X, y, data_container=None, pipeline_steps=None,
            params_cv=None, use_cohort_cv=True, n_splits_inner=5, **kwargs):
        """
        Fit Random Survival Forest with optional CV

        Parameters
        ----------
        X : array-like
            Features
        y : structured array
            Survival data
        data_container : DataContainer, optional
            Container with preprocessing functionality
        pipeline_steps : list, optional
            List of (name, transformer) tuples. If None, uses default pipeline
        params_cv : dict, optional
            Parameters for grid search, e.g.:
            {
                'rsf__n_estimators': [100, 200],
                'rsf__min_samples_split': [5, 10],
                'rsf__min_samples_leaf': [3, 5]
            }
        use_cohort_cv : bool, default=True
            Whether to use cohort-based CV
        n_splits_inner : int, default=5
            Number of inner CV splits if not cohort-based
        **kwargs : dict
            Additional parameters passed to base fit method
        """
        # Use default pipeline if none provided
        if pipeline_steps is None:
            pipeline_steps = self._get_default_pipeline()

        # Call parent fit method
        return super().fit(
            X=X,
            y=y,
            data_container=data_container,
            pipeline_steps=pipeline_steps,
            params_cv=params_cv,
            use_cohort_cv=use_cohort_cv,
            n_splits_inner=n_splits_inner,
            **kwargs
        )

    def get_feature_importance(self, feature_names=None):
        """Get feature importance from fitted model

        Parameters
        ----------
        feature_names : list, optional
            Names of features. If None, uses X0, X1, etc.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")

        # Get RSF from pipeline
        rsf = self.model.named_steps['rsf']
        importances = rsf.feature_importances_

        if feature_names is None:
            feature_names = [f'X{i}' for i in range(len(importances))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        return importance_df.sort_values('importance', ascending=False)