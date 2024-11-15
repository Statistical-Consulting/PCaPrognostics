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
        """Initialize RSF with default pipeline"""
        super().__init__()

        # Define default pipeline steps
        self.pipeline_steps = [
            ('scaler', StandardScaler()),
            ('rsf', RandomSurvivalForest(
                n_estimators=100,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=42
            ))
        ]

    def set_params(self, **params):
        """Set parameters for the RSF model

        This method allows setting parameters for the RSF estimator directly
        without needing to use the 'rsf__' prefix
        """
        if hasattr(self, 'model') and self.model is not None:
            self.model.named_steps['rsf'].set_params(**params)
        else:
            # If model isn't fitted yet, update the pipeline_steps
            rsf_params = self.pipeline_steps[1][1].get_params()
            rsf_params.update(params)
            self.pipeline_steps[1] = ('rsf', RandomSurvivalForest(**rsf_params))
        return self

    def fit(self, X, y, data_container=None, params_cv=None,
            use_cohort_cv=True, n_splits_inner=5, **kwargs):
        """Fit Random Survival Forest with optional CV

        Parameters
        ----------
        X : array-like
            Features
        y : structured array
            Survival data
        data_container : DataContainer, optional
            Container with preprocessing functionality
        params_cv : dict, optional
            Parameters for grid search, e.g.:
            {
                'n_estimators': [100, 200],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [3, 5]
            }
            Note: No need to prefix with 'rsf__', this is handled automatically
        use_cohort_cv : bool, default=True
            Whether to use cohort-based CV
        n_splits_inner : int, default=5
            Number of inner CV splits if not cohort-based
        **kwargs : dict
            Additional parameters passed to base fit method
        """
        logger.info("Starting RSF training...")
        logger.info(f"Parameter grid: {params_cv}")

        return super().fit(
            X=X,
            y=y,
            data_container=data_container,
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
            Names of features. If None, uses feature indices

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

    def predict_survival_function(self, X):
        """Predict survival function for samples in X

        Parameters
        ----------
        X : array-like
            Features to predict for

        Returns
        -------
        numpy.ndarray
            Predicted survival functions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        return self.model.named_steps['rsf'].predict_survival_function(X)