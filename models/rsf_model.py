"""
Random Survival Forest Implementation

Features:
- Standard RSF training
- Out-of-bag error estimation
- Feature importance
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sksurv.ensemble import RandomSurvivalForest
from sklearn.inspection import permutation_importance
from .base_model import BaseSurvivalModel

class RSFModel(BaseSurvivalModel):
    def __init__(self, config=None):
        super().__init__(config)

    def predict(self, X, output_type=None):
        """
        Make predictions

        Parameters
        ----------
        X : array-like
            Features
        output_type : str, optional
            One of:
            - None: Returns risk scores
            - 'survival_function': Returns survival function
            - 'mean_survival_time': Returns mean survival times
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        if output_type is None:
            return super().predict_model(X)
        elif output_type == 'survival_function':
            return self.predict_survival_function(X)
        elif output_type == 'mean_survival_time':
            return self.model.predict_survival_function(X)
        else:
            raise ValueError(f"Unknown output type: {output_type}")

    def get_feature_importance(self, X, y, n_repeats=5,
                             method='permutation'):
        """
        Get feature importance

        Parameters
        ----------
        X : array-like
            Features
        y : structured array
            Survival data
        n_repeats : int
            Number of permutation repeats
        method : str
            One of:
            - 'permutation': Permutation importance
            - 'impurity': Impurity-based importance (faster)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting importance")

        if method == 'permutation':
            result = permutation_importance(
                self.model, X, y,
                n_repeats=n_repeats,
                random_state=42
            )
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std
            }).sort_values('importance_mean', ascending=False)

        elif method == 'impurity':
            importance = self.model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance_mean': importance,
                'importance_std': np.zeros_like(importance)
            }).sort_values('importance_mean', ascending=False)

        else:
            raise ValueError(f"Unknown importance method: {method}")

        return importance_df

    def get_cv_summary(self):
        """Get summary of cross-validation results"""
        if self.cv_results_ is None:
            raise ValueError("No CV results available. Run fit_model with CV first.")

        summary = {
            'mean_score': self.cv_results_['mean_score'],
            'std_score': self.cv_results_['std_score'],
            'best_params': self.cv_results_['best_params'],
            'cohort_scores': {
                result['test_cohort']: result['test_score']
                for result in self.cv_results_['fold_results']
            }
        }
        return summary